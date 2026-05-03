"""Reward model trained from user-supplied per-trajectory scores.

Loss = MSE between predicted trajectory return (sum of per-(query-step)
predictions) and the user-supplied target score. The reward predictor reuses
the same encoder + PixelMultiplexer architecture as the SAC critic.
"""
from functools import partial
from typing import Any, Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders.impala_encoder import (ImpalaEncoder,
                                                     SmallerImpalaEncoder)
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.resnet_encoderv1 import (ResNet18, ResNet34,
                                                       ResNetSmall)
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.networks.values.state_action_value import StateActionValue


class _RewardTrainState(train_state.TrainState):
    batch_stats: Any


def _build_encoder(encoder_type: str, cnn_features, cnn_strides, cnn_padding,
                   encoder_norm, use_spatial_softmax, softmax_temperature):
    if encoder_type == "small":
        return Encoder(cnn_features, cnn_strides, cnn_padding)
    if encoder_type == "impala":
        return ImpalaEncoder()
    if encoder_type == "impala_small":
        return SmallerImpalaEncoder()
    if encoder_type == "resnet_small":
        return ResNetSmall(norm=encoder_norm,
                           use_spatial_softmax=use_spatial_softmax,
                           softmax_temperature=softmax_temperature)
    if encoder_type == "resnet_18_v1":
        return ResNet18(norm=encoder_norm,
                        use_spatial_softmax=use_spatial_softmax,
                        softmax_temperature=softmax_temperature)
    if encoder_type == "resnet_34_v1":
        return ResNet34(norm=encoder_norm,
                        use_spatial_softmax=use_spatial_softmax,
                        softmax_temperature=softmax_temperature)
    if encoder_type == "resnet_small_v2":
        return ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
    if encoder_type == "resnet_18_v2":
        return ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
    if encoder_type == "resnet_34_v2":
        return ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
    raise ValueError(f"unknown encoder_type {encoder_type!r}")


@partial(jax.jit, static_argnames=("apply_fn",))
def _update_step(rng, state, apply_fn, obs, actions, mask, targets):
    K, T = mask.shape

    def flatten(x):
        return x.reshape((K * T,) + x.shape[2:])

    obs_flat = jax.tree_util.tree_map(flatten, obs)
    act_flat = flatten(actions)

    def loss_fn(params):
        if state.batch_stats is None:
            r_flat = apply_fn({"params": params}, obs_flat, act_flat,
                              training=True)
            new_batch_stats = None
        else:
            r_flat, mutated = apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                obs_flat, act_flat, training=True,
                mutable=["batch_stats"])
            new_batch_stats = mutated["batch_stats"]
        r = r_flat.reshape(K, T)
        pred_returns = (r * mask).sum(axis=1)
        loss = jnp.mean((pred_returns - targets) ** 2)
        return loss, (pred_returns, new_batch_stats)

    grads, (pred_returns, new_batch_stats) = jax.grad(
        loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        new_state = new_state.replace(batch_stats=new_batch_stats)
    loss = jnp.mean((pred_returns - targets) ** 2)
    info = {
        "reward_model/loss": loss,
        "reward_model/pred_return_mean": pred_returns.mean(),
        "reward_model/pred_return_std": pred_returns.std(),
        "reward_model/target_return_mean": targets.mean(),
        "reward_model/target_return_std": targets.std(),
    }
    return rng, new_state, info


@partial(jax.jit, static_argnames=("apply_fn",))
def _update_step_per_step(rng, state, apply_fn, obs, actions, mask, targets):
    """Per-(query-step) MSE: loss = mean over masked positions of
    ``(r̂(o_t, a_t) - target_t)^2``.

    Shapes:
      mask     (K, T_max)   1.0 where target_t is valid, 0.0 otherwise
      targets  (K, T_max)
    """
    K, T = mask.shape

    def flatten(x):
        return x.reshape((K * T,) + x.shape[2:])

    obs_flat = jax.tree_util.tree_map(flatten, obs)
    act_flat = flatten(actions)

    def loss_fn(params):
        if state.batch_stats is None:
            r_flat = apply_fn({"params": params}, obs_flat, act_flat,
                              training=True)
            new_batch_stats = None
        else:
            r_flat, mutated = apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                obs_flat, act_flat, training=True,
                mutable=["batch_stats"])
            new_batch_stats = mutated["batch_stats"]
        r = r_flat.reshape(K, T)
        sq = ((r - targets) ** 2) * mask
        denom = jnp.maximum(mask.sum(), 1.0)
        loss = sq.sum() / denom
        return loss, (r, new_batch_stats)

    grads, (r, new_batch_stats) = jax.grad(
        loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        new_state = new_state.replace(batch_stats=new_batch_stats)
    sq = ((r - targets) ** 2) * mask
    denom = jnp.maximum(mask.sum(), 1.0)
    loss = sq.sum() / denom
    # Reduce per-step preds + targets only over masked positions for logging.
    pred_sum = (r * mask).sum(axis=1)
    target_sum = (targets * mask).sum(axis=1)
    info = {
        "reward_model/loss": loss,
        "reward_model/pred_step_mean": (r * mask).sum() / denom,
        "reward_model/pred_step_std": jnp.sqrt(
            ((((r - (r * mask).sum() / denom)) ** 2) * mask).sum() / denom),
        "reward_model/target_step_mean": (targets * mask).sum() / denom,
        "reward_model/target_step_std": jnp.sqrt(
            ((((targets - (targets * mask).sum() / denom)) ** 2) * mask).sum() / denom),
        "reward_model/pred_return_mean": pred_sum.mean(),
        "reward_model/pred_return_std": pred_sum.std(),
        "reward_model/target_return_mean": target_sum.mean(),
        "reward_model/target_return_std": target_sum.std(),
        "reward_model/coverage": mask.mean(),
    }
    return rng, new_state, info


@partial(jax.jit, static_argnames=("apply_fn",))
def _predict_step(state, apply_fn, obs, actions):
    if state.batch_stats is None:
        return apply_fn({"params": state.params}, obs, actions, training=False)
    return apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        obs, actions, training=False)


class RewardLearner:
    """Per-step reward predictor trained on trajectory-return targets.

    The user provides a callable ``score(traj) -> float`` that maps a full
    trajectory dict (the dict returned by ``collect_traj``) to a scalar.
    This class regresses ``Σ_t r̂(o_t, a_t)`` to that scalar.
    """

    def __init__(
        self,
        seed: int,
        sample_obs: Union[jnp.ndarray, DatasetDict],
        sample_action: jnp.ndarray,
        max_traj_len: int,
        lr: float = 3e-4,
        hidden_dims: Sequence[int] = (128, 128, 128),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        encoder_type: str = "small",
        encoder_norm: str = "group",
        use_spatial_softmax: bool = True,
        softmax_temperature: float = -1,
        use_bottleneck: bool = True,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)

        encoder_def = _build_encoder(encoder_type, cnn_features, cnn_strides,
                                     cnn_padding, encoder_norm,
                                     use_spatial_softmax, softmax_temperature)

        # Single-output head — no Q-ensemble; we want a scalar reward.
        head_def = StateActionValue(hidden_dims=hidden_dims)
        reward_def = PixelMultiplexer(encoder=encoder_def,
                                      network=head_def,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck)

        init_out = reward_def.init(init_key, sample_obs, sample_action)
        params = init_out["params"]
        batch_stats = init_out.get("batch_stats", None)

        state = _RewardTrainState.create(
            apply_fn=reward_def.apply,
            params=params,
            tx=optax.adam(learning_rate=lr),
            batch_stats=batch_stats,
        )

        self._rng = rng
        self._state = state
        self._apply_fn = reward_def.apply
        self.max_traj_len = int(max_traj_len)
        self.num_updates = 0

    # ------------------------------------------------------------------
    # Trajectory packing / padding
    # ------------------------------------------------------------------
    @staticmethod
    def _stack_traj(traj, T_max):
        """Pack one trajectory's (obs, action, mask) up to length T_max.

        Returns (obs_dict_of_arrays, actions_arr, mask_arr) with leading
        dim T_max (zero-padded past the true length).
        """
        actions = np.asarray(traj["actions"])  # (T, *action_shape)
        T = actions.shape[0]
        obs_list = traj["observations"][:T]  # drop the trailing terminal obs
        # Each obs is a dict of arrays already in the per-step shape used by
        # the replay buffer (after the [0] indexing in collect_traj's output
        # we still have leading batch dim of 1; strip it here so shapes
        # match the SAC critic init).
        keys = list(obs_list[0].keys())
        obs_stacked = {}
        for k in keys:
            arrs = [np.asarray(o[k][0]) for o in obs_list]
            obs_stacked[k] = np.stack(arrs, axis=0)  # (T, *)

        # pad
        if T < T_max:
            pad_t = T_max - T
            for k, v in obs_stacked.items():
                pad = np.zeros((pad_t, *v.shape[1:]), dtype=v.dtype)
                obs_stacked[k] = np.concatenate([v, pad], axis=0)
            act_pad = np.zeros((pad_t, *actions.shape[1:]),
                               dtype=actions.dtype)
            actions = np.concatenate([actions, act_pad], axis=0)
        elif T > T_max:
            for k, v in obs_stacked.items():
                obs_stacked[k] = v[:T_max]
            actions = actions[:T_max]
            T = T_max

        mask = np.zeros((T_max,), dtype=np.float32)
        mask[: min(T, T_max)] = 1.0
        return obs_stacked, actions, mask

    def _pack_batch(self, trajs):
        T_max = self.max_traj_len
        obs_list, act_list, mask_list = [], [], []
        for tr in trajs:
            o, a, m = self._stack_traj(tr, T_max)
            obs_list.append(o)
            act_list.append(a)
            mask_list.append(m)

        obs_batch = {k: np.stack([o[k] for o in obs_list], axis=0)
                     for k in obs_list[0].keys()}
        act_batch = np.stack(act_list, axis=0)
        mask_batch = np.stack(mask_list, axis=0)
        return obs_batch, act_batch, mask_batch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, trajs, targets) -> Dict[str, float]:
        """One gradient step of MSE on (Σ r̂(o,a) - target)^2 over K trajs."""
        obs_batch, act_batch, mask_batch = self._pack_batch(trajs)
        targets = jnp.asarray(np.asarray(targets, dtype=np.float32))
        obs_batch = {k: jnp.asarray(v) for k, v in obs_batch.items()}
        act_batch = jnp.asarray(act_batch)
        mask_batch = jnp.asarray(mask_batch)
        new_rng, new_state, info = _update_step(
            self._rng, self._state, self._apply_fn,
            obs_batch, act_batch, mask_batch, targets)
        self._rng = new_rng
        self._state = new_state
        self.num_updates += 1
        info = {k: float(jax.device_get(v)) for k, v in info.items()}
        info["reward_model/updates"] = float(self.num_updates)
        return info

    def update_per_step(self, trajs, per_step_targets,
                        per_step_masks) -> Dict[str, float]:
        """One gradient step of masked per-(query-step) MSE.

        per_step_targets / per_step_masks: lists of length K, each entry a
        (T,) numpy array (T may differ per traj; the trajectory mask from
        ``_pack_batch`` further AND-ed with these so padded positions are
        zeroed out automatically).
        """
        obs_batch, act_batch, traj_mask = self._pack_batch(trajs)
        K, T_max = traj_mask.shape

        tgt = np.zeros((K, T_max), dtype=np.float32)
        msk = np.zeros((K, T_max), dtype=np.float32)
        for k, (t, m) in enumerate(zip(per_step_targets, per_step_masks)):
            t = np.asarray(t, dtype=np.float32)
            m = np.asarray(m, dtype=np.float32)
            L = min(t.shape[0], T_max)
            tgt[k, :L] = t[:L]
            msk[k, :L] = m[:L]
        # Zero out padded positions even if the user-supplied mask says 1.
        msk = msk * traj_mask

        obs_batch = {k: jnp.asarray(v) for k, v in obs_batch.items()}
        act_batch = jnp.asarray(act_batch)
        msk_j = jnp.asarray(msk)
        tgt_j = jnp.asarray(tgt)

        new_rng, new_state, info = _update_step_per_step(
            self._rng, self._state, self._apply_fn,
            obs_batch, act_batch, msk_j, tgt_j)
        self._rng = new_rng
        self._state = new_state
        self.num_updates += 1
        info = {k: float(jax.device_get(v)) for k, v in info.items()}
        info["reward_model/updates"] = float(self.num_updates)
        return info

    def predict_per_step(self, traj) -> np.ndarray:
        """Per-(query-step) reward predictions for one trajectory.

        Returns a numpy array of shape (T,) where T = len(traj['actions']).
        """
        actions = np.asarray(traj["actions"])
        T = actions.shape[0]
        obs_list = traj["observations"][:T]
        keys = list(obs_list[0].keys())
        obs = {k: np.stack([np.asarray(o[k][0]) for o in obs_list], axis=0)
               for k in keys}
        obs = {k: jnp.asarray(v) for k, v in obs.items()}
        actions = jnp.asarray(actions)
        r = _predict_step(self._state, self._apply_fn, obs, actions)
        return np.asarray(jax.device_get(r))
