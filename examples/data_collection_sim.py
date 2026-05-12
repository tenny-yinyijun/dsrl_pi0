#! /usr/bin/env python
"""Continuous data-collection driver.

Mirrors ``examples.train_sim`` but plugs in ``data_collection_loop`` from
``examples.train_utils_collect``. Adds knobs for scene-reset frequency (X),
reward+policy update frequency (Y), and on-disk libero_processed saving.
"""
import json
import os

# Match train_sim.py: enable Triton GEMM before importing jax.
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import copy
import importlib
import pathlib
import tempfile
from functools import partial

import jax
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Dict
import gym_aloha  # noqa: F401  (registers gym envs)
import gymnasium as gym

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jax.experimental.compilation_cache import compilation_cache
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.agents.reward_model import RewardLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

from examples.train_sim import DummyEnv, _get_libero_env, shard_batch
from examples.train_utils_collect import data_collection_loop


home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    tf.config.set_visible_devices([], "GPU")

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    os.makedirs(outputdir, exist_ok=True)
    print('writing to output dir ', outputdir)

    # Default save_dir into the experiment folder if user didn't set one.
    if not variant.save_dir:
        variant.save_dir = os.path.join(outputdir, 'collected_data')
    os.makedirs(variant.save_dir, exist_ok=True)
    print('saving collected trajectories to', variant.save_dir)

    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[variant.task_suite_name]()
        task = task_suite.get_task(variant.task_id)
        env, task_description = _get_libero_env(task, variant.cam_resolution,
                                                variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.bddl_name = pathlib.Path(task.bddl_file).stem
        variant.env_max_reward = 1
        variant.max_timesteps = 400

        # Optional: load a pre-generated instruction list (multitask
        # variant). Each rollout's prompt to π₀ will be sampled uniformly
        # from this list — see train_utils_collect.data_collection_loop
        # where variant.task_description is overwritten before each call
        # to collect_traj_continuous.
        instruction_list_path = getattr(variant, 'instruction_list', '') or ''
        if instruction_list_path:
            with open(instruction_list_path) as f:
                _ilist_raw = json.load(f)
            if isinstance(_ilist_raw, dict) and 'instructions' in _ilist_raw:
                _instructions = list(_ilist_raw['instructions'])
            elif isinstance(_ilist_raw, list):
                _instructions = list(_ilist_raw)
            else:
                raise ValueError(
                    f"--instruction_list {instruction_list_path}: expected a "
                    f"JSON list or an object with key 'instructions'.")
            if not _instructions:
                raise ValueError(
                    f"--instruction_list {instruction_list_path} contains no "
                    f"instructions.")
            variant.instruction_list_data = _instructions
            print(f'[multitask] loaded {len(_instructions)} instructions from '
                  f'{instruction_list_path}; per-rollout prompt sampling enabled.')
        else:
            variant.instruction_list_data = []
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0",
                       obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.task_description = ''
        variant.bddl_name = ''
        variant.env_max_reward = 4
        variant.max_timesteps = 400
    else:
        raise NotImplementedError(variant.env)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != '', variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir,
        group_name=group_name)

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    policy_variant = getattr(variant, 'policy', 'pi0')
    if variant.env == 'libero':
        if policy_variant == 'pi05':
            config = openpi_config.get_config("pi05_libero")
            checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
        else:
            config = openpi_config.get_config("pi0_libero")
            checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif variant.env == 'aloha_cube':
        if policy_variant == 'pi05':
            raise NotImplementedError(
                "pi05 does not have an aloha_sim config; use --policy pi0 or add a pi05 sim config."
            )
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError(variant.env)
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    online_buffer_size = variant.max_steps // max(variant.multi_grad_step, 1)
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space, dummy_env.action_space,
        int(online_buffer_size))
    online_replay_buffer.seed(variant.seed)

    reward_learner = None
    score_fn = None
    score_fn_request = None
    score_fn_await = None
    score_fn_request_wm_only = None
    if int(getattr(variant, 'use_reward_model', 0)):
        spec = variant.reward_fn
        if ':' not in spec:
            raise ValueError(
                f"--reward_fn must be of the form 'module:callable', got {spec!r}")
        mod_path, attr = spec.split(':', 1)
        _mod = importlib.import_module(mod_path)
        score_fn = getattr(_mod, attr)
        # Optional async pair (same module, attr suffixes _request / _await).
        # When both exist, the collection loop drops .req as soon as a traj
        # is saved (parallel with the next rollout) and awaits at the
        # Y-batch boundary. Falls back to sync score_fn(traj) when missing.
        score_fn_request = getattr(_mod, attr + '_request', None)
        score_fn_await = getattr(_mod, attr + '_await', None)
        if score_fn_request is None or score_fn_await is None:
            score_fn_request = None
            score_fn_await = None
        # Optional 'wm-only' sibling. When present and --scored_per_round
        # < round_size, the unscored fraction of trajectories drops
        # .wm_only markers so the server still adds them to its WM
        # fine-tune buffer without paying the LPIPS-rollout cost.
        score_fn_request_wm_only = getattr(
            _mod, attr + '_request_wm_only', None)
        reward_kwargs = dict(
            hidden_dims=kwargs.get('hidden_dims', (128, 128, 128)),
            cnn_features=kwargs.get('cnn_features', (32, 32, 32, 32)),
            cnn_strides=kwargs.get('cnn_strides', (2, 1, 1, 1)),
            cnn_padding=kwargs.get('cnn_padding', 'VALID'),
            latent_dim=kwargs.get('latent_dim', 50),
            encoder_type=kwargs.get('encoder_type', 'small'),
            encoder_norm=kwargs.get('encoder_norm', 'group'),
            use_spatial_softmax=kwargs.get('use_spatial_softmax', True),
            softmax_temperature=kwargs.get('softmax_temperature', -1),
            use_bottleneck=kwargs.get('use_bottleneck', True),
        )
        max_traj_len = (variant.max_timesteps // max(1, variant.query_freq)) + 1
        reward_learner = RewardLearner(
            seed=variant.seed + 7,
            sample_obs=sample_obs,
            sample_action=sample_action,
            max_traj_len=max_traj_len,
            lr=float(variant.reward_lr),
            **reward_kwargs,
        )
        Y = int(getattr(variant, 'reward_update_freq',
                        variant.traj_batch_size))
        print(f"[reward] using learned reward model. Y={Y} "
              f"reward_grad_steps={variant.reward_grad_steps} "
              f"max_traj_len={max_traj_len} score_fn={spec}")

    # Optional latent-video encoder (e.g. a VAE) for libero_processed-style
    # `latent_videos/{cam}/<eid>.pt` files. Raw frames are saved regardless.
    encoder = None
    enc_spec = getattr(variant, 'latent_encoder', '') or ''
    if enc_spec:
        if ':' not in enc_spec:
            raise ValueError(
                f"--latent_encoder must be 'module:callable', got {enc_spec!r}")
        mod_path, attr = enc_spec.split(':', 1)
        encoder = getattr(importlib.import_module(mod_path), attr)
        print(f"[collect] latent encoder: {enc_spec}")

    data_collection_loop(
        variant, agent, env, eval_env, online_replay_buffer,
        wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp,
        reward_learner=reward_learner, score_fn=score_fn,
        score_fn_request=score_fn_request,
        score_fn_await=score_fn_await,
        score_fn_request_wm_only=score_fn_request_wm_only,
        encoder=encoder,
    )
