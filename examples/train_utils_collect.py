"""Continuous data-collection loop with periodic reward-model + policy updates.

This module is a thin extension of ``examples.train_utils_sim``:

* ``collect_traj_continuous`` is a variant of ``collect_traj`` that

  - can skip ``env.reset()`` and continue from a carried obs, and
  - additionally records the wrist image and the raw libero state we need
    to write the per-episode annotation in the ``libero_processed`` layout.

* ``data_collection_loop`` runs forever: every Y (= ``reward_update_freq``)
  trajectories it trains the reward model, re-labels rewards, saves the Y
  rollouts to disk, and runs ``len(rewards) * multi_grad_step`` SAC updates
  per trajectory. Scene reset is decoupled from Y via ``scene_reset_freq``.

Saved layout under ``variant.save_dir`` mirrors
``/n/fs/iromdata/project/open-world/data/libero_processed/<task_suite>/``::

    annotation/<split>/<eid>.json
    latent_videos/agentview/<eid>.pt   # (T, C, H, W) uint8 raw frames by default
    latent_videos/wrist/<eid>.pt
    <split>_sample.json                # appended index of {episode_id, frame_ids}

If the user later supplies a VAE encoder (``encoder`` argument), the raw
frames are replaced with the encoder output.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, Optional

import jax
import numpy as np
from tqdm import tqdm
import wandb

from examples.train_utils_sim import (
    _quat2axisangle,
    _relabel_buffer,
    add_online_data_to_buffer,
    obs_to_img,
    obs_to_pi_zero_input,
    obs_to_qpos,
    perform_control_eval,
)


# --------------------------------------------------------------------------- #
# Trajectory collection
# --------------------------------------------------------------------------- #
def collect_traj_continuous(variant, agent, env, i, agent_dp,
                            do_reset: bool = True,
                            carry_obs=None,
                            synthesize_sparse_reward: bool = True):
    """Roll out one trajectory.

    Parameters
    ----------
    do_reset:
        If True, call ``env.reset()`` at the start. If False, continue from
        ``carry_obs`` (which must be a valid observation from the previous
        rollout's last step).
    carry_obs:
        Observation to start from when ``do_reset`` is False. Required in
        that case.
    synthesize_sparse_reward:
        If True, fill ``rewards`` / ``masks`` with the sparse -1/0 scheme
        used by the SAC critic. If False, rewards are zeros and the caller
        is expected to overwrite them (e.g. with a learned reward model).

    Returns the same keys as ``collect_traj`` plus:
        ``wrist_images``  list[np.ndarray]   raw uint8 wrist frames (libero only)
        ``state_list``    list[dict]         {cartesian_position, gripper_position}
                                             per env step (libero only)
        ``done``          bool               whether the env returned done
        ``last_obs``      raw obs dict       last observation (for carry over)
    """
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)

    if do_reset or carry_obs is None:
        if 'libero' in variant.env:
            obs = env.reset()
            # Right after reset libero spawns objects above the table — they
            # then drop a few cm under gravity over the first physics ticks.
            # Without settling, the first recorded frame catches the drop and
            # objects appear to "jump". Advance a few zero-action env steps
            # so the scene is at rest before recording begins. libero's
            # default OSC controller is 7-dim (6 EEF delta + 1 gripper).
            settle_steps = int(getattr(variant, 'settle_steps', 10))
            if settle_steps > 0:
                zero_action = np.zeros(7, dtype=np.float32)
                for _ in range(settle_steps):
                    obs, _, _, _ = env.step(zero_action)
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
        else:
            raise NotImplementedError(variant.env)
    else:
        obs = carry_obs

    image_list = []         # resized DSRL pixels (matches existing collect_traj)
    raw_agentview_list = [] # full-resolution agentview frames for libero_processed
    wrist_image_list = []   # full-resolution wrist frames for libero_processed
    raw_state_list = []     # libero raw cartesian + gripper, per env step
    rewards = []
    action_list = []
    obs_list = []
    actions = None
    reward = 0.0
    done = False
    t = 0

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:
            assert agent_dp is not None
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            if i == 0:
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(
                    noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise_tail = np.repeat(
                    actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate(
                    [actions_noise, noise_tail], axis=0)[None]

            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)

        # Snapshot full-resolution observation pieces for libero_processed.
        if 'libero' in variant.env:
            raw_agentview_list.append(
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy())
            wrist_image_list.append(
                np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).copy())
            cart_pos = np.concatenate(
                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]))
            ).astype(np.float32)
            grip_pos = float(np.mean(np.asarray(obs["robot0_gripper_qpos"],
                                                dtype=np.float32)))
            raw_state_list.append({
                'cartesian_position': cart_pos,
                'gripper_position': grip_pos,
            })

        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated

        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # Trailing observation -------------------------------------------------- #
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    if 'libero' in variant.env:
        raw_agentview_list.append(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy())
        wrist_image_list.append(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).copy())
        cart_pos = np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]))
        ).astype(np.float32)
        grip_pos = float(np.mean(np.asarray(obs["robot0_gripper_qpos"],
                                            dtype=np.float32)))
        raw_state_list.append({
            'cartesian_position': cart_pos,
            'gripper_position': grip_pos,
        })

    rewards_arr = np.array(rewards)
    episode_return = float(np.sum(rewards_arr[rewards_arr != None]))
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: episode_return={episode_return}, '
          f'Success: {is_success}, env_done={done}')

    query_steps = len(action_list)
    if synthesize_sparse_reward:
        if is_success:
            r_out = np.concatenate([-np.ones(query_steps - 1), [0]])
            m_out = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            r_out = -np.ones(query_steps)
            m_out = np.ones(query_steps)
    else:
        r_out = np.zeros(query_steps, dtype=np.float32)
        m_out = np.ones(query_steps, dtype=np.float32)
        if is_success:
            m_out[-1] = 0.0

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': r_out,
        'masks': m_out,
        'is_success': bool(is_success),
        'episode_return': episode_return,
        'images': image_list,             # resized DSRL pixels (per env step)
        'raw_agentview': raw_agentview_list,  # full-res agentview (per env step)
        'wrist_images': wrist_image_list,     # full-res wrist (per env step)
        'state_list': raw_state_list,
        'env_steps': t + 1,
        'done': bool(done),
        'last_obs': obs,
        'task_description': str(getattr(variant, 'task_description', '')),
        'task_suite': str(getattr(variant, 'task_suite_name', '')),
        'task_id': int(getattr(variant, 'task_id', -1)),
        'bddl': str(getattr(variant, 'bddl_name', '')),
    }


# --------------------------------------------------------------------------- #
# libero_processed-format saving
# --------------------------------------------------------------------------- #
def find_next_episode_id(save_dir: str, split: str) -> int:
    ann_dir = os.path.join(save_dir, 'annotation', split)
    if not os.path.isdir(ann_dir):
        return 0
    existing = []
    for p in os.listdir(ann_dir):
        if p.endswith('.json'):
            stem = p[:-5]
            if stem.isdigit():
                existing.append(int(stem))
    return (max(existing) + 1) if existing else 0


def _write_mp4(path: str, frames_thwc_uint8: np.ndarray, fps: int = 20,
               codec: str = 'libx264', quality: int = 8):
    """Write (T, H, W, C) uint8 frames to ``path`` as an H.264 mp4.

    ``macro_block_size=1`` disables imageio-ffmpeg's automatic padding to
    multiples of 16, so any (H, W) — e.g. 64 or 256 — is encoded as-is.
    """
    import imageio.v2 as imageio
    if frames_thwc_uint8.dtype != np.uint8:
        frames_thwc_uint8 = frames_thwc_uint8.astype(np.uint8)
    writer = imageio.get_writer(
        path, fps=int(fps), codec=codec, quality=quality,
        macro_block_size=1, format='FFMPEG')
    try:
        for f in frames_thwc_uint8:
            writer.append_data(f)
    finally:
        writer.close()


def save_traj_libero_processed(traj: dict,
                               save_dir: str,
                               episode_id: int,
                               split: str = 'train',
                               fps: int = 20,
                               encoder: Optional[Callable[[np.ndarray], "torch.Tensor"]] = None):
    """Write one trajectory in the libero_processed layout.

    Always writes raw uint8 frames as (T, C, H, W) torch tensors under
    ``raw_videos/{agentview,wrist}/<eid>.pt``. When ``encoder`` is supplied,
    additionally writes its output (typically a VAE latent of shape
    (T, C', H', W')) to ``latent_videos/{agentview,wrist}/<eid>.pt``.

    The annotation JSON references both via ``raw_videos`` and
    ``latent_videos`` (the latter is empty when no encoder is configured).
    """
    import imageio.v2 as imageio  # ffmpeg-backed mp4 writer
    import torch  # local import; project uses jax primarily

    eid_str = f"{int(episode_id):06d}"
    ann_dir = os.path.join(save_dir, 'annotation', split)
    raw_agentview_dir = os.path.join(save_dir, 'raw_videos', 'agentview')
    raw_wrist_dir = os.path.join(save_dir, 'raw_videos', 'wrist')
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(raw_agentview_dir, exist_ok=True)
    os.makedirs(raw_wrist_dir, exist_ok=True)

    # ---- raw videos as mp4 (always) ----
    agentview_arr = np.stack(traj['raw_agentview'], axis=0)  # (T, H, W, C) uint8
    wrist_arr = np.stack(traj['wrist_images'], axis=0)
    _write_mp4(os.path.join(raw_agentview_dir, f'{eid_str}.mp4'),
               agentview_arr, fps=int(fps))
    _write_mp4(os.path.join(raw_wrist_dir, f'{eid_str}.mp4'),
               wrist_arr, fps=int(fps))
    raw_videos_meta = [
        {'video_path': f'raw_videos/agentview/{eid_str}.mp4', 'cam': 'agentview'},
        {'video_path': f'raw_videos/wrist/{eid_str}.mp4', 'cam': 'wrist'},
    ]

    # ---- latent videos (only if an encoder is provided) ----
    latent_videos_meta = []
    if encoder is not None:
        latent_agentview_dir = os.path.join(save_dir, 'latent_videos', 'agentview')
        latent_wrist_dir = os.path.join(save_dir, 'latent_videos', 'wrist')
        os.makedirs(latent_agentview_dir, exist_ok=True)
        os.makedirs(latent_wrist_dir, exist_ok=True)
        agentview_lat = encoder(agentview_arr)
        wrist_lat = encoder(wrist_arr)
        torch.save(agentview_lat, os.path.join(latent_agentview_dir, f'{eid_str}.pt'))
        torch.save(wrist_lat, os.path.join(latent_wrist_dir, f'{eid_str}.pt'))
        latent_videos_meta = [
            {'latent_video_path': f'latent_videos/agentview/{eid_str}.pt',
             'cam': 'agentview'},
            {'latent_video_path': f'latent_videos/wrist/{eid_str}.pt',
             'cam': 'wrist'},
        ]

    states = traj.get('state_list', [])
    cart = [list(map(float, s['cartesian_position'])) for s in states]
    grip = [float(s['gripper_position']) for s in states]

    annotation = {
        'texts': [traj.get('task_description', '')],
        'language_instruction': traj.get('task_description', ''),
        'task_suite': traj.get('task_suite', ''),
        'bddl': traj.get('bddl', ''),
        'fps': int(fps),
        'down_sample': 1,
        'observation.state.cartesian_position': cart,
        'observation.state.gripper_position': grip,
        'episode_return': float(traj.get('episode_return', 0.0)),
        'is_success': bool(traj.get('is_success', False)),
        'env_steps': int(traj.get('env_steps', 0)),
        'raw_videos': raw_videos_meta,
        'latent_videos': latent_videos_meta,
    }
    with open(os.path.join(ann_dir, f'{eid_str}.json'), 'w') as f:
        json.dump(annotation, f)


def append_sample_index(save_dir: str, split: str, episode_id: int,
                        num_frames: int, stride: int = 2,
                        start_offset: int = 6):
    """Append (episode_id, frame_ids) entries to ``<split>_sample.json``."""
    path = os.path.join(save_dir, f'{split}_sample.json')
    if os.path.exists(path):
        with open(path) as f:
            entries = json.load(f)
    else:
        entries = []
    eid_str = f"{int(episode_id):06d}"
    for fid in range(start_offset, num_frames, stride):
        entries.append({'episode_id': eid_str, 'frame_ids': [fid]})
    with open(path, 'w') as f:
        json.dump(entries, f)


# --------------------------------------------------------------------------- #
# Main collection loop
# --------------------------------------------------------------------------- #
def data_collection_loop(variant, agent, env, eval_env,
                         online_replay_buffer, wandb_logger,
                         shard_fn=None, agent_dp=None,
                         reward_learner=None, score_fn=None,
                         encoder=None,
                         perform_control_evals: bool = True):
    """Collect data continuously, train reward + policy every Y trajs.

    Knobs read from ``variant``:
      scene_reset_freq      X — call env.reset() every X trajectories.
      reward_update_freq    Y — train reward + run SAC updates every Y trajs.
      save_dir / save_split — where to write libero_processed-format data.
      max_trajs             stop after collecting this many trajectories.
      max_steps             stop after this many SAC updates (whichever first).
      fps, sample_stride, sample_start_offset — index-file controls.
      multi_grad_step, start_online_updates, log_interval, eval_interval,
      checkpoint_interval, traj_batch_size, reward_grad_steps,
      reward_relabel_buffer — same semantics as train_utils_sim.
    """
    replay_buffer_iterator = online_replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    save_dir = variant.save_dir
    save_split = getattr(variant, 'save_split', 'train')
    next_episode_id = find_next_episode_id(save_dir, save_split)
    print(f'[collect] resuming from episode_id={next_episode_id} in {save_dir}')

    scene_reset_freq = max(1, int(getattr(variant, 'scene_reset_freq', 1)))
    Y = max(1, int(getattr(variant, 'reward_update_freq',
                           getattr(variant, 'traj_batch_size', 1))))
    fps = int(getattr(variant, 'fps', 20))
    sample_stride = int(getattr(variant, 'sample_stride', 2))
    sample_start_offset = int(getattr(variant, 'sample_start_offset', 6))

    use_reward_model = (reward_learner is not None) and (score_fn is not None)

    pending_trajs = []
    num_trajs_collected = 0
    total_env_steps = 0
    i = 0
    last_obs = None  # carry obs forward when not resetting
    must_reset_after_eval = False

    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    max_trajs = int(getattr(variant, 'max_trajs', 10**9))

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps and num_trajs_collected < max_trajs:
            # Decide reset for this rollout.
            #   * Force reset if previous rollout terminated.
            #   * Force reset if eval just ran (it called env.reset internally).
            #   * Otherwise, reset only when num_trajs_collected % X == 0.
            do_reset = (
                last_obs is None
                or must_reset_after_eval
                or (num_trajs_collected % scene_reset_freq == 0)
            )
            must_reset_after_eval = False

            traj = collect_traj_continuous(
                variant, agent, env, i, agent_dp,
                do_reset=do_reset, carry_obs=last_obs,
                synthesize_sparse_reward=not use_reward_model,
            )
            num_trajs_collected += 1
            total_env_steps += traj['env_steps']
            last_obs = None if traj['done'] else traj.get('last_obs', None)

            # ----------------------------------------------------------- #
            # Per-trajectory disk save (raw frames always; latents if an
            # encoder is configured). Happens immediately so trajectories
            # are durable regardless of when the next reward/SAC update
            # runs.
            # ----------------------------------------------------------- #
            save_traj_libero_processed(
                traj, save_dir, next_episode_id, split=save_split,
                fps=fps, encoder=encoder)
            append_sample_index(
                save_dir, save_split, next_episode_id,
                num_frames=len(traj.get('raw_agentview', [])),
                stride=sample_stride,
                start_offset=sample_start_offset)
            print(f'[collect] saved episode_id={next_episode_id:06d} '
                  f'(env_steps={traj["env_steps"]}, success={traj["is_success"]}) '
                  f'to {save_dir}.')
            next_episode_id += 1

            pending_trajs.append(traj)
            if len(pending_trajs) < Y:
                print(f'[collect] buffered {len(pending_trajs)}/{Y} trajs '
                      f'for reward+policy update.')
                continue

            # ----------------------------------------------------------- #
            # Reward-model batch update (optional)
            # ----------------------------------------------------------- #
            if use_reward_model:
                targets = np.array(
                    [float(score_fn(t)) for t in pending_trajs],
                    dtype=np.float32)
                print(f'[reward] f-scores: mean={targets.mean():.4f} '
                      f'std={targets.std():.4f} min={targets.min():.4f} '
                      f'max={targets.max():.4f}')
                last_info = {}
                for _ in range(int(variant.reward_grad_steps)):
                    last_info = reward_learner.update(pending_trajs, targets)
                wandb_logger.log({
                    'reward_model/loss': last_info.get('reward_model/loss', 0.0),
                    'reward_model/pred_return_mean': last_info.get('reward_model/pred_return_mean', 0.0),
                    'reward_model/pred_return_std': last_info.get('reward_model/pred_return_std', 0.0),
                    'reward_model/target_return_mean': last_info.get('reward_model/target_return_mean', 0.0),
                    'reward_model/target_return_std': last_info.get('reward_model/target_return_std', 0.0),
                    'reward_model/updates': last_info.get('reward_model/updates', 0.0),
                    'reward_model/f_score_mean': float(targets.mean()),
                    'reward_model/f_score_std': float(targets.std()),
                }, step=i)

                # Re-label per-step rewards for the Y trajectories with r̂.
                for traj_k in pending_trajs:
                    r_hat = reward_learner.predict_per_step(traj_k)
                    T = len(traj_k['actions'])
                    is_succ = bool(traj_k.get('is_success', False))
                    masks = np.ones(T, dtype=np.float32)
                    if is_succ:
                        masks[-1] = 0.0
                    traj_k['rewards'] = np.asarray(r_hat[:T], dtype=np.float32)
                    traj_k['masks'] = masks

                if int(getattr(variant, 'reward_relabel_buffer', 0)):
                    _relabel_buffer(online_replay_buffer, reward_learner)
                    print('[reward] relabelled all transitions in replay buffer.')

            # ----------------------------------------------------------- #
            # (Trajectories were already saved to disk immediately after
            # collection — see above. The Y-batch path here only handles
            # the reward update, replay-buffer insertion, and SAC update.)
            # ----------------------------------------------------------- #
            # Insert into replay buffer + queue policy gradient steps
            # ----------------------------------------------------------- #
            gradsteps_acc = 0
            traj_id = 0
            for traj_k in pending_trajs:
                traj_id = online_replay_buffer._traj_counter
                add_online_data_to_buffer(variant, traj_k, online_replay_buffer)
                if variant.get('num_online_gradsteps_batch', -1) > 0:
                    gradsteps_acc += variant.num_online_gradsteps_batch
                else:
                    gradsteps_acc += len(traj_k['rewards']) * variant.multi_grad_step

            print(f'[collect] online buffer timesteps: {len(online_replay_buffer)}, '
                  f'num traj: {traj_id + 1}, total_env_steps: {total_env_steps}.')

            traj = pending_trajs[-1]  # for per-iter logging
            pending_trajs = []

            if len(online_replay_buffer) <= variant.start_online_updates:
                continue

            for _ in range(gradsteps_acc):
                if i == 0:
                    print('performing evaluation for initial checkpoint')
                    if perform_control_evals:
                        perform_control_eval(agent, eval_env, i, variant,
                                             wandb_logger, agent_dp)
                        must_reset_after_eval = True
                    if hasattr(agent, 'perform_eval'):
                        agent.perform_eval(variant, i, wandb_logger,
                                           online_replay_buffer,
                                           replay_buffer_iterator, eval_env)

                batch = next(replay_buffer_iterator)
                update_info = agent.update(batch)
                pbar.update()
                i += 1

                if i % variant.log_interval == 0:
                    update_info = {k: jax.device_get(v)
                                   for k, v in update_info.items()}
                    for k, v in update_info.items():
                        if v.ndim == 0:
                            wandb_logger.log({f'training/{k}': v}, step=i)
                        elif v.ndim <= 2:
                            wandb_logger.log_histogram(f'training/{k}', v, i)
                    wandb_logger.log({
                        'replay_buffer_size': len(online_replay_buffer),
                        'episode_return (exploration)': traj['episode_return'],
                        'is_success (exploration)': int(traj['is_success']),
                        'num_online_trajs': num_trajs_collected,
                        'env_steps': total_env_steps,
                        'episodes_saved': next_episode_id,
                    }, i)

                if i % variant.eval_interval == 0:
                    wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                    wandb_logger.log({'num_online_trajs': num_trajs_collected}, step=i)
                    wandb_logger.log({'env_steps': total_env_steps}, step=i)
                    if perform_control_evals:
                        perform_control_eval(agent, eval_env, i, variant,
                                             wandb_logger, agent_dp)
                        must_reset_after_eval = True
                    if hasattr(agent, 'perform_eval'):
                        agent.perform_eval(variant, i, wandb_logger,
                                           online_replay_buffer,
                                           replay_buffer_iterator, eval_env)

                if (variant.checkpoint_interval != -1
                        and i % variant.checkpoint_interval == 0):
                    agent.save_checkpoint(variant.outputdir, i,
                                          variant.checkpoint_interval)
