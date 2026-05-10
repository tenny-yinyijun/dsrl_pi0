#!/usr/bin/env python
"""Generate a libero_processed-format trajectory test set with stratified
noise levels for evaluating world models.

For each trajectory we sample a noise scale ``sigma`` (round-robin from
``--noise-scales``) and run pi0/pi05 with ``noise = N(0, I) * sigma`` per
query step. Larger sigma => greater perturbation off the base policy =>
more failures, so the resulting set has a natural mix of successes and
failures across the requested scale grid.

Output layout (under ``<save-dir>/<name>/``)::

    annotation/<split>/<eid>.json         libero_processed annotation
    raw_videos/agentview/<eid>.mp4
    raw_videos/wrist/<eid>.mp4
    <split>_sample.json                   sliding-window index
    manifest.json                         per-eid noise_scale + is_success

The output directory is byte-compatible with the layout the reward server
already consumes, so ``examples/scripts/eval_wm_on_testset.py`` (and the
existing reward_server / score_traj utilities) can score it directly.

Run from the dsrl_pi0 venv with libero env vars set::

    export DISPLAY=:0
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID=0
    source .venv/bin/activate
    python examples/scripts/collect_wm_testset.py \\
        --save-dir /n/fs/.../wm_testsets \\
        --name libero90_task57_v1 \\
        --num-trajs 40 \\
        --noise-scales 0.5,1.0,2.0,3.0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Match data_collection_sim.py: enable Triton GEMM before importing jax.
_xla_flags = os.environ.get("XLA_FLAGS", "")
_xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags

# Ensure repo root is on sys.path before any `from examples.*` imports
# (the venv ships a stub `examples` namespace that shadows ours otherwise).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

from examples.train_utils_sim import (
    _quat2axisangle,
    obs_to_pi_zero_input,
)
from examples.train_utils_collect import (
    append_sample_index,
    find_next_episode_id,
    save_traj_libero_processed,
)


# ----- per-policy noise chunk shape (action_horizon, action_dim=32) ------- #
_NOISE_SHAPE = {
    "pi0": (50, 32),
    "pi05": (10, 32),
}


def _get_libero_env(task, resolution: int, seed: int):
    """Mirror examples.train_sim._get_libero_env without the heavy imports
    that pull in the SAC stack."""
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language, pathlib.Path(task.bddl_file).stem


def _settle(env, n_steps: int):
    """Run a few zero-action steps so dropped objects come to rest before
    we start recording. Mirrors examples.train_utils_collect's behavior."""
    if n_steps <= 0:
        return
    zero_action = np.zeros(7, dtype=np.float32)
    obs = None
    for _ in range(n_steps):
        obs, _, _, _ = env.step(zero_action)
    return obs


def _record_obs(obs, raw_agentview, wrist_images, raw_state_list):
    # H-flip only — matches open-world's preprocess_libero_for_wm.py
    # convention (what the WM was trained on). The 180° rotation π₀
    # expects is applied at the policy boundary, not in saved frames.
    raw_agentview.append(
        np.ascontiguousarray(obs["agentview_image"][::-1]).copy()
    )
    wrist_images.append(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]).copy()
    )
    cart = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]))
    ).astype(np.float32)
    grip = float(np.mean(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)))
    raw_state_list.append({"cartesian_position": cart, "gripper_position": grip})


def collect_one(
    *,
    env,
    agent_dp,
    task_description: str,
    bddl_name: str,
    task_suite: str,
    task_id: int,
    noise_scale: float,
    noise_mode: str,
    query_freq: int,
    max_timesteps: int,
    settle_steps: int,
    rng,
    noise_shape,
    cam_resolution: int,
    np_rng: np.random.Generator,
):
    """One rollout with noise applied at the location set by ``noise_mode``.

    noise_mode:
        "seed"   — multiply the diffusion-ODE seed by sigma (mild; the policy
                   is largely invariant to seed scale).
        "action" — keep a clean N(0, I) seed; add iid N(0, sigma^2 I) to each
                   executed env action (most disruptive per unit sigma).
        "obs"    — keep a clean seed; add N(0, sigma^2 I) to the policy's
                   state input at every query step (lies to the policy).
    """
    obs = env.reset()
    obs = _settle(env, settle_steps) or obs

    raw_agentview, wrist_images, raw_state_list = [], [], []
    rewards = []
    actions = None
    reward = 0.0
    done = False

    # Build a lightweight "variant" namespace for obs_to_pi_zero_input.
    class _V:
        env = "libero"
        cam_resolution = 0  # unused
    _v = _V()
    _v.cam_resolution = cam_resolution
    _v.task_description = task_description

    seed_sigma = float(noise_scale) if noise_mode == "seed" else 1.0

    for t in tqdm(range(max_timesteps), leave=False):
        if t % query_freq == 0:
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (1, *noise_shape)) * seed_sigma
            obs_pi_zero = obs_to_pi_zero_input(obs, _v)
            if noise_mode == "obs":
                state = obs_pi_zero["observation/state"]
                state = state + np_rng.normal(
                    0.0, float(noise_scale), size=state.shape
                ).astype(state.dtype)
                obs_pi_zero["observation/state"] = state
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

        # Snapshot frames + raw state at each env step (pre-step).
        _record_obs(obs, raw_agentview, wrist_images, raw_state_list)

        action_t = np.asarray(actions[t % query_freq])
        if noise_mode == "action":
            action_t = action_t + np_rng.normal(
                0.0, float(noise_scale), size=action_t.shape
            ).astype(action_t.dtype)
        obs, reward, done, _ = env.step(action_t)
        rewards.append(reward)
        if done:
            break

    # Trailing observation (post-step) so frame count matches state count.
    _record_obs(obs, raw_agentview, wrist_images, raw_state_list)

    is_success = (reward == 1)
    env_steps = len(rewards)
    episode_return = float(np.sum(np.asarray(rewards, dtype=np.float32)))

    traj = {
        "raw_agentview": raw_agentview,
        "wrist_images": wrist_images,
        "state_list": raw_state_list,
        "episode_return": episode_return,
        "is_success": bool(is_success),
        "env_steps": int(env_steps),
        "task_description": task_description,
        "task_suite": task_suite,
        "task_id": int(task_id),
        "bddl": bddl_name,
    }
    return traj, rng


def parse_noise_scales(s: str) -> list[float]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("--noise-scales must contain at least one value")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-dir", required=True,
                        help="Parent directory for test sets.")
    parser.add_argument("--name", required=True,
                        help="Subdir name; final test set lives at "
                             "<save-dir>/<name>/")
    parser.add_argument("--num-trajs", type=int, required=True,
                        help="Total trajectories to collect across all "
                             "noise scales (round-robin).")
    parser.add_argument("--noise-scales", default="0.5,1.0,2.0,3.0",
                        help="Comma-separated sigmas. Larger = more "
                             "perturbation. Default mixes well-behaved "
                             "to chaotic. Meaningful range depends on "
                             "--noise-mode (seed: O(1-50), action: O(0.05-1.0), "
                             "obs: O(0.02-0.3)).")
    parser.add_argument("--noise-mode", default="seed",
                        choices=["seed", "action", "obs"],
                        help="Where to inject noise. 'seed' scales the "
                             "diffusion-ODE seed (mild — the policy is "
                             "largely invariant to seed scale). 'action' "
                             "adds iid gaussian to executed env actions "
                             "(most disruptive per unit sigma). 'obs' "
                             "perturbs the policy's state input at each "
                             "query step.")
    parser.add_argument("--policy", default="pi05", choices=["pi0", "pi05"])
    parser.add_argument("--task-suite", default="libero_90")
    parser.add_argument("--task-id", type=int, default=57)
    parser.add_argument("--cam-resolution", type=int, default=256)
    parser.add_argument("--max-timesteps", type=int, default=400)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--query-freq", type=int, default=-1,
                        help="Default = chunk length for the policy "
                             "(10 for pi05, 50 for pi0).")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--sample-start-offset", type=int, default=6)
    parser.add_argument("--split", default="train",
                        help="Annotation split name. The reward server "
                             "looks under 'train' and 'val'; 'train' is "
                             "the conventional choice for an eval set.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    noise_scales = parse_noise_scales(args.noise_scales)
    noise_shape = _NOISE_SHAPE[args.policy]
    chunk_len = noise_shape[0]
    query_freq = args.query_freq if args.query_freq > 0 else chunk_len
    if query_freq > chunk_len:
        raise ValueError(
            f"query_freq={query_freq} exceeds policy chunk length {chunk_len}")

    out_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[testset] writing to {out_dir}")
    print(f"[testset] policy={args.policy}  task={args.task_suite}/{args.task_id}  "
          f"noise_mode={args.noise_mode}  noise_scales={noise_scales}  "
          f"num_trajs={args.num_trajs}")

    # Load benchmark + task.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)

    # Load policy.
    if args.policy == "pi05":
        config = openpi_config.get_config("pi05_libero")
        ckpt_dir = download.maybe_download(
            "gs://openpi-assets/checkpoints/pi05_libero")
    else:
        config = openpi_config.get_config("pi0_libero")
        ckpt_dir = download.maybe_download(
            "s3://openpi-assets/checkpoints/pi0_libero")
    print(f"[testset] loading {args.policy} from {ckpt_dir}")
    agent_dp = policy_config.create_trained_policy(config, ckpt_dir)

    # Resume support: pick up where a prior run left off.
    next_eid = find_next_episode_id(out_dir, args.split)
    print(f"[testset] starting from episode_id={next_eid}")

    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "name": args.name,
            "policy": args.policy,
            "task_suite": args.task_suite,
            "task_id": int(args.task_id),
            "noise_mode": args.noise_mode,
            "noise_scales": noise_scales,
            "fps": int(args.fps),
            "entries": [],
        }

    rng = jax.random.PRNGKey(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))

    n_done = 0
    while n_done < args.num_trajs:
        sigma = noise_scales[n_done % len(noise_scales)]
        # Vary env seed per episode so initial conditions differ across
        # the test set (otherwise repeated trajs would be near-duplicates).
        env_seed = int(args.seed) + next_eid + n_done
        env, task_description, bddl_name = _get_libero_env(
            task, args.cam_resolution, env_seed)

        try:
            print(f"[testset] eid={next_eid:06d}  mode={args.noise_mode}  "
                  f"sigma={sigma}  env_seed={env_seed}")
            traj, rng = collect_one(
                env=env,
                agent_dp=agent_dp,
                task_description=task_description,
                bddl_name=bddl_name,
                task_suite=args.task_suite,
                task_id=args.task_id,
                noise_scale=sigma,
                noise_mode=args.noise_mode,
                query_freq=query_freq,
                max_timesteps=args.max_timesteps,
                settle_steps=args.settle_steps,
                rng=rng,
                noise_shape=noise_shape,
                cam_resolution=args.cam_resolution,
                np_rng=np_rng,
            )
        finally:
            env.close()

        save_traj_libero_processed(
            traj, out_dir, next_eid, split=args.split,
            fps=int(args.fps), encoder=None,
        )
        append_sample_index(
            out_dir, args.split, next_eid,
            num_frames=len(traj["raw_agentview"]),
            stride=int(args.sample_stride),
            start_offset=int(args.sample_start_offset),
        )

        manifest["entries"].append({
            "eid": f"{next_eid:06d}",
            "noise_mode": args.noise_mode,
            "noise_scale": float(sigma),
            "is_success": bool(traj["is_success"]),
            "env_steps": int(traj["env_steps"]),
            "env_seed": env_seed,
        })
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[testset] saved eid={next_eid:06d}  steps={traj['env_steps']}  "
              f"success={traj['is_success']}")

        next_eid += 1
        n_done += 1

    # Quick summary.
    by_sigma: dict[float, list[bool]] = {}
    for e in manifest["entries"][-args.num_trajs:]:
        by_sigma.setdefault(float(e["noise_scale"]), []).append(bool(e["is_success"]))
    print("\n[testset] summary of just-collected trajectories:")
    for s in sorted(by_sigma):
        succ = by_sigma[s]
        print(f"  sigma={s}  n={len(succ)}  success_rate={np.mean(succ):.2f}")
    print(f"[testset] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
