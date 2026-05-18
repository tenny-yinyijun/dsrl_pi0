#!/usr/bin/env python
"""Generate a libero_processed-format trajectory test set with a fixed
LIBERO scene (task_suite + task_id) but a list of language instructions
fed to the policy.

For each instruction in --instruction-list we collect --trajs-per-instruction
trajectories (full grid). Each rollout uses a fresh env seed so initial
conditions differ. The instruction is written into the saved annotation's
``texts`` and ``language_instruction`` fields, so when ``eval_wm_multitask.py``
later replays the trajectory through a world model it conditions on the
exact prompt the policy saw.

This is a sibling of ``examples/scripts/collect_wm_testset.py``: same
on-disk layout, same VAE-free annotation, same sample-index file. The
only axis of variation here is the instruction (no noise injection).

Output layout (under ``<save-dir>/<name>/``)::

    annotation/<split>/<eid>.json
    raw_videos/agentview/<eid>.mp4
    raw_videos/wrist/<eid>.mp4
    <split>_sample.json                   sliding-window index
    manifest.json                         per-eid instruction + is_success

Run from the dsrl_pi0 venv with the LIBERO env vars set::

    export DISPLAY=:0
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID=0
    source .venv/bin/activate
    python examples/scripts/collect_wm_multitask_testset.py \\
        --save-dir /scratch/gpfs/AM43/yy4041/wm_testsets \\
        --name libero_goal_1_multitask_v1 \\
        --instruction-list examples/scripts/libero_goal_1_instructions.json \\
        --task-suite libero_goal --task-id 1 \\
        --trajs-per-instruction 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Match data_collection_sim.py: enable Triton GEMM before importing jax.
_xla_flags = os.environ.get("XLA_FLAGS", "")
_xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags

# The .venv ships a stub `examples` namespace package that shadows the
# repo's examples/ unless the repo root is on sys.path before the imports
# below. Mirror what collect_wm_testset.py does.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import numpy as np

from libero.libero import benchmark

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

# Reuse helpers from the noise-axis collector so we cannot drift apart on
# env construction, frame recording, or save-format conventions.
from examples.scripts.collect_wm_testset import (
    _NOISE_SHAPE,
    _get_libero_env,
    collect_one,
)
from examples.train_utils_collect import (
    append_sample_index,
    find_next_episode_id,
    save_traj_libero_processed,
)


def _load_instructions(path: str) -> list[str]:
    """Load instruction strings. Accepts either a bare JSON list or an
    object with an ``instructions`` array (matches
    ``examples/scripts/libero_goal_1_instructions.json``)."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "instructions" in raw:
        instrs = list(raw["instructions"])
    elif isinstance(raw, list):
        instrs = list(raw)
    else:
        raise ValueError(
            f"--instruction-list {path}: expected a JSON list or an object "
            f"with key 'instructions'.")
    if not instrs:
        raise ValueError(f"--instruction-list {path} is empty.")
    # De-duplicate while preserving order so the manifest's by-instruction
    # buckets are clean.
    seen = set()
    out = []
    for s in instrs:
        s = str(s).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--save-dir", required=True,
                        help="Parent directory for test sets.")
    parser.add_argument("--name", required=True,
                        help="Subdir name; final test set lives at "
                             "<save-dir>/<name>/")
    parser.add_argument("--instruction-list", required=True,
                        help="JSON file with the language instructions to "
                             "feed pi0/pi05. Either a bare list or an "
                             "object with key 'instructions' (same format "
                             "as examples/scripts/libero_goal_1_instructions.json).")
    parser.add_argument("--trajs-per-instruction", type=int, required=True,
                        help="Number of trajectories collected for EACH "
                             "instruction. Total = N_instructions * this.")
    parser.add_argument("--policy", default="pi05", choices=["pi0", "pi05"])
    parser.add_argument("--task-suite", default="libero_goal",
                        help="LIBERO suite providing the scene + success "
                             "criterion. Note that is_success is judged by "
                             "the BDDL goal, not the instruction text.")
    parser.add_argument("--task-id", type=int, default=1)
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
                             "looks under both 'train' and 'val'; 'train' "
                             "is the conventional choice for an eval set.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the (instruction, replica) schedule "
                             "with the seed instead of collecting all "
                             "trajectories of one instruction in a row. "
                             "Useful if a partial run will still cover the "
                             "full grid uniformly.")

    # ----- optional per-rollout perturbation -----
    # Modeled on the MolmoBot policy's `ActionNoiseConfig` (see
    # /scratch/gpfs/AM43/yy4041/aim/molmospaces/molmo_spaces/configs/robot_configs.py)
    # but reduced to the dial set the existing collect_one already
    # implements:
    #   --perturb-mode action  -> per-env-step gaussian noise on the
    #                              executed action (analogous to MolmoBot's
    #                              ActionNoiseConfig, just without the
    #                              TCP-bounded Jacobian mapping)
    #   --perturb-mode obs     -> gaussian noise on the policy's state
    #                              input at each query step (lies to pi0)
    # A per-rollout coin flip with prob --perturb-prob decides whether
    # *this* rollout gets noise; the other rollouts stay clean. This
    # mixes "clean" and "perturbed" trajectories in one testset so the
    # WM eval can bucket performance by perturbed vs. clean.
    parser.add_argument("--perturb-prob", type=float, default=0.0,
                        help="Per-rollout probability of injecting "
                             "perturbation. 0 = always clean (default); "
                             "0.5 = roughly half perturbed.")
    parser.add_argument("--perturb-sigma", type=float, default=0.02,
                        help="Noise std for perturbed rollouts. Sensible "
                             "ranges differ by mode: action ~0.01-0.1 "
                             "(LIBERO 7-D delta actions live ~[-1,1] but "
                             "typical magnitudes per dim are O(0.01-0.1)), "
                             "obs ~0.02-0.3.")
    parser.add_argument("--perturb-mode", default="action",
                        choices=["action", "obs"],
                        help="Where to inject perturbation when a rollout "
                             "is sampled-perturbed. 'action' (default, "
                             "matches MolmoBot's action-noise hook) adds "
                             "iid gaussian to executed env actions. 'obs' "
                             "perturbs the policy's state input at each "
                             "query step.")
    args = parser.parse_args()

    if not 0.0 <= args.perturb_prob <= 1.0:
        raise ValueError(
            f"--perturb-prob must be in [0, 1], got {args.perturb_prob}")

    instructions = _load_instructions(args.instruction_list)
    noise_shape = _NOISE_SHAPE[args.policy]
    chunk_len = noise_shape[0]
    query_freq = args.query_freq if args.query_freq > 0 else chunk_len
    if query_freq > chunk_len:
        raise ValueError(
            f"query_freq={query_freq} exceeds policy chunk length {chunk_len}")

    out_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[testset] writing to {out_dir}")
    print(f"[testset] policy={args.policy}  task={args.task_suite}/{args.task_id}")
    print(f"[testset] {len(instructions)} instructions x "
          f"{args.trajs_per_instruction} reps = "
          f"{len(instructions) * args.trajs_per_instruction} trajs total")

    # Build the (instruction_index, replica_index) schedule. Stable order
    # by default; reproducibly shuffled when --shuffle is set.
    schedule = [
        (i, r)
        for i in range(len(instructions))
        for r in range(args.trajs_per_instruction)
    ]
    if args.shuffle:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(schedule)

    # Load benchmark + task.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)

    # Load policy once.
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
        # Skip schedule entries already covered by prior runs. We trust
        # next_eid as the ground truth for "how many we've done"; the
        # manifest entries with eid < next_eid were already saved.
    else:
        manifest = {
            "name": args.name,
            "policy": args.policy,
            "task_suite": args.task_suite,
            "task_id": int(args.task_id),
            "instructions": instructions,
            "trajs_per_instruction": int(args.trajs_per_instruction),
            "fps": int(args.fps),
            "perturb_prob": float(args.perturb_prob),
            "perturb_sigma": float(args.perturb_sigma),
            "perturb_mode": str(args.perturb_mode),
            "entries": [],
        }

    rng_jax = jax.random.PRNGKey(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))
    # Separate stream so the per-rollout coin flip is independent of the
    # per-step gaussian draws (and stays reproducible from --seed alone).
    perturb_rng = np.random.default_rng(int(args.seed) + 1)

    # Resume from the schedule offset matching next_eid (assumes the
    # schedule was the same in the prior run — same --seed and --shuffle).
    start_pos = next_eid
    if start_pos >= len(schedule):
        print(f"[testset] schedule already exhausted "
              f"(start_pos={start_pos} >= {len(schedule)} entries). "
              "Nothing to do.")
        return

    for pos in range(start_pos, len(schedule)):
        instr_idx, rep_idx = schedule[pos]
        instruction = instructions[instr_idx]
        env_seed = int(args.seed) + pos
        env, _, bddl_name = _get_libero_env(
            task, args.cam_resolution, env_seed)

        # Per-rollout coin flip: this rollout is either clean (seed mode,
        # noise_scale=1.0 -> obs/action untouched) or perturbed (the
        # configured mode + sigma).
        perturbed = bool(perturb_rng.random() < args.perturb_prob)
        if perturbed:
            rollout_mode = str(args.perturb_mode)
            rollout_sigma = float(args.perturb_sigma)
        else:
            rollout_mode = "seed"
            rollout_sigma = 1.0

        try:
            print(f"[testset] eid={next_eid:06d}  pos={pos}  "
                  f"instr_idx={instr_idx} rep={rep_idx}  "
                  f"env_seed={env_seed}  "
                  f"perturbed={perturbed} mode={rollout_mode} sigma={rollout_sigma}  "
                  f"instr={instruction!r}")
            traj, rng_jax = collect_one(
                env=env,
                agent_dp=agent_dp,
                task_description=instruction,
                bddl_name=bddl_name,
                task_suite=args.task_suite,
                task_id=args.task_id,
                noise_scale=rollout_sigma,
                noise_mode=rollout_mode,
                query_freq=query_freq,
                max_timesteps=args.max_timesteps,
                settle_steps=args.settle_steps,
                rng=rng_jax,
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
            "instruction": instruction,
            "instruction_idx": int(instr_idx),
            "replica_idx": int(rep_idx),
            "is_success": bool(traj["is_success"]),
            "env_steps": int(traj["env_steps"]),
            "env_seed": env_seed,
            "perturbed": bool(perturbed),
            "perturb_mode": rollout_mode if perturbed else "none",
            "perturb_sigma": float(rollout_sigma) if perturbed else 0.0,
        })
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[testset] saved eid={next_eid:06d}  steps={traj['env_steps']}  "
              f"success={traj['is_success']}")
        next_eid += 1

    # ---- summary ----
    by_instr: dict[str, list[bool]] = {}
    for e in manifest["entries"]:
        by_instr.setdefault(e["instruction"], []).append(bool(e["is_success"]))
    print("\n[testset] cumulative success rate by instruction:")
    for instr, succ in by_instr.items():
        print(f"  n={len(succ):3d}  rate={np.mean(succ):.2f}  {instr!r}")
    # Also break out by perturbed when used.
    pert = [e for e in manifest["entries"] if e.get("perturbed")]
    clean = [e for e in manifest["entries"] if not e.get("perturbed")]
    if pert or clean:
        def _rate(es):
            return float(np.mean([e["is_success"] for e in es])) if es else float("nan")
        print(f"[testset] clean: n={len(clean):3d}  success={_rate(clean):.2f}   "
              f"perturbed: n={len(pert):3d}  success={_rate(pert):.2f}")
    print(f"[testset] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
