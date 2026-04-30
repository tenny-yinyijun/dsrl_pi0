"""Reward-server daemon: holds the libero world model in memory and scores
trajectories submitted by the dsrl_pi0 trainer.

Communication is via the on-disk ``libero_processed`` layout that
``examples/data_collection_sim.py`` already produces. The daemon polls a
request directory for new ``.req`` files; for each one it loads the
trajectory, runs autoregressive replay-rollout, computes mean LPIPS vs the
recorded frames, and writes the scalar back as ``<eid>.score.json``.

Layout (configurable via --reward-root):

    <reward_root>/                         <- collector's save_dir
        annotation/<split>/<eid>.json
        raw_videos/{agentview,wrist}/<eid>.mp4
        latent_videos/{agentview,wrist}/<eid>.pt   (optional, daemon will
                                                    encode if missing)
        requests/<eid>.req                  <- touch-file = "score me"
        scores/<eid>.score.json             <- written by daemon

Run (open-world venv):

    /scratch/gpfs/AM43/yy4041/open-world/.venv/bin/python \\
        examples/reward_model/reward_server.py \\
        --reward-root /scratch/.../wm_reward \\
        --ckpt-path  /scratch/.../checkpoint-20000.pt \\
        --num-windows 8

The score is **higher = more novel** — explicitly the mean LPIPS between WM
prediction and the recorded rollout. When the WM thinks "I expected this", LPIPS
is low; when the trajectory is novel/unfamiliar, LPIPS is high. The dsrl_pi0
trainer uses this directly as ``f(traj)`` (no negation needed).

Fine-tuning is **not yet implemented** — the daemon just scores. A separate
periodic hook (every R requests) will be added in a follow-up.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch


OPEN_WORLD_ROOT = Path(
    os.environ.get("OPEN_WORLD_ROOT", "/scratch/gpfs/AM43/yy4041/open-world")
)

# Allow `import score_traj` from the same directory.
sys.path.insert(0, str(Path(__file__).parent))
from score_traj import (  # noqa: E402
    _load_libero_args,
    _load_stat,
    build_model,
    decode_latents_rgb,
    lpips_per_frame,
    score_episode,
    stack_cams,
)


# ---------------------------------------------------------------------------
# Trajectory loading from libero_processed format
# ---------------------------------------------------------------------------

def _read_annotation(ann_path: Path) -> dict:
    with ann_path.open() as f:
        return json.load(f)


def _resolve_traj_dir(reward_root: Path, eid: str) -> tuple[Path, dict, str]:
    """Find <eid>.json under <reward_root>/annotation/{train,val}/.

    Returns (traj_dir, annotation_dict, split).
    """
    for split in ("train", "val"):
        path = reward_root / "annotation" / split / f"{eid}.json"
        if path.exists():
            return reward_root, _read_annotation(path), split
    raise FileNotFoundError(f"No annotation for {eid} under {reward_root / 'annotation'}")


def _load_or_encode_latents(
    *,
    traj_dir: Path,
    annotation: dict,
    cfg,
    pipeline,
    device: torch.device,
    eid: str,
) -> torch.Tensor:
    """Return per-cam latents (T, num_cams, 4, h, w) float32 (CPU).

    If the annotation already references ``latent_videos``, just load them.
    Otherwise read ``raw_videos`` mp4s with decord, resize to (cfg.height,
    cfg.width), VAE-encode, save to disk, and update the annotation.
    """
    cam_specs = annotation.get("latent_videos") or []
    if len(cam_specs) >= cfg.num_cams:
        cam_latents = []
        for spec in cam_specs[: cfg.num_cams]:
            with (traj_dir / spec["latent_video_path"]).open("rb") as f:
                v = torch.load(f)
            v.requires_grad = False
            cam_latents.append(v)
        T = min(v.shape[0] for v in cam_latents)
        cam_latents = [v[:T] for v in cam_latents]
        return torch.stack(cam_latents, dim=1).float()

    # ----- encode from mp4 -----
    raw_specs = annotation.get("raw_videos") or []
    if len(raw_specs) < cfg.num_cams:
        raise RuntimeError(
            f"{eid}: annotation has neither latent_videos nor enough raw_videos"
        )

    import decord  # heavy import; defer
    decord.bridge.set_bridge("torch")

    cam_names = []
    cam_latents = []
    for spec in raw_specs[: cfg.num_cams]:
        cam = spec["cam"]
        mp4 = traj_dir / spec["video_path"]
        vr = decord.VideoReader(str(mp4))
        frames = vr[:].float() / 127.5 - 1.0  # (T, H, W, 3) in [-1, 1]
        # Resize to (cfg.height, cfg.width) per cam.
        frames = frames.permute(0, 3, 1, 2)  # (T, 3, H, W)
        if frames.shape[-2:] != (cfg.height, cfg.width):
            frames = torch.nn.functional.interpolate(
                frames, size=(cfg.height, cfg.width),
                mode="bilinear", align_corners=False,
            )
        # VAE-encode in chunks.
        chunk = max(1, int(cfg.decode_chunk_size))
        outs = []
        with torch.no_grad():
            for i in range(0, frames.shape[0], chunk):
                x = frames[i : i + chunk].to(device=device, dtype=pipeline.vae.dtype)
                lat = pipeline.vae.encode(x).latent_dist.mode()
                lat = lat * pipeline.vae.config.scaling_factor
                outs.append(lat.float().cpu())
        cam_lat = torch.cat(outs, dim=0)  # (T, 4, h, w)

        # Cache to disk.
        out_dir = traj_dir / "latent_videos" / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cam_lat, out_dir / f"{eid}.pt")
        cam_names.append(cam)
        cam_latents.append(cam_lat)

    # Update + persist annotation so future scorings hit the cache.
    annotation["latent_videos"] = [
        {"latent_video_path": f"latent_videos/{cam}/{eid}.pt", "cam": cam}
        for cam in cam_names
    ]
    for split in ("train", "val"):
        ann_path = traj_dir / "annotation" / split / f"{eid}.json"
        if ann_path.exists():
            with ann_path.open("w") as f:
                json.dump(annotation, f)
            break

    T = min(v.shape[0] for v in cam_latents)
    cam_latents = [v[:T] for v in cam_latents]
    return torch.stack(cam_latents, dim=1).float()


def _load_actions(annotation: dict, cfg, suite_root: Path, dataset_root: Path,
                   T_wm: int) -> torch.Tensor:
    """Load + normalize cartesian + gripper actions for the trajectory."""
    cart = np.asarray(annotation["observation.state.cartesian_position"], dtype=np.float32)
    grip = np.asarray(annotation["observation.state.gripper_position"], dtype=np.float32)
    if grip.ndim == 1:
        grip = grip[:, None]
    state = np.concatenate([cart, grip], axis=-1)
    idx = np.clip(np.arange(T_wm) * cfg.down_sample, 0, len(state) - 1)
    sampled = state[idx]
    p01, p99 = _load_stat(suite_root, dataset_root)
    sampled = np.clip(2 * (sampled - p01) / (p99 - p01 + 1e-8) - 1, -1, 1)
    return torch.tensor(sampled, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: write to .tmp, fsync, rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def serve(args, cfg, model, pipeline, pipeline_cls, lpips_fn, device, dataset_root):
    reward_root = Path(args.reward_root)
    requests_dir = reward_root / "requests"
    scores_dir = reward_root / "scores"
    requests_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    print(f"[server] watching {requests_dir} ; writing to {scores_dir}")
    print(f"[server] num_windows={args.num_windows}  start_frame={args.start_frame}")

    served = 0
    while True:
        reqs = sorted(p for p in requests_dir.glob("*.req") if p.is_file())
        if not reqs:
            time.sleep(args.poll_interval)
            continue

        req = reqs[0]
        eid = req.stem
        score_path = scores_dir / f"{eid}.score.json"
        if score_path.exists():
            # Already scored — clean up the stale request.
            req.unlink(missing_ok=True)
            continue

        try:
            t0 = time.perf_counter()
            traj_dir, annotation, split = _resolve_traj_dir(reward_root, eid)
            suite = annotation.get("task_suite", "libero_10")
            suite_root = traj_dir  # reward_root acts as the per-suite root for online data

            latents_per_cam = _load_or_encode_latents(
                traj_dir=traj_dir,
                annotation=annotation,
                cfg=cfg,
                pipeline=pipeline,
                device=device,
                eid=eid,
            )
            T_wm = latents_per_cam.shape[0]
            actions = _load_actions(annotation, cfg, suite_root, dataset_root, T_wm)
            text = (
                annotation["texts"][0]
                if annotation.get("texts")
                else annotation.get("language_instruction", "")
            )

            result = score_episode(
                model=model,
                pipeline=pipeline,
                pipeline_cls=pipeline_cls,
                cfg=cfg,
                latents_per_cam=latents_per_cam,
                actions=actions,
                text=text,
                start_frame=args.start_frame,
                num_windows=args.num_windows,
                skip_his=args.skip_his,
                lpips_fn=lpips_fn,
                device=device,
                autoregressive=True,
                verbose=False,
            )
            score = result["mean_lpips"]
            elapsed = time.perf_counter() - t0

            payload = {
                "score": score,
                "per_frame_lpips": result["per_frame_lpips"].tolist(),
                "windows_completed": result["windows_completed"],
                "elapsed_s": elapsed,
                "wm_ckpt": args.ckpt_path,
                "wm_step": _maybe_step(args.ckpt_path),
                "T_wm": int(T_wm),
                "suite": suite,
                "eid": eid,
            }
            _atomic_write_json(score_path, payload)
            req.unlink(missing_ok=True)
            served += 1
            print(
                f"[server] [{served:5d}] eid={eid}  score={score:.4f}  "
                f"windows={result['windows_completed']}  "
                f"elapsed={elapsed:.2f}s  T_wm={T_wm}"
            )
        except Exception:
            err_path = scores_dir / f"{eid}.error.json"
            tb = traceback.format_exc()
            print(f"[server] ERROR scoring {eid}:\n{tb}")
            try:
                _atomic_write_json(
                    err_path, {"eid": eid, "error": tb, "wm_ckpt": args.ckpt_path}
                )
            finally:
                req.unlink(missing_ok=True)


def _maybe_step(ckpt_path: str) -> Optional[int]:
    if not ckpt_path:
        return None
    name = Path(ckpt_path).stem  # e.g. checkpoint-20000
    if "-" in name:
        try:
            return int(name.rsplit("-", 1)[1])
        except ValueError:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-root",
        required=True,
        help="Directory containing online/ and scores/ (created if absent).",
    )
    parser.add_argument(
        "--config",
        default=str(OPEN_WORLD_ROOT / "configs/training/libero_wm.py"),
    )
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument(
        "--dataset-root",
        default=str(OPEN_WORLD_ROOT / "data/wm_training/libero_processed"),
        help="Where stat.json (action normalization) lives. Pretrain dataset root.",
    )
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--start-frame", type=int, default=6)
    parser.add_argument("--skip-his", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = _load_libero_args(Path(args.config))
    if not os.path.isabs(cfg.svd_model_path):
        cfg.svd_model_path = str(OPEN_WORLD_ROOT / cfg.svd_model_path)
    if not os.path.isabs(cfg.clip_model_path):
        cfg.clip_model_path = str(OPEN_WORLD_ROOT / cfg.clip_model_path)
    if args.num_inference_steps:
        cfg.num_inference_steps = args.num_inference_steps

    print(f"[cfg] flow_map_type={cfg.flow_map_type}  num_inference_steps={cfg.num_inference_steps}")

    model, pipeline, pipeline_cls = build_model(cfg, args.ckpt_path, device)

    print("[server] loading LPIPS...")
    import lpips as lpips_mod
    lpips_fn = lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()

    print(f"[server] ready. polling every {args.poll_interval}s")
    serve(
        args=args,
        cfg=cfg,
        model=model,
        pipeline=pipeline,
        pipeline_cls=pipeline_cls,
        lpips_fn=lpips_fn,
        device=device,
        dataset_root=Path(args.dataset_root),
    )


if __name__ == "__main__":
    main()
