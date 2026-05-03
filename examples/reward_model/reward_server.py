"""Reward-server daemon: holds the libero world model in memory and scores
trajectories submitted by the dsrl_pi0 trainer. Optionally fine-tunes the
WM online on the scored trajectories.

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

Optional fine-tuning is enabled with --enable-wm-finetune. Every
``--wm-update-every`` scored episodes the daemon runs ``--wm-grad-steps``
gradient steps using a small CPU buffer of recently-scored
``(latents, actions, text)`` tuples, then writes a fresh checkpoint to
``<reward_root>/wm_checkpoints/checkpoint-<step>.pt``. Subsequent scoring
calls reflect the updated weights.
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


# ---------------------------------------------------------------------------
# Online WM fine-tuner
# ---------------------------------------------------------------------------

class WMFineTuner:
    """Tiny online fine-tuner: keeps a CPU buffer of recently-scored
    trajectories and runs a few SGD steps every N scores.

    Training is done in the same dtype the model is loaded in (bf16 for
    unet/action_encoder, frozen bf16 for vae/image_encoder). AdamW moments
    stay in fp32 by default. This is intentionally minimal — for serious
    WM fine-tuning use scripts/train_libero_wm.py with proper accelerate.
    """

    def __init__(self, *, model, cfg, device, args):
        from collections import deque

        self.model = model
        self.cfg = cfg
        self.device = device

        # Only train unet + action_encoder. Freeze vae / image_encoder / text
        # encoder — they are the heavy frozen backbones and fine-tuning them
        # online with batch_size=1 would just make scoring noisier.
        self.trainable_params = []
        for name in ("unet", "action_encoder"):
            mod = getattr(model, name, None) or getattr(model.pipeline, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = True
                self.trainable_params.append(p)
        for name in ("vae", "image_encoder", "text_encoder"):
            mod = getattr(model.pipeline, name, None) or getattr(model, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=args.wm_lr, betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.update_every = int(args.wm_update_every)
        self.grad_steps = int(args.wm_grad_steps)
        self.batch_size = int(args.wm_batch_size)
        self.max_grad_norm = float(args.wm_max_grad_norm)
        self.buffer_max = int(args.wm_buffer_size)
        self.checkpoint_every = int(args.wm_checkpoint_every)

        self.ckpt_dir = Path(args.reward_root) / "wm_checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Buffer holds CPU tensors so we don't blow up GPU memory.
        # Each entry: (latents_per_cam_cpu_fp32, actions_cpu_fp32, text_str)
        self.buffer = deque(maxlen=self.buffer_max)
        self.global_step = 0   # number of grad steps taken
        self.cycles_done = 0
        self.scored_since_update = 0

        n_train = sum(p.numel() for p in self.trainable_params)
        print(f"[wm-ft] trainable params: {n_train/1e6:.2f}M  "
              f"buffer_max={self.buffer_max}  update_every={self.update_every}  "
              f"grad_steps={self.grad_steps}  batch_size={self.batch_size}  "
              f"lr={args.wm_lr}  ckpt_dir={self.ckpt_dir}")

    def add_sample(self, latents_per_cam, actions, text):
        """Buffer a freshly-scored trajectory for later training."""
        self.buffer.append(
            (latents_per_cam.detach().to("cpu", torch.float32),
             actions.detach().to("cpu", torch.float32),
             text)
        )
        self.scored_since_update += 1

    def _sample_window(self, latents_per_cam, actions, text):
        """Pick a random (num_history + num_frames)-length window and stack
        cameras vertically, matching what LiberoLatentDataset returns.

        latents_per_cam: (T, num_cams, 4, h, w)
        actions:         (T, action_dim)
        Returns dict with 'latent' (1,F,4,total_h,w), 'action' (1,F,A), 'text'.
        """
        cfg = self.cfg
        F = cfg.num_history + cfg.num_frames
        T = latents_per_cam.shape[0]
        if T < F:
            # Pad by repeating last frame.
            pad = F - T
            latents_per_cam = torch.cat(
                [latents_per_cam,
                 latents_per_cam[-1:].expand(pad, -1, -1, -1, -1)], dim=0)
            actions = torch.cat(
                [actions, actions[-1:].expand(pad, -1)], dim=0)
            T = F
        start = int(np.random.randint(0, T - F + 1))
        lat_win = latents_per_cam[start : start + F]   # (F, num_cams, 4, h, w)
        act_win = actions[start : start + F]           # (F, A)
        # Stack cams along H -> (F, 4, num_cams*h, w)
        lat_stacked = torch.cat(
            [lat_win[:, m] for m in range(lat_win.shape[1])], dim=-2)
        return {
            "latent": lat_stacked.unsqueeze(0),     # (1, F, 4, total_h, w)
            "action": act_win.unsqueeze(0),         # (1, F, A)
            "text": [text],
        }

    def maybe_step(self):
        """If enough scores have accumulated since the last cycle, run
        ``grad_steps`` updates, save a checkpoint, and reset counters."""
        if self.update_every <= 0:
            return None
        if self.scored_since_update < self.update_every:
            return None
        if len(self.buffer) == 0:
            self.scored_since_update = 0
            return None

        self.model.train()
        # vae/image_encoder/text_encoder were frozen in __init__, but
        # model.train() flips their training-mode flag too; re-eval them so
        # any internal eval-only behavior stays correct.
        for name in ("vae", "image_encoder", "text_encoder"):
            mod = getattr(self.model.pipeline, name, None) \
                  or getattr(self.model, name, None)
            if mod is not None:
                mod.eval()

        losses = []
        t0 = time.perf_counter()
        for k in range(self.grad_steps):
            self.optimizer.zero_grad(set_to_none=True)

            # Sample a batch of B (window) dicts and stack.
            batch_lat, batch_act, batch_txt = [], [], []
            for _ in range(self.batch_size):
                idx = int(np.random.randint(0, len(self.buffer)))
                lat, act, txt = self.buffer[idx]
                w = self._sample_window(lat, act, txt)
                batch_lat.append(w["latent"])
                batch_act.append(w["action"])
                batch_txt.append(w["text"][0])

            batch = {
                "latent": torch.cat(batch_lat, dim=0).to(self.device),
                "action": torch.cat(batch_act, dim=0).to(self.device),
                "text": batch_txt,
            }

            loss, _ = self.model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.trainable_params, self.max_grad_norm)
            self.optimizer.step()
            self.global_step += 1
            losses.append(float(loss.detach().to("cpu", torch.float32).item()))

        self.model.eval()
        self.scored_since_update = 0
        self.cycles_done += 1

        elapsed = time.perf_counter() - t0
        info = {
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_mean": float(np.mean(losses)),
            "global_step": self.global_step,
            "cycles_done": self.cycles_done,
            "buffer_size": len(self.buffer),
            "elapsed_s": elapsed,
        }
        print(
            f"[wm-ft] cycle={self.cycles_done}  step={self.global_step}  "
            f"loss {losses[0]:.4f} -> {losses[-1]:.4f} (mean {info['loss_mean']:.4f})  "
            f"buf={len(self.buffer)}  elapsed={elapsed:.1f}s"
        )

        # Save checkpoint.
        if self.checkpoint_every > 0 and (self.cycles_done % self.checkpoint_every) == 0:
            ckpt = self.ckpt_dir / f"checkpoint-{self.global_step}.pt"
            torch.save(self.model.state_dict(), ckpt)
            # Also write a "latest" pointer so external readers know the
            # current ckpt.
            (self.ckpt_dir / "latest.txt").write_text(str(ckpt.name) + "\n")
            print(f"[wm-ft] saved checkpoint {ckpt}")
            info["ckpt_path"] = str(ckpt)

        return info


def serve(args, cfg, model, pipeline, pipeline_cls, lpips_fn, device,
          dataset_root, finetuner=None):
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

            if finetuner is not None:
                finetuner.add_sample(latents_per_cam, actions, text)
                finetuner.maybe_step()
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

    # ----- Online WM fine-tuning -----
    parser.add_argument("--enable-wm-finetune", action="store_true",
                        help="Run online WM fine-tuning every "
                             "--wm-update-every scored episodes.")
    parser.add_argument("--wm-update-every", type=int, default=8,
                        help="Run a fine-tune cycle every N scored episodes.")
    parser.add_argument("--wm-grad-steps", type=int, default=25,
                        help="Gradient steps per fine-tune cycle.")
    parser.add_argument("--wm-batch-size", type=int, default=1,
                        help="Window-batch size per gradient step.")
    parser.add_argument("--wm-lr", type=float, default=1e-5,
                        help="AdamW learning rate for fine-tuning.")
    parser.add_argument("--wm-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--wm-buffer-size", type=int, default=64,
                        help="Max scored-episode tuples kept on CPU.")
    parser.add_argument("--wm-checkpoint-every", type=int, default=1,
                        help="Save checkpoint every N completed cycles. "
                             "0 disables checkpointing.")
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

    finetuner = None
    if args.enable_wm_finetune:
        finetuner = WMFineTuner(model=model, cfg=cfg, device=device, args=args)

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
        finetuner=finetuner,
    )


if __name__ == "__main__":
    main()
