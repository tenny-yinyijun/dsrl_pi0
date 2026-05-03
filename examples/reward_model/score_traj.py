"""Score one LIBERO trajectory by replay-rolling the world model and
computing per-frame LPIPS against ground truth.

For each prediction window of ``num_frames=5`` future frames:
  - condition on a sliding ``num_history=6`` history of latents
  - feed the GT actions for the next ``num_frames`` timesteps
  - decode predicted latents to RGB and compute LPIPS against GT RGB

The window slides by ``num_frames - 1`` frames each step, autoregressively:
the last predicted frame becomes the "current" frame for the next window,
and the history buffer is appended with the last predicted frame. After
``num_history`` windows the entire history is from the WM's own predictions
— this is how prediction error compounds and surfaces action sensitivity.

Run with the open-world venv. Example:

    /scratch/gpfs/AM43/yy4041/open-world/.venv/bin/python \\
        examples/reward_model/score_traj.py \\
        --suite libero_10 --episode-id 000000 \\
        --start-frame 6 --num-windows 8 \\
        --ckpt-path /scratch/.../checkpoint-20000.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


OPEN_WORLD_ROOT = Path(
    os.environ.get("OPEN_WORLD_ROOT", "/scratch/gpfs/AM43/yy4041/open-world")
)


# ---------------------------------------------------------------------------
# Config + data loading
# ---------------------------------------------------------------------------

def _load_libero_args(config_path: Path):
    spec = importlib.util.spec_from_file_location("user_libero_wm_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_args"):
        raise AttributeError(f"{config_path} must define get_args() -> LiberoWMArgs")
    return mod.get_args()


def _load_stat(suite_root: Path, dataset_root: Path):
    for p in (suite_root / "stat.json", dataset_root / "stat.json"):
        if p.exists():
            with p.open() as f:
                d = json.load(f)
            return (
                np.asarray(d["state_01"], dtype=np.float32),
                np.asarray(d["state_99"], dtype=np.float32),
            )
    raise FileNotFoundError(f"No stat.json under {suite_root} or {dataset_root}")


def load_full_episode(cfg, suite: str, episode_id: str, dataset_root: Path):
    """Load all latents + actions for one episode, normalized.

    Returns:
        latents:  (T_wm, num_cams, 4, h_per_cam, latent_w) float32 (CPU)
        actions:  (T_wm, action_dim) float32 (CPU), normalized to [-1, 1]
        text:     instruction string
    """
    suite_root = dataset_root / suite
    for split in ("train", "val"):
        ann_file = suite_root / cfg.annotation_name / split / f"{episode_id}.json"
        if ann_file.exists():
            break
    else:
        raise FileNotFoundError(
            f"No annotation for {suite}/{episode_id} under {suite_root}"
        )
    with ann_file.open() as f:
        label = json.load(f)

    cam_specs = label["latent_videos"]
    if len(cam_specs) < cfg.num_cams:
        raise ValueError(
            f"{ann_file} has {len(cam_specs)} cams, config wants {cfg.num_cams}"
        )

    cam_latents = []
    for spec in cam_specs[: cfg.num_cams]:
        path = suite_root / spec["latent_video_path"]
        with path.open("rb") as f:
            v = torch.load(f)
        v.requires_grad = False
        cam_latents.append(v)
    T_wm = min(v.shape[0] for v in cam_latents)
    cam_latents = [v[:T_wm] for v in cam_latents]
    # (T, num_cams, 4, h_per_cam, w)
    latents = torch.stack(cam_latents, dim=1).float()

    cart = np.asarray(label["observation.state.cartesian_position"], dtype=np.float32)
    grip = np.asarray(label["observation.state.gripper_position"], dtype=np.float32)
    if grip.ndim == 1:
        grip = grip[:, None]
    state_full = np.concatenate([cart, grip], axis=-1)
    state_idx = np.clip(np.arange(T_wm) * cfg.down_sample, 0, len(state_full) - 1)
    actions = state_full[state_idx]
    p01, p99 = _load_stat(suite_root, dataset_root)
    actions = np.clip(2 * (actions - p01) / (p99 - p01 + 1e-8) - 1, -1, 1)

    text = label["texts"][0] if label.get("texts") else label.get(
        "language_instruction", ""
    )
    return latents, torch.tensor(actions, dtype=torch.float32), text


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def stack_cams(latents: torch.Tensor) -> torch.Tensor:
    """(..., num_cams, 4, h, w) -> (..., 4, num_cams*h, w) by stacking cams along H."""
    M = latents.shape[-4]
    return torch.cat([latents[..., m, :, :, :] for m in range(M)], dim=-2)


def decode_latents_rgb(pipeline, latents: torch.Tensor, num_cams: int, chunk: int):
    """(B, T, 4, num_cams*h, w) -> uint8 (B, T, num_cams*h_px, w_px, 3)."""
    import einops

    B, T, C, H_stacked, W = latents.shape
    H_view = H_stacked // num_cams
    vae = pipeline.vae

    x = einops.rearrange(latents, "b t c (m h) w -> (b m) t c h w", m=num_cams)
    x = x.reshape(-1, C, H_view, W)
    chunks = []
    for i in range(0, x.shape[0], chunk):
        c = x[i : i + chunk] / vae.config.scaling_factor
        c = c.to(vae.dtype)
        kw = {}
        if hasattr(vae, "decoder") and hasattr(vae.decoder, "conv_in"):
            kw["num_frames"] = c.shape[0]
        chunks.append(vae.decode(c, **kw).sample)
    decoded = torch.cat(chunks, dim=0)
    decoded = einops.rearrange(
        decoded, "(b m t) c h w -> b t c (m h) w", b=B, m=num_cams, t=T
    )
    decoded = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255).detach().float().cpu().numpy()
    return decoded.transpose(0, 1, 3, 4, 2).astype(np.uint8)


def lpips_per_frame(lpips_fn, pred: np.ndarray, gt: np.ndarray, device) -> np.ndarray:
    """Inputs uint8 (T, H, W, 3); returns (T,) float32 LPIPS."""
    out = np.zeros(pred.shape[0], dtype=np.float32)
    for t in range(pred.shape[0]):
        a = torch.from_numpy(pred[t]).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        b = torch.from_numpy(gt[t]).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad():
            out[t] = float(lpips_fn(a.to(device), b.to(device)).squeeze().cpu())
    return out


def build_model(cfg, ckpt_path: str, device: torch.device):
    """Build CrtlWorld + load checkpoint, return (model, pipeline, pipeline_cls).

    The 9GB checkpoint is loaded with mmap=True so the CPU RAM peak stays low.
    """
    print("[model] importing CrtlWorld + pipeline...")
    t0 = time.perf_counter()
    from openworld.world_models.ctrl_world import CrtlWorld, CtrlWorldDiffusionPipeline
    print(f"[model] import time: {time.perf_counter() - t0:.2f}s")

    print("[model] instantiating CrtlWorld (loads SVD + CLIP from disk)...")
    t0 = time.perf_counter()
    model = CrtlWorld(cfg)
    print(f"[model] instantiate time: {time.perf_counter() - t0:.2f}s")

    if ckpt_path:
        print(f"[model] loading checkpoint {ckpt_path}")
        t0 = time.perf_counter()
        # mmap=True keeps the on-disk file paged in lazily — avoids a CPU-RAM
        # spike that has been seen to OOM-kill on shared/login nodes.
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        del state_dict
        import gc; gc.collect()
        print(
            f"[model] checkpoint load time: {time.perf_counter() - t0:.2f}s  "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    else:
        print("[model] WARNING: no checkpoint — LPIPS will be uninformative.")

    model.to(device).eval()
    pipeline = model.pipeline
    pipeline.vae.to(torch.bfloat16)
    pipeline.image_encoder.to(torch.bfloat16)
    pipeline.unet.to(torch.bfloat16)
    if hasattr(model, "action_encoder"):
        model.action_encoder.to(torch.bfloat16)
    return model, pipeline, CtrlWorldDiffusionPipeline


# ---------------------------------------------------------------------------
# Autoregressive multi-window scoring
# ---------------------------------------------------------------------------

def score_episode(
    *,
    model,
    pipeline,
    pipeline_cls,
    cfg,
    latents_per_cam: torch.Tensor,  # (T_wm, num_cams, 4, h, w) CPU float32
    actions: torch.Tensor,           # (T_wm, action_dim) CPU float32 normalized
    text: str,
    start_frame: int,
    num_windows: int,
    skip_his: int,
    lpips_fn,
    device: torch.device,
    autoregressive: bool = True,
    override_actions: Optional[torch.Tensor] = None,
    verbose: bool = True,
    return_rgb: bool = False,
):
    """Roll out num_windows prediction windows from start_frame, computing
    per-frame LPIPS for each window.

    The "effective" predictions per window are frames 1..num_frames-1 (frame 0
    of each pred window equals the input current_latent and is not scored).
    The cursor advances by (num_frames - 1) per window.

    Returns dict with:
        per_frame_lpips: (num_windows * (num_frames - 1),) ndarray
        mean_lpips:      float
        timings:         {'diffusion': [...], 'decode': [...], 'lpips': [...]}
        pred_rgb:        (num_windows * (num_frames - 1), H, W, 3) uint8
                         [only when return_rgb=True]
        gt_rgb:          same shape uint8 [only when return_rgb=True]
    """
    T_wm = latents_per_cam.shape[0]
    use_actions = override_actions if override_actions is not None else actions
    use_actions = use_actions.to(device=device, dtype=torch.bfloat16)

    # Stack cams along H once: (T, 4, num_cams*h, w)
    latents_stacked = stack_cams(latents_per_cam)  # CPU
    latents_stacked_bf = latents_stacked.to(device=device, dtype=torch.bfloat16)

    # Initialize history buffer: 6 latents at start_frame - i*skip_his (i=num_history..1).
    def _gt_at(i: int) -> torch.Tensor:
        i = max(0, min(i, T_wm - 1))
        return latents_stacked_bf[i : i + 1]  # (1, 4, num_cams*h, w)

    history_buf = [
        _gt_at(start_frame - i * skip_his) for i in range(cfg.num_history, 0, -1)
    ]
    history_indices = [start_frame - i * skip_his for i in range(cfg.num_history, 0, -1)]

    cursor = start_frame
    timings = {"diffusion": [], "decode": [], "lpips": []}
    all_pred_rgb = []
    all_gt_rgb = []
    per_frame_lpips_all = []

    for w in range(num_windows):
        # current_latent: GT for window 0 OR last predicted frame in autoregressive mode.
        if w == 0:
            current_latent = _gt_at(cursor)  # (1, 4, H, W)
        elif autoregressive:
            current_latent = history_buf[-1]  # last predicted
        else:
            current_latent = _gt_at(cursor)

        # history_latents: stack the buffer along time → (1, num_history, 4, H, W)
        history_latents = torch.cat(history_buf, dim=0).unsqueeze(0)

        # Build action chunk: history actions at history_indices + future actions at
        # cursor..cursor+num_frames-1.
        future_idx = list(range(cursor, cursor + cfg.num_frames))
        all_idx = history_indices + future_idx
        all_idx_clipped = [max(0, min(i, T_wm - 1)) for i in all_idx]
        action_chunk = use_actions[all_idx_clipped].unsqueeze(0)  # (1, num_h+num_f, 7)

        # ----- predict -----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            action_latent = model.action_encoder(
                action_chunk, [text], model.tokenizer, model.text_encoder,
                cfg.frame_level_cond,
            )
            _, pred_latents = pipeline_cls.__call__(
                pipeline,
                image=current_latent,
                text=action_latent,
                width=cfg.width,
                height=int(cfg.num_cams * cfg.height),
                num_frames=cfg.num_frames,
                history=history_latents,
                num_inference_steps=cfg.num_inference_steps,
                decode_chunk_size=cfg.decode_chunk_size,
                max_guidance_scale=cfg.guidance_scale,
                fps=cfg.fps,
                motion_bucket_id=cfg.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=cfg.frame_level_cond,
                his_cond_zero=cfg.his_cond_zero,
                flow_map_type=cfg.flow_map_type,
                flow_map_loss_type=cfg.flow_map_loss_type,
            )
        torch.cuda.synchronize()
        timings["diffusion"].append(time.perf_counter() - t0)

        # ----- decode pred + GT future, score frames 1..num_frames-1 -----
        # GT future is at indices cursor+1..cursor+num_frames-1 (skip frame 0 = input).
        gt_future_idx = [
            max(0, min(cursor + j, T_wm - 1)) for j in range(1, cfg.num_frames)
        ]
        gt_future_lat = torch.stack(
            [latents_stacked_bf[i] for i in gt_future_idx], dim=0
        ).unsqueeze(0)  # (1, num_frames-1, 4, H, W)
        pred_scoring_lat = pred_latents[:, 1:]  # skip frame 0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_rgb = decode_latents_rgb(pipeline, pred_scoring_lat, cfg.num_cams, cfg.decode_chunk_size)
            gt_rgb = decode_latents_rgb(pipeline, gt_future_lat, cfg.num_cams, cfg.decode_chunk_size)
        torch.cuda.synchronize()
        timings["decode"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        per_frame = lpips_per_frame(lpips_fn, pred_rgb[0], gt_rgb[0], device)
        timings["lpips"].append(time.perf_counter() - t0)
        per_frame_lpips_all.append(per_frame)
        all_pred_rgb.append(pred_rgb[0])
        all_gt_rgb.append(gt_rgb[0])

        if verbose:
            print(
                f"[window {w:02d}] cursor={cursor:3d}  "
                f"per-frame LPIPS={per_frame.tolist()}  mean={per_frame.mean():.4f}  "
                f"diff={timings['diffusion'][-1]*1000:.0f}ms"
            )

        # ----- advance: append last predicted frame to history, shift cursor -----
        history_buf = history_buf[1:] + [pred_latents[:, -1]]  # autoregressive
        history_indices = history_indices[1:] + [cursor + cfg.num_frames - 1]
        cursor += cfg.num_frames - 1
        if cursor + cfg.num_frames - 1 >= T_wm:
            if verbose:
                print(f"[score_episode] end of trajectory at window {w}, stopping early")
            break

    per_frame_arr = np.concatenate(per_frame_lpips_all, axis=0)
    out = {
        "per_frame_lpips": per_frame_arr,
        "mean_lpips": float(per_frame_arr.mean()),
        "timings": timings,
        "windows_completed": len(per_frame_lpips_all),
    }
    if return_rgb and all_pred_rgb:
        out["pred_rgb"] = np.concatenate(all_pred_rgb, axis=0)  # (T, H, W, 3) uint8
        out["gt_rgb"] = np.concatenate(all_gt_rgb, axis=0)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(OPEN_WORLD_ROOT / "configs/training/libero_wm.py"),
    )
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument(
        "--dataset-root",
        default=str(OPEN_WORLD_ROOT / "data/wm_training/libero_processed"),
    )
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--episode-id", default="000000")
    parser.add_argument("--start-frame", type=int, default=6)
    parser.add_argument(
        "--num-windows", type=int, default=8,
        help="Number of autoregressive prediction windows (each advances by num_frames-1).",
    )
    parser.add_argument(
        "--skip-his", type=int, default=4,
        help="Spacing between history frames at window 0. After that, autoregressive.",
    )
    parser.add_argument(
        "--no-autoregressive", action="store_true",
        help="Use sliding GT history instead of self-conditioned. Cheaper sanity check.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument(
        "--swap-actions-from", default="",
        help="Episode ID to take actions from (latents stay from --episode-id).",
    )
    parser.add_argument(
        "--swap-actions-suite", default="",
        help="Optional: suite for the donor episode if different from --suite.",
    )
    parser.add_argument(
        "--random-actions", action="store_true",
        help="Override actions with uniform noise in [-1, 1]. Most extreme OOD.",
    )
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
    print(f"[cfg] num_history={cfg.num_history}  num_frames={cfg.num_frames}  num_cams={cfg.num_cams}")
    print(f"[cfg] latent shape per frame: (4, {cfg.latent_h_total}, {cfg.latent_w})")

    # Load full episode latents + actions (CPU).
    latents_per_cam, actions, text = load_full_episode(
        cfg, args.suite, args.episode_id, Path(args.dataset_root)
    )
    print(f"[episode] {args.suite}/{args.episode_id}  T_wm={latents_per_cam.shape[0]}  text={text!r}")

    # OOD overrides.
    override_actions = None
    if args.swap_actions_from:
        donor_suite = args.swap_actions_suite or args.suite
        _, donor_actions, _ = load_full_episode(
            cfg, donor_suite, args.swap_actions_from, Path(args.dataset_root)
        )
        # Pad/truncate donor to match length of original.
        T = actions.shape[0]
        if donor_actions.shape[0] < T:
            pad = donor_actions[-1:].repeat(T - donor_actions.shape[0], 1)
            donor_actions = torch.cat([donor_actions, pad], dim=0)
        else:
            donor_actions = donor_actions[:T]
        override_actions = donor_actions
        print(f"[OOD] swapped actions from {donor_suite}/{args.swap_actions_from!r}")
    if args.random_actions:
        override_actions = torch.empty_like(actions).uniform_(-1.0, 1.0)
        print("[OOD] actions replaced with uniform noise in [-1, 1]")

    # Build model.
    model, pipeline, pipeline_cls = build_model(cfg, args.ckpt_path, device)

    # Lazily import LPIPS only after model is up.
    print("[score] loading LPIPS...")
    import lpips as lpips_mod
    lpips_fn = lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()

    # Run the scoring.
    print(
        f"[score] start_frame={args.start_frame}  num_windows={args.num_windows}  "
        f"autoregressive={not args.no_autoregressive}"
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
        autoregressive=not args.no_autoregressive,
        override_actions=override_actions,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"windows completed: {result['windows_completed']}")
    print(f"per-frame LPIPS (concat over windows): {result['per_frame_lpips'].tolist()}")
    print(f"mean LPIPS:        {result['mean_lpips']:.4f}")

    print()
    print("Timing (mean per window):")
    for k, vs in result["timings"].items():
        if vs:
            print(f"  {k:14s}  {np.mean(vs)*1000:.1f} ms")
    total = sum(np.mean(vs) for vs in result["timings"].values() if vs)
    print(f"  {'TOTAL/window':14s}  {total*1000:.1f} ms")
    print(f"  TOTAL EPISODE   {total * result['windows_completed']:.2f} s")


if __name__ == "__main__":
    main()
