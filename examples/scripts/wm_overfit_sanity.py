"""Standalone sanity check for the WM overfit experiment.

Loads a WM checkpoint, picks the trajectories sitting under a given
$REWARD_ROOT, and renders autoregressive rollouts with TWO temporal
layouts so you can directly tell whether the _sample_window fix worked:

  * --skip-his 4    matches score_episode at inference time (this is the
                    layout used by the in-server `wm_update_sanity_check/`
                    videos). After overfitting on the 2 trajs this should
                    look very close to GT.
  * --skip-his 1    the buggy "consecutive native frames" layout the old
                    _sample_window was sampling from. If you also see good
                    predictions here, the overfit worked but isn't picky
                    about layout — useful as a control.

Each call produces one mp4 per traj:
  <out_dir>/<eid>_gt_vs_pred_skiphis<N>.mp4
in (GT | pred) side-by-side, at the same fps as the WM config.

Run (open-world venv — needs the WM):

    /scratch/gpfs/AM43/yy4041/open-world/.venv/bin/python \\
        examples/scripts/wm_overfit_sanity.py \\
        --reward-root /scratch/gpfs/AM43/yy4041/playworld_rollout/<DATE>/<JOB_TAG>_overfit \\
        --ckpt-path   <same_reward_root>/wm_checkpoints/latest.txt   # or pass a .pt file directly
        --eids        000000,000001 \\
        --skip-his    4 \\
        --num-windows 8 \\
        --start-frame 24

Pass --ckpt-path pointing to a `latest.txt` (server writes one) or a `.pt`
file directly. Pass --eids to override which episodes to render; default
is the 2 most-recent episodes found under <reward_root>/annotation/train/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


OPEN_WORLD_ROOT = Path(
    os.environ.get("OPEN_WORLD_ROOT", "/scratch/gpfs/AM43/yy4041/open-world")
)
DSRL_ROOT = Path(
    os.environ.get("DSRL_ROOT", "/scratch/gpfs/AM43/yy4041/dsrl_pi0")
)

# Reuse the helpers the reward server uses so we score with the same code path.
sys.path.insert(0, str(DSRL_ROOT / "examples" / "reward_model"))
from score_traj import (  # noqa: E402
    _load_libero_args,
    build_model,
    score_episode,
)
from reward_server import (  # noqa: E402
    _load_actions,
    _load_or_encode_latents,
    _resolve_traj_dir,
    _write_mp4,
)


def _resolve_ckpt(p: Path) -> Path:
    """Accept either a .pt file or a latest.txt pointer file."""
    if p.suffix == ".txt":
        name = p.read_text().strip()
        return p.parent / name
    return p


def _pick_eids(reward_root: Path, k: int) -> list[str]:
    ann_dir = reward_root / "annotation" / "train"
    if not ann_dir.exists():
        raise FileNotFoundError(f"No {ann_dir}; pass --eids explicitly")
    files = sorted(ann_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation/train/*.json under {reward_root}")
    return [f.stem for f in files[-k:]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward-root", required=True,
                    help="The overfit run's $REWARD_ROOT.")
    ap.add_argument("--ckpt-path", required=True,
                    help="Path to a WM .pt checkpoint OR a latest.txt pointer.")
    ap.add_argument("--config", default=str(OPEN_WORLD_ROOT / "configs/training/libero_wm.py"))
    ap.add_argument("--dataset-root",
                    default=str(OPEN_WORLD_ROOT / "data/wm_training/libero_processed"),
                    help="Where stat.json lives (action normalization).")
    ap.add_argument("--eids", default="",
                    help="Comma-separated episode IDs to render. Default: 2 most-recent.")
    ap.add_argument("--num-eids", type=int, default=2,
                    help="When --eids is empty, how many recent eids to use.")
    ap.add_argument("--skip-his", type=int, default=4,
                    help="History stride in native frames. 4 = inference default; "
                         "1 = the (buggy) old training stride.")
    ap.add_argument("--start-frame", type=int, default=24,
                    help="First cursor (native units). Must be >= num_history*skip_his "
                         "or the GT history clips to frame 0.")
    ap.add_argument("--num-windows", type=int, default=8)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="",
                    help="Where to drop the mp4s. Default: "
                         "<reward_root>/wm_overfit_sanity/<ckpt_stem>/")
    args = ap.parse_args()

    reward_root = Path(args.reward_root)
    if not reward_root.exists():
        raise FileNotFoundError(reward_root)

    ckpt_path = _resolve_ckpt(Path(args.ckpt_path))
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    eids = ([s.strip() for s in args.eids.split(",") if s.strip()]
            or _pick_eids(reward_root, args.num_eids))
    print(f"[sanity] reward_root={reward_root}")
    print(f"[sanity] ckpt={ckpt_path}")
    print(f"[sanity] eids={eids}")
    print(f"[sanity] start_frame={args.start_frame}  skip_his={args.skip_his}  "
          f"num_windows={args.num_windows}")

    out_dir = (Path(args.out_dir) if args.out_dir
               else reward_root / "wm_overfit_sanity" / ckpt_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    device = torch.device(args.device)
    cfg = _load_libero_args(Path(args.config))
    if not os.path.isabs(cfg.svd_model_path):
        cfg.svd_model_path = str(OPEN_WORLD_ROOT / cfg.svd_model_path)
    if not os.path.isabs(cfg.clip_model_path):
        cfg.clip_model_path = str(OPEN_WORLD_ROOT / cfg.clip_model_path)
    if args.num_inference_steps:
        cfg.num_inference_steps = args.num_inference_steps

    model, pipeline, pipeline_cls = build_model(cfg, str(ckpt_path), device)
    model.eval()

    import lpips as lpips_mod
    lpips_fn = lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()

    dataset_root = Path(args.dataset_root)
    fps = int(getattr(cfg, "fps", 10))

    summary: list[dict] = []
    for eid in eids:
        traj_dir, annotation, split = _resolve_traj_dir(reward_root, eid)
        latents = _load_or_encode_latents(
            traj_dir=traj_dir, annotation=annotation, cfg=cfg,
            pipeline=pipeline, device=device, eid=eid,
        )
        T_wm = latents.shape[0]
        actions = _load_actions(annotation, cfg, traj_dir, dataset_root, T_wm)
        text = (annotation["texts"][0] if annotation.get("texts")
                else annotation.get("language_instruction", ""))
        print(f"[sanity] {eid}: T_wm={T_wm}  text={text!r}")

        with torch.no_grad():
            res = score_episode(
                model=model, pipeline=pipeline, pipeline_cls=pipeline_cls,
                cfg=cfg,
                latents_per_cam=latents, actions=actions, text=text,
                start_frame=args.start_frame,
                num_windows=args.num_windows,
                skip_his=args.skip_his,
                lpips_fn=lpips_fn,
                device=device,
                autoregressive=True,
                verbose=False,
                return_rgb=True,
            )
        pred_rgb = res["pred_rgb"]      # (T, H, W, 3) uint8
        gt_rgb = res["gt_rgb"]
        mean_lpips = float(res["mean_lpips"])
        T = min(pred_rgb.shape[0], gt_rgb.shape[0])
        side_by_side = np.concatenate([gt_rgb[:T], pred_rgb[:T]], axis=2)
        out_path = out_dir / f"{eid}_gt_vs_pred_skiphis{args.skip_his}.mp4"
        _write_mp4(out_path, side_by_side, fps=fps)
        print(f"[sanity] {eid}: mean_lpips={mean_lpips:.4f}  -> {out_path}")
        summary.append({
            "eid": eid,
            "skip_his": args.skip_his,
            "start_frame": args.start_frame,
            "num_windows": args.num_windows,
            "mean_lpips": mean_lpips,
            "per_frame_lpips": [float(x) for x in res["per_frame_lpips"]],
            "video_path": str(out_path),
        })

    summary_path = out_dir / f"summary_skiphis{args.skip_his}.json"
    with summary_path.open("w") as f:
        json.dump({
            "ckpt": str(ckpt_path),
            "reward_root": str(reward_root),
            "skip_his": args.skip_his,
            "start_frame": args.start_frame,
            "num_windows": args.num_windows,
            "results": summary,
        }, f, indent=2)
    print(f"[sanity] summary -> {summary_path}")


if __name__ == "__main__":
    main()
