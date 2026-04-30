"""Fine-tune the libero world model on a mix of pretrain + online data.

Designed to run periodically (every N online trajectories or every K minutes).
Reads a starting checkpoint, does ``--num-steps`` gradient steps with a small
LR on a pretrain∪online batch mix, optionally validates on a pretrain holdout
to gate the new checkpoint, and writes the result.

Layout assumed:
    <pretrain_root>/<suite>/{annotation,latent_videos,...}   (already there)
    <pretrain_root>/<suite>/{train,val}_sample.json
    <pretrain_root>/stat.json

    <online_root>/{annotation,latent_videos,raw_videos,...}  (collector output)
    <online_root>/train_sample.json                          (collector output)

Run (open-world venv):

    /scratch/gpfs/AM43/yy4041/open-world/.venv/bin/python \\
        examples/reward_model/finetune_wm.py \\
        --ckpt-in   /scratch/.../checkpoint-20000.pt \\
        --ckpt-out  /scratch/.../wm_reward_step_001.pt \\
        --pretrain-root /scratch/.../data/wm_training/libero_processed \\
        --online-root  /scratch/.../wm_reward \\
        --num-steps 200 \\
        --batch-size 4 \\
        --lr 5e-6 \\
        --mix-online 0.5 \\
        --validate

The "mix" knob is the *probability per sample* of drawing from online data;
the rest comes from pretrain. Start with 0.5 and turn down if the model drifts
on the validation set.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


OPEN_WORLD_ROOT = Path(
    os.environ.get("OPEN_WORLD_ROOT", "/scratch/gpfs/AM43/yy4041/open-world")
)


def _load_libero_args(config_path: Path):
    spec = importlib.util.spec_from_file_location("user_libero_wm_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_args"):
        raise AttributeError(f"{config_path} must define get_args() -> LiberoWMArgs")
    return mod.get_args()


# ---------------------------------------------------------------------------
# Datasets — reuse open-world's LiberoLatentDataset as-is for both sources
# ---------------------------------------------------------------------------

def _make_dataset(args_template, dataset_root: str, dataset_name: str, mode: str):
    """Build a single-suite LiberoLatentDataset by mutating the args template."""
    from openworld.training.world_model.dataset import LiberoLatentDataset
    import copy
    a = copy.copy(args_template)
    a.dataset_root_path = dataset_root
    a.dataset_meta_info_path = dataset_root
    a.dataset_names = dataset_name
    a.dataset_cfgs = dataset_name
    a.prob = (1.0,)
    return LiberoLatentDataset(a, mode=mode)


class MixedDataset(torch.utils.data.Dataset):
    """Stochastically draws a sample from one of N child datasets per __getitem__."""

    def __init__(self, datasets, probs, length=None):
        super().__init__()
        assert len(datasets) == len(probs)
        s = float(sum(probs))
        self.probs = [p / s for p in probs]
        self.datasets = datasets
        self.length = length or sum(len(d) for d in datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        which = int(np.random.choice(len(self.datasets), p=self.probs))
        d = self.datasets[which]
        return d[np.random.randint(len(d))]


# ---------------------------------------------------------------------------
# Training step (single GPU; no accelerate to stay simple)
# ---------------------------------------------------------------------------

def run_finetune(args) -> dict:
    from openworld.world_models.ctrl_world import CrtlWorld

    cfg = _load_libero_args(Path(args.config))
    if not os.path.isabs(cfg.svd_model_path):
        cfg.svd_model_path = str(OPEN_WORLD_ROOT / cfg.svd_model_path)
    if not os.path.isabs(cfg.clip_model_path):
        cfg.clip_model_path = str(OPEN_WORLD_ROOT / cfg.clip_model_path)
    cfg.train_batch_size = int(args.batch_size)

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    pretrain_suites = args.pretrain_suites.split(",")
    pretrain_datasets = [
        _make_dataset(cfg, args.pretrain_root, s, mode="train")
        for s in pretrain_suites
    ]
    online_dataset = _make_dataset(cfg, args.online_root, args.online_suite, mode="train")

    train_dataset = MixedDataset(
        datasets=pretrain_datasets + [online_dataset],
        probs=[(1 - args.mix_online) / len(pretrain_datasets)] * len(pretrain_datasets)
              + [args.mix_online],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=False,  # the mix is already random; shuffling would no-op
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    print(f"[finetune] building CrtlWorld and loading {args.ckpt_in}")
    t0 = time.perf_counter()
    model = CrtlWorld(cfg)
    try:
        sd = torch.load(args.ckpt_in, map_location="cpu", mmap=True)
    except TypeError:
        sd = torch.load(args.ckpt_in, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    import gc; gc.collect()
    print(
        f"[finetune] model ready ({time.perf_counter() - t0:.1f}s) "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Optional pre-validation (canary loss on pretrain val)
    # ------------------------------------------------------------------
    val_before = None
    if args.validate:
        val_before = _val_loss(model, cfg, args.pretrain_root,
                                args.pretrain_suites.split(",")[0],
                                args.num_workers, args.val_batches, device)
        print(f"[finetune] pre-finetune val_loss = {val_before:.4f}")

    # ------------------------------------------------------------------
    # Gradient steps
    # ------------------------------------------------------------------
    print(f"[finetune] running {args.num_steps} steps  lr={args.lr}  bs={cfg.train_batch_size}")
    losses = []
    step = 0
    iterator = iter(train_loader)
    t_loop_start = time.perf_counter()
    while step < args.num_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _move_batch(batch, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
        step += 1
        if step % max(1, args.num_steps // 10) == 0:
            print(f"[finetune] step {step}/{args.num_steps}  loss={np.mean(losses[-20:]):.4f}")

    train_time = time.perf_counter() - t_loop_start
    print(f"[finetune] {args.num_steps} steps done in {train_time:.1f}s "
          f"({train_time/args.num_steps:.2f}s/step)")

    # ------------------------------------------------------------------
    # Optional post-validation gate
    # ------------------------------------------------------------------
    val_after = None
    accepted = True
    if args.validate:
        model.eval()
        val_after = _val_loss(model, cfg, args.pretrain_root,
                               args.pretrain_suites.split(",")[0],
                               args.num_workers, args.val_batches, device)
        model.train()
        print(f"[finetune] post-finetune val_loss = {val_after:.4f}  (was {val_before:.4f})")
        if val_after > val_before * (1 + args.val_regress_tol):
            accepted = False
            print(f"[finetune] REJECTED: val_loss regressed by more than {args.val_regress_tol*100:.0f}%")

    # ------------------------------------------------------------------
    # Save (only if accepted)
    # ------------------------------------------------------------------
    if accepted:
        out_path = Path(args.ckpt_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"[finetune] saved checkpoint to {out_path}")
    else:
        print(f"[finetune] checkpoint NOT saved")

    return {
        "accepted": accepted,
        "ckpt_out": args.ckpt_out if accepted else None,
        "num_steps": args.num_steps,
        "train_time_s": train_time,
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "val_before": val_before,
        "val_after": val_after,
    }


def _move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _val_loss(model, cfg, pretrain_root, pretrain_suite, num_workers, n_batches, device):
    """Average forward loss on a pretrain val mini-set."""
    val_ds = _make_dataset(cfg, pretrain_root, pretrain_suite, mode="val")
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.train_batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    losses = []
    model.eval()
    with torch.no_grad():
        for k, batch in enumerate(loader):
            if k >= n_batches:
                break
            batch = _move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(batch)
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(OPEN_WORLD_ROOT / "configs/training/libero_wm.py"),
    )
    parser.add_argument("--ckpt-in", required=True)
    parser.add_argument("--ckpt-out", required=True)
    parser.add_argument(
        "--pretrain-root",
        default=str(OPEN_WORLD_ROOT / "data/wm_training/libero_processed"),
    )
    parser.add_argument(
        "--pretrain-suites",
        default="libero_10,libero_object,libero_goal,libero_spatial",
        help="Comma-separated pretrain suites to mix in.",
    )
    parser.add_argument(
        "--online-root",
        required=True,
        help="Where the dsrl_pi0 collector wrote its data (+train_sample.json).",
    )
    parser.add_argument(
        "--online-suite",
        default=".",
        help="Subdir under --online-root, or '.' if data lives directly in root.",
    )
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--mix-online", type=float, default=0.5,
        help="Per-sample probability of drawing from online vs. pretrain.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--validate", action="store_true",
                        help="Run pre/post-finetune validation on pretrain val.")
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument(
        "--val-regress-tol", type=float, default=0.05,
        help="Reject the new ckpt if val_loss > (1 + tol) * pre-finetune val_loss.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    summary = run_finetune(args)

    # Print machine-readable summary so a daemon/cron can parse it.
    summary_path = Path(args.ckpt_out).with_suffix(".finetune.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[finetune] summary written to {summary_path}")


if __name__ == "__main__":
    main()
