# World-Model Novelty Reward (DSRL-π₀ + open-world)

This is the orchestration layer that uses the libero world model in
`/scratch/gpfs/AM43/yy4041/open-world` as a reward signal for DSRL-π₀:
**how surprised is the WM by this trajectory?**. Higher LPIPS between
WM-predicted frames and the actually-recorded frames → higher reward,
because novelty produces unfamiliar training data for the WM.

This sits alongside the existing learned-reward pipeline documented in
[`CUSTOM_REWARD.md`](CUSTOM_REWARD.md). Same `--reward_fn` plug-in point —
just a different `score(traj)` callable on the receiving side.

---

## Architecture

Two processes, one shared on-disk dataset:

```
   dsrl_pi0 trainer             reward server                    (maybe later)
   (jaxrl2 venv)                (open-world venv)                fine-tune cron
   ────────────────             ──────────────────               ──────────────
   data_collection_loop         loads WM once                    triggered every
   collect_traj                 watches <reward_root>/requests/   N trajs / hours
       │                                │                              │
       ├─ save_traj_libero_processed    │                              │
       │   writes annotation + mp4s     │                              │
       │                                │                              │
       ├─ traj['_eid'] = '000123'       │                              │
       │                                │                              │
       └─ score_fn(traj) ─────► drops   │                              │
                                requests/000123.req                    │
                                         │                             │
                                         ▼                             │
                                 load mp4s → VAE encode                │
                                 autoregressive WM rollout             │
                                 LPIPS(pred, recorded) ─────► writes   │
                                 scores/000123.score.json              │
                                         │                             │
   blocks until score appears ◄─────────┘                              │
   returns float                                                       │
                                                                       ▼
                                                           finetune_wm.py
                                                           reads pretrain ∪ online
                                                           writes new ckpt
                                                           (daemon picks up on
                                                            restart, or via
                                                            symlink-watching —
                                                            not yet implemented)
```

The two processes never talk over network or shared memory — only files. No
ports to allocate, no SSH/Slurm pain, atomic writes via tmp-rename.

---

## Files

| File | Role |
|---|---|
| `examples/reward_fn.py:wm_score` | the `score(traj)` callable; runs in the trainer's venv (jax). Drops a request file and polls for the score. |
| `examples/reward_model/reward_server.py` | long-running daemon (open-world venv). Watches `requests/`, scores trajectories, writes `scores/`. |
| `examples/reward_model/score_traj.py` | standalone scorer (no daemon, one-shot). Used to validate cost and LPIPS spread before wiring up the full system. |
| `examples/reward_model/finetune_wm.py` | optional periodic fine-tuner. Mixes pretrain and online data, gates new checkpoint with a val-loss regression check. |
| `examples/train_utils_collect.py` | the existing collector — minor edit to inject `_eid` / `_save_dir` / `_save_split` into `traj` so `wm_score` can find the files. |

---

## Setup

### TL;DR (Slurm + offline compute nodes)

```bash
# 1. Once on a login node (has internet) — pre-fetch all caches:
bash examples/scripts/setup_caches.sh

# 2. Submit the loop:
sbatch examples/scripts/run_wm_loop.sh
```

`setup_caches.sh` fails fast if anything is missing and tells you what to
do. `run_wm_loop.sh` re-checks every cache before launching, spawns the
reward-server daemon, waits for it to finish loading, then runs the
dsrl_pi0 trainer. Both processes share a single GPU by default; for two
GPUs, change `#SBATCH --gres=gpu:1` to `gpu:2` and set `REWARD_GPU=1`
when submitting.

The rest of this section explains the underlying pieces in case you want to
run them by hand.

### 1. Pick a reward root

```bash
export REWARD_ROOT=/scratch/gpfs/AM43/yy4041/wm_reward
mkdir -p "$REWARD_ROOT"
```

This is also the collector's `save_dir`. The on-disk layout will be:

```
$REWARD_ROOT/
  annotation/{train,val}/<eid>.json     # collector writes
  raw_videos/{agentview,wrist}/<eid>.mp4 # collector writes
  latent_videos/{agentview,wrist}/<eid>.pt  # daemon writes (after first encode)
  requests/<eid>.req                     # trainer writes, daemon deletes
  scores/<eid>.score.json                # daemon writes
  scores/<eid>.error.json                # daemon writes on failure
```

### 2. Start the reward server (compute node)

```bash
cd /scratch/gpfs/AM43/yy4041/open-world
.venv/bin/python /scratch/gpfs/AM43/yy4041/dsrl_pi0/examples/reward_model/reward_server.py \
    --reward-root "$REWARD_ROOT" \
    --ckpt-path /scratch/gpfs/AM43/yy4041/open-world/models/wm_training/libero_0429/checkpoint-20000.pt \
    --num-windows 8 \
    --start-frame 6 \
    --device cuda:0
```

The first request blocks for ~90s while SVD loads. Subsequent requests are
~2 minutes each at default settings (8 windows × ~15s diffusion each, plus
decode + LPIPS).

### 3. Start the dsrl_pi0 trainer (different node, same reward root)

```bash
export DSRL_REWARD_ROOT="$REWARD_ROOT"
export DSRL_REWARD_TIMEOUT_S=900   # default 600

source /scratch/gpfs/AM43/yy4041/dsrl_pi0/.venv/bin/activate
cd /scratch/gpfs/AM43/yy4041/dsrl_pi0

python examples/launch_collect.py \
    --save_dir "$REWARD_ROOT" \
    --use_reward_model 1 \
    --reward_fn examples.reward_fn:wm_score \
    --traj_batch_size 8 \
    ...
```

The `--save_dir` MUST equal `$DSRL_REWARD_ROOT` (or be the same physical
location — symlinks are fine). The collector and the daemon read/write the
same tree.

If you need to run the daemon on a different machine/share, set
`DSRL_REWARD_ROOT` to a path the trainer can also see (a network mount).
The two can be on different nodes as long as the filesystem is shared.

---

## Periodic fine-tuning (optional)

The fine-tune step is a standalone script — invoke it however you like
(cron, slurm sbatch, or a script that the daemon spawns as a subprocess).

```bash
.venv/bin/python /scratch/gpfs/AM43/yy4041/dsrl_pi0/examples/reward_model/finetune_wm.py \
    --ckpt-in   /scratch/.../checkpoint-20000.pt \
    --ckpt-out  /scratch/.../wm_reward_step_001.pt \
    --pretrain-root /scratch/.../data/wm_training/libero_processed \
    --online-root  "$REWARD_ROOT" \
    --num-steps 200 \
    --batch-size 4 \
    --lr 5e-6 \
    --mix-online 0.5 \
    --validate
```

`--validate` runs a pretrain-val canary before and after fine-tuning and
**rejects** the new checkpoint if val loss regresses by more than 5%. This
is the anti-forgetting gate — without it, naïve fine-tuning on online data
will collapse the WM's pretrain knowledge over a few iterations.

To switch the daemon over to a new checkpoint: stop and restart it with the
new `--ckpt-path`. (Live hot-swap is not yet implemented; symlink-watching
would be a small addition if you find restarting too disruptive.)

---

## How the score is computed

For each trajectory:

1. Daemon loads the recorded frames (encoded once to VAE latents and cached
   under `latent_videos/`). Loads the same actions the policy chose.
2. Starting at `start_frame=6` (default), runs the WM autoregressively for
   `num_windows=8` windows (each is `num_frames=5` future frames given
   `num_history=6` past frames). Sliding by 4 frames per window.
3. After each window, the **last predicted frame** is appended to the
   history buffer (instead of the GT frame). This is the autoregressive
   path — predictions compound over the trajectory, surfacing the
   action-sensitivity that a single-window prediction can't.
4. For each predicted frame, decodes the latent to RGB and computes LPIPS
   against the recorded RGB. The trajectory score is the mean of all
   per-frame LPIPS values.

Higher = more novel = higher reward.

### Why autoregressive (and not single-window)?

Validated empirically on this checkpoint at default settings: a single
5-frame prediction is near-perfect for any plausible action sequence
(in-dist LPIPS ~0.058, random-action LPIPS ~0.038 — model is essentially
action-blind at 1 second of motion). Stretching the prediction across the
whole trajectory via autoregression is what makes the LPIPS sensitive to
which actions the policy chose.

---

## Cost notes

Per trajectory, default settings (8 windows × ~13s diffusion + decode +
LPIPS): ~2 minutes per scored trajectory at `B=1`.

If this is too slow:
- Drop `num_inference_steps` from 50 → 25 (≈2× speedup, modest LPIPS shift)
- Drop `num_windows` (less novelty resolution but faster)
- Batch K trajectories per forward (the pipeline supports `B>1`; the
  current daemon serves one at a time — straightforward extension)

The cold-start (~90s) is one-time per daemon process. Don't restart it
casually.

---

## Troubleshooting

**"wm_score: traj is missing '_eid' / '_save_dir'"**
You're running `examples.train_sim` (which doesn't save trajectories to
disk) instead of `examples.data_collection_sim`. Switch to the latter.

**"No annotation for <eid>"**
The collector wrote to a different `save_dir` than the daemon is watching.
Check that `--save_dir` (collector) and `--reward-root` (daemon) point to
the same path.

**Score timeout**
The daemon is either down or processing a backlog. Check
`scores/<eid>.error.json` for failures, and look at the daemon's stdout. If
the daemon is alive but slow, raise `DSRL_REWARD_TIMEOUT_S`.

**Daemon OOMs on the login node**
Don't run the daemon on a login node — use a compute node. The 9GB SVD
checkpoint plus VAE forwards will get killed by shared cgroup pressure
even if `free` shows plenty of RAM.
