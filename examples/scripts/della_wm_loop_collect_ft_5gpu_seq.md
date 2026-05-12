# `della_wm_loop_collect_ft_5gpu_seq.sh` — Architecture Doc

This document specifies, **stage by stage**, what the orchestrator script does on each GPU and how the components interact through the filesystem. It exists so a reviewing agent can sanity-check the implementation in `examples/scripts/della_wm_loop_collect_ft_5gpu_seq.sh` against a precise reference.

If anything in the script disagrees with this doc, the doc is the intended behavior — flag it as a bug.

---

## 0. Why this design exists

The original `della_wm_loop_collect_ft_5gpu.sh` runs only **2 of 5** allocated GPUs:
- GPU 0: single reward-server process that does both LPIPS scoring AND in-process WM fine-tuning.
- GPU 4: SAC trainer / data collection.
- GPUs 1, 2, 3: idle.

The reason for idling those three GPUs is that the in-process `WMFineTuner` (`examples/reward_model/reward_server.py:253`) is single-process. Running it on N reward-server workers in parallel would diverge the WM weights across workers (each has its own optimizer / buffer / grad-step schedule), and a reward signal stitched across N divergent WMs is incoherent.

**This script (`_seq`) replaces concurrent score+FT with sequential phases on the same 4-GPU pool**, so all 4 WM GPUs are saturated in both phases:

| Phase | GPUs 0-3 | GPU 4 | Concurrent? |
|---|---|---|---|
| **A — scoring** | 4 parallel `reward_server.py` workers (FT disabled) | SAC trainer rolling out + dropping `.req` files + (between Y-boundaries) doing SAC grad steps | Yes |
| **B — DDP FT** | `accelerate launch` of `train_wm.py`, 4-way DDP | SAC trainer rolls + blocks at next Y-boundary `gather_fn` await (no workers serving) | Yes — but trainer effectively idle once it hits a Y-boundary |

Phase B is **not** the in-process `WMFineTuner` — it is `openworld.training.world_model.train_wm` (in the `open-world` repo), the same DDP trainer used in `della_wm_loop_overfit50_5gpu.sh` phase 2 and the original pretrain pipeline. Effective batch size = `WM_TRAIN_BATCH_PER_GPU × 4`.

---

## 1. GPU layout

5 GPUs allocated by `#SBATCH --gres=gpu:5`.

| GPU id | Role | Process during phase A | Process during phase B |
|---|---|---|---|
| 0 | WM GPU | `reward_server.py` worker `c<N>_w0_gpu0` | `accelerate` rank 0 of `train_wm.py` |
| 1 | WM GPU | `reward_server.py` worker `c<N>_w1_gpu1` | `accelerate` rank 1 of `train_wm.py` |
| 2 | WM GPU | `reward_server.py` worker `c<N>_w2_gpu2` | `accelerate` rank 2 of `train_wm.py` |
| 3 | WM GPU | `reward_server.py` worker `c<N>_w3_gpu3` | `accelerate` rank 3 of `train_wm.py` |
| 4 | Trainer GPU | `launch_collect.py` (continuous) | `launch_collect.py` (continuous; blocks at next Y-boundary) |

`TRAINER_GPU` (default `4`) MUST NOT be in `REWARD_GPUS` (default `0,1,2,3`). The script asserts this at boot.

Worker tag format: `c<cycle_n>_w<idx>_gpu<gpu_id>`. Cycle 0 = the first phase A wave (before any FT has happened); the cycle counter increments at the start of every phase B.

---

## 2. Process lifecycle (high-level)

```
sbatch start
  │
  ├─ trainer launches (GPU 4) ───────────────────────────────────────────────► (runs continuously until job ends)
  │
  └─ orchestrator loop:
       cycle 0:  start phase A (4 workers, ckpt=WM_CKPT_INITIAL)
                 ┌─ wait until count_scores() ≥ WM_UPDATE_EVERY
                 ├─ stop_phase_a()  (SIGTERM, SIGKILL, recover orphaned .taken-* claims)
                 └─ run_phase_b()   (accelerate launch DDP train_wm.py for WM_TRAIN_STEPS_PER_CYCLE)
                                    new ckpt → wm_checkpoints/cycle_1/checkpoint-*.pt
       cycle 1:  start phase A (4 workers, ckpt=latest from cycle_1)
                 ┌─ wait until count_scores() ≥ 2 × WM_UPDATE_EVERY
                 ├─ stop_phase_a()
                 └─ run_phase_b()   → wm_checkpoints/cycle_2/
       cycle 2:  ...
       ...
       (loop continues until SLURM time limit triggers SIGTERM → cleanup trap)
```

The orchestrator polls `count_scores()` every `PHASE_TRIGGER_POLL_S=15` seconds. Phase B is launched as a foreground child of the orchestrator (the orchestrator blocks on it).

---

## 3. Filesystem contracts

All paths below are relative to `REWARD_ROOT = /scratch/gpfs/AM43/yy4041/playworld_rollout/<MMDD>/<JOB_TAG>`.

### 3.1 Trajectory artifacts (written by trainer + reward_server)

| Path | Producer | Format |
|---|---|---|
| `annotation/train/<eid>.json` | trainer | per-traj metadata (texts, latent_videos manifest, raw_videos manifest) |
| `raw_videos/agentview/<eid>.mp4` | trainer | 256×256 mp4 |
| `raw_videos/wrist/<eid>.mp4` | trainer | 256×256 mp4 |
| `latent_videos/agentview/<eid>.pt` | reward_server (`_load_or_encode_latents`) | cached VAE latents, written on first score |
| `latent_videos/wrist/<eid>.pt` | reward_server | same |
| `train_sample.json` | trainer (`append_sample_index`) | canonical running list of all collected (frame_id, eid) pairs; appended after every traj; NEVER read directly by phase B |
| `_ft_cycles/cycle_<N>/annotation` | orchestrator (phase B prep) | symlink → `annotation/` so loader's path resolution works |
| `_ft_cycles/cycle_<N>/latent_videos` | orchestrator (phase B prep) | symlink → `latent_videos/` |
| `_ft_cycles/cycle_<N>/raw_videos` | orchestrator (phase B prep) | symlink → `raw_videos/` (if present) |
| `_ft_cycles/cycle_<N>/train_sample.json` | orchestrator (phase B prep) | **filtered** copy of canonical `train_sample.json`: only entries whose `latent_videos/{agentview,wrist}/<eid>.pt` exist on disk. This is what phase B's DataLoader actually reads. |
| `_ft_cycles/cycle_<N>/val_sample.json` | orchestrator (phase B prep) | same content as the filtered train_sample (we accept train==val) |

### 3.2 Score IPC

| Path | Producer | Consumer | Lifecycle |
|---|---|---|---|
| `requests/<eid>.req` | trainer (`score_fn_request`) | reward_server worker (claim via atomic rename) | created when trainer wants this eid scored; deleted by worker after score is written |
| `requests/<eid>.req.taken-<worker_tag>` | worker | (cleanup) | transient: worker has claimed the request and is scoring it; deleted by worker after the corresponding `.score.json` is written |
| `requests/<eid>.wm_only` | trainer | worker | scored-this=False path (currently unused with `SCORED_PER_ROUND=SAC_UPDATE_EVERY`) |
| `scores/<eid>.score.json` | reward_server worker | trainer (`score_fn_await`) | created when scoring completes; trainer reads then keeps; **this file's existence is what the orchestrator counts** |
| `scores/<eid>.error.json` | reward_server worker | trainer | created on per-traj scoring exceptions |

### 3.3 WM checkpoints

| Path | Producer | Consumer |
|---|---|---|
| `wm_checkpoints/cycle_<N>/checkpoint-<step>.pt` | phase B `train_wm.py` | next cycle's phase A reward_servers |
| (initial) `WM_CKPT_INITIAL` = `/scratch/gpfs/.../libero_0429/checkpoint-36000.pt` | (pretrain) | cycle 0 phase A reward_servers |

`latest_wm_checkpoint()` scans `wm_checkpoints/` recursively and picks the file with the largest mtime. After a successful phase B cycle, the orchestrator sets `CURRENT_WM_CKPT = latest_wm_checkpoint()` and passes that to the next `start_phase_a` call as `--ckpt-path`.

### 3.4 Logs

| Path | Content |
|---|---|
| `_logs/trainer.log` | full stdout/stderr of `launch_collect.py`. Includes `[collect] online buffer timesteps: ...`, `[reward] ...`, `[round-time] ...` lines. |
| `_logs/reward_server_cycle<N>_w<idx>_gpu<g>.log` | one log per phase-A worker per cycle. Includes `[server:<worker_tag>] [<served>] eid=<eid> score=...` lines. |
| `_logs/wm_ft_cycle_<N>.log` | full output of one `accelerate launch ... train_wm.py` invocation. |
| `_logs/wm_train_config_cycle_<N>.py` | auto-generated `LiberoWMArgs` config for cycle N. |
| `_logs/sanity_watcher.log` | NOT produced by this script (overfit50_5gpu has one; this script does not). |

---

## 4. Phase A — concurrent scoring + collection

### 4.1 Workers

`start_phase_a(ckpt, cycle_n)` spawns one `reward_server.py` per entry in `REWARD_GPU_ARR` (default 4). Each worker:

- Has its own `CUDA_VISIBLE_DEVICES=<g>` (one GPU each).
- Is invoked with `--worker-id "c<cycle_n>_w<idx>_gpu<g>"` so log lines and `.taken-*` rename suffixes are uniquely tagged.
- Is invoked with `--ckpt-path "$ckpt"` — the same WM weights across all 4 workers in a given cycle.
- Is invoked **WITHOUT** `--enable-wm-finetune`. This is the critical contract that makes 4-worker scoring coherent (no per-worker drift).
- Polls `requests/` for `*.req` files, claims via atomic rename (`_try_claim` in reward_server.py), runs LPIPS scoring (see `scoring_mode=spread`, `num_passes=5`, `windows_per_call=1`), writes `scores/<eid>.score.json`, deletes the `.req.taken-*` claim.

The "ready" signal is the line `"ready. polling"` in the worker's log (reward_server.py:1263). The orchestrator waits up to `SERVER_READY_TIMEOUT_S=2400` for ALL `NUM_REWARD_WORKERS` workers to print this. If any worker dies before being ready, the orchestrator FAILs the whole job.

### 4.2 Trainer (concurrent)

The trainer runs `launch_collect.py` on GPU 4 and is **oblivious to phase boundaries**. It just keeps:

1. Rolling out trajs (one at a time on GPU 4).
2. After each traj, calling `score_fn_request(traj)`, which drops a `.req` file (and the traj's video/annotation files) into `REWARD_ROOT`.
3. Every `Y = SAC_UPDATE_EVERY = 50` trajs, hitting a "round boundary":
   - Calls `gather_fn(t) = score_fn_await(t)` for each of the 50 trajs in the round. This blocks on `scores/<eid>.score.json` existing, with a per-traj timeout of `DSRL_REWARD_TIMEOUT_S=3600` (set by this script).
   - Fits the per-step reward model for `REWARD_GRAD_STEPS=200` steps.
   - Inserts the 50 trajs into the SAC replay buffer.
   - If buffer > `START_ONLINE_UPDATES`, runs `gradsteps_acc = sum(len(rewards) * MULTI_GRAD_STEP)` SAC updates.
   - Prints a `[round-time] await=… reward_grad=… sac=… total=…` line (added by `examples/train_utils_collect.py` in this branch).
4. Loops back to (1).

During phase A: workers drain the request queue at ~30s/traj × 1/4 workers = ~7.5s/traj/worker steady state, so the trainer's await usually completes quickly.

### 4.3 Scored-count milestone trigger

The orchestrator polls `count_scores() = find scores/ -name '*.score.json' | wc -l` every `PHASE_TRIGGER_POLL_S=15` seconds. When `count_scores() ≥ next_threshold` (initially `WM_UPDATE_EVERY=200`), phase A is stopped and phase B begins. The threshold is incremented by `WM_UPDATE_EVERY` after each cycle, so cycle N fires at `N × WM_UPDATE_EVERY` total scored trajs.

**Important**: the trigger is *cumulative* total scores, not per-cycle deltas. This is intentional — it means "after another 200 scored trajs have accumulated, fine-tune again." If the orchestrator is killed and restarted on the same `REWARD_ROOT`, the count picks up where the previous run left off (though resume support is not otherwise implemented; see §7).

---

## 5. Phase transition: A → B

`stop_phase_a()`:

1. Sends `SIGTERM` to each of the 4 worker PIDs.
2. Sleeps 3 seconds for graceful shutdown.
3. Sends `SIGKILL -9` to any still-alive.
4. **Orphan recovery**: scans `requests/` for files matching `*.req.taken-*` and `*.wm_only.taken-*`. Each such file is a claim a (now-dead) worker held when killed. For each:
   - Strip the `.taken-<worker_tag>` suffix to recover the original `.req` / `.wm_only` name.
   - If the unsuffixed name does NOT already exist, rename the taken file back. The next phase-A wave will see it as an unclaimed request.
   - If the unsuffixed name DOES exist (rare edge case: trainer re-dropped the same eid), delete the taken file as a duplicate.
5. Resets `SERVER_PIDS` and `SERVER_LOGS` to empty.

Why orphan recovery is necessary: `_try_claim` (reward_server.py:762) renames a `.req` to `.req.taken-<worker>` atomically. If the worker dies before writing the score, the rename is permanent — no `.score.json` ever appears, and no other worker can re-claim the request (the original `.req` name is gone). Without recovery, those trajs would never be scored, and the trainer's `gather_fn` would time out after `DSRL_REWARD_TIMEOUT_S=3600`.

---

## 6. Phase B — 4-GPU DDP fine-tune

`run_phase_b(cycle_n, ckpt_in)`:

1. **Prepare per-cycle dataset view.**
   - Create `_ft_cycles/cycle_<N>/`.
   - Symlink `annotation`, `latent_videos`, and (if present) `raw_videos` into the cycle view so the loader's path resolution finds the canonical data.
   - Ensure `annotation/val` exists as a symlink to `annotation/train` (idempotent; we accept train==val).
   - **Filter** the canonical `train_sample.json` in two steps:
     1. Drop every entry whose `latent_videos/agentview/<eid>.pt` or `latent_videos/wrist/<eid>.pt` does not exist on disk (un-scored / mid-claim trajs).
     2. Apply the rolling-buffer cap: sort surviving eids numerically (eids are zero-padded monotonic ints) and keep only the **largest** `WM_BUFFER_SIZE` of them. `WM_BUFFER_SIZE=0` disables the cap.
   - Write the filtered list to `_ft_cycles/cycle_<N>/train_sample.json` AND `_ft_cycles/cycle_<N>/val_sample.json` (val mirrors train). If fewer than 5 eids survive, abort with rc=3 — the FT batch would be too small to be meaningful.
2. **Write per-cycle config.** `_logs/wm_train_config_cycle_<N>.py` defines a `get_args()` returning a `LiberoWMArgs`:
   - `ckpt_path = ckpt_in` (the current WM, loaded as the starting point for this cycle).
   - `dataset_root_path = _ft_cycles/cycle_<N>` and `dataset_names=["."]` so the loader resolves traj paths against the per-cycle filtered view, not the canonical (possibly inconsistent) root.
   - `dataset_meta_info_path = WM_DATASET_ROOT` so `stat.json` is found via the loader's meta_root fallback.
   - `train_batch_size = WM_TRAIN_BATCH_PER_GPU`; effective batch = `WM_TRAIN_BATCH_PER_GPU × NUM_REWARD_WORKERS`.
   - `max_train_steps = WM_TRAIN_STEPS_PER_CYCLE = 1000`.
   - `checkpointing_steps = WM_CKPT_EVERY_STEPS` (default 1000 — one ckpt at the end of the cycle).
   - `validation_steps = WM_VAL_EVERY_STEPS` (default 1000 — one val pass at the end).
3. **Launch.** `accelerate launch --num_processes=$NUM_REWARD_WORKERS --mixed_precision fp16 -m openworld.training.world_model.train_wm --config <path> --output_dir wm_checkpoints/cycle_<N>/`. `CUDA_VISIBLE_DEVICES=$REWARD_GPUS` restricts to the 4 WM GPUs; accelerate sees them as ranks 0..3.
4. **Wait synchronously.** The orchestrator does NOT background this — it waits for accelerate to exit (`set +e; (...); rc=$?; set -e`). If `rc ≠ 0`, the orchestrator prints the failure and exits 1 (the cleanup trap kills the trainer).
5. **Pick the new checkpoint.** `latest_wm_checkpoint()` returns the most-recent-mtime `checkpoint-*.pt` anywhere under `wm_checkpoints/`. This will be the file `cycle_<N>` just wrote. The orchestrator assigns it to `CURRENT_WM_CKPT` for the next phase A wave.

### 6.1 Phase B feature parity with WMFineTuner

The in-process `WMFineTuner` (reward_server.py:253) has features `train_wm.py` does NOT have. This script trades them for DDP-throughput:

| feature | WMFineTuner | phase B (train_wm.py) | notes |
|---|---|---|---|
| online buffer (rolling deque) | yes (`WM_BUFFER_SIZE=400`) | no — trains on ALL trajs in `train_sample.json` | see §7.1 |
| pretrain-replay mixing (anti-forgetting) | yes (`--wm-pretrain-root`/`--wm-mix-online`) | not invoked | known limitation |
| before/after sanity-check mp4s | yes (`--wm-sanity-check`) | not invoked | known limitation |
| `wm_finetune.jsonl` metrics for trainer's wandb tail | yes | no | `examples/train_utils_collect.py:832` will find no new lines, so `wm_finetune/*` wandb panels stay empty during phase B. Phase B's own loss curves are in `_logs/wm_ft_cycle_<N>.log` and in `train_wm.py`'s internal wandb tag. |
| ckpt-every-N-cycles | yes (`--wm-checkpoint-every 5`) | every cycle saves at least one ckpt | acceptable |

If you need any of those features, see §7 for follow-ups.

---

## 7. Known limitations / explicit deferrals

### 7.1 Phase B uses a rolling buffer of the last `WM_BUFFER_SIZE` eids

The filter step sorts the eids with cached latents numerically (eids are zero-padded monotonic ints assigned at collection time) and keeps only the largest `WM_BUFFER_SIZE` (default `400`). Older eids are dropped from the per-cycle manifest but their files stay on disk.

`WM_BUFFER_SIZE=400` matches the original `WMFineTuner` deque in `della_wm_loop_collect_ft_5gpu.sh:79`. Set `WM_BUFFER_SIZE=200` to make the FT window equal to the cycle trigger (`WM_UPDATE_EVERY`) so each cycle sees exactly the *newest* 200 trajs and no overlap with the previous cycle's data. Set `WM_BUFFER_SIZE=0` to disable the cap and train on every traj with cached latents.

Disk usage: `latent_videos/<cam>/<eid>.pt` files outside the rolling buffer are NOT deleted automatically. If disk fills up over a long run, manually prune old `latent_videos/<cam>/<eid>.pt` (and matching `annotation/train/<eid>.json` + `raw_videos/<cam>/<eid>.mp4` if you want to fully reclaim) between sbatch jobs.

### 7.2 No resume support

If sbatch is interrupted mid-cycle, the orchestrator does not pick up state from a previous run. The next invocation creates a fresh `JOB_TAG`-named `REWARD_ROOT` and starts cycle 0 over.

### 7.3 Trainer's `wm_finetune/*` wandb panels show per-cycle summaries

The orchestrator appends one JSON record to `_logs/wm_finetune.jsonl` after every successful phase B (`run_phase_b` body). Each record has:

- `cycle_n` / `cycles_done` — increments each FT cycle (so the wandb panel `wm_finetune/cycles_done` is a step function you can use to mark cycle boundaries on the SAC plots).
- `global_step` = `cycle_n × WM_TRAIN_STEPS_PER_CYCLE` — FT step counter.
- `elapsed_s` — wall time of that phase B (DDP launch overhead + training + ckpt save).
- `buffer_size` = `WM_BUFFER_SIZE`.
- `loss_first`, `loss_last`, `loss_mean` — best-effort parse from `_logs/wm_ft_cycle_<N>.log` via a regex over `loss[: =]<float>`. If the log format changes upstream the loss fields become 0.0 but the cycle markers still show up.

The trainer's existing tail at `examples/train_utils_collect.py:832-857` picks these up at the next round boundary and pushes them to `wm_finetune/*` in the same wandb run as the SAC metrics.

### 7.4 Phase B does not render sanity-check mp4s automatically

Run `examples/scripts/wm_overfit_sanity.sh` against any of the saved checkpoints manually. (The `overfit50_5gpu` template has a background watcher that does this — porting it here would conflict with the trainer using GPU 4.)

### 7.5 DDP throughput at `batch_size=4`

Default is `WM_TRAIN_BATCH_PER_GPU=4`, effective batch = 16, matching `della_wm_loop_overfit50_5gpu.sh`. If you see CUDA OOM during phase B (less likely than at pretrain time since the WM is already loaded once before scoring), drop to `WM_TRAIN_BATCH_PER_GPU=2` or `1`. At batch=1 per rank, all-reduce overhead dominates and you get a fraction of theoretical speedup, but it still works.

### 7.6 The trainer can stall mid-round during phase B

If the trainer happens to be on a Y-boundary while phase B is running, it blocks at `gather_fn`. `DSRL_REWARD_TIMEOUT_S=3600` (1 hour) covers the worst-case phase-B duration (≤ ~20 min DDP + ~1-2 min worker restart). If phase B + restart takes > 1 hour, trainer crashes. Bump the env var if you raise `WM_TRAIN_STEPS_PER_CYCLE` substantially.

### 7.7 Scoring inconsistency at the cycle boundary

The orchestrator triggers phase B based on `count_scores()`, not on Y-boundary alignment. A single trainer round (50 trajs) can straddle a cycle boundary — some scored under WM_n, some under WM_{n+1}. The reward-model fitter sees a mixed-target distribution for that one round. Subsequent rounds are clean.

If you need strict per-round WM-consistency: pick `WM_UPDATE_EVERY = k × SAC_UPDATE_EVERY` (integer multiple, currently 200 = 4 × 50, which IS a multiple — but trigger timing is mtime-driven, not exact-boundary-driven, so you still get a 1-2 traj race window).

---

## 8. Variable reference (must match script)

| variable | default | role |
|---|---|---|
| `TASK_SUITE` | `libero_goal` | LIBERO suite |
| `TASK_ID` | `1` | LIBERO task within suite |
| `POLICY` | `pi05` | base policy (`pi05` or `pi0`) |
| `MAX_TRAJS` | `1000000` | hard cap on total trajs; effectively "never reached" |
| `SAC_UPDATE_EVERY` | `50` | Y — trainer's round size (also `--reward_update_freq` and `--traj_batch_size` and `--scored_per_round`) |
| `WM_UPDATE_EVERY` | `200` | phase-B trigger interval (scored-trajs cumulative) |
| `WM_BUFFER_SIZE` | `400` | rolling-buffer cap: phase B sees at most the last N eids by numeric eid order. `0` disables the cap. Independent of `WM_UPDATE_EVERY`. |
| `WM_TRAIN_STEPS_PER_CYCLE` | `1000` | DDP grad steps per phase B |
| `WM_TRAIN_BATCH_PER_GPU` | `4` | DDP per-rank batch size; effective batch = 16 across 4 GPUs (matches `overfit50_5gpu`) |
| `WM_LR` | `1e-5` | DDP optimizer LR |
| `WM_CKPT_EVERY_STEPS` | `=WM_TRAIN_STEPS_PER_CYCLE` | `train_wm.py` checkpoint cadence |
| `WM_VAL_EVERY_STEPS` | `=WM_TRAIN_STEPS_PER_CYCLE` | `train_wm.py` val cadence |
| `WM_NUM_WORKERS` | `0` | DataLoader workers (avoid CPU OOM with fork copies of 9GB WM) |
| `WARMUP_TRAJS` | `20` | trainer's warmup; with `TRANSITIONS_PER_TRAJ=10`, gives `start_online_updates=200` so SAC fires by round 2 |
| `MULTI_GRAD_STEP` | `20` | trainer's grad steps per transition per round |
| `BASE_POLICY_PROB` | `0.5` | trainer's mixing |
| `TARGET_ENTROPY` | `3.5` | SAC entropy target |
| `SCORING_MODE` | `spread` | LPIPS scoring mode |
| `NUM_PASSES` | `5` | autoregressive starts per traj |
| `WINDOWS_PER_CALL` | `1` | windows per call (spread mode = 1) |
| `NUM_INFERENCE_STEPS` | `50` | diffusion denoising steps per window |
| `REWARD_GPUS` | `0,1,2,3` | WM GPU pool |
| `TRAINER_GPU` | `4` | SAC GPU |
| `DSRL_REWARD_TIMEOUT_S` | `3600` (env) | per-score await deadline; must cover worst-case phase-B duration |
| `PHASE_TRIGGER_POLL_S` | `15` | orchestrator's poll interval |
| `SERVER_READY_TIMEOUT_S` | `2400` | per-phase-A wave readiness deadline |

---

## 9. Invariants the reviewing agent should verify

1. ✅ `TRAINER_GPU` does not appear in `REWARD_GPUS`. Asserted at line ~211.
2. ✅ `--enable-wm-finetune` is NOT passed to phase-A workers. (Look at `start_phase_a` body.)
3. ✅ All 4 phase-A workers in cycle N use the **same** `--ckpt-path` (the cycle-N-1 output, or `WM_CKPT_INITIAL` for cycle 0).
4. ✅ Phase B is launched **synchronously** from the orchestrator (no `&`), so it blocks the loop until DDP finishes.
5. ✅ Trainer is launched **once** at script start and runs across all cycles; phase boundaries do NOT restart it.
6. ✅ `stop_phase_a()` recovers orphaned `.req.taken-*` claims so phase A restart can re-score them. (Without this, those eids would deadlock the trainer's await.)
7. ✅ `latest_wm_checkpoint()` finds files under `wm_checkpoints/cycle_<N>/` after phase B, not just files at the top-level (the script writes them in subdirs).
8. ✅ `DSRL_REWARD_TIMEOUT_S=3600` is set in the trainer's environment, not just declared in bash (the trainer reads it via `os.environ`).
9. ✅ Worker log filename includes `cycle<N>` so cycle-N logs don't overwrite cycle-(N-1) logs.
10. ✅ Phase B reads from `_ft_cycles/cycle_<N>/train_sample.json` (a filtered subset), NOT directly from `REWARD_ROOT/train_sample.json` (which the trainer appends to during the FT window).
11. ✅ The filter step DROPS any eid whose `latent_videos/{agentview,wrist}/<eid>.pt` does not exist on disk, so phase B's DataLoader can't trip on un-cached latents.
12. ✅ The filter step aborts (rc=3) if fewer than 5 eids survive — protects against accidentally fine-tuning on a tiny batch when the queue ran dry.
13. ✅ The cleanup trap kills both the trainer and any still-alive phase-A workers when the script exits (SLURM SIGTERM, error, or normal exit).
14. ✅ Phase B uses a rolling buffer of the most recent `WM_BUFFER_SIZE` eids (default 400). Older eids are dropped from the per-cycle manifest but their files stay on disk (see §7.1 for cleanup).

---

## 10. Quick-test checklist before letting it run for hours

After the first phase B completes (cycle 1), verify:

```bash
# 1. New checkpoint exists.
ls -lh "$REWARD_ROOT/wm_checkpoints/cycle_1/"
# Expect: at least one checkpoint-<N>.pt file.

# 2. Latest-ckpt picker found it.
grep "new WM checkpoint" "$LOG_DIR/"*.out  # or the slurm out
# Expect: a line like "new WM checkpoint: <REWARD_ROOT>/wm_checkpoints/cycle_1/checkpoint-1000.pt"

# 3. Phase A restart used the new ckpt.
grep "PHASE A start  cycle=1" "$LOG_DIR/"*.out
grep "ckpt=" "$LOG_DIR/reward_server_cycle1_w0_gpu0.log" | head -3
# Expect: the cycle_1 ckpt path.

# 4. No orphaned .taken-* leftover.
ls "$REWARD_ROOT/requests/"*.taken-* 2>/dev/null
# Expect: no files (all should be drained / restored to .req).

# 5. Trainer didn't time out on the FT-window await.
grep -i "timed out" "$LOG_DIR/trainer.log"
# Expect: empty.
```

If any of those fail, the orchestrator has a bug — DO NOT let it accumulate hours of compute.
