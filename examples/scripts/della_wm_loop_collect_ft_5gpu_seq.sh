#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --job-name=dsrl-wm-loop-collect-ft-seq
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yy4041@princeton.edu

source ~/.bashrc

# ============================================================================
# CONTINUOUS COLLECT + ALTERNATING 4-GPU SCORE / 4-GPU DDP WM FINE-TUNE
# ----------------------------------------------------------------------------
# Sequential-phase variant of della_wm_loop_collect_ft_5gpu.sh. The trainer
# (SAC + collection) runs on GPU 4 continuously. The other 4 GPUs alternate
# between two phases of equal GPU footprint:
#
#   PHASE A (scoring):  4 reward_server.py workers, one per GPU 0-3, share
#     <reward_root>/requests/ and produce <reward_root>/scores/<eid>.score.json
#     via the atomic-rename claim mechanism (_try_claim, reward_server.py).
#     WM fine-tune is DISABLED inside the servers — see the design doc for
#     why (single-process FT would diverge across 4 workers).
#
#   PHASE B (DDP FT):   accelerate launch --num_processes=4 of
#     openworld.training.world_model.train_wm on GPUs 0-3 with effective
#     batch = 4 × WM_TRAIN_BATCH_PER_GPU. Reads the same on-disk dataset
#     the scoring servers cached (latent_videos/{agentview,wrist}/<eid>.pt,
#     annotation/train/<eid>.json, train_sample.json). Outputs new
#     checkpoint-*.pt to <reward_root>/wm_checkpoints/cycle_<N>/.
#
# The bash orchestrator below alternates A → B → A → ... triggered by
# the count of .score.json files crossing the next N × WM_UPDATE_EVERY
# milestone. Phase A workers are killed cleanly between cycles; any
# .req.taken-* claims orphaned by the kill are renamed back to .req so
# the next phase-A wave picks them up under the new WM checkpoint.
#
# Trainer-side: launch_collect.py is unmodified. It drops .req files and
# awaits .score.json files. During phase B no workers serve those .reqs,
# so the trainer eventually blocks at its Y-boundary gather_fn() call.
# DSRL_REWARD_TIMEOUT_S is bumped to 3600s so a single phase-B window
# (~10-20 min wallclock for 1000 grad steps DDP) can't time out a wait.
#
# Companion doc: della_wm_loop_collect_ft_5gpu_seq.md
# References:
#   - della_wm_loop_policy_5gpu.sh         (4-GPU scoring template)
#   - della_wm_loop_overfit50_5gpu.sh      (4-GPU DDP FT template)
#   - della_wm_loop_collect_ft_5gpu.sh     (1-GPU score + 1-GPU FT predecessor)
# ============================================================================

# ----------------------------------------------------------------------------
# Task & run length
# ----------------------------------------------------------------------------
TASK_SUITE=libero_goal
TASK_ID="${TASK_ID:-1}"
POLICY="${POLICY:-pi05}"

MAX_TRAJS=1000000
MAX_STEPS=10000000

# ----------------------------------------------------------------------------
# Round structure (SAC + reward refit fires every SAC_UPDATE_EVERY trajs)
# ----------------------------------------------------------------------------
SAC_UPDATE_EVERY=50
TRAJ_BATCH=$SAC_UPDATE_EVERY
SCORED_PER_ROUND=$SAC_UPDATE_EVERY

# ----------------------------------------------------------------------------
# WM fine-tune cadence (orchestrator-driven, NOT in-server)
# ----------------------------------------------------------------------------
# Every WM_UPDATE_EVERY scored trajs the orchestrator stops the scoring
# workers and launches a DDP FT cycle for WM_TRAIN_STEPS_PER_CYCLE steps.
WM_UPDATE_EVERY=200
WM_TRAIN_STEPS_PER_CYCLE=1000
WM_TRAIN_BATCH_PER_GPU="${WM_TRAIN_BATCH_PER_GPU:-4}"   # effective batch = 4 × 4 = 16; matches overfit50_5gpu
WM_LR="${WM_LR:-1e-5}"
WM_MAX_GRAD_NORM=1.0
WM_NUM_WORKERS="${WM_NUM_WORKERS:-0}"  # see overfit50_5gpu for the RSS rationale
# Rolling-buffer size for phase B: each cycle trains on at most the last
# WM_BUFFER_SIZE eids with cached latents (sorted by eid). 0 = no cap
# (train on everything seen so far). Default 400 matches the original
# WMFineTuner deque in della_wm_loop_collect_ft_5gpu.sh.
WM_BUFFER_SIZE="${WM_BUFFER_SIZE:-400}"
# Checkpoint cadence within a cycle. Default: save once at the end of each
# cycle (= WM_TRAIN_STEPS_PER_CYCLE). Reduce to checkpoint more frequently
# inside a cycle (e.g. for resumability).
WM_CKPT_EVERY_STEPS="${WM_CKPT_EVERY_STEPS:-$WM_TRAIN_STEPS_PER_CYCLE}"
WM_VAL_EVERY_STEPS="${WM_VAL_EVERY_STEPS:-$WM_TRAIN_STEPS_PER_CYCLE}"

# ----------------------------------------------------------------------------
# SAC / data-collection policy
# ----------------------------------------------------------------------------
# 100 env steps / query_freq is what we observe per traj in practice (the
# 400-step LIBERO cap is rarely hit); set WARMUP_TRAJS accordingly so SAC
# fires by round 2.
WARMUP_TRAJS="${WARMUP_TRAJS:-20}"
MULTI_GRAD_STEP=20
ACTION_MAGNITUDE=1.0
BASE_POLICY_PROB=0.5
TARGET_ENTROPY="${TARGET_ENTROPY:-3.5}"

# ----------------------------------------------------------------------------
# Reward model (per-step LPIPS regressor on top of WM scores)
# ----------------------------------------------------------------------------
REWARD_GRAD_STEPS=200
REWARD_LR=3e-4
REWARD_LOSS_MODE=per_step

# ----------------------------------------------------------------------------
# WM scoring
# ----------------------------------------------------------------------------
SCORING_MODE=spread
NUM_PASSES=2               # 2 autoregressive rollouts per traj (samples)
WINDOWS_PER_CALL=3         # each rollout = 3 autoregressive windows = 12 predicted frames
RANDOM_SPREAD=1
START_FRAME=6
NUM_INFERENCE_STEPS=50
NUM_WINDOWS=6              # = NUM_PASSES × WINDOWS_PER_CALL (kept for legacy code paths)

# ----------------------------------------------------------------------------
# GPU layout (5 GPUs)
# ----------------------------------------------------------------------------
# REWARD_GPUS = the WM GPUs (shared between phase A scoring and phase B DDP).
# TRAINER_GPU = the SAC/data-collection GPU. Must not appear in REWARD_GPUS.
REWARD_GPUS="${REWARD_GPUS:-0,1,2,3}"
TRAINER_GPU="${TRAINER_GPU:-4}"

IFS=',' read -r -a REWARD_GPU_ARR <<< "$REWARD_GPUS"
NUM_REWARD_WORKERS=${#REWARD_GPU_ARR[@]}

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DSRL_ROOT="${DSRL_ROOT:-/scratch/gpfs/AM43/yy4041/dsrl_pi0}"
OPEN_WORLD_ROOT="${OPEN_WORLD_ROOT:-/scratch/gpfs/AM43/yy4041/open-world}"
WM_CKPT_INITIAL="${WM_CKPT:-/scratch/gpfs/AM43/yy4041/open-world/models/wm_training/libero_0429/checkpoint-36000.pt}"
WM_DATASET_ROOT="${WM_DATASET_ROOT:-/scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_processed}"

# ============================================================================
# Derived
# ============================================================================
DATE_DIR=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
JOB_TAG="${SLURM_JOB_ID:-local}_${TIME_TAG}_collect_ft_seq"
REWARD_ROOT="${REWARD_ROOT:-/scratch/gpfs/AM43/yy4041/playworld_rollout/$DATE_DIR/$JOB_TAG}"

if [ -z "${QUERY_FREQ:-}" ]; then
    if [ "$POLICY" = "pi05" ]; then
        QUERY_FREQ=10
    else
        QUERY_FREQ=20
    fi
fi

TRANSITIONS_PER_TRAJ=$(( 100 / QUERY_FREQ ))
SAC_CKPT_TRAJS=200
CHECKPOINT_INTERVAL=$(( SAC_CKPT_TRAJS * TRANSITIONS_PER_TRAJ * MULTI_GRAD_STEP ))

MAX_TRANSITIONS_PER_TRAJ=$TRANSITIONS_PER_TRAJ
START_ONLINE_UPDATES=$(( WARMUP_TRAJS * MAX_TRANSITIONS_PER_TRAJ ))

SERVER_READY_TIMEOUT_S=2400
PHASE_TRIGGER_POLL_S=15
PHASE_B_DRAIN_TIMEOUT_S=600

# ---------------------------------------------------------------------------
# Offline env
# ---------------------------------------------------------------------------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TORCH_HOME=/scratch/gpfs/AM43/yy4041/.cache/torch
export HF_HOME=/scratch/gpfs/AM43/yy4041/.cache/huggingface

# ---------------------------------------------------------------------------
# Verify caches (fail fast)
# ---------------------------------------------------------------------------
PI_CACHE="/scratch/gpfs/AM43/yy4041/.cache/openpi/openpi-assets/checkpoints/${POLICY}_libero"
REQUIRED=(
    "$DSRL_ROOT/.venv/bin/python"
    "$OPEN_WORLD_ROOT/.venv/bin/python"
    "$OPEN_WORLD_ROOT/.venv/bin/accelerate"
    "$OPEN_WORLD_ROOT/external/stable-video-diffusion-img2vid"
    "$OPEN_WORLD_ROOT/external/clip-vit-base-patch32"
    "$TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth"
    "$WM_CKPT_INITIAL"
    "$WM_DATASET_ROOT/stat.json"
    "$PI_CACHE"
)
for f in "${REQUIRED[@]}"; do
    if [ ! -e "$f" ]; then
        echo "[seq] FATAL: missing cached artifact: $f"
        echo "[seq] run setup_caches.sh on a login node first."
        exit 1
    fi
done

# ---- GPU sanity ----
declare -A GPU_SEEN
for g in "${REWARD_GPU_ARR[@]}"; do
    if [ "$g" = "$TRAINER_GPU" ]; then
        echo "[seq] FATAL: TRAINER_GPU=$TRAINER_GPU appears in REWARD_GPUS=$REWARD_GPUS"
        exit 1
    fi
    if [ -n "${GPU_SEEN[$g]:-}" ]; then
        echo "[seq] FATAL: REWARD_GPUS has duplicate id $g"
        exit 1
    fi
    GPU_SEEN[$g]=1
done

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
NEED=$(( NUM_REWARD_WORKERS + 1 ))
if [ "$NUM_GPUS" -lt "$NEED" ]; then
    echo "[seq] FATAL: only $NUM_GPUS GPU(s) visible, need $NEED."
    exit 1
fi

mkdir -p "$REWARD_ROOT"
LOG_DIR="$REWARD_ROOT/_logs"
mkdir -p "$LOG_DIR"
WM_CKPT_ROOT="$REWARD_ROOT/wm_checkpoints"
mkdir -p "$WM_CKPT_ROOT"

# ---------------------------------------------------------------------------
# Snapshot config
# ---------------------------------------------------------------------------
DSRL_GIT_SHA=$(git -C "$DSRL_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
OPEN_WORLD_GIT_SHA=$(git -C "$OPEN_WORLD_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
CONFIG_PATH="$REWARD_ROOT/config.json"
cat > "$CONFIG_PATH" <<EOF
{
  "meta": {
    "job_tag": "$JOB_TAG",
    "experiment_type": "continuous_collect_seq_4gpu_score_then_4gpu_ddp_ft",
    "started_at": "$(date -Iseconds)",
    "host": "$(hostname)",
    "script": "$0",
    "dsrl_git_sha": "$DSRL_GIT_SHA",
    "open_world_git_sha": "$OPEN_WORLD_GIT_SHA"
  },
  "paths": {
    "dsrl_root": "$DSRL_ROOT",
    "open_world_root": "$OPEN_WORLD_ROOT",
    "reward_root": "$REWARD_ROOT",
    "wm_ckpt_initial": "$WM_CKPT_INITIAL",
    "wm_dataset_root": "$WM_DATASET_ROOT"
  },
  "task": {"task_suite": "$TASK_SUITE", "task_id": $TASK_ID},
  "policy": {
    "name": "$POLICY", "query_freq": $QUERY_FREQ,
    "base_policy_prob": $BASE_POLICY_PROB, "action_magnitude": $ACTION_MAGNITUDE
  },
  "sac": {
    "update_every_trajs": $SAC_UPDATE_EVERY,
    "multi_grad_step": $MULTI_GRAD_STEP,
    "target_entropy": "$TARGET_ENTROPY",
    "ckpt_every_trajs": $SAC_CKPT_TRAJS,
    "checkpoint_interval_sac_steps": $CHECKPOINT_INTERVAL,
    "warmup_trajs": $WARMUP_TRAJS,
    "start_online_updates": $START_ONLINE_UPDATES
  },
  "wm_finetune": {
    "mode": "orchestrated_phase_b_ddp",
    "update_every_trajs": $WM_UPDATE_EVERY,
    "train_steps_per_cycle": $WM_TRAIN_STEPS_PER_CYCLE,
    "per_gpu_batch": $WM_TRAIN_BATCH_PER_GPU,
    "effective_batch": $((WM_TRAIN_BATCH_PER_GPU * NUM_REWARD_WORKERS)),
    "lr": $WM_LR,
    "checkpoint_every_steps": $WM_CKPT_EVERY_STEPS,
    "validation_every_steps": $WM_VAL_EVERY_STEPS
  },
  "wm_scoring": {
    "scoring_mode": "$SCORING_MODE",
    "num_passes": $NUM_PASSES,
    "windows_per_call": $WINDOWS_PER_CALL,
    "random_spread": $RANDOM_SPREAD,
    "start_frame": $START_FRAME,
    "num_inference_steps": $NUM_INFERENCE_STEPS
  },
  "gpus": {
    "reward_gpus": "$REWARD_GPUS",
    "trainer_gpu": $TRAINER_GPU,
    "num_reward_workers": $NUM_REWARD_WORKERS
  }
}
EOF
echo "[seq] wrote experiment config to $CONFIG_PATH"

echo "[seq] REWARD_ROOT=$REWARD_ROOT"
echo "[seq] POLICY=$POLICY  QUERY_FREQ=$QUERY_FREQ"
echo "[seq] SAC update_every=$SAC_UPDATE_EVERY trajs  multi_grad_step=$MULTI_GRAD_STEP  target_entropy=$TARGET_ENTROPY"
echo "[seq] SAC ckpt every ~$SAC_CKPT_TRAJS trajs  (checkpoint_interval=$CHECKPOINT_INTERVAL sac steps)"
echo "[seq] WM FT cycle every $WM_UPDATE_EVERY scored trajs, $WM_TRAIN_STEPS_PER_CYCLE DDP steps per cycle"
echo "[seq] WM FT per-gpu batch=$WM_TRAIN_BATCH_PER_GPU  effective=$((WM_TRAIN_BATCH_PER_GPU * NUM_REWARD_WORKERS))  lr=$WM_LR"
echo "[seq] GPUs: REWARD_GPUS=$REWARD_GPUS  TRAINER_GPU=$TRAINER_GPU"
nvidia-smi -L | head -8 || true

# ---------------------------------------------------------------------------
# Trainer (continuous, runs across all cycles)
# ---------------------------------------------------------------------------
TRAINER_LOG="$LOG_DIR/trainer.log"
echo "[seq] starting trainer on GPU $TRAINER_GPU  (logs -> $TRAINER_LOG)"

cd "$DSRL_ROOT"
export PYTHONPATH="$DSRL_ROOT:${PYTHONPATH:-}"
export DSRL_REWARD_ROOT="$REWARD_ROOT"
# Bumped to cover phase-B duration (DDP FT ~10-20 min + restart). Phase A
# absent → trainer blocks at gather_fn for the worst-case phase-B wallclock.
export DSRL_REWARD_TIMEOUT_S=3600
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$TRAINER_GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS="ignore::RuntimeWarning:subprocess"

source "$DSRL_ROOT/.venv/bin/activate"

CUDA_VISIBLE_DEVICES=$TRAINER_GPU \
EXP="$DSRL_ROOT/logs/dsrl_wm_$JOB_TAG" \
PYTHONUNBUFFERED=1 \
python3 examples/launch_collect.py \
    --algorithm pixel_sac \
    --env libero \
    --policy "$POLICY" \
    --prefix "dsrl_pi0_libero_wm_$JOB_TAG" \
    --wandb_project DSRL_pi0_libero_wm_collect_ft_seq \
    --batch_size 256 \
    --discount 0.999 \
    --seed 0 \
    --max_steps "$MAX_STEPS" \
    --eval_interval 10000 \
    --log_interval 200 \
    --eval_episodes 5 \
    --multi_grad_step "$MULTI_GRAD_STEP" \
    --start_online_updates "$START_ONLINE_UPDATES" \
    --resize_image 64 \
    --action_magnitude "$ACTION_MAGNITUDE" \
    --query_freq "$QUERY_FREQ" \
    --hidden_dims 128 \
    --use_reward_model 1 \
    --reward_fn examples.reward_fn:wm_score \
    --traj_batch_size "$TRAJ_BATCH" \
    --reward_grad_steps "$REWARD_GRAD_STEPS" \
    --reward_lr "$REWARD_LR" \
    --reward_relabel_buffer 0 \
    --reward_loss_mode "$REWARD_LOSS_MODE" \
    --scene_reset_freq 1 \
    --reward_update_freq "$SAC_UPDATE_EVERY" \
    --scored_per_round "$SCORED_PER_ROUND" \
    --base_policy_prob "$BASE_POLICY_PROB" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --save_dir "$REWARD_ROOT" \
    --save_split train \
    --task_suite_name "$TASK_SUITE" \
    --task_id "$TASK_ID" \
    --cam_resolution 256 \
    --fps 20 \
    --sample_stride 2 \
    --sample_start_offset 6 \
    --sample_wm_down_sample 4 \
    --max_trajs "$MAX_TRAJS" \
    --target_entropy "$TARGET_ENTROPY" \
    > "$TRAINER_LOG" 2>&1 &
TRAINER_PID=$!
echo "[seq] trainer pid=$TRAINER_PID"

# ---------------------------------------------------------------------------
# Phase-A / Phase-B helpers
# ---------------------------------------------------------------------------
SERVER_PIDS=()
SERVER_LOGS=()

start_phase_a() {
    local ckpt="$1" cycle_n="$2"
    SERVER_PIDS=()
    SERVER_LOGS=()
    echo "[seq] ====== PHASE A start  cycle=$cycle_n  ckpt=$(basename "$ckpt") ======"
    for idx in "${!REWARD_GPU_ARR[@]}"; do
        local g="${REWARD_GPU_ARR[$idx]}"
        local log="$LOG_DIR/reward_server_cycle${cycle_n}_w${idx}_gpu${g}.log"
        SERVER_LOGS+=("$log")
        echo "[seq] starting worker $idx on GPU $g  (logs -> $log)"
        (
            cd "$OPEN_WORLD_ROOT"
            CUDA_VISIBLE_DEVICES=$g \
            OPEN_WORLD_ROOT="$OPEN_WORLD_ROOT" \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            PYTHONUNBUFFERED=1 \
            "$OPEN_WORLD_ROOT/.venv/bin/python" -u \
                "$DSRL_ROOT/examples/reward_model/reward_server.py" \
                --reward-root "$REWARD_ROOT" \
                --ckpt-path "$ckpt" \
                --dataset-root "$WM_DATASET_ROOT" \
                --num-windows "$NUM_WINDOWS" \
                --num-passes "$NUM_PASSES" \
                --windows-per-call "$WINDOWS_PER_CALL" \
                $( [ "$RANDOM_SPREAD" = "1" ] && echo "--random-spread" ) \
                --start-frame "$START_FRAME" \
                --num-inference-steps "$NUM_INFERENCE_STEPS" \
                --scoring-mode "$SCORING_MODE" \
                --device "cuda:0" \
                --worker-id "c${cycle_n}_w${idx}_gpu${g}" \
                > "$log" 2>&1
        ) &
        SERVER_PIDS+=("$!")
        echo "[seq]   worker $idx pid=${SERVER_PIDS[$idx]}"
    done

    echo "[seq] waiting up to ${SERVER_READY_TIMEOUT_S}s for all $NUM_REWARD_WORKERS workers to load..."
    local deadline=$(($(date +%s) + SERVER_READY_TIMEOUT_S))
    local ready=0
    while [ "$ready" -lt "$NUM_REWARD_WORKERS" ]; do
        ready=0
        for i in "${!SERVER_PIDS[@]}"; do
            local pid="${SERVER_PIDS[$i]}"
            local log="${SERVER_LOGS[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[seq] FATAL: phase-A worker $i died before ready"
                tail -80 "$log" || true
                exit 1
            fi
            if grep -q "ready. polling" "$log" 2>/dev/null; then
                ready=$((ready + 1))
            fi
        done
        if [ $(date +%s) -gt $deadline ]; then
            echo "[seq] FATAL: only $ready/$NUM_REWARD_WORKERS workers ready after ${SERVER_READY_TIMEOUT_S}s"
            for log in "${SERVER_LOGS[@]}"; do
                echo "----- $log -----"; tail -40 "$log" || true
            done
            exit 1
        fi
        [ "$ready" -lt "$NUM_REWARD_WORKERS" ] && sleep 5
    done
    echo "[seq] all $NUM_REWARD_WORKERS phase-A workers ready (cycle=$cycle_n)."
}

stop_phase_a() {
    echo "[seq] ====== PHASE A stop ======"
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 3
    for pid in "${SERVER_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    # Recover claims that died mid-scoring: rename *.req.taken-* and
    # *.wm_only.taken-* back so the next wave of workers picks them up.
    local requests_dir="$REWARD_ROOT/requests"
    if [ -d "$requests_dir" ]; then
        local recovered=0
        for f in "$requests_dir"/*.req.taken-* "$requests_dir"/*.wm_only.taken-*; do
            [ -e "$f" ] || continue
            # Strip the .taken-<worker> suffix.
            local restored="${f%.taken-*}"
            if [ ! -e "$restored" ]; then
                mv "$f" "$restored" 2>/dev/null && recovered=$((recovered + 1))
            else
                rm -f "$f" 2>/dev/null || true
            fi
        done
        echo "[seq] recovered $recovered orphaned claim(s) in $requests_dir/"
    fi
    SERVER_PIDS=()
    SERVER_LOGS=()
}

# Latest checkpoint on disk (across all cycles). Returns absolute path on
# stdout, empty string if none.
latest_wm_checkpoint() {
    # Sort by mtime, newest first; works across cycle_<N>/ subdirs.
    find "$WM_CKPT_ROOT" -mindepth 1 -name "checkpoint-*.pt" -printf "%T@ %p\n" 2>/dev/null \
        | sort -nr | awk 'NR==1{print $2}'
}

# Phase B writes a per-cycle config.py and invokes accelerate launch.
run_phase_b() {
    local cycle_n="$1" ckpt_in="$2"
    echo "[seq] ====== PHASE B start  cycle=$cycle_n  ckpt_in=$(basename "$ckpt_in") ======"

    local out_dir="$WM_CKPT_ROOT/cycle_${cycle_n}"
    mkdir -p "$out_dir"

    # Per-cycle dataset view. The trainer continuously appends to the
    # canonical train_sample.json, but the latent .pt files are only
    # written by the scoring path. So train_sample.json can contain eids
    # whose latents don't exist yet (mid-scoring at kill time, or freshly
    # rolled-out but un-scored). To keep phase B's DataLoader from
    # crashing on those, we build a per-cycle filtered view:
    #   <cycle_view>/annotation         -> symlink to REWARD_ROOT/annotation
    #   <cycle_view>/latent_videos      -> symlink to REWARD_ROOT/latent_videos
    #   <cycle_view>/raw_videos         -> symlink to REWARD_ROOT/raw_videos
    #   <cycle_view>/train_sample.json  -> filtered: only eids with cached latents
    #   <cycle_view>/val_sample.json    -> same as train_sample.json (we accept train==val)
    # phase B's dataset_root_path then points at <cycle_view>.
    local cycle_view="$REWARD_ROOT/_ft_cycles/cycle_${cycle_n}"
    mkdir -p "$cycle_view"
    ln -sfn "$REWARD_ROOT/annotation"    "$cycle_view/annotation"
    ln -sfn "$REWARD_ROOT/latent_videos" "$cycle_view/latent_videos"
    if [ -d "$REWARD_ROOT/raw_videos" ]; then
        ln -sfn "$REWARD_ROOT/raw_videos" "$cycle_view/raw_videos"
    fi
    # annotation/val: dataset loader looks for annotation/val/<eid>.json;
    # we point val at train (overfit50_5gpu does the same).
    if [ ! -e "$REWARD_ROOT/annotation/val" ]; then
        ln -sfn "train" "$REWARD_ROOT/annotation/val"
    fi

    # Filter train_sample.json:
    #   1. drop entries whose latents aren't on disk yet (un-scored / mid-claim).
    #   2. apply rolling-buffer cap: keep only the last WM_BUFFER_SIZE
    #      eids (by numeric eid order). 0 = no cap.
    REWARD_ROOT="$REWARD_ROOT" CYCLE_VIEW="$cycle_view" WM_BUFFER_SIZE="$WM_BUFFER_SIZE" \
    "$DSRL_ROOT/.venv/bin/python" - <<'PYEOF'
import json, os, sys
reward_root = os.environ["REWARD_ROOT"]
cycle_view = os.environ["CYCLE_VIEW"]
buf_cap = int(os.environ.get("WM_BUFFER_SIZE", "0"))
src = os.path.join(reward_root, "train_sample.json")
if not os.path.exists(src):
    print(f"[seq:filter] FATAL: {src} does not exist; trainer hasn't written any trajs yet?", flush=True)
    sys.exit(2)
with open(src) as f:
    entries = json.load(f)
agentview_dir = os.path.join(reward_root, "latent_videos", "agentview")
wrist_dir = os.path.join(reward_root, "latent_videos", "wrist")
eids_seen = set()
eids_with_latents = set()
for entry in entries:
    eid = entry["episode_id"]
    eids_seen.add(eid)
    if eid in eids_with_latents:
        continue
    if (os.path.isfile(os.path.join(agentview_dir, f"{eid}.pt"))
            and os.path.isfile(os.path.join(wrist_dir, f"{eid}.pt"))):
        eids_with_latents.add(eid)

# Rolling-buffer cap: sort by numeric eid (eids are zero-padded ints), keep
# the largest N. This selects the most recently collected trajs because
# eids are assigned monotonically.
eids_sorted = sorted(eids_with_latents, key=lambda s: int(s))
if buf_cap > 0 and len(eids_sorted) > buf_cap:
    keep = set(eids_sorted[-buf_cap:])
else:
    keep = set(eids_sorted)

filtered = [e for e in entries if e["episode_id"] in keep]
print(f"[seq:filter] kept {len(filtered)} entries / {len(entries)} total  "
      f"(eids: {len(keep)} kept / {len(eids_with_latents)} with latents / "
      f"{len(eids_seen)} in manifest; cap={buf_cap or 'unlimited'})", flush=True)
if len(keep) < 5:
    print(f"[seq:filter] FATAL: only {len(keep)} eids would be used; refusing to fine-tune.", flush=True)
    sys.exit(3)
dst_train = os.path.join(cycle_view, "train_sample.json")
dst_val = os.path.join(cycle_view, "val_sample.json")
with open(dst_train, "w") as f:
    json.dump(filtered, f)
with open(dst_val, "w") as f:
    json.dump(filtered, f)
PYEOF
    local filter_rc=$?
    if [ "$filter_rc" -ne 0 ]; then
        echo "[seq] FATAL: filtering train_sample.json failed rc=$filter_rc"
        return $filter_rc
    fi

    local cfg_path="$LOG_DIR/wm_train_config_cycle_${cycle_n}.py"
    cat > "$cfg_path" <<PYEOF
"""Auto-generated FT config for cycle ${cycle_n}.

Written by della_wm_loop_collect_ft_5gpu_seq.sh. Points
openworld.training.world_model.train_wm at the trajs collected so far in
${REWARD_ROOT} as a single-suite dataset (name="."). Resumes the WM from
${ckpt_in}. New checkpoints land under ${out_dir}/.
"""

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    return LiberoWMArgs(
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path="${ckpt_in}",

        # Single-suite dataset rooted at the per-cycle filtered view so the
        # loader only sees eids whose latents are on disk. stat.json is
        # resolved via the loader's meta_root fallback (pretrain root).
        dataset_root_path="${cycle_view}",
        dataset_meta_info_path="${WM_DATASET_ROOT}",
        dataset_names=".",
        dataset_cfgs=".",
        prob=(1.0,),

        train_batch_size=${WM_TRAIN_BATCH_PER_GPU},
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        num_workers=${WM_NUM_WORKERS},

        learning_rate=${WM_LR},
        max_train_steps=${WM_TRAIN_STEPS_PER_CYCLE},
        checkpointing_steps=${WM_CKPT_EVERY_STEPS},
        validation_steps=${WM_VAL_EVERY_STEPS},
        max_grad_norm=${WM_MAX_GRAD_NORM},

        num_cams=2,
        num_frames=5,
        num_history=6,
        action_dim=7,
        down_sample=4,

        flow_map_type="flow_matching",
        distance_conditioning=False,

        tag="collect_ft_seq_${JOB_TAG}_cycle${cycle_n}",
    )
PYEOF
    echo "[seq] wrote $cfg_path"

    local phase_b_log="$LOG_DIR/wm_ft_cycle_${cycle_n}.log"
    echo "[seq] launching $NUM_REWARD_WORKERS-way DDP on REWARD_GPUS=$REWARD_GPUS (logs -> $phase_b_log)"

    local phase_b_t0=$(date +%s)
    set +e
    (
        cd "$OPEN_WORLD_ROOT"
        CUDA_VISIBLE_DEVICES=$REWARD_GPUS \
        OPEN_WORLD_ROOT="$OPEN_WORLD_ROOT" \
        PYTHONPATH="$OPEN_WORLD_ROOT:${PYTHONPATH:-}" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PYTHONUNBUFFERED=1 \
        "$OPEN_WORLD_ROOT/.venv/bin/accelerate" launch \
            --num_processes "$NUM_REWARD_WORKERS" \
            --mixed_precision fp16 \
            -m openworld.training.world_model.train_wm \
            --config "$cfg_path" \
            --output_dir "$out_dir" \
            > "$phase_b_log" 2>&1
    )
    local rc=$?
    set -e
    local phase_b_elapsed=$(( $(date +%s) - phase_b_t0 ))
    if [ "$rc" -ne 0 ]; then
        echo "[seq] !!! phase B cycle=$cycle_n failed (rc=$rc) after ${phase_b_elapsed}s. See $phase_b_log"
        echo "[seq] aborting orchestrator; trainer will be cleaned up."
        return $rc
    fi

    # Emit one summary record to wm_finetune.jsonl so the trainer's wandb
    # tail (examples/train_utils_collect.py around line 832) picks it up
    # and the wm_finetune/* panels show progress. Best-effort loss parsing
    # from the phase B log — if format changes upstream we still get the
    # cycle event markers (cycles_done, global_step, elapsed_s).
    LOG_PATH="$phase_b_log" \
    OUT_PATH="$LOG_DIR/wm_finetune.jsonl" \
    CYCLE_N="$cycle_n" \
    ELAPSED_S="$phase_b_elapsed" \
    GLOBAL_STEP=$((cycle_n * WM_TRAIN_STEPS_PER_CYCLE)) \
    BUFFER_SIZE="$WM_BUFFER_SIZE" \
    "$DSRL_ROOT/.venv/bin/python" - <<'PYEOF'
import json, os, re
log_path = os.environ["LOG_PATH"]
out_path = os.environ["OUT_PATH"]
losses = []
if os.path.exists(log_path):
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            # Match 'loss: 0.123', 'loss=0.123', "'loss': 0.123", 'loss":0.123', etc.
            for m in re.finditer(r"\bloss[\"'\s:=]+(-?\d+\.\d+(?:[eE][+\-]?\d+)?)", line):
                try:
                    losses.append(float(m.group(1)))
                except ValueError:
                    pass
rec = {
    "cycle_n": int(os.environ["CYCLE_N"]),
    "cycles_done": int(os.environ["CYCLE_N"]),
    "global_step": int(os.environ["GLOBAL_STEP"]),
    "elapsed_s": float(os.environ["ELAPSED_S"]),
    "buffer_size": int(os.environ["BUFFER_SIZE"]),
    "loss_first": losses[0] if losses else 0.0,
    "loss_last":  losses[-1] if losses else 0.0,
    "loss_mean":  (sum(losses) / len(losses)) if losses else 0.0,
    "loss_n_parsed": len(losses),
}
with open(out_path, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"[seq:metrics] cycle {rec['cycle_n']} summary: "
      f"elapsed={rec['elapsed_s']:.0f}s  parsed_losses={rec['loss_n_parsed']}  "
      f"loss_first={rec['loss_first']:.4f}  loss_last={rec['loss_last']:.4f}  "
      f"-> {out_path}", flush=True)
PYEOF

    echo "[seq] phase B cycle=$cycle_n complete in ${phase_b_elapsed}s."
    return 0
}

count_scores() {
    find "$REWARD_ROOT/scores" -maxdepth 1 -name "*.score.json" 2>/dev/null | wc -l
}

# ---------------------------------------------------------------------------
# Cleanup trap
# ---------------------------------------------------------------------------
cleanup_all() {
    echo "[seq] cleanup_all firing"
    if [ -n "${TRAINER_PID:-}" ] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "[seq] stopping trainer (pid=$TRAINER_PID)"
        kill "$TRAINER_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$TRAINER_PID" 2>/dev/null || true
    fi
    stop_phase_a || true
}
trap cleanup_all INT TERM EXIT

# Stream trainer log to stdout so slurm output is informative.
tail -n +1 -F "$TRAINER_LOG" --pid="$TRAINER_PID" &
TAIL_PID=$!

# ---------------------------------------------------------------------------
# Main orchestrator loop
# ---------------------------------------------------------------------------
CURRENT_WM_CKPT="$WM_CKPT_INITIAL"
cycle_n=0
next_threshold="$WM_UPDATE_EVERY"

# Phase A starts immediately. The trainer is alive and dropping .req files;
# without workers nothing gets scored, so kick off phase A before idling.
start_phase_a "$CURRENT_WM_CKPT" "$cycle_n"

echo "[seq] orchestrator entering main loop. next FT threshold = $next_threshold scores."
while kill -0 "$TRAINER_PID" 2>/dev/null; do
    n_scores=$(count_scores)
    if [ "$n_scores" -ge "$next_threshold" ]; then
        cycle_n=$((cycle_n + 1))
        echo "[seq] threshold hit: scores=$n_scores >= $next_threshold  → trigger phase B cycle $cycle_n"
        stop_phase_a
        if ! run_phase_b "$cycle_n" "$CURRENT_WM_CKPT"; then
            echo "[seq] phase B failed; exiting orchestrator."
            exit 1
        fi
        # Pick the freshest checkpoint produced this cycle.
        new_ckpt=$(latest_wm_checkpoint)
        if [ -z "$new_ckpt" ]; then
            echo "[seq] FATAL: phase B produced no checkpoint under $WM_CKPT_ROOT/cycle_${cycle_n}/"
            exit 1
        fi
        CURRENT_WM_CKPT="$new_ckpt"
        next_threshold=$((next_threshold + WM_UPDATE_EVERY))
        echo "[seq] new WM checkpoint: $CURRENT_WM_CKPT  → restarting phase A; next threshold=$next_threshold"
        start_phase_a "$CURRENT_WM_CKPT" "$cycle_n"
    else
        sleep "$PHASE_TRIGGER_POLL_S"
    fi
done

echo "[seq] trainer exited; orchestrator loop done."
kill "$TAIL_PID" 2>/dev/null || true
echo "[seq] final ls of WM ckpts (latest cycle dirs):"
ls -lh "$WM_CKPT_ROOT" 2>/dev/null | tail -20 || true
