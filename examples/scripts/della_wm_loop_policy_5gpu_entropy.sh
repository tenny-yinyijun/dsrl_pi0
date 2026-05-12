source ~/.bashrc
# set -euo pipefail

# ============================================================================
# POLICY-ONLY STABILITY TEST (frozen WM) — 5-GPU + ENTROPY-BONUS VARIANT
# ----------------------------------------------------------------------------
# Identical to della_wm_loop_policy_5gpu.sh, plus an explicit entropy
# bonus on the SAC actor to encourage more diverse exploration noise.
#
# How the bonus works: SAC already has auto-tuned temperature
# (`alpha`) that drives the actor entropy toward `target_entropy`. The
# default is `target_entropy = -action_dim / 2` (i.e. a relatively
# deterministic policy). This script overrides `target_entropy` to a
# HIGHER value (less negative / positive), which makes the auto-tuner
# raise `alpha` until the actor maintains higher entropy in its action
# distribution. Net effect: the SAC explores more, producing a more
# diverse set of (state, action) pairs in the replay buffer.
#
# Knob: TARGET_ENTROPY (env-overridable).
#   * auto (default in non-entropy variant): -action_dim/2 ≈ -3.5 for a
#     7-d action — fairly deterministic at convergence.
#   *  0.0 : moderate diversity (this script's default).
#   * +5.0 to +10.0 : strong exploration; risks instability.
#   * -3.5 reproduces the non-entropy variant.
# Larger values → more diverse actions, but harder to converge to a
# tight optimum. Sweep in [0, 5] first; only push beyond if rollouts
# still look too similar.
#
# Everything else identical to della_wm_loop_policy_5gpu.sh:
#   * 4 reward_server workers on REWARD_GPUS, 1 trainer on TRAINER_GPU.
#   * Parallel scoring via atomic-claim rename (see reward_server.py).
#   * WM fine-tuning DISABLED.
# ============================================================================

# ----------------------------------------------------------------------------
# Task & run length
# ----------------------------------------------------------------------------
# TASK_SUITE=libero_90; TASK_ID=57
TASK_SUITE=libero_goal
TASK_ID="${TASK_ID:-1}"          # env-overridable for task sweeps

POLICY="${POLICY:-pi05}"         # pi0 | pi05  (also sets QUERY_FREQ below); env-overridable

MAX_TRAJS=1000000
MAX_STEPS=500000

# ----------------------------------------------------------------------------
# Round structure
# ----------------------------------------------------------------------------
# Per-round: ROUND_SIZE freshly collected trajs, all scored, refit reward
# model + SAC update. With 4 parallel scoring workers the round budget can
# be bigger than the single-GPU variant — each traj's scoring overlaps
# with the others.
ROUND_SIZE=10
SCORED_PER_ROUND=$ROUND_SIZE   # score every traj in the round
TRAJ_BATCH=$ROUND_SIZE         # reward-model batch = the round

# ----------------------------------------------------------------------------
# Warmup
# ----------------------------------------------------------------------------
WARMUP_TRAJS="${WARMUP_TRAJS:-20}"

# ----------------------------------------------------------------------------
# SAC / data-collection policy
# ----------------------------------------------------------------------------
MULTI_GRAD_STEP=10             # SAC gradient updates per env transition
ACTION_MAGNITUDE=1.0
BASE_POLICY_PROB=0.5
CHECKPOINT_INTERVAL=5000

# ----- Entropy bonus -----
# Target entropy for SAC's auto-tuned alpha. Higher = more diverse
# actions. See header for value guidance. Default `auto` reproduces
# stock SAC behavior; we set 0.0 here to nudge exploration up.
TARGET_ENTROPY="${TARGET_ENTROPY:-3.5}"

# ----------------------------------------------------------------------------
# Reward model
# ----------------------------------------------------------------------------
REWARD_GRAD_STEPS=200
REWARD_LR=3e-4
REWARD_LOSS_MODE=per_step      # per_step | traj

# ----------------------------------------------------------------------------
# World-model scoring (frozen WM)
# ----------------------------------------------------------------------------
SCORING_MODE=spread
NUM_PASSES=5
WINDOWS_PER_CALL=1
RANDOM_SPREAD=1
START_FRAME=6
NUM_INFERENCE_STEPS=50
NUM_WINDOWS=5

# ----------------------------------------------------------------------------
# WM fine-tuning — DISABLED in this script
# ----------------------------------------------------------------------------
ENABLE_WM_FINETUNE=0
WM_SANITY_CHECK=0

# ----------------------------------------------------------------------------
# GPU layout
# ----------------------------------------------------------------------------
# REWARD_GPUS is a comma-separated list (any number of GPUs); one worker
# per entry. TRAINER_GPU must not appear in REWARD_GPUS.
REWARD_GPUS="${REWARD_GPUS:-0,1,2,3}"
TRAINER_GPU="${TRAINER_GPU:-4}"

# Parse REWARD_GPUS into an array so we can iterate.
IFS=',' read -r -a REWARD_GPU_ARR <<< "$REWARD_GPUS"
NUM_REWARD_WORKERS=${#REWARD_GPU_ARR[@]}

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DSRL_ROOT="${DSRL_ROOT:-/scratch/gpfs/AM43/yy4041/dsrl_pi0}"
OPEN_WORLD_ROOT="${OPEN_WORLD_ROOT:-/scratch/gpfs/AM43/yy4041/open-world}"
WM_CKPT="${WM_CKPT:-/scratch/gpfs/AM43/yy4041/open-world/models/wm_training/libero_0429/checkpoint-36000.pt}"
WM_DATASET_ROOT="${WM_DATASET_ROOT:-/scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_processed}"

# ============================================================================
# Derived (rarely needs editing)
# ============================================================================
DATE_DIR=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
JOB_TAG="${SLURM_JOB_ID:-local}_${TIME_TAG}_policy_5gpu_entropy"
REWARD_ROOT="${REWARD_ROOT:-/scratch/gpfs/AM43/yy4041/playworld_rollout/$DATE_DIR/$JOB_TAG}"

# QUERY_FREQ defaults to 10 for pi05, 20 for pi0; override via env.
if [ -z "${QUERY_FREQ:-}" ]; then
    if [ "$POLICY" = "pi05" ]; then
        QUERY_FREQ=10
    else
        QUERY_FREQ=20
    fi
fi

MAX_TRANSITIONS_PER_TRAJ=$(( 400 / QUERY_FREQ ))
START_ONLINE_UPDATES=$(( WARMUP_TRAJS * MAX_TRANSITIONS_PER_TRAJ ))

SERVER_READY_TIMEOUT_S=2400

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
    "$OPEN_WORLD_ROOT/external/stable-video-diffusion-img2vid"
    "$OPEN_WORLD_ROOT/external/clip-vit-base-patch32"
    "$TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth"
    "$WM_CKPT"
    "$WM_DATASET_ROOT/stat.json"
    "$PI_CACHE"
)
for f in "${REQUIRED[@]}"; do
    if [ ! -e "$f" ]; then
        echo "[policy5gpu] FATAL: missing cached artifact: $f"
        echo "[policy5gpu] run setup_caches.sh on a login node first."
        exit 1
    fi
done

# ---- GPU sanity ----
# REWARD_GPUS must not contain TRAINER_GPU; all must be distinct.
declare -A GPU_SEEN
for g in "${REWARD_GPU_ARR[@]}"; do
    if [ "$g" = "$TRAINER_GPU" ]; then
        echo "[policy5gpu] FATAL: TRAINER_GPU=$TRAINER_GPU appears in REWARD_GPUS=$REWARD_GPUS"
        exit 1
    fi
    if [ -n "${GPU_SEEN[$g]:-}" ]; then
        echo "[policy5gpu] FATAL: REWARD_GPUS has duplicate id $g"
        exit 1
    fi
    GPU_SEEN[$g]=1
done

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
NEED=$(( NUM_REWARD_WORKERS + 1 ))
if [ "$NUM_GPUS" -lt "$NEED" ]; then
    echo "[policy5gpu] FATAL: only $NUM_GPUS GPU(s) visible, need $NEED ($NUM_REWARD_WORKERS reward + 1 trainer)."
    exit 1
fi

mkdir -p "$REWARD_ROOT"
LOG_DIR="$REWARD_ROOT/_logs"
mkdir -p "$LOG_DIR"

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
    "experiment_type": "policy_only_5gpu_parallel_scoring_entropy_bonus",
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
    "wm_ckpt": "$WM_CKPT",
    "wm_dataset_root": "$WM_DATASET_ROOT"
  },
  "task": {
    "task_suite": "$TASK_SUITE",
    "task_id": $TASK_ID
  },
  "policy": {
    "name": "$POLICY",
    "query_freq": $QUERY_FREQ,
    "base_policy_prob": $BASE_POLICY_PROB,
    "action_magnitude": $ACTION_MAGNITUDE
  },
  "wm_scoring": {
    "scoring_mode": "$SCORING_MODE",
    "num_passes": $NUM_PASSES,
    "windows_per_call": $WINDOWS_PER_CALL,
    "num_windows_legacy": $NUM_WINDOWS,
    "random_spread": $RANDOM_SPREAD,
    "start_frame": $START_FRAME,
    "num_inference_steps": $NUM_INFERENCE_STEPS,
    "num_workers": $NUM_REWARD_WORKERS,
    "reward_gpus": "$REWARD_GPUS"
  },
  "wm_finetune": {
    "enabled": $ENABLE_WM_FINETUNE
  },
  "sac": {
    "episodes_per_policy_update": $TRAJ_BATCH,
    "start_online_updates": $START_ONLINE_UPDATES,
    "multi_grad_step": $MULTI_GRAD_STEP,
    "checkpoint_interval": $CHECKPOINT_INTERVAL,
    "batch_size": 256,
    "discount": 0.999,
    "seed": 0,
    "hidden_dims": 128,
    "resize_image": 64,
    "eval_interval": 10000,
    "log_interval": 200,
    "eval_episodes": 5,
    "target_entropy": "$TARGET_ENTROPY"
  },
  "reward_model": {
    "use_reward_model": 1,
    "reward_fn": "examples.reward_fn:wm_score",
    "loss_mode": "$REWARD_LOSS_MODE",
    "grad_steps": $REWARD_GRAD_STEPS,
    "lr": $REWARD_LR,
    "episodes_per_update": $TRAJ_BATCH,
    "relabel_buffer": 0,
    "scene_reset_freq": 1
  },
  "run_length": {
    "max_steps": $MAX_STEPS,
    "max_trajs": $MAX_TRAJS
  },
  "video_capture": {
    "cam_resolution": 256,
    "fps": 20,
    "sample_stride": 2,
    "sample_start_offset": 6
  },
  "gpus": {
    "reward_gpus": "$REWARD_GPUS",
    "trainer_gpu": $TRAINER_GPU
  }
}
EOF
echo "[policy5gpu] wrote experiment config to $CONFIG_PATH"

echo "[policy5gpu] DSRL_ROOT=$DSRL_ROOT"
echo "[policy5gpu] REWARD_ROOT=$REWARD_ROOT"
echo "[policy5gpu] POLICY=$POLICY  QUERY_FREQ=$QUERY_FREQ"
echo "[policy5gpu] WARMUP_TRAJS=$WARMUP_TRAJS  -> START_ONLINE_UPDATES=$START_ONLINE_UPDATES transitions"
echo "[policy5gpu] ROUND_SIZE=$ROUND_SIZE  SCORED_PER_ROUND=$SCORED_PER_ROUND  MULTI_GRAD_STEP=$MULTI_GRAD_STEP"
echo "[policy5gpu] WM scoring: $NUM_PASSES passes x $WINDOWS_PER_CALL windows/call = $((NUM_PASSES * WINDOWS_PER_CALL)) windows/traj"
echo "[policy5gpu] REWARD_GPUS=$REWARD_GPUS  ($NUM_REWARD_WORKERS workers)  TRAINER_GPU=$TRAINER_GPU"
echo "[policy5gpu] REWARD_GRAD_STEPS=$REWARD_GRAD_STEPS"
echo "[policy5gpu] ACTION_MAGNITUDE=$ACTION_MAGNITUDE  BASE_POLICY_PROB=$BASE_POLICY_PROB"
echo "[policy5gpu] CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "[policy5gpu] WM fine-tune: DISABLED (frozen WM)"
echo "[policy5gpu] SAC target_entropy=$TARGET_ENTROPY  (higher = more diverse actions; 'auto' = -action_dim/2)"
nvidia-smi -L | head -8 || true

# ---------------------------------------------------------------------------
# Reward servers — one process per REWARD_GPU
# ---------------------------------------------------------------------------
SERVER_PIDS=()
SERVER_LOGS=()
for idx in "${!REWARD_GPU_ARR[@]}"; do
    g="${REWARD_GPU_ARR[$idx]}"
    LOG="$LOG_DIR/reward_server_w${idx}_gpu${g}.log"
    SERVER_LOGS+=("$LOG")
    echo "[policy5gpu] starting reward worker $idx on GPU $g  (logs -> $LOG)"
    (
        cd "$OPEN_WORLD_ROOT"
        CUDA_VISIBLE_DEVICES=$g \
        OPEN_WORLD_ROOT="$OPEN_WORLD_ROOT" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PYTHONUNBUFFERED=1 \
        "$OPEN_WORLD_ROOT/.venv/bin/python" -u \
            "$DSRL_ROOT/examples/reward_model/reward_server.py" \
            --reward-root "$REWARD_ROOT" \
            --ckpt-path "$WM_CKPT" \
            --dataset-root "$WM_DATASET_ROOT" \
            --num-windows "$NUM_WINDOWS" \
            --num-passes "$NUM_PASSES" \
            --windows-per-call "$WINDOWS_PER_CALL" \
            $( [ "$RANDOM_SPREAD" = "1" ] && echo "--random-spread" ) \
            --start-frame "$START_FRAME" \
            --num-inference-steps "$NUM_INFERENCE_STEPS" \
            --scoring-mode "$SCORING_MODE" \
            --device "cuda:0" \
            --worker-id "w${idx}_gpu${g}" \
            > "$LOG" 2>&1
    ) &
    SERVER_PIDS+=("$!")
    echo "[policy5gpu]   worker $idx pid=${SERVER_PIDS[$idx]}"
done

cleanup() {
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[policy5gpu] stopping reward worker pid=$pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${SERVER_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM EXIT

echo "[policy5gpu] waiting up to ${SERVER_READY_TIMEOUT_S}s for all $NUM_REWARD_WORKERS workers to load..."
DEADLINE=$(($(date +%s) + SERVER_READY_TIMEOUT_S))
ready_count=0
while [ "$ready_count" -lt "$NUM_REWARD_WORKERS" ]; do
    ready_count=0
    for idx in "${!SERVER_PIDS[@]}"; do
        pid="${SERVER_PIDS[$idx]}"
        log="${SERVER_LOGS[$idx]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[policy5gpu] FATAL: reward worker $idx (pid=$pid) died before becoming ready"
            tail -80 "$log" || true
            exit 1
        fi
        if grep -q "ready. polling" "$log" 2>/dev/null; then
            ready_count=$((ready_count + 1))
        fi
    done
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[policy5gpu] FATAL: only $ready_count/$NUM_REWARD_WORKERS workers ready after ${SERVER_READY_TIMEOUT_S}s"
        for log in "${SERVER_LOGS[@]}"; do
            echo "----- $log -----"
            tail -40 "$log" || true
        done
        exit 1
    fi
    [ "$ready_count" -lt "$NUM_REWARD_WORKERS" ] && sleep 5
done
echo "[policy5gpu] all $NUM_REWARD_WORKERS reward workers ready."

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
echo "[policy5gpu] starting trainer on GPU $TRAINER_GPU..."
cd "$DSRL_ROOT"
export PYTHONPATH="$DSRL_ROOT:${PYTHONPATH:-}"
export DSRL_REWARD_ROOT="$REWARD_ROOT"
export DSRL_REWARD_TIMEOUT_S=900

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$TRAINER_GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export PYTHONWARNINGS="ignore::RuntimeWarning:subprocess"

source "$DSRL_ROOT/.venv/bin/activate"

CUDA_VISIBLE_DEVICES=$TRAINER_GPU \
EXP="$DSRL_ROOT/logs/dsrl_wm_$JOB_TAG" \
python3 examples/launch_collect.py \
    --algorithm pixel_sac \
    --env libero \
    --policy "$POLICY" \
    --prefix "dsrl_pi0_libero_wm_$JOB_TAG" \
    --wandb_project DSRL_pi0_libero_wm_policy_5gpu \
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
    --reward_update_freq "$TRAJ_BATCH" \
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
    --max_trajs "$MAX_TRAJS" \
    --target_entropy "$TARGET_ENTROPY"

echo "[policy5gpu] trainer exited. cleaning up server workers."
