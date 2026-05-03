#!/bin/bash
# DSRL-π₀ + WM-reward FULL RUN on TWO GPUs.
#
# Tuned to be observably training (vs the production script which warms up
# slowly) but not as drastic as run_wm_loop_2gpu_smoketest.sh — i.e. the
# robot stays vaguely sane and we keep some clean π₀ trajectories in the mix.
#
# vs run_wm_loop_2gpu_smoketest.sh:
#   * action_magnitude=1.0 (was 3.0) — SAC noise tanh-bounded to typical
#     gaussian range, so perturbations are no wilder than a fresh sample.
#   * multi_grad_step=50 (was 200) — each batch of trajectories yields
#     fewer SAC updates, the actor moves more gradually.
#   * traj_batch_size=4 (was 2) — smoother reward-model targets.
#   * start_online_updates=10 (was 1) — small warmup so SAC sees ≥1 batch.
#   * base_policy_prob=0.5 — every episode flips a fair coin: heads = roll
#     out with fresh gaussian noise (pure π₀), tails = use SAC-chosen noise.
#     Halves the rate at which the buffer fills with garbage data while SAC
#     learns.
#   * checkpoint_interval=500 — SAC checkpoints land on disk in $EXP/...
#   * Online WM fine-tuning enabled: every 8 scored episodes the reward
#     server runs 25 grad steps and writes a fresh checkpoint to
#     $REWARD_ROOT/wm_checkpoints/checkpoint-<step>.pt (a "latest.txt"
#     pointer too).
#
# Usage: from inside a 2-GPU interactive allocation,
#     bash examples/scripts/run_wm_loop_2gpu_full.sh

source ~/.bashrc
set -euo pipefail

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
DSRL_ROOT="${DSRL_ROOT:-/n/fs/iromdata/project/dsrl_pi0}"
OPEN_WORLD_ROOT="${OPEN_WORLD_ROOT:-/n/fs/iromdata/project/open-world}"

JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}_full"
REWARD_ROOT="${REWARD_ROOT:-/n/fs/iromdata/project/shared/playworld_rollout/$JOB_TAG}"

WM_CKPT="/n/fs/iromdata/project/open-world/checkpoints/wm/libero/checkpoint-32000.pt"
WM_DATASET_ROOT="${WM_DATASET_ROOT:-/n/fs/iromdata/project/open-world/data/libero_processed}"

POLICY="${POLICY:-pi05}"
if [ -z "${QUERY_FREQ:-}" ]; then
    if [ "$POLICY" = "pi05" ]; then
        QUERY_FREQ=10
    else
        QUERY_FREQ=20
    fi
fi

REWARD_GPU="${REWARD_GPU:-0}"
TRAINER_GPU="${TRAINER_GPU:-1}"

# WM scoring (production-ish, but a touch faster than the 50-step / 8-window
# default so each request returns in ~30s instead of ~2min).
NUM_WINDOWS="${NUM_WINDOWS:-4}"
START_FRAME="${START_FRAME:-6}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"

# WM fine-tuning controls (enabled by default in this script).
ENABLE_WM_FINETUNE="${ENABLE_WM_FINETUNE:-1}"
WM_UPDATE_EVERY="${WM_UPDATE_EVERY:-8}"
WM_GRAD_STEPS="${WM_GRAD_STEPS:-25}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1}"
WM_LR="${WM_LR:-1e-5}"
WM_MAX_GRAD_NORM="${WM_MAX_GRAD_NORM:-1.0}"
WM_BUFFER_SIZE="${WM_BUFFER_SIZE:-64}"
WM_CHECKPOINT_EVERY="${WM_CHECKPOINT_EVERY:-1}"

# SAC training.
TRAJ_BATCH="${TRAJ_BATCH:-4}"
START_ONLINE_UPDATES="${START_ONLINE_UPDATES:-10}"
MULTI_GRAD_STEP="${MULTI_GRAD_STEP:-50}"
REWARD_GRAD_STEPS="${REWARD_GRAD_STEPS:-200}"
REWARD_LR="${REWARD_LR:-3e-4}"
ACTION_MAGNITUDE="${ACTION_MAGNITUDE:-1.0}"  # hard boundary on SAC noise
BASE_POLICY_PROB="${BASE_POLICY_PROB:-0.5}"  # 50% pure π₀ episodes
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"

# Run length (real run, not a smoketest).
MAX_TRAJS="${MAX_TRAJS:-1000000}"
MAX_STEPS="${MAX_STEPS:-500000}"

TASK_SUITE="${TASK_SUITE:-libero_90}"
TASK_ID="${TASK_ID:-57}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1200}"

# ---------------------------------------------------------------------------
# Offline env
# ---------------------------------------------------------------------------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TORCH_HOME=/n/fs/tom-project/.cache/torch
export HF_HOME=/n/fs/tom-project/.cache/huggingface

# ---------------------------------------------------------------------------
# Verify caches (fail fast)
# ---------------------------------------------------------------------------
PI_CACHE="/n/fs/tom-project/.cache/openpi/openpi-assets/checkpoints/${POLICY}_libero"
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
        echo "[full] FATAL: missing cached artifact: $f"
        echo "[full] run setup_caches.sh on a login node first."
        exit 1
    fi
done

if [ "$REWARD_GPU" = "$TRAINER_GPU" ]; then
    echo "[full] FATAL: REWARD_GPU and TRAINER_GPU are both $REWARD_GPU."
    exit 1
fi
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "[full] FATAL: only $NUM_GPUS GPU(s) visible, need 2."
    exit 1
fi

mkdir -p "$REWARD_ROOT"
LOG_DIR="$REWARD_ROOT/_logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/reward_server.log"

echo "[full] DSRL_ROOT=$DSRL_ROOT"
echo "[full] REWARD_ROOT=$REWARD_ROOT"
echo "[full] POLICY=$POLICY  QUERY_FREQ=$QUERY_FREQ"
echo "[full] TRAJ_BATCH=$TRAJ_BATCH  START_ONLINE=$START_ONLINE_UPDATES"
echo "[full] MULTI_GRAD_STEP=$MULTI_GRAD_STEP  REWARD_GRAD_STEPS=$REWARD_GRAD_STEPS"
echo "[full] ACTION_MAGNITUDE=$ACTION_MAGNITUDE  BASE_POLICY_PROB=$BASE_POLICY_PROB"
echo "[full] CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "[full] WM ft: enabled=$ENABLE_WM_FINETUNE every=$WM_UPDATE_EVERY"
echo "[full]        steps=$WM_GRAD_STEPS bs=$WM_BATCH_SIZE lr=$WM_LR"
nvidia-smi -L | head -4 || true

# ---------------------------------------------------------------------------
# Reward server
# ---------------------------------------------------------------------------
echo "[full] starting reward server on GPU $REWARD_GPU (logs -> $SERVER_LOG)"

# Build the optional --enable-wm-finetune flag conditionally.
WM_FT_ARGS=()
if [ "$ENABLE_WM_FINETUNE" = "1" ]; then
    WM_FT_ARGS=(
        "--enable-wm-finetune"
        "--wm-update-every" "$WM_UPDATE_EVERY"
        "--wm-grad-steps" "$WM_GRAD_STEPS"
        "--wm-batch-size" "$WM_BATCH_SIZE"
        "--wm-lr" "$WM_LR"
        "--wm-max-grad-norm" "$WM_MAX_GRAD_NORM"
        "--wm-buffer-size" "$WM_BUFFER_SIZE"
        "--wm-checkpoint-every" "$WM_CHECKPOINT_EVERY"
    )
fi

(
    cd "$OPEN_WORLD_ROOT"
    CUDA_VISIBLE_DEVICES=$REWARD_GPU \
    OPEN_WORLD_ROOT="$OPEN_WORLD_ROOT" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    "$OPEN_WORLD_ROOT/.venv/bin/python" -u \
        "$DSRL_ROOT/examples/reward_model/reward_server.py" \
        --reward-root "$REWARD_ROOT" \
        --ckpt-path "$WM_CKPT" \
        --dataset-root "$WM_DATASET_ROOT" \
        --num-windows "$NUM_WINDOWS" \
        --start-frame "$START_FRAME" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --device "cuda:0" \
        "${WM_FT_ARGS[@]}" \
        > "$SERVER_LOG" 2>&1
) &
SERVER_PID=$!
echo "[full] reward server pid=$SERVER_PID"

cleanup() {
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "[full] stopping reward server (pid=$SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

echo "[full] waiting up to ${SERVER_READY_TIMEOUT_S}s for server to load..."
DEADLINE=$(($(date +%s) + SERVER_READY_TIMEOUT_S))
while ! grep -q "ready. polling" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[full] FATAL: reward server died before becoming ready"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[full] FATAL: server didn't print 'ready. polling' in ${SERVER_READY_TIMEOUT_S}s"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[full] reward server is ready."

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
echo "[full] starting trainer on GPU $TRAINER_GPU..."
cd "$DSRL_ROOT"
export PYTHONPATH="$DSRL_ROOT:${PYTHONPATH:-}"
export DSRL_REWARD_ROOT="$REWARD_ROOT"
export DSRL_REWARD_TIMEOUT_S=900

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$TRAINER_GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source "$DSRL_ROOT/.venv/bin/activate"

CUDA_VISIBLE_DEVICES=$TRAINER_GPU \
EXP="$DSRL_ROOT/logs/dsrl_wm_$JOB_TAG" \
python3 examples/launch_collect.py \
    --algorithm pixel_sac \
    --env libero \
    --policy "$POLICY" \
    --prefix "dsrl_pi0_libero_wm_$JOB_TAG" \
    --wandb_project DSRL_pi0_libero_wm_full \
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
    --scene_reset_freq 1 \
    --reward_update_freq "$TRAJ_BATCH" \
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
    --max_trajs "$MAX_TRAJS"

echo "[full] trainer exited. cleaning up server."
