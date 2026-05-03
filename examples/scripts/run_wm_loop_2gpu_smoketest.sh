#!/bin/bash
# DSRL-π₀ + WM-reward SMOKETEST on TWO GPUs.
#
# This is a deliberately aggressive variant of run_wm_loop_2gpu.sh designed to
# confirm in ~10 episodes that BOTH the SAC actor and the reward model are
# updating. It is NOT a real training run.
#
# Aggressiveness vs the production script:
#   * SAC starts after the FIRST traj (start_online_updates=1) instead of 500
#     transitions of warmup.
#   * Reward + SAC update every 2 trajs (Y=2) instead of 8 — more frequent.
#   * 200 SAC grad steps per env transition (multi_grad_step=200) instead of 20.
#   * 1000 reward-model grad steps per cycle (reward_grad_steps=1000) instead
#     of 200.
#   * action_magnitude=3.0 so SAC perturbations are visibly large.
#   * max_trajs=12 and max_steps=50000 so the run actually exits.
#   * NUM_INFERENCE_STEPS=10 and NUM_WINDOWS=2 so each WM scoring call returns
#     in seconds rather than ~2min.
#   * Writes to a "_smoketest" subdir so it doesn't mix with real runs.
#
# What to look for:
#   * "[reward] f-scores: mean=X std=Y" — printed every 2 trajs. If std is
#     non-trivial across trajectories, the reward model has signal to learn.
#   * "reward_model/loss" in wandb dropping over the first few cycles =
#     reward model is learning.
#   * "training/actor_loss" / "training/critic_loss" appearing in wandb after
#     the first batch = SAC is updating.
#   * Per-episode env_steps and behavior visibly diverging by episode 6-12.
#
# Usage: from inside a 2-GPU interactive allocation,
#     bash examples/scripts/run_wm_loop_2gpu_smoketest.sh

source ~/.bashrc
set -euo pipefail

# ---------------------------------------------------------------------------
# Knobs — same defaults as run_wm_loop_2gpu.sh, but with _smoketest suffix.
# ---------------------------------------------------------------------------
DSRL_ROOT="${DSRL_ROOT:-/n/fs/iromdata/project/dsrl_pi0}"
OPEN_WORLD_ROOT="${OPEN_WORLD_ROOT:-/n/fs/iromdata/project/open-world}"

JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}_smoketest"
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

# WM scoring — much faster than production so each request returns in seconds.
NUM_WINDOWS="${NUM_WINDOWS:-2}"
START_FRAME="${START_FRAME:-6}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

# Aggressive learning knobs.
TRAJ_BATCH="${TRAJ_BATCH:-2}"            # update every 2 trajs
START_ONLINE_UPDATES="${START_ONLINE_UPDATES:-1}"   # SAC kicks in after first batch
MULTI_GRAD_STEP="${MULTI_GRAD_STEP:-200}"           # SAC steps per traj transition
REWARD_GRAD_STEPS="${REWARD_GRAD_STEPS:-1000}"      # reward-model steps per cycle
REWARD_LR="${REWARD_LR:-1e-3}"                       # higher LR than prod (3e-4)
ACTION_MAGNITUDE="${ACTION_MAGNITUDE:-3.0}"          # bigger SAC perturbations

# Stop after this many episodes — we just want to see things move.
MAX_TRAJS="${MAX_TRAJS:-12}"
MAX_STEPS="${MAX_STEPS:-50000}"

TASK_SUITE="${TASK_SUITE:-libero_90}"
TASK_ID="${TASK_ID:-57}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1200}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TORCH_HOME=/n/fs/tom-project/.cache/torch
export HF_HOME=/n/fs/tom-project/.cache/huggingface

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
        echo "[smoketest] FATAL: missing cached artifact: $f"
        echo "[smoketest] run setup_caches.sh on a login node first."
        exit 1
    fi
done

if [ "$REWARD_GPU" = "$TRAINER_GPU" ]; then
    echo "[smoketest] FATAL: REWARD_GPU and TRAINER_GPU both = $REWARD_GPU."
    exit 1
fi
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "[smoketest] FATAL: only $NUM_GPUS GPU(s) visible, need 2."
    exit 1
fi

mkdir -p "$REWARD_ROOT"
LOG_DIR="$REWARD_ROOT/_logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/reward_server.log"

echo "[smoketest] DSRL_ROOT=$DSRL_ROOT"
echo "[smoketest] REWARD_ROOT=$REWARD_ROOT"
echo "[smoketest] POLICY=$POLICY  QUERY_FREQ=$QUERY_FREQ"
echo "[smoketest] TRAJ_BATCH=$TRAJ_BATCH  START_ONLINE=$START_ONLINE_UPDATES"
echo "[smoketest] MULTI_GRAD_STEP=$MULTI_GRAD_STEP  REWARD_GRAD_STEPS=$REWARD_GRAD_STEPS"
echo "[smoketest] ACTION_MAGNITUDE=$ACTION_MAGNITUDE  MAX_TRAJS=$MAX_TRAJS"
echo "[smoketest] NUM_WINDOWS=$NUM_WINDOWS  NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS"
nvidia-smi -L | head -4 || true

# ---------------------------------------------------------------------------
# Reward server (REWARD_GPU)
# ---------------------------------------------------------------------------
echo "[smoketest] starting reward server on GPU $REWARD_GPU (logs -> $SERVER_LOG)"
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
        > "$SERVER_LOG" 2>&1
) &
SERVER_PID=$!
echo "[smoketest] reward server pid=$SERVER_PID"

cleanup() {
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "[smoketest] stopping reward server (pid=$SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

echo "[smoketest] waiting up to ${SERVER_READY_TIMEOUT_S}s for server to load..."
DEADLINE=$(($(date +%s) + SERVER_READY_TIMEOUT_S))
while ! grep -q "ready. polling" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[smoketest] FATAL: reward server died before becoming ready"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[smoketest] FATAL: server didn't print 'ready. polling' in ${SERVER_READY_TIMEOUT_S}s"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[smoketest] reward server is ready."

# ---------------------------------------------------------------------------
# Trainer (TRAINER_GPU)
# ---------------------------------------------------------------------------
echo "[smoketest] starting trainer on GPU $TRAINER_GPU..."
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
    --wandb_project DSRL_pi0_libero_wm_smoketest \
    --batch_size 256 \
    --discount 0.999 \
    --seed 0 \
    --max_steps "$MAX_STEPS" \
    --eval_interval 999999999 \
    --log_interval 50 \
    --eval_episodes 2 \
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
    --save_dir "$REWARD_ROOT" \
    --save_split train \
    --task_suite_name "$TASK_SUITE" \
    --task_id "$TASK_ID" \
    --cam_resolution 256 \
    --fps 20 \
    --sample_stride 2 \
    --sample_start_offset 6 \
    --max_trajs "$MAX_TRAJS"

echo "[smoketest] trainer exited. cleaning up server."
