#!/bin/bash
# Slurm submission for the full DSRL-π₀ + WM-reward loop on a compute node.
#
# Spawns the reward-server daemon in the background, waits for it to load
# (~90s for SVD), then runs the dsrl_pi0 collector/trainer in the foreground
# with --reward_fn examples.reward_fn:wm_score wired up.
#
# Usage:
#     sbatch examples/scripts/run_wm_loop.sh
#   or, on an interactive compute node:
#     bash   examples/scripts/run_wm_loop.sh
#
# Run setup_caches.sh on a LOGIN NODE first — compute nodes have no internet
# and the script will fail-fast if any artifact is missing.

#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=dsrl-wm
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yy4041@princeton.edu

# Source bashrc before enabling strict mode: /etc/bashrc references unbound
# BASHRCSOURCED (trips `set -u`), and ~/.bashrc has `[ -s nvm.sh ] && . nvm.sh`
# style chains that return non-zero when the file is missing on compute nodes
# (trips `set -e`).
source ~/.bashrc
set -euo pipefail

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
DSRL_ROOT="${DSRL_ROOT:-/scratch/gpfs/AM43/yy4041/dsrl_pi0}"
OPEN_WORLD_ROOT="${OPEN_WORLD_ROOT:-/scratch/gpfs/AM43/yy4041/open-world}"

# Where the reward server and the trainer share trajectories. One per run.
JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
REWARD_ROOT="${REWARD_ROOT:-/scratch/gpfs/AM43/yy4041/wm_reward_runs/$JOB_TAG}"

# WM checkpoint the daemon scores against.
WM_CKPT="${WM_CKPT:-$OPEN_WORLD_ROOT/models/wm_training/libero_0429/checkpoint-20000.pt}"

# Pi0 variant. Must match what setup_caches.sh fetched.
POLICY="${POLICY:-pi05}"

# Single-GPU mode: both processes share GPU 0 (memory-tight but simplest).
# For two GPUs: bump #SBATCH --gres=gpu:2 above and set REWARD_GPU=1.
REWARD_GPU="${REWARD_GPU:-0}"
TRAINER_GPU="${TRAINER_GPU:-0}"

# Daemon scoring knobs.
NUM_WINDOWS="${NUM_WINDOWS:-8}"
START_FRAME="${START_FRAME:-6}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"

# Trainer batching.
TRAJ_BATCH="${TRAJ_BATCH:-8}"
TASK_SUITE="${TASK_SUITE:-libero_90}"
TASK_ID="${TASK_ID:-57}"

# Wait up to this long for the reward server to print its "ready" line.
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-600}"

# ---------------------------------------------------------------------------
# Offline env (compute node has no internet)
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
    "$PI_CACHE"
)
for f in "${REQUIRED[@]}"; do
    if [ ! -e "$f" ]; then
        echo "[run_wm_loop] FATAL: missing cached artifact: $f"
        echo "[run_wm_loop] run setup_caches.sh on a login node first."
        exit 1
    fi
done

mkdir -p "$REWARD_ROOT"
LOG_DIR="$REWARD_ROOT/_logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/reward_server.log"

echo "[run_wm_loop] DSRL_ROOT=$DSRL_ROOT"
echo "[run_wm_loop] OPEN_WORLD_ROOT=$OPEN_WORLD_ROOT"
echo "[run_wm_loop] REWARD_ROOT=$REWARD_ROOT"
echo "[run_wm_loop] WM_CKPT=$WM_CKPT"
echo "[run_wm_loop] POLICY=$POLICY  REWARD_GPU=$REWARD_GPU  TRAINER_GPU=$TRAINER_GPU"
nvidia-smi -L | head -2 || true

# ---------------------------------------------------------------------------
# Start reward server in the background
# ---------------------------------------------------------------------------
echo "[run_wm_loop] starting reward server (logs -> $SERVER_LOG)"
(
    cd "$OPEN_WORLD_ROOT"
    CUDA_VISIBLE_DEVICES=$REWARD_GPU \
    OPEN_WORLD_ROOT="$OPEN_WORLD_ROOT" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$OPEN_WORLD_ROOT/.venv/bin/python" \
        "$DSRL_ROOT/examples/reward_model/reward_server.py" \
        --reward-root "$REWARD_ROOT" \
        --ckpt-path "$WM_CKPT" \
        --num-windows "$NUM_WINDOWS" \
        --start-frame "$START_FRAME" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --device "cuda:0" \
        > "$SERVER_LOG" 2>&1
) &
SERVER_PID=$!
echo "[run_wm_loop] reward server pid=$SERVER_PID"

# Make sure we kill the server when this script exits (any reason).
cleanup() {
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "[run_wm_loop] stopping reward server (pid=$SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

# ---------------------------------------------------------------------------
# Wait for "ready. polling" line
# ---------------------------------------------------------------------------
echo "[run_wm_loop] waiting up to ${SERVER_READY_TIMEOUT_S}s for server to load..."
DEADLINE=$(($(date +%s) + SERVER_READY_TIMEOUT_S))
while ! grep -q "ready. polling" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[run_wm_loop] FATAL: reward server died before becoming ready"
        echo "----- last 80 lines of $SERVER_LOG -----"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[run_wm_loop] FATAL: server didn't print 'ready. polling' in ${SERVER_READY_TIMEOUT_S}s"
        echo "----- last 80 lines of $SERVER_LOG -----"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[run_wm_loop] reward server is ready."

# ---------------------------------------------------------------------------
# Run the dsrl_pi0 collector + trainer (foreground)
# ---------------------------------------------------------------------------
echo "[run_wm_loop] starting trainer..."
cd "$DSRL_ROOT"
export PYTHONPATH="$DSRL_ROOT:${PYTHONPATH:-}"
export DSRL_REWARD_ROOT="$REWARD_ROOT"
# Each scoring call may take ~2 minutes. Be generous.
export DSRL_REWARD_TIMEOUT_S=900

# LIBERO env vars (mirrors run_libero_collect.sh).
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
    --wandb_project DSRL_pi0_libero_wm \
    --batch_size 256 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 500000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --multi_grad_step 20 \
    --start_online_updates 500 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 20 \
    --hidden_dims 128 \
    --use_reward_model 1 \
    --reward_fn examples.reward_fn:wm_score \
    --traj_batch_size "$TRAJ_BATCH" \
    --reward_grad_steps 200 \
    --reward_lr 3e-4 \
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
    --max_trajs 1000000

echo "[run_wm_loop] trainer exited. cleaning up server."
# trap will fire on exit
