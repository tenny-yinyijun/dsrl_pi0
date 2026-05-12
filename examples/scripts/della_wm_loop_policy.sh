source ~/.bashrc
# set -euo pipefail

# ============================================================================
# POLICY-ONLY STABILITY TEST (frozen WM)
# ----------------------------------------------------------------------------
# Single-run smoke test: does the SAC exploration-policy update *stably* drive
# π₀ toward more explorative behavior? No grid/sweep — fixed knobs.
#
# Loop:
#   1. Warmup: collect WARMUP_TRAJS trajectories to populate the replay
#      buffer; no SAC gradient updates yet.
#   2. Round: collect ROUND_SIZE=10 trajectories; score each with the frozen
#      WM using 5 windows/traj (10 × 5 = 50 WM windows scored per round);
#      refit reward model on the round; take SAC gradient steps. Repeat.
#
# WM fine-tuning is disabled. Scoring is the runtime bottleneck (~10s/window),
# so we cap at 50 windows/round.
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
# model + SAC update.
ROUND_SIZE=10
SCORED_PER_ROUND=$ROUND_SIZE   # score every traj in the round
TRAJ_BATCH=$ROUND_SIZE         # reward-model batch = the round

# ----------------------------------------------------------------------------
# Warmup
# ----------------------------------------------------------------------------
# Collect WARMUP_TRAJS trajectories before SAC updates begin. The trainer
# gates SAC updates on len(buffer) > START_ONLINE_UPDATES, so convert trajs
# → transitions using the worst-case per-traj transition count
# (max_timesteps / query_freq). Real trajs may terminate early, so this is a
# conservative *upper* bound on the # of warmup trajs actually collected.
WARMUP_TRAJS="${WARMUP_TRAJS:-20}"

# ----------------------------------------------------------------------------
# SAC / data-collection policy
# ----------------------------------------------------------------------------
MULTI_GRAD_STEP=10             # SAC gradient updates per env transition
ACTION_MAGNITUDE=1.0           # hard boundary on SAC noise
BASE_POLICY_PROB=0.5           # fraction of pure-π₀ episodes
CHECKPOINT_INTERVAL=5000

# ----------------------------------------------------------------------------
# Reward model
# ----------------------------------------------------------------------------
# REWARD_LOSS_MODE='per_step' supervises r̂ with per-WM-frame LPIPS at the
# corresponding query-step (finer credit assignment than the legacy 'traj'
# loss, which only fits Σ r̂ = mean LPIPS per trajectory).
REWARD_GRAD_STEPS=200
REWARD_LR=3e-4
REWARD_LOSS_MODE=per_step      # per_step | traj

# ----------------------------------------------------------------------------
# World-model scoring (frozen WM)
# ----------------------------------------------------------------------------
# Each traj scored with NUM_PASSES * WINDOWS_PER_CALL = 5 * 1 = 5 windows
# (stratified spread), so a round of 10 trajs = 50 WM windows total —
# the scoring budget we're trying to stay within (~10s/window).
SCORING_MODE=spread
NUM_PASSES=5
WINDOWS_PER_CALL=1
RANDOM_SPREAD=1
START_FRAME=6
NUM_INFERENCE_STEPS=50
NUM_WINDOWS=5                  # legacy total; ignored when NUM_PASSES & WINDOWS_PER_CALL set.

# ----------------------------------------------------------------------------
# WM fine-tuning — DISABLED in this script
# ----------------------------------------------------------------------------
ENABLE_WM_FINETUNE=0
WM_SANITY_CHECK=0

# ----------------------------------------------------------------------------
# GPUs (must be different)
# ----------------------------------------------------------------------------
REWARD_GPU="${REWARD_GPU:-0}"
TRAINER_GPU="${TRAINER_GPU:-1}"

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
# Two-level layout under playworld_rollout: <MMDD>/<jobid>_<HHMMSS>_policy_stability/.
DATE_DIR=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
JOB_TAG="${SLURM_JOB_ID:-local}_${TIME_TAG}_policy_stability"
REWARD_ROOT="${REWARD_ROOT:-/scratch/gpfs/AM43/yy4041/playworld_rollout/$DATE_DIR/$JOB_TAG}"

# QUERY_FREQ defaults to 10 for pi05, 20 for pi0; override via env.
if [ -z "${QUERY_FREQ:-}" ]; then
    if [ "$POLICY" = "pi05" ]; then
        QUERY_FREQ=10
    else
        QUERY_FREQ=20
    fi
fi

# Convert WARMUP_TRAJS (in trajectories) → START_ONLINE_UPDATES (in
# transitions) using the worst-case per-traj transition count. libero caps
# trajectories at max_timesteps=400 env steps; one buffer transition per
# QUERY_FREQ env steps → at most 400/QUERY_FREQ transitions/traj.
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
        echo "[policy] FATAL: missing cached artifact: $f"
        echo "[policy] run setup_caches.sh on a login node first."
        exit 1
    fi
done

if [ "$REWARD_GPU" = "$TRAINER_GPU" ]; then
    echo "[policy] FATAL: REWARD_GPU and TRAINER_GPU are both $REWARD_GPU."
    exit 1
fi
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "[policy] FATAL: only $NUM_GPUS GPU(s) visible, need 2."
    exit 1
fi

mkdir -p "$REWARD_ROOT"
LOG_DIR="$REWARD_ROOT/_logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/reward_server.log"

# ---------------------------------------------------------------------------
# Snapshot the experiment config so $REWARD_ROOT is self-describing.
# ---------------------------------------------------------------------------
DSRL_GIT_SHA=$(git -C "$DSRL_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
OPEN_WORLD_GIT_SHA=$(git -C "$OPEN_WORLD_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
CONFIG_PATH="$REWARD_ROOT/config.json"
cat > "$CONFIG_PATH" <<EOF
{
  "meta": {
    "job_tag": "$JOB_TAG",
    "experiment_type": "policy_only",
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
    "num_inference_steps": $NUM_INFERENCE_STEPS
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
    "eval_episodes": 5
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
    "reward_gpu": $REWARD_GPU,
    "trainer_gpu": $TRAINER_GPU
  }
}
EOF
echo "[policy] wrote experiment config to $CONFIG_PATH"

echo "[policy] DSRL_ROOT=$DSRL_ROOT"
echo "[policy] REWARD_ROOT=$REWARD_ROOT"
echo "[policy] POLICY=$POLICY  QUERY_FREQ=$QUERY_FREQ"
echo "[policy] WARMUP_TRAJS=$WARMUP_TRAJS  -> START_ONLINE_UPDATES=$START_ONLINE_UPDATES transitions"
echo "[policy] ROUND_SIZE=$ROUND_SIZE  SCORED_PER_ROUND=$SCORED_PER_ROUND  MULTI_GRAD_STEP=$MULTI_GRAD_STEP"
echo "[policy] WM scoring: $NUM_PASSES passes x $WINDOWS_PER_CALL windows/call = $((NUM_PASSES * WINDOWS_PER_CALL)) windows/traj"
echo "[policy] REWARD_GRAD_STEPS=$REWARD_GRAD_STEPS"
echo "[policy] ACTION_MAGNITUDE=$ACTION_MAGNITUDE  BASE_POLICY_PROB=$BASE_POLICY_PROB"
echo "[policy] CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "[policy] WM fine-tune: DISABLED (frozen WM)"
nvidia-smi -L | head -4 || true

# ---------------------------------------------------------------------------
# Reward server (scoring only — WM fine-tune flags omitted)
# ---------------------------------------------------------------------------
echo "[policy] starting reward server on GPU $REWARD_GPU (logs -> $SERVER_LOG)"

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
        --num-passes "$NUM_PASSES" \
        --windows-per-call "$WINDOWS_PER_CALL" \
        $( [ "$RANDOM_SPREAD" = "1" ] && echo "--random-spread" ) \
        --start-frame "$START_FRAME" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --scoring-mode "$SCORING_MODE" \
        --device "cuda:0" \
        > "$SERVER_LOG" 2>&1
) &
SERVER_PID=$!
echo "[policy] reward server pid=$SERVER_PID"

cleanup() {
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "[policy] stopping reward server (pid=$SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

echo "[policy] waiting up to ${SERVER_READY_TIMEOUT_S}s for server to load..."
DEADLINE=$(($(date +%s) + SERVER_READY_TIMEOUT_S))
while ! grep -q "ready. polling" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[policy] FATAL: reward server died before becoming ready"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[policy] FATAL: server didn't print 'ready. polling' in ${SERVER_READY_TIMEOUT_S}s"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[policy] reward server is ready."

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
echo "[policy] starting trainer on GPU $TRAINER_GPU..."
cd "$DSRL_ROOT"
export PYTHONPATH="$DSRL_ROOT:${PYTHONPATH:-}"
export DSRL_REWARD_ROOT="$REWARD_ROOT"
export DSRL_REWARD_TIMEOUT_S=900

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$TRAINER_GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Silence JAX's "os.fork() was called" warning. Fired by every subprocess.Popen
# (mp4 writer, wandb offline run, etc.) because Python's audit hook can't see
# that exec() follows immediately. Fork-exec doesn't deadlock JAX.
export PYTHONWARNINGS="ignore::RuntimeWarning:subprocess"

source "$DSRL_ROOT/.venv/bin/activate"

CUDA_VISIBLE_DEVICES=$TRAINER_GPU \
EXP="$DSRL_ROOT/logs/dsrl_wm_$JOB_TAG" \
python3 examples/launch_collect.py \
    --algorithm pixel_sac \
    --env libero \
    --policy "$POLICY" \
    --prefix "dsrl_pi0_libero_wm_$JOB_TAG" \
    --wandb_project DSRL_pi0_libero_wm_policy \
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
    --max_trajs "$MAX_TRAJS"

echo "[policy] trainer exited. cleaning up server."
