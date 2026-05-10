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

# WM scoring. SCORING_MODE=spread + RANDOM_SPREAD=1 places NUM_PASSES
# autoregressive chunks at stratified-random positions across the
# trajectory. Each chunk runs WINDOWS_PER_CALL contiguous windows, so a
# single chunk covers WINDOWS_PER_CALL * (cfg.num_frames-1) WM frames
# with the original WM error-compounding behavior (good signal). Across
# chunks, history is reset to GT — no compounding across the trajectory.
#   defaults: 5 chunks × 4 windows × 4 frames = 80 WM frames per traj
#             ≈ same coverage as the prior 20-pass setup, but each chunk
#             is autoregressive so the LPIPS values are richer per call.
NUM_WINDOWS="${NUM_WINDOWS:-20}"        # legacy total; ignored when
                                        # NUM_PASSES & WINDOWS_PER_CALL set.
NUM_PASSES="${NUM_PASSES:-5}"
WINDOWS_PER_CALL="${WINDOWS_PER_CALL:-4}"
RANDOM_SPREAD="${RANDOM_SPREAD:-1}"
START_FRAME="${START_FRAME:-6}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
SCORING_MODE="${SCORING_MODE:-spread}"

# WM fine-tuning controls (enabled by default in this script). Cranked up
# vs the prior conservative defaults — for a 3B-param video model the old
# settings produced ~100 examples per cycle, which barely moves the
# weights. New defaults: cycles every 10 episodes (was 20), 200 grad
# steps per cycle (was 50), batch 4 (was 2), buffer 128 (was 64). That's
# ~16x more total update work per episode while keeping lr unchanged.
# Combined with WM_CHECKPOINT_EVERY=1 this yields ~1 checkpoint per 10
# episodes.
ENABLE_WM_FINETUNE="${ENABLE_WM_FINETUNE:-1}"
WM_UPDATE_EVERY="${WM_UPDATE_EVERY:-10}"
WM_GRAD_STEPS="${WM_GRAD_STEPS:-200}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-4}"
WM_LR="${WM_LR:-1e-5}"
WM_MAX_GRAD_NORM="${WM_MAX_GRAD_NORM:-1.0}"
WM_BUFFER_SIZE="${WM_BUFFER_SIZE:-128}"
WM_CHECKPOINT_EVERY="${WM_CHECKPOINT_EVERY:-1}"

# WM update sanity-check: render WM rollouts on the K most-recent training
# trajectories both before and after each fine-tune cycle, and save
# side-by-side videos to $REWARD_ROOT/wm_update_sanity_check/update_<n>/.
# Lets you visually verify the WM is actually moving each cycle.
WM_SANITY_CHECK="${WM_SANITY_CHECK:-1}"
WM_SANITY_NUM_TRAJS="${WM_SANITY_NUM_TRAJS:-2}"
WM_SANITY_WINDOWS="${WM_SANITY_WINDOWS:-8}"

# SAC training.
# MULTI_GRAD_STEP=10 (was 50) — 5× fewer SAC updates per env transition,
# so the actor moves more gradually and is less likely to diverge from π₀.
# CHECKPOINT_INTERVAL=5000 (was 500) — at the new step rate that lands
# roughly one SAC ckpt every ~12 episodes, comparable to the WM cadence.
TRAJ_BATCH="${TRAJ_BATCH:-4}"
START_ONLINE_UPDATES="${START_ONLINE_UPDATES:-10}"
MULTI_GRAD_STEP="${MULTI_GRAD_STEP:-10}"
REWARD_GRAD_STEPS="${REWARD_GRAD_STEPS:-200}"
REWARD_LR="${REWARD_LR:-3e-4}"
# 'per_step' = supervise the reward model with per-WM-frame LPIPS at the
# corresponding query-step (finer credit assignment than the legacy
# 'traj' loss, which only fits Σ r̂ = mean LPIPS per trajectory).
REWARD_LOSS_MODE="${REWARD_LOSS_MODE:-per_step}"
ACTION_MAGNITUDE="${ACTION_MAGNITUDE:-1.0}"  # hard boundary on SAC noise
BASE_POLICY_PROB="${BASE_POLICY_PROB:-0.5}"  # 50% pure π₀ episodes
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"

# Run length (real run, not a smoketest).
MAX_TRAJS="${MAX_TRAJS:-1000000}"
MAX_STEPS="${MAX_STEPS:-500000}"

# TASK_SUITE="${TASK_SUITE:-libero_90}"
# TASK_ID="${TASK_ID:-57}"

TASK_SUITE="${TASK_SUITE:-libero_goal}"
TASK_ID="${TASK_ID:-1}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-2400}"

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

# ---------------------------------------------------------------------------
# Snapshot the experiment config so $REWARD_ROOT is self-describing.
# Captures every knob this script controls (incl. trainer-side constants)
# plus git SHAs for reproducibility.
# ---------------------------------------------------------------------------
DSRL_GIT_SHA=$(git -C "$DSRL_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
OPEN_WORLD_GIT_SHA=$(git -C "$OPEN_WORLD_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
CONFIG_PATH="$REWARD_ROOT/config.json"
cat > "$CONFIG_PATH" <<EOF
{
  "meta": {
    "job_tag": "$JOB_TAG",
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
    "enabled": $ENABLE_WM_FINETUNE,
    "episodes_per_update": $WM_UPDATE_EVERY,
    "grad_steps": $WM_GRAD_STEPS,
    "batch_size": $WM_BATCH_SIZE,
    "lr": $WM_LR,
    "max_grad_norm": $WM_MAX_GRAD_NORM,
    "buffer_size": $WM_BUFFER_SIZE,
    "checkpoint_every_updates": $WM_CHECKPOINT_EVERY,
    "sanity_check": {
      "enabled": $WM_SANITY_CHECK,
      "num_trajs_per_update": $WM_SANITY_NUM_TRAJS,
      "windows_per_replay": $WM_SANITY_WINDOWS
    }
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
echo "[full] wrote experiment config to $CONFIG_PATH"

echo "[full] DSRL_ROOT=$DSRL_ROOT"
echo "[full] REWARD_ROOT=$REWARD_ROOT"
echo "[full] POLICY=$POLICY  QUERY_FREQ=$QUERY_FREQ"
echo "[full] TRAJ_BATCH=$TRAJ_BATCH  START_ONLINE=$START_ONLINE_UPDATES"
echo "[full] MULTI_GRAD_STEP=$MULTI_GRAD_STEP  REWARD_GRAD_STEPS=$REWARD_GRAD_STEPS"
echo "[full] ACTION_MAGNITUDE=$ACTION_MAGNITUDE  BASE_POLICY_PROB=$BASE_POLICY_PROB"
echo "[full] CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "[full] WM ft: enabled=$ENABLE_WM_FINETUNE every=$WM_UPDATE_EVERY"
echo "[full]        steps=$WM_GRAD_STEPS bs=$WM_BATCH_SIZE lr=$WM_LR  buf=$WM_BUFFER_SIZE"
echo "[full] WM sanity: enabled=$WM_SANITY_CHECK trajs=$WM_SANITY_NUM_TRAJS windows=$WM_SANITY_WINDOWS"
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
        "--wm-sanity-check" "$WM_SANITY_CHECK"
        "--wm-sanity-num-trajs" "$WM_SANITY_NUM_TRAJS"
        "--wm-sanity-windows" "$WM_SANITY_WINDOWS"
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
        --num-passes "$NUM_PASSES" \
        --windows-per-call "$WINDOWS_PER_CALL" \
        $( [ "$RANDOM_SPREAD" = "1" ] && echo "--random-spread" ) \
        --start-frame "$START_FRAME" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --scoring-mode "$SCORING_MODE" \
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
    --reward_loss_mode "$REWARD_LOSS_MODE" \
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
