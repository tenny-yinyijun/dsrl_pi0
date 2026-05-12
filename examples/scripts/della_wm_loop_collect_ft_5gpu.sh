#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --job-name=dsrl-wm-loop-collect-ft
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yy4041@princeton.edu

source ~/.bashrc
# set -euo pipefail   # leave off: bashrc on compute nodes trips set -u

# ============================================================================
# CONTINUOUS COLLECT + ONLINE WM FINE-TUNE — 5-hour 5-GPU run
# ----------------------------------------------------------------------------
# Single coherent run: ONE reward server (with in-process WMFineTuner) +
# ONE trainer that collects continuously. SAC reward signal uses the
# (live-updating) WM, so policy improvements compound with WM improvements.
#
#   * Every WM_UPDATE_EVERY=200 scored trajs, the reward server pauses
#     scoring and runs WM_GRAD_STEPS=1000 grad steps. Sanity videos
#     (before/after side-by-side mp4s) render at the end of every cycle,
#     i.e. every 1000 grad steps. A WM checkpoint is saved every
#     WM_CHECKPOINT_EVERY=5 cycles = 5000 grad steps.
#   * SAC + reward refit fires every SAC_UPDATE_EVERY=50 trajs
#     (4× more often than the overfit50_5gpu template). MULTI_GRAD_STEP
#     is bumped from 10 → 20 so each round triggers more SAC updates,
#     and the auto-tuned entropy target matches policy_5gpu_entropy
#     (TARGET_ENTROPY=3.5) so the actor doesn't collapse to deterministic.
#   * SAC checkpoint cadence chosen so a ckpt lands roughly every 200
#     trajs (= TRAJS_PER_ROUND × transitions/traj × multi_grad_step SAC
#     update steps).
#
# GPU layout (5 GPUs total, but only 2 active — the WM fine-tuner is
# single-GPU so adding parallel reward workers would diverge their model
# state). GPUs 1-3 are reserved but idle; sbatch's --gres=gpu:5 keeps the
# node allocation consistent with the rest of the loop scripts. If you
# want to reclaim those GPUs, drop --gres to gpu:2 and update
# REWARD_GPU / TRAINER_GPU below.
#
# References:
#   - della_wm_loop_overfit50_5gpu.sh — collection knobs, env setup
#   - della_wm_loop_policy_5gpu_entropy.sh — entropy bonus, target_entropy
#   - della_wm_loop_overfit50.sh — in-server WMFineTuner template (single-gpu)
# ============================================================================

# ----------------------------------------------------------------------------
# Task & run length
# ----------------------------------------------------------------------------
TASK_SUITE=libero_goal
TASK_ID="${TASK_ID:-1}"
POLICY="${POLICY:-pi05}"

MAX_TRAJS=1000000                # cap, won't realistically hit in 5h
MAX_STEPS=10000000

# ----------------------------------------------------------------------------
# Round structure
# ----------------------------------------------------------------------------
# Collection: SAC + reward refit fires every SAC_UPDATE_EVERY trajs.
SAC_UPDATE_EVERY=50              # data collection policy updates every 50 trajs
TRAJ_BATCH=$SAC_UPDATE_EVERY
SCORED_PER_ROUND=$SAC_UPDATE_EVERY  # score every traj in the SAC round

# ----------------------------------------------------------------------------
# WM finetune cadence (in-server WMFineTuner)
# ----------------------------------------------------------------------------
WM_UPDATE_EVERY=200              # collect 200 trajs between each WM update
WM_GRAD_STEPS=1000               # 1000 grad steps per cycle → val every 1000
WM_CHECKPOINT_EVERY=5            # save WM ckpt every 5 cycles = 5000 grad steps
WM_BATCH_SIZE=1
WM_LR=1e-5
WM_MAX_GRAD_NORM=1.0
WM_BUFFER_SIZE=400               # roomy rolling buffer; not frozen
WM_BUFFER_FREEZE_AT=0            # 0 = rolling deque (no freeze)
WM_SANITY_CHECK=1                # render before/after videos each cycle
WM_SANITY_NUM_TRAJS=2

# ----------------------------------------------------------------------------
# SAC / data-collection policy (more aggressive than the overfit template)
# ----------------------------------------------------------------------------
WARMUP_TRAJS="${WARMUP_TRAJS:-20}"
MULTI_GRAD_STEP=20               # was 10 — previous runs barely moved
ACTION_MAGNITUDE=1.0
BASE_POLICY_PROB=0.5
TARGET_ENTROPY="${TARGET_ENTROPY:-3.5}"  # entropy bonus, from policy_5gpu_entropy

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
# GPU layout
# ----------------------------------------------------------------------------
# Single reward server (with WMFineTuner) on REWARD_GPU; trainer on
# TRAINER_GPU. GPUs 1-3 idle — see header for why.
REWARD_GPU="${REWARD_GPU:-0}"
TRAINER_GPU="${TRAINER_GPU:-4}"

if [ "$REWARD_GPU" = "$TRAINER_GPU" ]; then
    echo "[loop] FATAL: REWARD_GPU and TRAINER_GPU must differ ($REWARD_GPU)"
    exit 1
fi

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DSRL_ROOT="${DSRL_ROOT:-/scratch/gpfs/AM43/yy4041/dsrl_pi0}"
OPEN_WORLD_ROOT="${OPEN_WORLD_ROOT:-/scratch/gpfs/AM43/yy4041/open-world}"
WM_CKPT="${WM_CKPT:-/scratch/gpfs/AM43/yy4041/open-world/models/wm_training/libero_0429/checkpoint-36000.pt}"
WM_DATASET_ROOT="${WM_DATASET_ROOT:-/scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_processed}"

# ============================================================================
# Derived
# ============================================================================
DATE_DIR=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
JOB_TAG="${SLURM_JOB_ID:-local}_${TIME_TAG}_collect_ft_5gpu"
REWARD_ROOT="${REWARD_ROOT:-/scratch/gpfs/AM43/yy4041/playworld_rollout/$DATE_DIR/$JOB_TAG}"

if [ -z "${QUERY_FREQ:-}" ]; then
    if [ "$POLICY" = "pi05" ]; then
        QUERY_FREQ=10
    else
        QUERY_FREQ=20
    fi
fi

# SAC checkpoint cadence — target one ckpt every 200 trajs.
# Per-traj SAC update count ≈ transitions/traj × multi_grad_step.
# transitions/traj ≈ ~100 env steps / QUERY_FREQ. Empirically trajs
# terminate well before the 400-step LIBERO cap (see slurm log: ~93
# env steps/traj observed), so a 100-step estimate matches reality.
TRANSITIONS_PER_TRAJ=$(( 100 / QUERY_FREQ ))
SAC_CKPT_TRAJS=200
CHECKPOINT_INTERVAL=$(( SAC_CKPT_TRAJS * TRANSITIONS_PER_TRAJ * MULTI_GRAD_STEP ))

MAX_TRANSITIONS_PER_TRAJ=$TRANSITIONS_PER_TRAJ
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
        echo "[loop] FATAL: missing cached artifact: $f"
        echo "[loop] run setup_caches.sh on a login node first."
        exit 1
    fi
done

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "[loop] FATAL: only $NUM_GPUS GPU(s) visible, need >= 2."
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
    "experiment_type": "continuous_collect_plus_in_server_wm_finetune",
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
    "update_every_trajs": $WM_UPDATE_EVERY,
    "grad_steps_per_cycle": $WM_GRAD_STEPS,
    "val_videos_every_steps": $WM_GRAD_STEPS,
    "checkpoint_every_cycles": $WM_CHECKPOINT_EVERY,
    "checkpoint_every_steps": $((WM_GRAD_STEPS * WM_CHECKPOINT_EVERY)),
    "batch_size": $WM_BATCH_SIZE,
    "lr": $WM_LR,
    "buffer_size": $WM_BUFFER_SIZE,
    "sanity_check": $WM_SANITY_CHECK
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
    "reward_gpu": $REWARD_GPU,
    "trainer_gpu": $TRAINER_GPU,
    "note": "GPUs 1-3 (or other non-reward/trainer ids) are idle — see header"
  }
}
EOF
echo "[loop] wrote experiment config to $CONFIG_PATH"

echo "[loop] REWARD_ROOT=$REWARD_ROOT"
echo "[loop] POLICY=$POLICY  QUERY_FREQ=$QUERY_FREQ"
echo "[loop] SAC update_every=$SAC_UPDATE_EVERY trajs  multi_grad_step=$MULTI_GRAD_STEP  target_entropy=$TARGET_ENTROPY"
echo "[loop] SAC ckpt every ~$SAC_CKPT_TRAJS trajs  (checkpoint_interval=$CHECKPOINT_INTERVAL sac steps)"
echo "[loop] WM finetune: every $WM_UPDATE_EVERY trajs, $WM_GRAD_STEPS grad steps per cycle"
echo "[loop] WM ckpt every $WM_CHECKPOINT_EVERY cycles = $((WM_GRAD_STEPS * WM_CHECKPOINT_EVERY)) grad steps"
echo "[loop] WM val videos every $WM_GRAD_STEPS grad steps (per-cycle sanity check)"
echo "[loop] GPUs:  reward=$REWARD_GPU  trainer=$TRAINER_GPU"
nvidia-smi -L | head -8 || true

# ---------------------------------------------------------------------------
# Reward server (single worker, WM fine-tune enabled)
# ---------------------------------------------------------------------------
SERVER_LOG="$LOG_DIR/reward_server.log"
echo "[loop] starting reward server (WM fine-tune on) on GPU $REWARD_GPU  (logs -> $SERVER_LOG)"

WM_FT_ARGS=(
    "--enable-wm-finetune"
    "--wm-update-every" "$WM_UPDATE_EVERY"
    "--wm-grad-steps" "$WM_GRAD_STEPS"
    "--wm-batch-size" "$WM_BATCH_SIZE"
    "--wm-lr" "$WM_LR"
    "--wm-max-grad-norm" "$WM_MAX_GRAD_NORM"
    "--wm-buffer-size" "$WM_BUFFER_SIZE"
    "--wm-buffer-freeze-at" "$WM_BUFFER_FREEZE_AT"
    "--wm-checkpoint-every" "$WM_CHECKPOINT_EVERY"
    "--wm-sanity-check" "$WM_SANITY_CHECK"
    "--wm-sanity-num-trajs" "$WM_SANITY_NUM_TRAJS"
)

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
        --worker-id "primary" \
        "${WM_FT_ARGS[@]}" \
        > "$SERVER_LOG" 2>&1
) &
SERVER_PID=$!
echo "[loop] reward server pid=$SERVER_PID"

cleanup() {
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "[loop] stopping reward server (pid=$SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

echo "[loop] waiting up to ${SERVER_READY_TIMEOUT_S}s for reward server to load..."
DEADLINE=$(($(date +%s) + SERVER_READY_TIMEOUT_S))
while ! grep -q "ready. polling" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[loop] FATAL: reward server died before becoming ready"
        tail -120 "$SERVER_LOG" || true
        exit 1
    fi
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[loop] FATAL: server didn't become ready in ${SERVER_READY_TIMEOUT_S}s"
        tail -120 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[loop] reward server ready."

# ---------------------------------------------------------------------------
# Trainer (continuous; collects until SLURM time limit)
# ---------------------------------------------------------------------------
TRAINER_LOG="$LOG_DIR/trainer.log"
echo "[loop] starting trainer on GPU $TRAINER_GPU  (logs -> $TRAINER_LOG)"

cd "$DSRL_ROOT"
export PYTHONPATH="$DSRL_ROOT:${PYTHONPATH:-}"
export DSRL_REWARD_ROOT="$REWARD_ROOT"
export DSRL_REWARD_TIMEOUT_S=1800
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
    --wandb_project DSRL_pi0_libero_wm_collect_ft_5gpu \
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
echo "[loop] trainer pid=$TRAINER_PID"

cleanup_all() {
    if [ -n "${TRAINER_PID:-}" ] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "[loop] stopping trainer (pid=$TRAINER_PID)"
        kill "$TRAINER_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$TRAINER_PID" 2>/dev/null || true
    fi
    cleanup
}
trap cleanup_all INT TERM EXIT

echo "[loop] ----------------------------------------------------------------"
echo "[loop] streaming trainer log. Ctrl+C (or SLURM timeout) -> cleanup."
echo "[loop] WM ckpts:           $REWARD_ROOT/wm_checkpoints/"
echo "[loop] WM sanity videos:   $REWARD_ROOT/wm_update_sanity_check/update_<n>/"
echo "[loop] SAC ckpts:          $DSRL_ROOT/logs/dsrl_wm_$JOB_TAG/"
echo "[loop] Collected trajs:    $REWARD_ROOT/{annotation,raw_videos,latent_videos}/"
echo "[loop] ----------------------------------------------------------------"

tail -n +1 -F "$TRAINER_LOG" --pid="$TRAINER_PID" &
TAIL_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
set -e
kill "$TAIL_PID" 2>/dev/null || true

echo "[loop] trainer exited rc=$TRAINER_RC"
echo "[loop] final ls of WM ckpts:"
ls -lh "$REWARD_ROOT/wm_checkpoints/" 2>/dev/null | tail -20 || true
echo "[loop] final ls of sanity dirs:"
ls "$REWARD_ROOT/wm_update_sanity_check/" 2>/dev/null | tail -10 || true
exit "$TRAINER_RC"
