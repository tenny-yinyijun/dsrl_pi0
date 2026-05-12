source ~/.bashrc
# set -euo pipefail

# ============================================================================
# WM FINE-TUNE OVERFIT EXPERIMENT — 50-TRAJ, 5-GPU MULTI-GPU TRAINING VARIANT
# ----------------------------------------------------------------------------
# Same diagnostic intent as della_wm_loop_overfit50.sh (collect N=50 π₀
# trajs, then fine-tune the WM on that frozen buffer) but with WM updates
# parallelized across 4 GPUs via accelerate-DDP, giving an effective batch
# size of 4 × WM_TRAIN_BATCH_PER_GPU. The 5th GPU is the SAC/data
# collector. Runs in two phases:
#
# Phase 1 — Collection (2 GPUs in use, 3 idle):
#   * GPU = REWARD_GPU0 : reward_server.py with WM finetune nominally
#     enabled but WM_UPDATE_EVERY set so high it never fires. The only
#     reason the server runs at all in this phase is that the wm_only
#     code path encodes raw mp4s -> VAE latents and CACHES them to disk
#     (see _load_or_encode_latents in reward_server.py). Phase 2 needs
#     those latents on disk to feed train_wm.py.
#   * GPU = TRAINER_GPU : launch_collect.py collects MAX_TRAJS=50 trajs
#     of pure π₀ (BASE_POLICY_PROB=1.0), no SAC updates, no scoring.
#
# Phase 2 — Multi-GPU fine-tune (4 GPUs):
#   * accelerate launch on REWARD_GPUS (4 GPUs) running train_wm.py on the
#     50 collected trajs as a single-suite dataset. Effective batch =
#     WM_TRAIN_BATCH_PER_GPU × 4. Checkpoints land in
#     $REWARD_ROOT/wm_checkpoints/ on the same schedule as the single-GPU
#     overfit variant.
#
# The reward server's in-process WMFineTuner is NOT used for training in
# this variant — it'd be single-GPU. That class is fine for the original
# 2-GPU overfit script but doesn't parallelize.
#
# After phase 2 finishes, run examples/scripts/wm_overfit_sanity.sh
# pointing at a saved checkpoint to render before/after videos. (Phase 2
# itself emits validation videos via train_wm.py's built-in validation
# loop, under $REWARD_ROOT/wm_checkpoints/steps_<num_inference>/samples/.)
# ============================================================================

# ----------------------------------------------------------------------------
# Task & run length
# ----------------------------------------------------------------------------
TASK_SUITE=libero_goal
TASK_ID="${TASK_ID:-1}"

POLICY="${POLICY:-pi05}"

MAX_TRAJS=50                   # collect 50 trajs, then exit
MAX_STEPS=500000

# ----------------------------------------------------------------------------
# Round structure (phase 1 only — disabled effects since no SAC/scoring)
# ----------------------------------------------------------------------------
ROUND_SIZE=2
SCORED_PER_ROUND=0             # no LPIPS scoring; all trajs go via wm_only
WM_UPDATE_EVERY_ROUNDS=999999  # gate phase-1 in-server fine-tune cycles
                               # off (we only want latent caching here).

TRAJ_BATCH=$ROUND_SIZE
WM_UPDATE_EVERY=$((ROUND_SIZE * WM_UPDATE_EVERY_ROUNDS))

# ----------------------------------------------------------------------------
# SAC / data-collection policy
# ----------------------------------------------------------------------------
START_ONLINE_UPDATES=10
MULTI_GRAD_STEP=0              # no SAC updates
ACTION_MAGNITUDE=1.0
BASE_POLICY_PROB=1.0           # all rollouts use base π₀
CHECKPOINT_INTERVAL=5000

# ----------------------------------------------------------------------------
# Reward model (not used; phase 1 has SCORED_PER_ROUND=0)
# ----------------------------------------------------------------------------
REWARD_GRAD_STEPS=200
REWARD_LR=3e-4
REWARD_LOSS_MODE=per_step

# ----------------------------------------------------------------------------
# WM scoring (not used in phase 1; we still pass these to keep the server
# arg-parsing happy)
# ----------------------------------------------------------------------------
SCORING_MODE=spread
NUM_PASSES=5
WINDOWS_PER_CALL=4
RANDOM_SPREAD=1
START_FRAME=24
NUM_INFERENCE_STEPS=50
NUM_WINDOWS=20

# ----------------------------------------------------------------------------
# WM fine-tuning — PHASE-1 SETTINGS (latent-caching only) + PHASE-2 SETTINGS
# (multi-GPU training)
# ----------------------------------------------------------------------------
# Phase 1: in-server WMFineTuner is created (so wm_only requests trigger
# latent encoding) but WM_UPDATE_EVERY=huge so no gradient cycles fire.
ENABLE_WM_FINETUNE_PHASE1=1
WM_GRAD_STEPS_PHASE1=1         # ignored; kept >0 so finetuner constructs OK
WM_BATCH_SIZE_PHASE1=1
WM_LR_PHASE1=1e-6
WM_MAX_GRAD_NORM=1.0
WM_BUFFER_SIZE=50
WM_BUFFER_FREEZE_AT=50
WM_CHECKPOINT_EVERY=999999     # disable phase-1 ckpt saving (no cycles fire anyway)
WM_SANITY_CHECK_PHASE1=0       # phase 1 doesn't render sanity (no fine-tune
                               # happens there); use wm_overfit_sanity.sh
                               # on the phase-2 checkpoints.

# Phase 2: accelerate-DDP across 4 GPUs.
WM_TRAIN_BATCH_PER_GPU="${WM_TRAIN_BATCH_PER_GPU:-4}"   # effective batch = 4 × 4 = 16
WM_LR="${WM_LR:-2e-5}"                                  # match the 1-GPU overfit50
WM_TOTAL_STEPS="${WM_TOTAL_STEPS:-5000}"
WM_CKPT_EVERY_STEPS="${WM_CKPT_EVERY_STEPS:-100}"
WM_VAL_EVERY_STEPS="${WM_VAL_EVERY_STEPS:-100}"
# num_workers=0 avoids fork-time RSS amplification — each DataLoader worker
# is a forked child whose copy-on-write pages can blow up CPU RAM when the
# parent has the 9GB WM loaded. With 4 ranks × N workers that's 4N extra
# forks, each a candidate for the OOM killer. Pick 0 unless you've
# explicitly confirmed there's headroom.
WM_NUM_WORKERS="${WM_NUM_WORKERS:-0}"

# ----------------------------------------------------------------------------
# GPU layout (5 GPUs)
# ----------------------------------------------------------------------------
# REWARD_GPUS used in phase 2 for accelerate-DDP. Phase 1 reuses
# REWARD_GPUS[0] as the lone reward server GPU.
REWARD_GPUS="${REWARD_GPUS:-0,1,2,3}"
TRAINER_GPU="${TRAINER_GPU:-4}"

IFS=',' read -r -a REWARD_GPU_ARR <<< "$REWARD_GPUS"
NUM_REWARD_WORKERS=${#REWARD_GPU_ARR[@]}
PHASE1_REWARD_GPU="${REWARD_GPU_ARR[0]}"

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
JOB_TAG="${SLURM_JOB_ID:-local}_${TIME_TAG}_overfit50_5gpu"
REWARD_ROOT="${REWARD_ROOT:-/scratch/gpfs/AM43/yy4041/playworld_rollout/$DATE_DIR/$JOB_TAG}"

if [ -z "${QUERY_FREQ:-}" ]; then
    if [ "$POLICY" = "pi05" ]; then
        QUERY_FREQ=10
    else
        QUERY_FREQ=20
    fi
fi

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
# Verify caches
# ---------------------------------------------------------------------------
PI_CACHE="/scratch/gpfs/AM43/yy4041/.cache/openpi/openpi-assets/checkpoints/${POLICY}_libero"
REQUIRED=(
    "$DSRL_ROOT/.venv/bin/python"
    "$OPEN_WORLD_ROOT/.venv/bin/python"
    "$OPEN_WORLD_ROOT/.venv/bin/accelerate"
    "$OPEN_WORLD_ROOT/external/stable-video-diffusion-img2vid"
    "$OPEN_WORLD_ROOT/external/clip-vit-base-patch32"
    "$TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth"
    "$WM_CKPT"
    "$WM_DATASET_ROOT/stat.json"
    "$PI_CACHE"
)
for f in "${REQUIRED[@]}"; do
    if [ ! -e "$f" ]; then
        echo "[ov50-5gpu] FATAL: missing cached artifact: $f"
        exit 1
    fi
done

# ---- GPU sanity ----
declare -A GPU_SEEN
for g in "${REWARD_GPU_ARR[@]}"; do
    if [ "$g" = "$TRAINER_GPU" ]; then
        echo "[ov50-5gpu] FATAL: TRAINER_GPU=$TRAINER_GPU appears in REWARD_GPUS=$REWARD_GPUS"
        exit 1
    fi
    if [ -n "${GPU_SEEN[$g]:-}" ]; then
        echo "[ov50-5gpu] FATAL: REWARD_GPUS has duplicate id $g"
        exit 1
    fi
    GPU_SEEN[$g]=1
done

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
NEED=$(( NUM_REWARD_WORKERS + 1 ))
if [ "$NUM_GPUS" -lt "$NEED" ]; then
    echo "[ov50-5gpu] FATAL: only $NUM_GPUS GPU(s) visible, need $NEED."
    exit 1
fi

mkdir -p "$REWARD_ROOT"
LOG_DIR="$REWARD_ROOT/_logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Snapshot experiment config
# ---------------------------------------------------------------------------
DSRL_GIT_SHA=$(git -C "$DSRL_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
OPEN_WORLD_GIT_SHA=$(git -C "$OPEN_WORLD_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
CONFIG_PATH="$REWARD_ROOT/config.json"
cat > "$CONFIG_PATH" <<EOF
{
  "meta": {
    "job_tag": "$JOB_TAG",
    "experiment_type": "overfit50_5gpu_multi_gpu_finetune",
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
  "phase1_collection": {
    "max_trajs": $MAX_TRAJS,
    "scored_per_round": $SCORED_PER_ROUND,
    "phase1_reward_gpu": $PHASE1_REWARD_GPU,
    "trainer_gpu": $TRAINER_GPU
  },
  "phase2_multi_gpu_finetune": {
    "reward_gpus": "$REWARD_GPUS",
    "num_processes": $NUM_REWARD_WORKERS,
    "per_gpu_batch": $WM_TRAIN_BATCH_PER_GPU,
    "effective_batch": $((WM_TRAIN_BATCH_PER_GPU * NUM_REWARD_WORKERS)),
    "lr": $WM_LR,
    "max_train_steps": $WM_TOTAL_STEPS,
    "checkpointing_steps": $WM_CKPT_EVERY_STEPS,
    "validation_steps": $WM_VAL_EVERY_STEPS
  }
}
EOF
echo "[ov50-5gpu] wrote experiment config to $CONFIG_PATH"

echo "[ov50-5gpu] REWARD_ROOT=$REWARD_ROOT"
echo "[ov50-5gpu] PHASE1 reward GPU=$PHASE1_REWARD_GPU  trainer GPU=$TRAINER_GPU"
echo "[ov50-5gpu] PHASE2 reward GPUs=$REWARD_GPUS  ($NUM_REWARD_WORKERS-way DDP)"
echo "[ov50-5gpu] PHASE2 per-gpu batch=$WM_TRAIN_BATCH_PER_GPU  effective batch=$((WM_TRAIN_BATCH_PER_GPU * NUM_REWARD_WORKERS))"
echo "[ov50-5gpu] PHASE2 lr=$WM_LR  total_steps=$WM_TOTAL_STEPS  ckpt_every=$WM_CKPT_EVERY_STEPS"
nvidia-smi -L | head -8 || true

# ============================================================================
# PHASE 1: collect 50 trajs (reward server only encodes latents)
# ============================================================================
# Skip phase 1 if a prior run already has all $MAX_TRAJS annotations +
# latents on disk under $REWARD_ROOT. Useful when iterating on phase-2
# settings — point REWARD_ROOT at the previous run and rerun. Set
# FORCE_PHASE1=1 to bypass this fast-path.
existing_ann=$(find "$REWARD_ROOT/annotation/train" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
existing_agent=$(find "$REWARD_ROOT/latent_videos/agentview" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
existing_wrist=$(find "$REWARD_ROOT/latent_videos/wrist" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)

# MIN_TRAJS_PHASE2: minimum trajectory count required to proceed to phase 2.
# Defaults to MAX_TRAJS (strict). Set lower to allow phase 2 to run on a
# partial collection (e.g. after Ctrl+C during phase 1 — set to whatever
# the "phase-1 results: annotations=N" line printed).
MIN_TRAJS_PHASE2="${MIN_TRAJS_PHASE2:-$MAX_TRAJS}"

# REMAINING_TRAJS: how many more trajs phase 1 should try to collect.
# - existing >= MIN_TRAJS_PHASE2: skip phase 1 entirely.
# - 0 < existing < MAX_TRAJS:    resume — collect (MAX_TRAJS - existing) more.
# - existing == 0:                fresh collection of MAX_TRAJS trajs.
# The trainer's launch_collect.py already calls find_next_episode_id to pick
# the next eid; we just need to bound how many MORE it should collect.
if [ "${FORCE_PHASE1:-0}" != "1" ] \
   && [ "$existing_ann" -ge "$MIN_TRAJS_PHASE2" ] \
   && [ "$existing_agent" -ge "$MIN_TRAJS_PHASE2" ] \
   && [ "$existing_wrist" -ge "$MIN_TRAJS_PHASE2" ]; then
    echo "[ov50-5gpu] ========== PHASE 1: SKIPPED (>= MIN_TRAJS_PHASE2 already present) =========="
    echo "[ov50-5gpu] found ann=$existing_ann  agent_lat=$existing_agent  wrist_lat=$existing_wrist  (min=$MIN_TRAJS_PHASE2)"
    echo "[ov50-5gpu] set FORCE_PHASE1=1 to re-collect."
    SKIP_PHASE1=1
    REMAINING_TRAJS=0
else
    SKIP_PHASE1=0
    if [ "$existing_ann" -gt 0 ] && [ "${FORCE_PHASE1:-0}" != "1" ]; then
        REMAINING_TRAJS=$((MAX_TRAJS - existing_ann))
        if [ "$REMAINING_TRAJS" -lt 0 ]; then REMAINING_TRAJS=0; fi
        echo "[ov50-5gpu] resuming phase 1: $existing_ann existing trajs, collecting $REMAINING_TRAJS more to reach $MAX_TRAJS"
    else
        REMAINING_TRAJS=$MAX_TRAJS
    fi
fi

if [ "$SKIP_PHASE1" = "0" ]; then
echo "[ov50-5gpu] ========== PHASE 1: COLLECTION =========="
SERVER_LOG="$LOG_DIR/reward_server_phase1.log"
echo "[ov50-5gpu] starting reward server (latent-caching only) on GPU $PHASE1_REWARD_GPU"

WM_FT_ARGS_PHASE1=(
    "--enable-wm-finetune"
    "--wm-update-every" "$WM_UPDATE_EVERY"
    "--wm-grad-steps" "$WM_GRAD_STEPS_PHASE1"
    "--wm-batch-size" "$WM_BATCH_SIZE_PHASE1"
    "--wm-lr" "$WM_LR_PHASE1"
    "--wm-max-grad-norm" "$WM_MAX_GRAD_NORM"
    "--wm-buffer-size" "$WM_BUFFER_SIZE"
    "--wm-buffer-freeze-at" "$WM_BUFFER_FREEZE_AT"
    "--wm-checkpoint-every" "$WM_CHECKPOINT_EVERY"
    "--wm-sanity-check" "$WM_SANITY_CHECK_PHASE1"
)

(
    cd "$OPEN_WORLD_ROOT"
    CUDA_VISIBLE_DEVICES=$PHASE1_REWARD_GPU \
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
        --worker-id "phase1" \
        "${WM_FT_ARGS_PHASE1[@]}" \
        > "$SERVER_LOG" 2>&1
) &
SERVER_PID=$!
echo "[ov50-5gpu] reward server pid=$SERVER_PID"

cleanup_phase1() {
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "[ov50-5gpu] stopping reward server (pid=$SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup_phase1 INT TERM EXIT

echo "[ov50-5gpu] waiting up to ${SERVER_READY_TIMEOUT_S}s for server to load..."
DEADLINE=$(($(date +%s) + SERVER_READY_TIMEOUT_S))
while ! grep -q "ready. polling" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[ov50-5gpu] FATAL: reward server died before becoming ready"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[ov50-5gpu] FATAL: server didn't become ready in ${SERVER_READY_TIMEOUT_S}s"
        tail -80 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[ov50-5gpu] reward server ready."

TRAINER_LOG="$LOG_DIR/trainer_phase1.log"
echo "[ov50-5gpu] starting trainer on GPU $TRAINER_GPU (logs -> $TRAINER_LOG)"
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
PYTHONUNBUFFERED=1 \
python3 examples/launch_collect.py \
    --algorithm pixel_sac \
    --env libero \
    --policy "$POLICY" \
    --prefix "dsrl_pi0_libero_wm_$JOB_TAG" \
    --wandb_project DSRL_pi0_libero_wm_overfit50_5gpu \
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
    --max_trajs "$REMAINING_TRAJS" \
    > "$TRAINER_LOG" 2>&1

echo "[ov50-5gpu] trainer (phase 1) exited."

# Drain any pending wm_only requests so all 50 trajs have latents on disk
# before phase 2 starts. The server runs the wm_only encode in a tight loop;
# we wait until no .wm_only or .taken-* claims remain.
echo "[ov50-5gpu] draining pending wm_only requests..."
DRAIN_DEADLINE=$(($(date +%s) + 600))
while :; do
    pending=$(find "$REWARD_ROOT/requests" -maxdepth 1 \( -name "*.wm_only" -o -name "*.wm_only.taken-*" \) 2>/dev/null | wc -l)
    if [ "$pending" -eq 0 ]; then break; fi
    if [ $(date +%s) -gt $DRAIN_DEADLINE ]; then
        echo "[ov50-5gpu] WARN: still $pending wm_only pending after 10min; proceeding anyway"
        break
    fi
    sleep 3
done

# Verify we got 50 latents per cam (otherwise phase 2 would dataset-fail).
n_ann=$(find "$REWARD_ROOT/annotation/train" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
n_lat_agent=$(find "$REWARD_ROOT/latent_videos/agentview" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
n_lat_wrist=$(find "$REWARD_ROOT/latent_videos/wrist" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
echo "[ov50-5gpu] phase-1 results: annotations=$n_ann  agent_latents=$n_lat_agent  wrist_latents=$n_lat_wrist"
if [ "$n_ann" -lt "$MIN_TRAJS_PHASE2" ] \
   || [ "$n_lat_agent" -lt "$MIN_TRAJS_PHASE2" ] \
   || [ "$n_lat_wrist" -lt "$MIN_TRAJS_PHASE2" ]; then
    echo "[ov50-5gpu] FATAL: phase 1 produced $n_ann trajs but MIN_TRAJS_PHASE2=$MIN_TRAJS_PHASE2."
    echo "[ov50-5gpu] options:"
    echo "[ov50-5gpu]   1) re-run with the same REWARD_ROOT to resume collection up to MAX_TRAJS=$MAX_TRAJS"
    echo "[ov50-5gpu]   2) re-run with MIN_TRAJS_PHASE2=$n_ann to proceed with what's already on disk"
    exit 1
fi

echo "[ov50-5gpu] stopping phase-1 reward server"
cleanup_phase1
trap - INT TERM EXIT  # phase 2 installs its own trap

fi  # /SKIP_PHASE1

# ============================================================================
# PHASE 2: multi-GPU fine-tune via accelerate launch + train_wm.py
# ============================================================================
echo "[ov50-5gpu] ========== PHASE 2: MULTI-GPU FINE-TUNE =========="

# LiberoLatentDataset wants:
#   <dataset_root>/<suite>/annotation/{train,val}/<eid>.json
#   <dataset_root>/<suite>/latent_videos/<cam>/<eid>.pt
#   <dataset_root>/<suite>/{train,val}_sample.json
# Phase 1 only writes the "train" half. Make val mirror train so train_wm.py's
# validation step has something to load. We're overfitting on training data
# anyway, so train==val is intentional.
if [ ! -e "$REWARD_ROOT/annotation/val" ]; then
    ln -s "train" "$REWARD_ROOT/annotation/val"
fi
if [ ! -e "$REWARD_ROOT/val_sample.json" ]; then
    cp "$REWARD_ROOT/train_sample.json" "$REWARD_ROOT/val_sample.json"
fi

# Generate a config .py for train_wm.py that points at $REWARD_ROOT as a
# single-suite dataset. Suite "." resolves to $REWARD_ROOT/. directly.
TRAIN_CFG="$LOG_DIR/wm_train_config.py"
WM_OUT_DIR="$REWARD_ROOT/wm_checkpoints"
mkdir -p "$WM_OUT_DIR"

cat > "$TRAIN_CFG" <<PYEOF
"""Auto-generated config for phase-2 multi-GPU fine-tune.

Written by della_wm_loop_overfit50_5gpu.sh. Points train_wm.py at the
$MAX_TRAJS trajs collected in phase 1 ($REWARD_ROOT) as a single-suite
dataset. stat.json is reused from the pretrain root via the loader's
<meta_root>/stat.json fallback (since this suite has no stat.json of its
own).
"""

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    return LiberoWMArgs(
        # Paths
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path="$WM_CKPT",

        # Dataset: $REWARD_ROOT is the single suite (name="."), so paths
        # resolve as <REWARD_ROOT>/./annotation/{train,val}/<eid>.json etc.
        # meta_info_path points at the pretrain root so stat.json is found
        # via the loader's fallback.
        dataset_root_path="$REWARD_ROOT",
        dataset_meta_info_path="$WM_DATASET_ROOT",
        dataset_names=".",
        dataset_cfgs=".",
        prob=(1.0,),

        # Compute
        train_batch_size=$WM_TRAIN_BATCH_PER_GPU,
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        num_workers=$WM_NUM_WORKERS,

        # Schedule
        learning_rate=$WM_LR,
        max_train_steps=$WM_TOTAL_STEPS,
        checkpointing_steps=$WM_CKPT_EVERY_STEPS,
        validation_steps=$WM_VAL_EVERY_STEPS,
        max_grad_norm=1.0,

        # Architecture (LIBERO defaults)
        num_cams=2,
        num_frames=5,
        num_history=6,
        action_dim=7,
        down_sample=4,

        # Loss
        flow_map_type="flow_matching",
        distance_conditioning=False,

        tag="overfit50_5gpu_$JOB_TAG",
    )
PYEOF
echo "[ov50-5gpu] wrote $TRAIN_CFG"

# Run accelerate-DDP. CUDA_VISIBLE_DEVICES restricts to the 4 reward GPUs;
# accelerate picks them up as devices 0..3.
PHASE2_LOG="$LOG_DIR/wm_train_phase2.log"
echo "[ov50-5gpu] launching $NUM_REWARD_WORKERS-way DDP on REWARD_GPUS=$REWARD_GPUS"
echo "[ov50-5gpu] (logs -> $PHASE2_LOG)"

cleanup_phase2() {
    if [ -n "${SANITY_PID:-}" ] && kill -0 "$SANITY_PID" 2>/dev/null; then
        echo "[ov50-5gpu] stopping sanity watcher (pid=$SANITY_PID)"
        # Kill the watcher AND any in-flight render subprocess (they live in
        # the watcher's process group).
        kill -- "-$SANITY_PID" 2>/dev/null || kill "$SANITY_PID" 2>/dev/null || true
    fi
    if [ -n "${PHASE2_PID:-}" ] && kill -0 "$PHASE2_PID" 2>/dev/null; then
        echo "[ov50-5gpu] stopping phase-2 trainer (pid=$PHASE2_PID)"
        kill "$PHASE2_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$PHASE2_PID" 2>/dev/null || true
    fi
}
trap cleanup_phase2 INT TERM EXIT

(
    cd "$OPEN_WORLD_ROOT"
    # train_wm.py is invoked via `-m openworld...`; needs OPEN_WORLD_ROOT
    # on PYTHONPATH so spawned worker procs can import the package.
    CUDA_VISIBLE_DEVICES=$REWARD_GPUS \
    OPEN_WORLD_ROOT="$OPEN_WORLD_ROOT" \
    PYTHONPATH="$OPEN_WORLD_ROOT:${PYTHONPATH:-}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    "$OPEN_WORLD_ROOT/.venv/bin/accelerate" launch \
        --num_processes "$NUM_REWARD_WORKERS" \
        --mixed_precision fp16 \
        -m openworld.training.world_model.train_wm \
        --config "$TRAIN_CFG" \
        --output_dir "$WM_OUT_DIR" \
        > "$PHASE2_LOG" 2>&1
) &
PHASE2_PID=$!
echo "[ov50-5gpu] phase-2 trainer pid=$PHASE2_PID"

# ----------------------------------------------------------------
# Background sanity-render watcher.
# Polls $WM_OUT_DIR for new checkpoint-*.pt files; whenever one shows up,
# runs wm_overfit_sanity.py on it using GPU $TRAINER_GPU (idle during
# phase 2 because launch_collect.py is done). Skips already-rendered
# checkpoints. Output lands at $REWARD_ROOT/wm_overfit_sanity/<stem>/.
# Quits when the script's cleanup trap fires.
# ----------------------------------------------------------------
SANITY_LOG="$LOG_DIR/sanity_watcher.log"
SANITY_DONE_DIR="$REWARD_ROOT/wm_overfit_sanity"
mkdir -p "$SANITY_DONE_DIR"
echo "[ov50-5gpu] sanity watcher: GPU $TRAINER_GPU → $SANITY_DONE_DIR/ (log: $SANITY_LOG)"

(
    # New process group so cleanup can SIGKILL the watcher + any in-flight
    # render with a single `kill -- -<pid>`.
    set -m 2>/dev/null || true
    POLL_S=30
    while true; do
        for ckpt in "$WM_OUT_DIR"/checkpoint-*.pt; do
            [ -f "$ckpt" ] || continue
            stem=$(basename "$ckpt" .pt)
            done_dir="$SANITY_DONE_DIR/$stem"
            # Marker dir means we already rendered this checkpoint.
            if [ -d "$done_dir" ]; then continue; fi
            # Wait briefly to let torch.save finish writing.
            sleep 5
            echo "[sanity] rendering $ckpt on GPU $TRAINER_GPU" >> "$SANITY_LOG"
            CUDA_VISIBLE_DEVICES=$TRAINER_GPU \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            OPEN_WORLD_ROOT="$OPEN_WORLD_ROOT" \
            DSRL_ROOT="$DSRL_ROOT" \
            "$OPEN_WORLD_ROOT/.venv/bin/python" \
                "$DSRL_ROOT/examples/scripts/wm_overfit_sanity.py" \
                --reward-root "$REWARD_ROOT" \
                --ckpt-path "$ckpt" \
                --skip-his 4 \
                --start-frame 24 \
                --num-windows 8 \
                --num-eids 2 \
                --num-inference-steps 50 \
                --device "cuda:0" \
                --out-dir "$done_dir" \
                >> "$SANITY_LOG" 2>&1 || \
                echo "[sanity] render failed for $stem (see $SANITY_LOG)" >> "$SANITY_LOG"
        done
        sleep "$POLL_S"
    done
) &
SANITY_PID=$!
echo "[ov50-5gpu] sanity watcher pid=$SANITY_PID"

echo "[ov50-5gpu] ----------------------------------------------------------------"
echo "[ov50-5gpu] streaming phase-2 log. Ctrl+C to stop (cleanup trap will kill the trainer + watcher)."
echo "[ov50-5gpu] checkpoints will land in $WM_OUT_DIR/"
echo "[ov50-5gpu] sanity videos:  $SANITY_DONE_DIR/<ckpt_stem>/<eid>_gt_vs_pred_skiphis4.mp4"
echo "[ov50-5gpu] ----------------------------------------------------------------"

tail -n +1 -F "$PHASE2_LOG" --pid="$PHASE2_PID" &
TAIL_PID=$!

# Wait on the trainer specifically and capture its exit code. `tail -F`
# happens to exit when --pid dies, so we can let it run alongside.
set +e
wait "$PHASE2_PID"
PHASE2_RC=$?
set -e
kill "$TAIL_PID" 2>/dev/null || true

if [ "$PHASE2_RC" -ne 0 ]; then
    echo "[ov50-5gpu] !!!! phase 2 FAILED with exit code $PHASE2_RC"
    echo "[ov50-5gpu] inspect $PHASE2_LOG and the accelerate stack trace above."
    echo "[ov50-5gpu] common culprits:"
    echo "[ov50-5gpu]   * SIGKILL (-9) right after pipeline load = CPU OOM from"
    echo "[ov50-5gpu]     N ranks each materializing a 9GB checkpoint. Make sure"
    echo "[ov50-5gpu]     train_wm.py loads the ckpt with mmap=True, or drop to"
    echo "[ov50-5gpu]     fewer ranks: REWARD_GPUS=0,1 bash $0"
    echo "[ov50-5gpu]   * CUDA OOM = lower WM_TRAIN_BATCH_PER_GPU"
    exit "$PHASE2_RC"
fi

echo "[ov50-5gpu] phase 2 complete."
