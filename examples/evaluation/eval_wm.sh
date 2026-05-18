#!/bin/bash
# Evaluate one world-model checkpoint on the multitask test set generated
# by examples/evaluation/generate_test_case.sh. Computes per-traj +
# averaged LPIPS / MSE / PSNR / SSIM, bucketed overall, by success, by
# perturbed-vs-clean, and by instruction.
#
# Run from any cwd; this script cds into the open-world repo so the
# pipeline finds its config + asset paths.
#   bash /scratch/gpfs/AM43/yy4041/dsrl_pi0/examples/evaluation/eval_wm.sh

set -e

OPEN_WORLD_ROOT=/scratch/gpfs/AM43/yy4041/open-world
DSRL_ROOT=/scratch/gpfs/AM43/yy4041/dsrl_pi0

NAME=libero_goal_1_multitask_v1
TESTSET_ROOT=/scratch/gpfs/AM43/yy4041/playworld_tests
EVAL_ROOT=/scratch/gpfs/AM43/yy4041/playworld_tests/wm_eval

# stat.json (state percentiles for action normalization) lives here for
# the 0518 collect_ft wm_checkpoints. Must match what the WM was trained
# against, otherwise the action input is mis-scaled and predictions drift.
DATASET_ROOT=/scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_processed

CKPT_CYCLE=cycle_1
CKPT_PATH=/scratch/gpfs/AM43/yy4041/playworld_rollout/0518/8387623_090651_collect_ft_seq_mt/wm_checkpoints/${CKPT_CYCLE}/checkpoint-2000.pt

NUM_WINDOWS=20
NUM_INFERENCE_STEPS=50
# Per-traj cursor advances by (num_frames-1)=4 per window; with
# --random-start-frame each trajectory samples START_FRAME uniformly from
#   [num_history*SKIP_HIS, T_wm - 1 - NUM_WINDOWS*(num_frames-1)]
# so all 6 history frames AND all NUM_WINDOWS prediction windows stay in
# bounds. START_FRAME below is the FALLBACK when not using random mode.
START_FRAME=20
SKIP_HIS=3
DEVICE=cuda:0

# NOTE on action stride: libero_wm.py now uses down_sample=1 (latents and
# state aligned 1:1, matching the actual preprocessed data layout), so
# state_id = cursor directly. No action clipping until cursor reaches
# T_state - 1 = ~199. With NUM_WINDOWS=20 and start_frame in [18, 119],
# final cursor maxes at ~199 - all in-bounds for both latents AND actions.
# (The legacy down_sample=4 version had a hard cliff at cursor=50.)

export OPEN_WORLD_ROOT
export CUDA_VISIBLE_DEVICES=0

cd "${OPEN_WORLD_ROOT}"

.venv/bin/python "${DSRL_ROOT}/examples/scripts/eval_wm_multitask.py" \
    --testset-dir "${TESTSET_ROOT}/${NAME}" \
    --ckpt-path "${CKPT_PATH}" \
    --output "${EVAL_ROOT}/${CKPT_CYCLE}_on_${NAME}" \
    --dataset-root "${DATASET_ROOT}" \
    --num-windows "${NUM_WINDOWS}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --start-frame "${START_FRAME}" \
    --skip-his "${SKIP_HIS}" \
    --device "${DEVICE}" \
    --random-start-frame \
    --save-predictions
