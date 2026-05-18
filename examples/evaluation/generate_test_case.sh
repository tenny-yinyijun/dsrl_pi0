#!/bin/bash
# Generate a multitask trajectory test set for evaluating world-model
# checkpoints via replay-LPIPS / MSE / PSNR / SSIM.
#
# Run from the dsrl_pi0 repo root:
#   bash examples/evaluation/generate_test_case.sh

set -e

NAME=libero_goal_1_multitask_v1
SAVE_ROOT=/scratch/gpfs/AM43/yy4041/playworld_tests
INSTRUCTION_LIST=examples/scripts/libero_goal_1_instructions.json
TASK_SUITE=libero_goal
TASK_ID=1
POLICY=pi05
TRAJS_PER_INSTRUCTION=2
CAM_RESOLUTION=256
MAX_TIMESTEPS=200
SETTLE_STEPS=10
FPS=20
SEED=0

# Per-rollout observation perturbation: with probability PERTURB_PROB,
# add N(0, PERTURB_SIGMA^2) gaussian noise to the policy's state input at
# each query step. Set PERTURB_PROB=0 to disable.
PERTURB_PROB=0.5
PERTURB_SIGMA=0.05
PERTURB_MODE=obs

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$(pwd):${PYTHONPATH}"

source .venv/bin/activate

python examples/scripts/collect_wm_multitask_testset.py \
    --save-dir "${SAVE_ROOT}" \
    --name "${NAME}" \
    --instruction-list "${INSTRUCTION_LIST}" \
    --task-suite "${TASK_SUITE}" \
    --task-id "${TASK_ID}" \
    --policy "${POLICY}" \
    --trajs-per-instruction "${TRAJS_PER_INSTRUCTION}" \
    --cam-resolution "${CAM_RESOLUTION}" \
    --max-timesteps "${MAX_TIMESTEPS}" \
    --settle-steps "${SETTLE_STEPS}" \
    --fps "${FPS}" \
    --seed "${SEED}" \
    --perturb-prob "${PERTURB_PROB}" \
    --perturb-sigma "${PERTURB_SIGMA}" \
    --perturb-mode "${PERTURB_MODE}"
