#!/bin/bash
# Collect 200 pi0 rollouts on libero_goal task=1 (put_the_bowl_on_the_stove)
# in libero_processed format, for fine-tuning a world model.
#
# Output: /scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_goal_1_ft/
#   annotation/train/<eid>.json + annotation/val/<eid>.json
#   latent_videos/agentview/<eid>.pt + latent_videos/wrist/<eid>.pt
#   train_sample.json + val_sample.json
#
# Run from repo root: bash examples/scripts/run_collect_libero_goal_1_ft.sh

set -e

device_id=${CUDA_VISIBLE_DEVICES:-0}

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The .venv ships a stub namespace `examples` package that shadows the repo's
# examples/ unless the repo root is on PYTHONPATH. Force it.
export PYTHONPATH="$(pwd):${PYTHONPATH}"

source .venv/bin/activate

python examples/scripts/collect_wm_ft_data.py \
    --save-dir /scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_goal_1_ft \
    --task-suite libero_goal \
    --task-id 1 \
    --num-trajs 200 \
    --val-fraction 0.1 \
    --policy pi0 \
    --cam-resolution 256 \
    --max-timesteps 400 \
    --settle-steps 10 \
    --fps 20 \
    --seed 0
