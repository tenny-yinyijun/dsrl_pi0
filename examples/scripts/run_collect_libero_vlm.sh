#!/bin/bash
# Collect pi05 rollouts on libero_goal task=1 with per-rollout task
# instructions sampled from a PRE-GENERATED list of VLM outputs, in the
# libero_processed (SVD-latent) format used by the world-model trainer.
#
# Requires 1 GPU.  NO internet needed at runtime — the instruction list
# must already exist (generate it once on a connected node via
#   python examples/scripts/generate_libero_instructions.py \
#       --task-suite libero_goal --task-id 1 --num-instructions 20 \
#       --output examples/scripts/libero_goal_1_instructions.json
# ).
#
# Output: /scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_goal_1_vlm/
#   annotation/train/<eid>.json  ← `language_instruction` = sampled instruction
#   annotation/val/<eid>.json
#   latent_videos/agentview/<eid>.pt + latent_videos/wrist/<eid>.pt
#   train_sample.json + val_sample.json
#
# Run from repo root: bash examples/scripts/run_collect_libero_vlm.sh

set -e

INSTRUCTION_LIST="${INSTRUCTION_LIST:-examples/scripts/libero_goal_1_instructions.json}"
if [ ! -f "$INSTRUCTION_LIST" ]; then
    echo "[run_collect_libero_vlm] ERROR: instruction list not found at" >&2
    echo "  $INSTRUCTION_LIST" >&2
    echo "Generate it first on an internet-connected node:" >&2
    echo "  python examples/scripts/generate_libero_instructions.py \\" >&2
    echo "      --task-suite libero_goal --task-id 1 --num-instructions 20 \\" >&2
    echo "      --output $INSTRUCTION_LIST" >&2
    exit 1
fi

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
    --save-dir /scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_goal_1_vlm \
    --task-suite libero_goal \
    --task-id 1 \
    --num-trajs 200 \
    --val-fraction 0.1 \
    --policy pi05 \
    --cam-resolution 256 \
    --max-timesteps 400 \
    --settle-steps 10 \
    --fps 20 \
    --seed 0 \
    --instruction-list "$INSTRUCTION_LIST" \
    --save-mp4
