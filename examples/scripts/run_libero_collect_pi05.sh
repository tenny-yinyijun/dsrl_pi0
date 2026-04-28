#!/bin/bash
# Continuous data-collection + reward-update + policy-update loop on Libero,
# using the pi05 base policy (instead of pi0).
#
# Differences from run_libero_collect.sh:
#   --policy pi05  — selects pi05_libero config + gs://openpi-assets/checkpoints/pi05_libero
#   prefix / proj_name updated to reflect pi05
# Note: pi05_libero uses action_horizon=10 (vs pi0_libero's 50). The hardcoded
# noise-padding in train_utils_sim.py was switched to 10 to match.

proj_name=DSRL_pi05_Libero_Collect
# Pick a GPU that is currently empty (`nvidia-smi` to check). pi05 has
# max_token_len=200 (vs pi0's 48) so the prefix KV-cache is ~4x larger —
# may need a less-loaded card.
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name;
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The .venv ships a stub namespace `examples` package that shadows the repo's
# examples/ unless the repo root is on PYTHONPATH. Force it.
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Optional: point your scoring fn at a reference trajectory file
# export DSRL_REFERENCE_TRAJ_PATH=/path/to/reference.npz

source .venv/bin/activate


python3 examples/launch_collect.py \
    --policy pi05 \
    --algorithm pixel_sac \
    --env libero \
    --prefix dsrl_pi05_libero_collect \
    --wandb_project ${proj_name} \
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
    --query_freq 10 \
    --hidden_dims 128 \
    --use_reward_model 1 \
    --reward_fn examples.reward_fn:score \
    --traj_batch_size 8 \
    --reward_grad_steps 200 \
    --reward_lr 3e-4 \
    --reward_relabel_buffer 0 \
    --scene_reset_freq 1 \
    --reward_update_freq 8 \
    --save_dir ./collected_data/libero/test1_pi05 \
    --save_split train \
    --task_suite_name libero_90 \
    --task_id 57 \
    --cam_resolution 256 \
    --fps 20 \
    --sample_stride 2 \
    --sample_start_offset 6 \
    --max_trajs 1000000
