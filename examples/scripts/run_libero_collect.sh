#!/bin/bash
# Continuous data-collection + reward-update + policy-update loop on Libero.
#
# Mirrors run_libero_reward.sh and adds the new collection flags:
#   --scene_reset_freq   X — reset scene every X trajectories
#   --reward_update_freq Y — update reward model + run SAC every Y trajectories
#   --save_dir             — libero_processed-style output directory
#   --save_split           — annotation split to write (default "train")

proj_name=DSRL_pi0_Libero_Collect
# Pick a GPU that is currently empty (`nvidia-smi` to check). GPU 0 was busy
# the last time we tried — pi0 materialization was killed mid-load.
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
    --algorithm pixel_sac \
    --env libero \
    --prefix dsrl_pi0_libero_collect \
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
    --query_freq 20 \
    --hidden_dims 128 \
    --use_reward_model 1 \
    --reward_fn examples.reward_fn:score \
    --traj_batch_size 8 \
    --reward_grad_steps 200 \
    --reward_lr 3e-4 \
    --reward_relabel_buffer 0 \
    --scene_reset_freq 1 \
    --reward_update_freq 8 \
    --save_dir ./collected_data/libero/test1 \
    --save_split train \
    --task_suite_name libero_90 \
    --task_id 57 \
    --cam_resolution 256 \
    --fps 20 \
    --sample_stride 2 \
    --sample_start_offset 6 \
    --max_trajs 1000000
