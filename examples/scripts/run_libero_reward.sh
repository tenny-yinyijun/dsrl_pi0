#!/bin/bash
proj_name=DSRL_pi0_Libero_CustomReward
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export EXP=./logs/$proj_name;
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# If you have not installed mujoco 3.3.1 in this venv yet:
# uv pip install mujoco==3.3.1

# Optional: point your scoring fn at a reference trajectory file
# export DSRL_REFERENCE_TRAJ_PATH=/path/to/reference.npz

python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi0_libero_reward \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
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
--reward_relabel_buffer 0
