"""Argparse entrypoint for continuous data collection.

Mirrors ``examples.launch_train_sim`` and adds a few new flags:

  --scene_reset_freq        X — reset env.reset() every X trajectories.
  --reward_update_freq      Y — train reward + run SAC every Y trajectories.
  --save_dir                where to write libero_processed-format data.
  --save_split              annotation split name ("train"/"val"/...).
  --max_trajs               stop once this many trajectories are collected.
  --task_suite_name         libero benchmark name (default libero_90).
  --task_id                 task index within the suite.
  --cam_resolution          libero camera resolution.
  --fps                     fps written into annotation JSON.
  --sample_stride           stride for *_sample.json index entries.
  --sample_start_offset     start offset for *_sample.json index entries.
"""
import argparse
import os
import sys

# When invoked as `python examples/launch_collect.py`, sys.path[0] is the
# script's directory (examples/) — not the repo root. The venv ships a
# namespace `examples` package (numpyro stubs) that shadows ours unless the
# repo root is on sys.path. Prepend it so `from examples.*` resolves locally.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.data_collection_sim import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -------- Existing flags (kept identical to launch_train_sim.py) -------- #
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--launch_group_id', default='')
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--env', default='libero')
    parser.add_argument('--policy', default='pi0', choices=['pi0', 'pi05'],
                        help='Pi-zero variant to load as the base policy (pi0 or pi05).')
    parser.add_argument('--log_interval', default=1000, type=int)
    parser.add_argument('--eval_interval', default=5000, type=int)
    parser.add_argument('--checkpoint_interval', default=-1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_steps', default=int(1e6), type=int)
    parser.add_argument('--add_states', default=1, type=int)
    parser.add_argument('--wandb_project', default='dsrl_pi0_collect')
    parser.add_argument('--start_online_updates', default=1000, type=int)
    parser.add_argument('--algorithm', default='pixel_sac')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--multi_grad_step', default=1, type=int)
    parser.add_argument('--resize_image', default=-1, type=int)
    parser.add_argument('--query_freq', default=-1, type=int)
    parser.add_argument('--base_policy_prob', default=0.0, type=float,
                        help='Per-episode probability of using fresh gaussian '
                             'noise (i.e. base pi0 behavior) instead of the '
                             'SAC-chosen noise. 0.0 = always use SAC once it '
                             'has updates, 1.0 = always pi0. Useful to keep '
                             'some clean-data trajectories while exploring.')

    # -------- Reward-model flags (carried over) -------- #
    parser.add_argument('--use_reward_model', default=0, type=int)
    parser.add_argument('--reward_fn', default='examples.reward_fn:score')
    parser.add_argument('--traj_batch_size', default=8, type=int,
                        help='Reward-model batch size; defaults Y to this '
                             'value if --reward_update_freq is unset.')
    parser.add_argument('--reward_grad_steps', default=200, type=int)
    parser.add_argument('--reward_lr', default=3e-4, type=float)
    parser.add_argument('--reward_relabel_buffer', default=0, type=int)
    parser.add_argument('--reward_loss_mode', default='per_step',
                        choices=['traj', 'per_step'],
                        help='How to fit the reward model. "per_step" uses '
                             'per-WM-frame LPIPS as masked per-(query-step) '
                             'targets (finer credit assignment). "traj" '
                             'uses the legacy sum-of-rewards = trajectory-'
                             'target loss. Falls back to "traj" silently if '
                             'WM payload is missing.')

    # -------- New flags for continuous collection -------- #
    parser.add_argument('--scene_reset_freq', default=1, type=int,
                        help='Call env.reset() every X trajectories. '
                             'Default 1 = reset before every traj.')
    parser.add_argument('--reward_update_freq', default=-1, type=int,
                        help='Train reward + run SAC updates every Y '
                             'trajectories. -1 means use --traj_batch_size.')
    parser.add_argument('--scored_per_round', default=-1, type=int,
                        help='If >=0, exactly this many randomly-chosen '
                             'trajectories per round (= --traj_batch_size '
                             'trajs) are scored by the WM reward server; '
                             'the rest are sent to the WM finetune buffer '
                             '(server-side) without scoring. -1 (default) '
                             '= score every trajectory.')
    parser.add_argument('--save_dir', default='',
                        help='Output directory in libero_processed layout. '
                             'Empty => <EXP>/<expname>/collected_data.')
    parser.add_argument('--save_split', default='train')
    parser.add_argument('--max_trajs', default=10**9, type=int)
    parser.add_argument('--task_suite_name', default='libero_90')
    parser.add_argument('--task_id', default=57, type=int)
    parser.add_argument('--instruction_list', default='',
                        help='Optional path to a JSON file of pre-generated '
                             'task instructions (list of strings, or '
                             "{'instructions': [...]}). When set, each "
                             "rollout's prompt to π₀ is sampled uniformly "
                             'from this list instead of using the fixed BDDL '
                             'task language. Use to multi-task on instruction '
                             'variations within a single LIBERO scene. '
                             'Generate the file with '
                             'examples/scripts/generate_libero_instructions.py.')
    parser.add_argument('--cam_resolution', default=256, type=int)
    parser.add_argument('--fps', default=20, type=int)
    parser.add_argument('--sample_stride', default=2, type=int)
    parser.add_argument('--sample_start_offset', default=6, type=int)
    parser.add_argument('--sample_wm_down_sample', default=4, type=int,
                        help='Must match LiberoWMArgs.down_sample. Caps '
                             'frame_now to num_frames // wm_down_sample so '
                             'WM training samples are non-degenerate.')
    parser.add_argument('--settle_steps', default=10, type=int,
                        help='Number of zero-action env steps to take after '
                             'every libero env.reset() so objects settle to '
                             'rest before recording. Set 0 to disable.')
    parser.add_argument('--latent_encoder', default='',
                        help='Optional "module:callable" that maps a '
                             'uint8 (T, H, W, C) frame array to a torch '
                             'Tensor of shape (T, C\', H\', W\'). When set, '
                             'each saved trajectory also writes encoded '
                             'latents to latent_videos/{cam}/<eid>.pt.')

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128, 128),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding='VALID',
        latent_dim=50,
        discount=0.999,
        tau=0.005,
        critic_reduction='mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        actor_grad_clip=0.0,  # >0 enables optax.clip_by_global_norm on actor
    )

    variant, args = parse_training_args(train_args_dict, parser)

    # Resolve Y default -> traj_batch_size if user didn't set it.
    if int(variant.reward_update_freq) <= 0:
        variant.reward_update_freq = int(variant.traj_batch_size)

    print(variant)
    main(variant)
    sys.exit()
