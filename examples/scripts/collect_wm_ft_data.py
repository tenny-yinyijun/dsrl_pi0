#!/usr/bin/env python
"""Collect pi0 policy rollouts in the libero_processed layout used by the
world-model trainer (SVD-VAE-encoded `.pt` latents, no raw mp4).

Output layout under ``<save-dir>/`` (byte-compatible with
``open-world/data/wm_training/libero_processed/<suite>/``)::

    annotation/<split>/<eid>.json
    latent_videos/agentview/<eid>.pt
    latent_videos/wrist/<eid>.pt
    <split>_sample.json

Annotation schema mirrors ``scripts/preprocess_libero_for_wm.py`` in the
open-world repo: ``texts``, ``language_instruction``, ``task_suite``,
``bddl``, ``fps``, ``down_sample``, ``observation.state.cartesian_position``,
``observation.state.gripper_position``, ``latent_videos``.

Run from the dsrl_pi0 venv with libero env vars set::

    export DISPLAY=:0
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID=0
    source .venv/bin/activate
    python examples/scripts/collect_wm_ft_data.py \\
        --save-dir /scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_goal_1_ft \\
        --task-suite libero_goal --task-id 1 --num-trajs 200 --val-fraction 0.1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import sys

# Match data_collection_sim.py: enable Triton GEMM before importing jax.
_xla_flags = os.environ.get("XLA_FLAGS", "")
_xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags

# Ensure repo root is on sys.path before any `from examples.*` imports
# (the venv ships a stub `examples` namespace that shadows ours otherwise).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

from examples.train_utils_sim import _quat2axisangle, obs_to_pi_zero_input


# pi0 / pi05 noise input shapes (action_horizon, action_dim=32).
_NOISE_SHAPE = {"pi0": (50, 32), "pi05": (10, 32)}


# --------------------------------------------------------------------------- #
# SVD VAE latent encoder (mirrors open-world preprocess_libero_for_wm.py)
# --------------------------------------------------------------------------- #
class LatentEncoder:
    """Wraps the SVD AutoencoderKLTemporalDecoder VAE.

    ``encode(frames_uint8)`` takes (T, H, W, 3) uint8 frames, resizes to
    (target_h, target_w), and returns a (T, 4, target_h//8, target_w//8)
    float16 cpu tensor (matches the libero_processed `.pt` layout).
    """

    def __init__(self, svd_path: str, device: str = "cuda",
                 target_h: int = 320, target_w: int = 320, chunk: int = 8):
        from diffusers import AutoencoderKLTemporalDecoder

        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.chunk = int(chunk)
        self.device = device
        print(f"[encoder] loading SVD VAE from {svd_path}")
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_path, subfolder="vae", torch_dtype=torch.float16
        ).to(device)
        self.vae.eval()
        self.scale = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, frames_uint8: np.ndarray) -> torch.Tensor:
        T = int(frames_uint8.shape[0])
        out = torch.empty(
            (T, 4, self.target_h // 8, self.target_w // 8), dtype=torch.float16
        )
        for start in range(0, T, self.chunk):
            end = min(T, start + self.chunk)
            tile = []
            for i in range(start, end):
                img = Image.fromarray(frames_uint8[i])
                img = img.resize((self.target_w, self.target_h), Image.BICUBIC)
                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                tile.append(arr)
            tensor = torch.tensor(
                np.stack(tile), dtype=torch.float16, device=self.device
            ).permute(0, 3, 1, 2)
            latent = self.vae.encode(tensor).latent_dist.mean * self.scale
            out[start:end] = latent.cpu()
        return out


# --------------------------------------------------------------------------- #
# VLM instruction generator (gpt-5-mini etc.)
# --------------------------------------------------------------------------- #
_DEFAULT_VLM_USER_TEXT = (
    "You are a manipulation robot in a Libero tabletop scene. Look at the two "
    "images (overhead agentview + wrist camera) and output ONE short, "
    "concrete instruction describing a plausible single-step manipulation "
    "task for the robot to perform in this scene. Use the visible objects, "
    "with colors when helpful. Keep it under 12 words. Examples: "
    "'put the black bowl on the stove', 'pick up the red mug and place it on "
    "the plate', 'move the bowl to the right of the stove'. Do not include "
    "quotation marks. Output only the instruction, nothing else."
)


def _img_to_b64_jpeg(image_uint8: np.ndarray) -> str:
    if image_uint8.dtype != np.uint8:
        image_uint8 = image_uint8.astype(np.uint8)
    pil = Image.fromarray(image_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class ListInstructor:
    """Samples a per-rollout instruction uniformly from a pre-generated list.

    Used on offline GPU nodes (no internet) — the list is produced once on a
    connected node via ``examples/scripts/generate_libero_instructions.py``.
    """

    def __init__(self, list_path: str, rng_seed: int = 0):
        with open(list_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "instructions" in data:
            self.instructions = list(data["instructions"])
        elif isinstance(data, list):
            self.instructions = list(data)
        else:
            raise ValueError(
                f"{list_path}: expected a JSON list or an object with key "
                f"'instructions'.")
        if not self.instructions:
            raise ValueError(f"{list_path} contains no instructions.")
        self._rng = np.random.default_rng(int(rng_seed))
        print(f"[instr-list] loaded {len(self.instructions)} instructions "
              f"from {list_path}")

    # Mirrors VLMInstructor.generate signature; images are ignored.
    def generate(self, agentview: np.ndarray, wrist: np.ndarray) -> str:
        return str(self._rng.choice(self.instructions))


class VLMInstructor:
    """Generates a per-rollout language instruction from agentview+wrist."""

    def __init__(self, model: str, user_text: str):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY env var is required when --vlm-instructions "
                "is set."
            )
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.user_text = user_text

    def generate(self, agentview: np.ndarray, wrist: np.ndarray) -> str:
        b64_agent = _img_to_b64_jpeg(agentview)
        b64_wrist = _img_to_b64_jpeg(wrist)
        resp = self.client.responses.create(
            model=self.model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": self.user_text},
                    {"type": "input_image",
                     "image_url": f"data:image/jpeg;base64,{b64_agent}"},
                    {"type": "input_image",
                     "image_url": f"data:image/jpeg;base64,{b64_wrist}"},
                ],
            }],
        )
        return resp.output_text.strip().strip('"').strip("'")


# --------------------------------------------------------------------------- #
# libero env + rollout
# --------------------------------------------------------------------------- #
def _get_libero_env(task, resolution: int, seed: int):
    bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language, pathlib.Path(task.bddl_file).stem


def _settle(env, n_steps: int):
    if n_steps <= 0:
        return None
    zero_action = np.zeros(7, dtype=np.float32)
    obs = None
    for _ in range(n_steps):
        obs, _, _, _ = env.step(zero_action)
    return obs


def _record_obs(obs, raw_agentview, wrist_images, raw_state_list):
    # H-flip only — matches the open-world preprocess convention the WM
    # was trained on. (The 180° rotation pi0 expects is applied at the
    # policy boundary in obs_to_pi_zero_input, not in saved frames.)
    raw_agentview.append(
        np.ascontiguousarray(obs["agentview_image"][::-1]).copy()
    )
    wrist_images.append(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]).copy()
    )
    cart = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]))
    ).astype(np.float32)
    grip = float(np.mean(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)))
    raw_state_list.append({"cartesian_position": cart, "gripper_position": grip})


def collect_one(
    *,
    env,
    agent_dp,
    task_description: str,
    bddl_name: str,
    task_suite: str,
    task_id: int,
    query_freq: int,
    max_timesteps: int,
    settle_steps: int,
    rng,
    noise_shape,
    cam_resolution: int,
    vlm_instructor=None,
):
    """One plain pi0 rollout (fresh Gaussian noise per query step, sigma=1).

    If ``vlm_instructor`` is provided, the task description is replaced per
    rollout with the VLM's output (conditioned on the initial agentview +
    wrist images after env reset/settle).
    """
    obs = env.reset()
    obs = _settle(env, settle_steps) or obs

    if vlm_instructor is not None:
        # Same H-flip convention as _record_obs (matches what is saved to disk).
        agent_init = np.ascontiguousarray(obs["agentview_image"][::-1]).copy()
        wrist_init = np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1]
        ).copy()
        task_description = vlm_instructor.generate(agent_init, wrist_init)
        print(f"[vlm] generated instruction: {task_description!r}")

    raw_agentview, wrist_images, raw_state_list = [], [], []
    rewards = []
    actions = None
    reward = 0.0
    done = False

    class _V:
        env = "libero"
        cam_resolution = 0
    _v = _V()
    _v.cam_resolution = cam_resolution
    _v.task_description = task_description

    for t in tqdm(range(max_timesteps), leave=False):
        if t % query_freq == 0:
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (1, *noise_shape))
            obs_pi_zero = obs_to_pi_zero_input(obs, _v)
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

        _record_obs(obs, raw_agentview, wrist_images, raw_state_list)
        action_t = np.asarray(actions[t % query_freq])
        obs, reward, done, _ = env.step(action_t)
        rewards.append(reward)
        if done:
            break

    # Trailing observation so frame count matches state count.
    _record_obs(obs, raw_agentview, wrist_images, raw_state_list)

    return {
        "raw_agentview": raw_agentview,
        "wrist_images": wrist_images,
        "state_list": raw_state_list,
        "is_success": bool(reward == 1),
        "env_steps": int(len(rewards)),
        "task_description": task_description,
        "task_suite": task_suite,
        "task_id": int(task_id),
        "bddl": bddl_name,
    }, rng


# --------------------------------------------------------------------------- #
# Save in libero_processed (latent-only) format
# --------------------------------------------------------------------------- #
def _write_mp4(path: str, frames_uint8: np.ndarray, fps: int):
    """Write a (T, H, W, 3) uint8 array to ``path`` as H.264 mp4."""
    import imageio

    imageio.mimwrite(
        path, list(frames_uint8), fps=int(fps),
        codec="libx264", quality=8, macro_block_size=None,
    )


def save_traj_wm_format(
    traj: dict,
    save_dir: str,
    episode_id: int,
    split: str,
    fps: int,
    encoder: LatentEncoder,
    save_mp4: bool = False,
):
    eid = f"{int(episode_id):06d}"
    ann_dir = os.path.join(save_dir, "annotation", split)
    agent_dir = os.path.join(save_dir, "latent_videos", "agentview")
    wrist_dir = os.path.join(save_dir, "latent_videos", "wrist")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(wrist_dir, exist_ok=True)

    agentview_arr = np.stack(traj["raw_agentview"], axis=0)  # (T, H, W, 3) uint8
    wrist_arr = np.stack(traj["wrist_images"], axis=0)
    agent_lat = encoder.encode(agentview_arr)
    wrist_lat = encoder.encode(wrist_arr)
    torch.save(agent_lat, os.path.join(agent_dir, f"{eid}.pt"))
    torch.save(wrist_lat, os.path.join(wrist_dir, f"{eid}.pt"))

    if save_mp4:
        videos_agent_dir = os.path.join(save_dir, "videos", "agentview")
        videos_wrist_dir = os.path.join(save_dir, "videos", "wrist")
        os.makedirs(videos_agent_dir, exist_ok=True)
        os.makedirs(videos_wrist_dir, exist_ok=True)
        _write_mp4(os.path.join(videos_agent_dir, f"{eid}.mp4"),
                   agentview_arr, fps)
        _write_mp4(os.path.join(videos_wrist_dir, f"{eid}.mp4"),
                   wrist_arr, fps)

    states = traj["state_list"]
    cart = [list(map(float, s["cartesian_position"])) for s in states]
    grip = [float(s["gripper_position"]) for s in states]

    annotation = {
        "texts": [traj.get("task_description", "")],
        "language_instruction": traj.get("task_description", ""),
        "task_suite": traj.get("task_suite", ""),
        "bddl": traj.get("bddl", ""),
        "fps": int(fps),
        "down_sample": 1,
        "observation.state.cartesian_position": cart,
        "observation.state.gripper_position": grip,
        "latent_videos": [
            {"latent_video_path": f"latent_videos/agentview/{eid}.pt",
             "cam": "agentview"},
            {"latent_video_path": f"latent_videos/wrist/{eid}.pt",
             "cam": "wrist"},
        ],
    }
    with open(os.path.join(ann_dir, f"{eid}.json"), "w") as f:
        json.dump(annotation, f)


def write_sample_list(
    save_dir: str,
    split: str,
    episode_ids: list,
    num_history: int = 6,
    num_frames: int = 5,
    down_sample: int = 4,
):
    """Rebuild ``<split>_sample.json`` from the annotations on disk.

    Mirrors open-world's preprocess_libero_for_wm.write_sample_list: frame
    indices are in *downsampled* units (raw_fps / down_sample), and each
    sample is a single starting frame consumed by the WM dataset loader.
    """
    samples = []
    for eid in episode_ids:
        ann_path = os.path.join(save_dir, "annotation", split, f"{eid}.json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            ann = json.load(f)
        T = len(ann["observation.state.cartesian_position"])
        max_start = max(1, (T // down_sample) - num_frames - 1)
        for start in range(num_history, max_start, max(1, num_frames // 2)):
            samples.append({"episode_id": eid, "frame_ids": [start]})
    out = os.path.join(save_dir, f"{split}_sample.json")
    with open(out, "w") as f:
        json.dump(samples, f)
    print(f"[sample] wrote {len(samples)} entries to {out}")


def _existing_eids(save_dir: str, split: str) -> list:
    ann_dir = os.path.join(save_dir, "annotation", split)
    if not os.path.isdir(ann_dir):
        return []
    eids = []
    for p in os.listdir(ann_dir):
        if p.endswith(".json") and p[:-5].isdigit():
            eids.append(p[:-5])
    return sorted(eids)


def _next_episode_id(save_dir: str) -> int:
    eids = _existing_eids(save_dir, "train") + _existing_eids(save_dir, "val")
    if not eids:
        return 0
    return max(int(e) for e in eids) + 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--save-dir", required=True,
                        help="Output directory (final libero_processed-format "
                             "data lives directly inside this path).")
    parser.add_argument("--num-trajs", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--policy", default="pi0", choices=["pi0", "pi05"])
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--cam-resolution", type=int, default=256)
    parser.add_argument("--max-timesteps", type=int, default=400)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--query-freq", type=int, default=-1,
                        help="Default = chunk length for the policy "
                             "(50 for pi0, 10 for pi05).")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--svd-path", type=str,
                        default="/scratch/gpfs/AM43/yy4041/open-world/external/"
                                "stable-video-diffusion-img2vid",
                        help="Path to the SVD model directory (with "
                             "subfolder=vae). Defaults to the open-world repo's "
                             "checkpoint on della.")
    parser.add_argument("--encoder-device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vlm-instructions", action="store_true",
                        help="If set, generate a per-rollout task instruction "
                             "live from a VLM (OpenAI) instead of using the "
                             "BDDL task language. Requires OPENAI_API_KEY and "
                             "internet access at runtime. Mutually exclusive "
                             "with --instruction-list.")
    parser.add_argument("--vlm-model", default="gpt-5-mini",
                        help="OpenAI model id used to generate instructions.")
    parser.add_argument("--vlm-prompt-text", default=_DEFAULT_VLM_USER_TEXT,
                        help="System/user prompt text fed to the VLM along "
                             "with the agentview+wrist images.")
    parser.add_argument("--instruction-list", default=None,
                        help="Path to a JSON file (list of strings, or "
                             "{'instructions': [...]}) pre-generated by "
                             "generate_libero_instructions.py. When set, "
                             "each rollout's task description is sampled "
                             "uniformly from this list — no live VLM call.")
    parser.add_argument("--save-mp4", action="store_true",
                        help="Also write raw H.264 mp4s alongside the SVD "
                             "latents, under <save-dir>/videos/{agentview,"
                             "wrist}/<eid>.mp4. Useful for debug viewing.")
    args = parser.parse_args()

    noise_shape = _NOISE_SHAPE[args.policy]
    chunk_len = noise_shape[0]
    query_freq = args.query_freq if args.query_freq > 0 else chunk_len
    if query_freq > chunk_len:
        raise ValueError(
            f"query_freq={query_freq} exceeds policy chunk length {chunk_len}")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[collect] writing to {args.save_dir}")
    print(f"[collect] policy={args.policy}  "
          f"task={args.task_suite}/{args.task_id}  "
          f"num_trajs={args.num_trajs}  val_fraction={args.val_fraction}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)

    if args.policy == "pi05":
        config = openpi_config.get_config("pi05_libero")
        ckpt_dir = download.maybe_download(
            "gs://openpi-assets/checkpoints/pi05_libero")
    else:
        config = openpi_config.get_config("pi0_libero")
        ckpt_dir = download.maybe_download(
            "s3://openpi-assets/checkpoints/pi0_libero")
    print(f"[collect] loading {args.policy} from {ckpt_dir}")
    agent_dp = policy_config.create_trained_policy(config, ckpt_dir)

    encoder = LatentEncoder(args.svd_path, device=args.encoder_device)

    if args.vlm_instructions and args.instruction_list:
        raise ValueError(
            "--vlm-instructions and --instruction-list are mutually exclusive.")

    vlm_instructor = None
    if args.instruction_list:
        vlm_instructor = ListInstructor(
            args.instruction_list, rng_seed=args.seed)
    elif args.vlm_instructions:
        vlm_instructor = VLMInstructor(
            model=args.vlm_model, user_text=args.vlm_prompt_text)
        print(f"[vlm] enabled — model={args.vlm_model}")

    next_eid = _next_episode_id(args.save_dir)
    print(f"[collect] starting from episode_id={next_eid:06d}")

    rng = jax.random.PRNGKey(int(args.seed))
    split_rng = np.random.default_rng(int(args.seed) + 17)

    n_done = 0
    n_success = 0
    while n_done < args.num_trajs:
        env_seed = int(args.seed) + next_eid + n_done
        env, task_description, bddl_name = _get_libero_env(
            task, args.cam_resolution, env_seed)
        try:
            split = "val" if split_rng.random() < args.val_fraction else "train"
            print(f"[collect] eid={next_eid:06d}  split={split}  "
                  f"env_seed={env_seed}")
            traj, rng = collect_one(
                env=env,
                agent_dp=agent_dp,
                task_description=task_description,
                bddl_name=bddl_name,
                task_suite=args.task_suite,
                task_id=args.task_id,
                query_freq=query_freq,
                max_timesteps=args.max_timesteps,
                settle_steps=args.settle_steps,
                rng=rng,
                noise_shape=noise_shape,
                cam_resolution=args.cam_resolution,
                vlm_instructor=vlm_instructor,
            )
        finally:
            env.close()

        save_traj_wm_format(
            traj, args.save_dir, next_eid, split=split,
            fps=int(args.fps), encoder=encoder, save_mp4=args.save_mp4,
        )
        n_success += int(traj["is_success"])
        print(f"[collect] saved eid={next_eid:06d}  steps={traj['env_steps']}  "
              f"success={traj['is_success']}  "
              f"running_success_rate={n_success / (n_done + 1):.2f}")
        next_eid += 1
        n_done += 1

    # Rebuild both sample lists from everything currently on disk
    # (so resumed runs end up with consistent indices).
    write_sample_list(args.save_dir, "train", _existing_eids(args.save_dir, "train"))
    write_sample_list(args.save_dir, "val", _existing_eids(args.save_dir, "val"))
    print(f"[collect] done. total success rate this run: "
          f"{n_success}/{args.num_trajs} = {n_success / max(1, args.num_trajs):.2f}")


if __name__ == "__main__":
    main()
