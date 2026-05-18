#!/usr/bin/env python
"""Render the initial agentview frame for every task in a LIBERO suite,
annotated with the default task instruction.

For each task_id in the given --task-suite this:
  - constructs an OffScreenRenderEnv from the task's BDDL,
  - resets, settles a few zero-action steps so the scene stops jittering,
  - grabs the agentview frame (and optionally the wrist frame),
  - burns the task language and "task_id=<i>" header onto the image,
  - writes one PNG per task and, optionally, a grid montage of all of them.

Usage::

    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
    source .venv/bin/activate
    python examples/scripts/preview_libero_tasks.py \\
        --task-suite libero_goal \\
        --output-dir examples/scripts/libero_goal_previews
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure repo root on sys.path before any examples.* imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def _settle(env, n_steps: int):
    if n_steps <= 0:
        return None
    zero = np.zeros(7, dtype=np.float32)
    obs = None
    for _ in range(n_steps):
        obs, _, _, _ = env.step(zero)
    return obs


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.ImageFont, max_width: int,
          draw: ImageDraw.ImageDraw) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _annotate(image_uint8: np.ndarray, header: str, instruction: str) -> Image.Image:
    base = Image.fromarray(image_uint8).convert("RGB")
    W, _ = base.size

    pad = max(6, W // 32)
    header_font = _load_font(max(12, W // 18))
    body_font = _load_font(max(12, W // 22))

    tmp_draw = ImageDraw.Draw(base)
    max_text_w = W - 2 * pad
    header_lines = _wrap(header, header_font, max_text_w, tmp_draw)
    body_lines = _wrap(instruction, body_font, max_text_w, tmp_draw)

    def _line_h(font):
        bbox = font.getbbox("Ag")
        return bbox[3] - bbox[1]

    header_h = _line_h(header_font)
    body_h = _line_h(body_font)
    banner_h = pad + len(header_lines) * (header_h + 2) + 4 + len(body_lines) * (body_h + 2) + pad

    canvas = Image.new("RGB", (W, base.height + banner_h), color=(0, 0, 0))
    canvas.paste(base, (0, banner_h))
    draw = ImageDraw.Draw(canvas)

    y = pad
    for line in header_lines:
        draw.text((pad, y), line, fill=(255, 220, 0), font=header_font)
        y += header_h + 2
    y += 4
    for line in body_lines:
        draw.text((pad, y), line, fill=(255, 255, 255), font=body_font)
        y += body_h + 2

    return canvas


def _make_grid(images: List[Image.Image], cols: int) -> Image.Image:
    if not images:
        raise ValueError("no images to grid")
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(0, 0, 0))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * cell_w, r * cell_h))
    return grid


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-suite", default="libero_goal",
                        help="One of: libero_spatial, libero_object, "
                             "libero_goal, libero_10, libero_90, libero_100.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write the annotated PNGs into.")
    parser.add_argument("--cam-resolution", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-ids", default=None,
                        help="Optional comma-separated subset of task_ids "
                             "(e.g. '0,3,5'). Default: every task in the suite.")
    parser.add_argument("--include-wrist", action="store_true",
                        help="Also save the wrist-camera frame next to the "
                             "agentview frame.")
    parser.add_argument("--grid", action="store_true",
                        help="Also save a grid montage of all rendered frames.")
    parser.add_argument("--grid-cols", type=int, default=5)
    args = parser.parse_args()

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_dict:
        raise SystemExit(
            f"unknown task suite '{args.task_suite}'. "
            f"available: {sorted(benchmark_dict.keys())}")
    suite = benchmark_dict[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    print(f"[preview] suite '{args.task_suite}' has {n_tasks} tasks")

    if args.task_ids:
        task_ids = [int(s) for s in args.task_ids.split(",") if s.strip()]
        for tid in task_ids:
            if tid < 0 or tid >= n_tasks:
                raise SystemExit(f"task_id {tid} out of range [0, {n_tasks})")
    else:
        task_ids = list(range(n_tasks))

    os.makedirs(args.output_dir, exist_ok=True)
    bddl_root = pathlib.Path(get_libero_path("bddl_files"))

    annotated_imgs: List[Image.Image] = []
    summary_rows: List[str] = ["task_id\tlanguage\tbddl_file"]

    for tid in task_ids:
        task = suite.get_task(tid)
        bddl = bddl_root / task.problem_folder / task.bddl_file
        print(f"[preview] task_id={tid:2d}  language={task.language!r}")
        print(f"[preview]           bddl={bddl}")

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=args.cam_resolution,
            camera_widths=args.cam_resolution,
        )
        try:
            env.seed(int(args.seed))
            obs = env.reset()
            obs = _settle(env, args.settle_steps) or obs

            agent = np.ascontiguousarray(obs["agentview_image"][::-1]).copy()
            if args.include_wrist:
                wrist = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1]).copy()
                combined = np.concatenate([agent, wrist], axis=1)
            else:
                combined = agent
        finally:
            env.close()

        header = f"{args.task_suite}  task_id={tid}"
        annotated = _annotate(combined, header=header, instruction=task.language)

        safe_lang = task.language.replace(" ", "_").replace("/", "-")[:80]
        out_name = f"task_{tid:02d}_{safe_lang}.png"
        out_path = os.path.join(args.output_dir, out_name)
        annotated.save(out_path)
        annotated_imgs.append(annotated)
        summary_rows.append(f"{tid}\t{task.language}\t{task.bddl_file}")

    summary_path = os.path.join(args.output_dir, "tasks.tsv")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_rows) + "\n")
    print(f"[preview] wrote summary -> {summary_path}")

    if args.grid and annotated_imgs:
        grid = _make_grid(annotated_imgs, cols=max(1, args.grid_cols))
        grid_path = os.path.join(args.output_dir, f"{args.task_suite}_grid.png")
        grid.save(grid_path)
        print(f"[preview] wrote grid -> {grid_path}")

    print(f"[preview] done. {len(annotated_imgs)} image(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
