#!/usr/bin/env python
"""One-off: pre-generate diverse libero task instructions from a VLM.

Designed to run on a node with internet access. Renders the initial
agentview + wrist images for a given (task_suite, task_id), sends them to
``--vlm-model`` with a request for N diverse single-step manipulation
instructions for that scene, and writes the parsed list to ``--output``
as JSON.

The resulting JSON is consumed by ``collect_wm_ft_data.py
--instruction-list <path>`` on an offline GPU node (uniform random sampling
per rollout — no live API call).

Usage::

    export OPENAI_API_KEY=sk-...
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
    source .venv/bin/activate
    python examples/scripts/generate_libero_instructions.py \\
        --task-suite libero_goal --task-id 1 \\
        --num-instructions 20 \\
        --output examples/scripts/libero_goal_1_instructions.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import re
import sys

import numpy as np
from PIL import Image

# Ensure repo root on sys.path before any examples.* imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# Ground-truth object list for libero_goal task 1 (put_the_bowl_on_the_stove).
# The renderer's textures can confuse a VLM (e.g. "cream_cheese" in the BDDL
# is visually butter), so we don't let it identify objects — we tell it.
# Cabinet drawers are split into separate entries so the VLM is forced to
# reference them explicitly (instead of saying "the cabinet").
_DEFAULT_OBJECTS = [
    "top drawer of the cabinet",
    "bottom drawer of the cabinet",
    "top of the cabinet",
    "black bowl",
    "plate",
    "rack",
    "wine bottle",
    "stove",
    "butter",
]

_PROMPT_TEMPLATE = (
    "You are a manipulation robot in a Libero tabletop kitchen scene. The "
    "two attached images are the overhead agentview camera and the wrist "
    "camera at the scene's initial state.\n\n"
    "The scene contains EXACTLY these objects and NOTHING else:\n"
    "{OBJECTS_BULLETED}\n"
    "Do not invent or reference any object that is not in this list "
    "(e.g. no spoon, no pot, no mug, no cream cheese, no apple, no knife, "
    "etc.). Use the names from the list verbatim. In particular, when "
    "referring to the cabinet's drawers ALWAYS say 'top drawer of the "
    "cabinet' or 'bottom drawer of the cabinet' — never just 'the cabinet' "
    "or 'the drawer'. Include at least one open and one close instruction "
    "for each drawer.\n\n"
    "Output {N} DIVERSE single-step manipulation instructions the robot "
    "could attempt in this exact scene. Each instruction MUST:\n"
    "  - be one short imperative sentence (under 12 words),\n"
    "  - reference ONLY objects from the list above,\n"
    "  - be physically plausible (pick / place / push / move / open / close "
    "/ turn on / put inside / put on top of / put next to / put in front of "
    "/ etc.),\n"
    "  - span a wide range of object-action combinations (avoid repeating "
    "the same object pair across many lines).\n\n"
    "Output ONLY the numbered list, one instruction per line, in the format:\n"
    "1. <instruction>\n"
    "2. <instruction>\n"
    "...\n"
    "{N}. <instruction>\n"
    "No prose, no quotation marks, no extra commentary."
)


def _build_prompt(objects: list, n: int) -> str:
    bulleted = "\n".join(f"  - {o}" for o in objects)
    return _PROMPT_TEMPLATE.format(OBJECTS_BULLETED=bulleted, N=n)


def _img_to_b64_jpeg(image_uint8: np.ndarray) -> str:
    if image_uint8.dtype != np.uint8:
        image_uint8 = image_uint8.astype(np.uint8)
    pil = Image.fromarray(image_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _settle(env, n_steps: int):
    if n_steps <= 0:
        return None
    zero = np.zeros(7, dtype=np.float32)
    obs = None
    for _ in range(n_steps):
        obs, _, _, _ = env.step(zero)
    return obs


def _parse_numbered_list(raw: str) -> list:
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*\d+\s*[\).:\-]\s*(.+?)\s*$", line)
        if m:
            cleaned = m.group(1).strip().strip('"').strip("'").rstrip(".")
            if cleaned:
                out.append(cleaned)
    # Dedup preserving order (case-insensitive).
    seen = set()
    dedup = []
    for s in out:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(s)
    return dedup


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--num-instructions", type=int, default=20)
    parser.add_argument("--output", required=True,
                        help="Path to write the JSON instruction list.")
    parser.add_argument("--vlm-model", default="gpt-5-mini")
    parser.add_argument("--cam-resolution", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-images", default=None,
                        help="Optional dir to dump the rendered "
                             "agentview/wrist JPEGs (for inspection).")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="If the VLM returns fewer than "
                             "--num-instructions parseable lines, re-call up "
                             "to this many extra times and merge.")
    parser.add_argument("--objects", default=None,
                        help="Comma-separated closed-set of allowed objects "
                             "for instructions. Overrides the built-in "
                             "libero_goal_1 default. Example: "
                             "'cabinet,black bowl,plate,wine bottle,stove'.")
    args = parser.parse_args()

    if args.objects is None:
        objects = list(_DEFAULT_OBJECTS)
    else:
        objects = [s.strip() for s in args.objects.split(",") if s.strip()]
    if not objects:
        raise ValueError("--objects parsed to an empty list.")
    print(f"[generate] allowed objects ({len(objects)}): {objects}")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY must be set in the environment to call the VLM.")

    benchmark_dict = benchmark.get_benchmark_dict()
    task = benchmark_dict[args.task_suite]().get_task(args.task_id)
    bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    print(f"[generate] task: {args.task_suite}/{args.task_id} "
          f"({task.language})")
    print(f"[generate] bddl: {bddl}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.cam_resolution,
        camera_widths=args.cam_resolution,
    )
    env.seed(int(args.seed))
    obs = env.reset()
    obs = _settle(env, args.settle_steps) or obs

    # H-flip — matches what collect_wm_ft_data.py saves to disk.
    agent = np.ascontiguousarray(obs["agentview_image"][::-1]).copy()
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]).copy()
    env.close()

    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
        Image.fromarray(agent).save(
            os.path.join(args.save_images, "agentview.jpg"))
        Image.fromarray(wrist).save(
            os.path.join(args.save_images, "wrist.jpg"))
        print(f"[generate] wrote initial images to {args.save_images}")

    from openai import OpenAI

    client = OpenAI()
    prompt_text = _build_prompt(objects, args.num_instructions)
    b64_agent = _img_to_b64_jpeg(agent)
    b64_wrist = _img_to_b64_jpeg(wrist)

    instructions: list = []
    for attempt in range(args.max_retries + 1):
        print(f"[generate] calling {args.vlm_model} "
              f"(attempt {attempt + 1}/{args.max_retries + 1})...")
        resp = client.responses.create(
            model=args.vlm_model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image",
                     "image_url": f"data:image/jpeg;base64,{b64_agent}"},
                    {"type": "input_image",
                     "image_url": f"data:image/jpeg;base64,{b64_wrist}"},
                ],
            }],
        )
        parsed = _parse_numbered_list(resp.output_text)
        seen = {s.lower() for s in instructions}
        for s in parsed:
            if s.lower() not in seen:
                instructions.append(s)
                seen.add(s.lower())
        if len(instructions) >= args.num_instructions:
            break

    instructions = instructions[: args.num_instructions]
    print(f"[generate] got {len(instructions)} instructions:")
    for i, s in enumerate(instructions):
        print(f"  {i + 1:2d}. {s}")

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "task_suite": args.task_suite,
        "task_id": int(args.task_id),
        "bddl": task.bddl_file,
        "base_language": task.language,
        "vlm_model": args.vlm_model,
        "objects": objects,
        "num_instructions": len(instructions),
        "instructions": instructions,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[generate] wrote {out_path}")


if __name__ == "__main__":
    main()
