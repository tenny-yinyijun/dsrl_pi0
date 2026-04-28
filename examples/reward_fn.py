"""User-supplied trajectory scoring function.

The training loop calls ``score(traj)`` once per rollout when launched with
``--use_reward_model --reward_fn examples.reward_fn:score``. ``traj`` is the
dict returned by ``collect_traj`` in ``examples/train_utils_sim.py``:

    traj = {
        "observations":  list[dict],   # query-step obs dicts (pixels, state)
        "actions":       list[ndarray],  # noise-action chunks per query step
        "rewards":       np.ndarray,   # raw env rewards (you can ignore)
        "is_success":    bool,
        "episode_return": float,
        "images":        list[ndarray],  # raw camera frames, uint8
        "env_steps":     int,
    }

The returned float is the *target trajectory return* used to fit the reward
model. Higher = better. The scale does NOT need to be normalized.

The example below is the design the user described: a pixel-based discrepancy
between the rollout and a fixed reference trajectory loaded once at module
import time. Lower discrepancy → higher score (we negate). Replace
``_load_reference()`` and ``_pixel_discrepancy()`` with your own.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Reference trajectory loading (done once at import time so it is shared
# across all calls to score()). Set REFERENCE_TRAJ_PATH to a .npz/.npy/.pkl
# file or override _load_reference() entirely.
# ---------------------------------------------------------------------------
REFERENCE_TRAJ_PATH = os.environ.get("DSRL_REFERENCE_TRAJ_PATH", "")


def _load_reference() -> Optional[List[np.ndarray]]:
    """Return a list of uint8 frames (H, W, C) for the reference trajectory.

    Returning None disables the reference comparison and falls back to the
    placeholder heuristic at the bottom of score().
    """
    if not REFERENCE_TRAJ_PATH:
        return None
    if REFERENCE_TRAJ_PATH.endswith(".npz"):
        with np.load(REFERENCE_TRAJ_PATH) as npz:
            frames = npz["images"] if "images" in npz else npz[npz.files[0]]
        return [f for f in frames]
    if REFERENCE_TRAJ_PATH.endswith(".npy"):
        frames = np.load(REFERENCE_TRAJ_PATH)
        return [f for f in frames]
    raise NotImplementedError(
        f"Add a loader for reference path {REFERENCE_TRAJ_PATH!r}")


_REFERENCE: Optional[List[np.ndarray]] = _load_reference()


def _pixel_discrepancy(rollout_frames: List[np.ndarray],
                       ref_frames: List[np.ndarray]) -> float:
    """Pixel-space discrepancy between two trajectories.

    Default: mean absolute pixel difference after time-uniform resampling
    of the rollout to the reference length. Replace with whatever metric
    you want (LPIPS, DINO feature distance, etc.).
    """
    T_rollout = len(rollout_frames)
    T_ref = len(ref_frames)
    if T_rollout == 0 or T_ref == 0:
        return float("inf")

    # Time-uniform resample rollout to T_ref frames (nearest-neighbor index).
    idx = np.linspace(0, T_rollout - 1, T_ref).round().astype(int)
    diffs = []
    for i, j in enumerate(idx):
        a = rollout_frames[j].astype(np.float32)
        b = ref_frames[i].astype(np.float32)
        if a.shape != b.shape:
            # Tolerate (H, W, C) vs (H, W, C) shape mismatch by center-crop.
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = a[:h, :w]
            b = b[:h, :w]
        diffs.append(np.mean(np.abs(a - b)))
    return float(np.mean(diffs))


def score(traj) -> float:
    """Score one trajectory. Higher = better."""
    if _REFERENCE is not None:
        d = _pixel_discrepancy(traj["images"], _REFERENCE)
        # Negate so that "closer to the reference" → larger score.
        return -d

    # Fallback placeholder so the pipeline is runnable end-to-end without a
    # reference. Replace before any real run.
    return float(traj.get("episode_return", 0.0))
