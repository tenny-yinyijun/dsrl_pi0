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


# ---------------------------------------------------------------------------
# World-model novelty score (file-based IPC with the reward server)
# ---------------------------------------------------------------------------

import json
import time
import os.path

DSRL_REWARD_ROOT = os.environ.get("DSRL_REWARD_ROOT", "")
DSRL_REWARD_TIMEOUT_S = float(os.environ.get("DSRL_REWARD_TIMEOUT_S", "600"))
DSRL_REWARD_POLL_S = float(os.environ.get("DSRL_REWARD_POLL_S", "0.5"))


def wm_score(traj) -> float:
    """Score one trajectory by how surprised the libero world model is.

    Higher = more novel = better (matches the reward intuition: novel actions
    generate training data the WM hasn't seen, so the policy gets credit for
    producing them).

    **Requires** the trajectory to have already been saved to disk by the
    continuous-collection loop (``data_collection_loop`` in
    ``examples/train_utils_collect.py``). The collector injects three keys
    into the traj dict before calling this function::

        traj['_save_dir']    str — absolute path to libero_processed dir
        traj['_save_split']  str — "train" or "val"
        traj['_eid']         str — zero-padded 6-digit episode id

    Communication with the reward server is via files under
    ``$DSRL_REWARD_ROOT`` (must equal ``traj['_save_dir']`` or be a sibling
    pointing at the same data, e.g. via symlink)::

        <reward_root>/online/requests/<eid>.req       <- we drop this
        <reward_root>/scores/<eid>.score.json         <- we wait for this

    The reward server is the long-running open-world process; it loads the
    WM once at startup and watches the requests dir.

    Env knobs:
        DSRL_REWARD_ROOT       (required) where the server is watching
        DSRL_REWARD_TIMEOUT_S  max wait for a single score (default 600)
        DSRL_REWARD_POLL_S     poll interval (default 0.5)
    """
    eid = traj.get("_eid")
    save_dir = traj.get("_save_dir")
    save_split = traj.get("_save_split", "train")
    if not eid or not save_dir:
        raise RuntimeError(
            "wm_score: traj is missing '_eid' / '_save_dir'. The dsrl_pi0 "
            "collector must run via examples.data_collection_sim (which "
            "saves trajectories to disk and injects these keys). The plain "
            "examples.train_sim path does not save trajectories."
        )

    # The reward server watches <reward_root>; by default it is the same as
    # the collector's save_dir. Override via DSRL_REWARD_ROOT if you want the
    # daemon to read from a different (e.g. symlinked) tree.
    reward_root = DSRL_REWARD_ROOT or save_dir
    requests_dir = os.path.join(reward_root, "requests")
    scores_dir = os.path.join(reward_root, "scores")
    os.makedirs(requests_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    ann_path = os.path.join(save_dir, "annotation", save_split, f"{eid}.json")
    if not os.path.exists(ann_path):
        raise RuntimeError(f"wm_score: annotation not found at {ann_path}")

    score_path = os.path.join(scores_dir, f"{eid}.score.json")
    err_path = os.path.join(scores_dir, f"{eid}.error.json")
    req_path = os.path.join(requests_dir, f"{eid}.req")

    # Drop the request (atomic via tmp-rename in case the server is mid-poll).
    tmp = req_path + ".tmp"
    with open(tmp, "w") as f:
        f.write("")  # empty marker file
    os.replace(tmp, req_path)

    # Poll for the response.
    deadline = time.time() + DSRL_REWARD_TIMEOUT_S
    while time.time() < deadline:
        if os.path.exists(score_path):
            with open(score_path) as f:
                payload = json.load(f)
            return float(payload["score"])
        if os.path.exists(err_path):
            with open(err_path) as f:
                payload = json.load(f)
            raise RuntimeError(
                f"wm_score: reward server failed for eid={eid}: "
                f"{payload.get('error', 'unknown error')[:500]}"
            )
        time.sleep(DSRL_REWARD_POLL_S)

    raise TimeoutError(
        f"wm_score: timed out after {DSRL_REWARD_TIMEOUT_S:.0f}s waiting for "
        f"{score_path}. Is the reward server running and watching {reward_root!r}?"
    )
