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


def _wm_paths(traj) -> dict:
    """Resolve the file-IPC paths for a trajectory's score request.

    Returns a dict with all the paths needed by wm_score_request /
    wm_score_await. Raises RuntimeError if the trajectory hasn't been
    saved to disk yet (no _eid / _save_dir).
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
    reward_root = DSRL_REWARD_ROOT or save_dir
    return {
        "eid": eid,
        "reward_root": reward_root,
        "requests_dir": os.path.join(reward_root, "requests"),
        "scores_dir": os.path.join(reward_root, "scores"),
        "ann_path": os.path.join(save_dir, "annotation", save_split, f"{eid}.json"),
        "score_path": os.path.join(reward_root, "scores", f"{eid}.score.json"),
        "err_path": os.path.join(reward_root, "scores", f"{eid}.error.json"),
        "req_path": os.path.join(reward_root, "requests", f"{eid}.req"),
    }


def wm_score_request(traj) -> str:
    """Drop a `.req` marker file for the trajectory; do NOT block.

    Idempotent: if a `.score.json` (or `.error.json`) is already present
    for this eid (e.g. from a resumed run), no new request is dropped.
    Returns the eid for logging.

    The companion ``wm_score_await(traj)`` blocks until the result lands
    and returns the scalar score. ``wm_score(traj)`` is the sync wrapper
    that calls request + await — preserves the existing single-call API.
    """
    p = _wm_paths(traj)
    os.makedirs(p["requests_dir"], exist_ok=True)
    os.makedirs(p["scores_dir"], exist_ok=True)
    if not os.path.exists(p["ann_path"]):
        raise RuntimeError(
            f"wm_score_request: annotation not found at {p['ann_path']}")
    # Skip if already scored (or already requested). The server treats
    # both as no-ops too, but skipping here saves the IPC round-trip.
    if os.path.exists(p["score_path"]) or os.path.exists(p["err_path"]):
        return p["eid"]
    if os.path.exists(p["req_path"]):
        return p["eid"]
    # Drop the request (atomic via tmp-rename so the server can't see a
    # half-written file mid-poll).
    tmp = p["req_path"] + ".tmp"
    with open(tmp, "w") as f:
        f.write("")  # empty marker file
    os.replace(tmp, p["req_path"])
    return p["eid"]


def wm_score_request_wm_only(traj) -> str:
    """Drop a `.wm_only` marker for the trajectory; do NOT block.

    Asks the reward server to encode the trajectory's latents and add it
    to its WM fine-tune buffer WITHOUT scoring it. No `.score.json` will
    be written, no LPIPS rollout will be performed, and the trainer must
    NOT call ``wm_score_await`` for this eid. Use this for trajectories
    that the trainer wants to feed the WM but doesn't need a reward
    target for (i.e. the unscored fraction under --score_prob < 1.0).

    Note the marker uses extension ``.wm_only`` (NOT ``.wm_only.req``) so
    it doesn't collide with the server's ``*.req`` glob.

    Idempotent in the same way as ``wm_score_request``.
    """
    p = _wm_paths(traj)
    os.makedirs(p["requests_dir"], exist_ok=True)
    if not os.path.exists(p["ann_path"]):
        raise RuntimeError(
            f"wm_score_request_wm_only: annotation not found at {p['ann_path']}")
    wm_only_req = os.path.join(p["requests_dir"], f"{p['eid']}.wm_only")
    # If it's already been scored or a normal request is already pending,
    # there's nothing to do — the WM buffer add will happen via the
    # scoring path in either case.
    if os.path.exists(p["score_path"]) or os.path.exists(p["err_path"]):
        return p["eid"]
    if os.path.exists(p["req_path"]) or os.path.exists(wm_only_req):
        return p["eid"]
    tmp = wm_only_req + ".tmp"
    with open(tmp, "w") as f:
        f.write("")
    os.replace(tmp, wm_only_req)
    return p["eid"]


def wm_score_await(traj) -> float:
    """Block until the score for ``traj`` is ready; return the scalar.

    Stashes the full payload on ``traj['_wm_payload']`` for downstream
    consumers (per-step reward target derivation in particular).
    """
    p = _wm_paths(traj)
    eid = p["eid"]
    score_path, err_path = p["score_path"], p["err_path"]
    deadline = time.time() + DSRL_REWARD_TIMEOUT_S
    while time.time() < deadline:
        if os.path.exists(score_path):
            with open(score_path) as f:
                payload = json.load(f)
            try:
                traj["_wm_payload"] = payload
            except TypeError:
                pass
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
        f"wm_score: timed out after {DSRL_REWARD_TIMEOUT_S:.0f}s waiting "
        f"for {score_path}. Is the reward server running and watching "
        f"{p['reward_root']!r}?"
    )


def wm_score(traj) -> float:
    """Score one trajectory by how surprised the libero world model is.

    Higher = more novel = better. Backward-compatible synchronous
    interface: equivalent to ``wm_score_request(traj)`` immediately
    followed by ``wm_score_await(traj)``. The collection loop auto-
    detects the ``_request`` / ``_await`` siblings and pipelines them
    against rollout when both are present.

    **Requires** the trajectory to have already been saved to disk by the
    continuous-collection loop (``data_collection_loop`` in
    ``examples/train_utils_collect.py``). The collector injects three keys
    into the traj dict before calling this function::

        traj['_save_dir']    str — absolute path to libero_processed dir
        traj['_save_split']  str — "train" or "val"
        traj['_eid']         str — zero-padded 6-digit episode id

    File-IPC layout under ``$DSRL_REWARD_ROOT`` (defaults to ``_save_dir``)::

        <reward_root>/requests/<eid>.req            <- we drop this
        <reward_root>/scores/<eid>.score.json       <- we wait for this
        <reward_root>/scores/<eid>.error.json       <- raised as RuntimeError

    Env knobs:
        DSRL_REWARD_ROOT       (required) where the server is watching
        DSRL_REWARD_TIMEOUT_S  max wait for a single score (default 600)
        DSRL_REWARD_POLL_S     poll interval (default 0.5)
    """
    wm_score_request(traj)
    return wm_score_await(traj)
