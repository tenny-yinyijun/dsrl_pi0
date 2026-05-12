#!/bin/bash
# Run the standalone WM overfit sanity check on the LATEST overfit run.
# Renders the most recent fine-tuned checkpoint against the 2 collected
# trajectories with BOTH temporal layouts:
#   skip_his=4  — matches score_episode (in-server sanity videos).
#                 After the _sample_window fix, this should converge to GT.
#   skip_his=1  — the layout the OLD _sample_window trained on. A control:
#                 if BOTH look fine, training generalizes; if only this one
#                 looks fine, the model overfit to the wrong distribution.
#
# Optional env vars:
#   REWARD_ROOT  — pin a specific run (default: newest *_overfit dir under
#                  /scratch/gpfs/AM43/yy4041/playworld_rollout/).
#   CKPT         — pin a specific .pt or latest.txt (default: latest.txt
#                  under $REWARD_ROOT/wm_checkpoints).
#   EIDS         — comma-separated; default: 2 most-recent under
#                  annotation/train/.
#   START_FRAME  — default 24.
#   NUM_WINDOWS  — default 8.
#
# Usage:
#   bash examples/scripts/wm_overfit_sanity.sh
#   REWARD_ROOT=/scratch/.../<jobtag>_overfit bash examples/scripts/wm_overfit_sanity.sh

set -u

DSRL_ROOT="${DSRL_ROOT:-/scratch/gpfs/AM43/yy4041/dsrl_pi0}"
OPEN_WORLD_ROOT="${OPEN_WORLD_ROOT:-/scratch/gpfs/AM43/yy4041/open-world}"
PLAYWORLD_ROOT="${PLAYWORLD_ROOT:-/scratch/gpfs/AM43/yy4041/playworld_rollout}"

# ---- pick reward_root ----
if [ -z "${REWARD_ROOT:-}" ]; then
    REWARD_ROOT=$(ls -1dt "$PLAYWORLD_ROOT"/*/*_overfit 2>/dev/null | head -n1 || true)
    if [ -z "$REWARD_ROOT" ]; then
        echo "[sanity] FATAL: no *_overfit run found under $PLAYWORLD_ROOT and REWARD_ROOT not set"
        exit 1
    fi
fi
echo "[sanity] REWARD_ROOT=$REWARD_ROOT"

# ---- pick checkpoint ----
if [ -z "${CKPT:-}" ]; then
    if [ -f "$REWARD_ROOT/wm_checkpoints/latest.txt" ]; then
        CKPT="$REWARD_ROOT/wm_checkpoints/latest.txt"
    else
        CKPT=$(ls -1t "$REWARD_ROOT/wm_checkpoints"/checkpoint-*.pt 2>/dev/null | head -n1 || true)
    fi
fi
if [ -z "$CKPT" ] || [ ! -e "$CKPT" ]; then
    echo "[sanity] FATAL: no checkpoint found (CKPT=$CKPT)"
    exit 1
fi
echo "[sanity] CKPT=$CKPT"

START_FRAME="${START_FRAME:-24}"
NUM_WINDOWS="${NUM_WINDOWS:-8}"
EIDS="${EIDS:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPEN_WORLD_ROOT
export DSRL_ROOT

PY="$OPEN_WORLD_ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "[sanity] FATAL: open-world venv python not found at $PY"
    exit 1
fi

run_one() {
    local skip_his="$1"
    echo "[sanity] ---- rendering with skip_his=$skip_his ----"
    "$PY" "$DSRL_ROOT/examples/scripts/wm_overfit_sanity.py" \
        --reward-root "$REWARD_ROOT" \
        --ckpt-path "$CKPT" \
        --skip-his "$skip_his" \
        --start-frame "$START_FRAME" \
        --num-windows "$NUM_WINDOWS" \
        ${EIDS:+--eids "$EIDS"}
}

run_one 4   # inference layout — the one that matters
run_one 1   # control: old training layout

echo "[sanity] DONE. videos under $REWARD_ROOT/wm_overfit_sanity/"
ls -1 "$REWARD_ROOT/wm_overfit_sanity/" 2>/dev/null || true
