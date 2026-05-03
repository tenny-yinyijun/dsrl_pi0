#!/bin/bash
# One-time pre-download of every artifact the WM-reward loop needs.
# Run this on a LOGIN NODE (compute nodes have no internet).
#
# Idempotent — re-running only fetches things that aren't already cached.
#
#   bash examples/scripts/setup_caches.sh
#
# Knobs (override on the command line if needed):
#   POLICY={pi0,pi05}   — which Pi0 variant to fetch (default: pi0)
#   DSRL_ROOT=...       — defaults to /scratch/gpfs/AM43/yy4041/dsrl_pi0
#   OPEN_WORLD_ROOT=... — defaults to /scratch/gpfs/AM43/yy4041/open-world

set -euo pipefail

DSRL_ROOT="${DSRL_ROOT:-/n/fs/iromdata/project/dsrl_pi0}"
OPEN_WORLD_ROOT="${OPEN_WORLD_ROOT:-/n/fs/iromdata/project/open-world}"
POLICY="${POLICY:-pi05}"
TORCH_HOME="${TORCH_HOME:-/n/fs/tom-project/.cache/torch}"

ok() { printf '  \033[32mOK\033[0m  %s\n' "$1"; }
miss() { printf '  \033[31mMISSING\033[0m  %s\n' "$1"; }
note() { printf '\033[33m[setup]\033[0m %s\n' "$1"; }

note "verifying / fetching all caches needed by the WM-reward loop..."
echo

# ---- 1. dsrl_pi0 venv ----
note "1) dsrl_pi0 venv"
if [ -x "$DSRL_ROOT/.venv/bin/python" ]; then
    ok "$DSRL_ROOT/.venv/bin/python"
else
    miss "$DSRL_ROOT/.venv — run README install steps first"
    exit 1
fi

# ---- 2. open-world venv ----
note "2) open-world venv"
if [ -x "$OPEN_WORLD_ROOT/.venv/bin/python" ]; then
    ok "$OPEN_WORLD_ROOT/.venv/bin/python"
else
    miss "$OPEN_WORLD_ROOT/.venv — run uv sync inside open-world first"
    exit 1
fi

# ---- 3. SVD + CLIP (open-world's external/) ----
note "3) SVD + CLIP"
for d in stable-video-diffusion-img2vid clip-vit-base-patch32; do
    if [ -d "$OPEN_WORLD_ROOT/external/$d" ]; then
        ok "$OPEN_WORLD_ROOT/external/$d"
    else
        miss "$OPEN_WORLD_ROOT/external/$d"
        echo "        try: cd $OPEN_WORLD_ROOT && bash external/download_models.sh"
        exit 1
    fi
done

# ---- 4. AlexNet for LPIPS ----
note "4) torchvision AlexNet (used by lpips)"
ALEXNET="$TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth"
if [ -f "$ALEXNET" ]; then
    ok "$ALEXNET"
else
    note "downloading AlexNet weights..."
    mkdir -p "$(dirname "$ALEXNET")"
    "$OPEN_WORLD_ROOT/.venv/bin/python" - <<PY
import os, torch
os.environ["TORCH_HOME"] = "$TORCH_HOME"
import lpips; lpips.LPIPS(net="alex", verbose=False)
PY
    [ -f "$ALEXNET" ] && ok "$ALEXNET" || { miss "AlexNet still not at $ALEXNET"; exit 1; }
fi

# ---- 5. Pi055 LIBERO checkpoint ----
note "5) Pi05 LIBERO checkpoint ($POLICY)"
PI_VARIANT="${POLICY}_libero"
PI_CACHE="/n/fs/tom-project/.cache/openpi/openpi-assets/checkpoints/$PI_VARIANT"
if [ -d "$PI_CACHE" ] && [ -d "$PI_CACHE/params" ]; then
    ok "$PI_CACHE"
else
    note "downloading $PI_VARIANT (this can take a while)..."
    cd "$DSRL_ROOT"
    "$DSRL_ROOT/.venv/bin/python" - <<PY
from openpi.shared import download
# openpi-assets is a GCS bucket; pi05_libero in particular is GCS-only
# (the S3 mirror only has pi0_libero / pi0_fast_libero). openpi's
# download.maybe_download routes gs://openpi-assets through gsutil.
print(download.maybe_download("gs://openpi-assets/checkpoints/$PI_VARIANT"))
PY
    if [ -d "$PI_CACHE" ]; then
        ok "$PI_CACHE"
    else
        miss "Pi05 download did not land at expected path $PI_CACHE"
        exit 1
    fi
fi

# ---- 6. WM checkpoint ----
note "6) libero WM checkpoint"
WM_DEFAULT="$OPEN_WORLD_ROOT/models/wm_training/libero_0429/checkpoint-20000.pt"
if [ -f "$WM_DEFAULT" ]; then
    ok "$WM_DEFAULT"
else
    miss "$WM_DEFAULT — train one with bash_scripts/train_libero_wm.sh"
    exit 1
fi

# ---- 7. pretrain dataset (for fine-tune; optional) ----
note "7) pretrain libero_processed dataset (optional, only for finetune)"
PRETRAIN="$OPEN_WORLD_ROOT/data/wm_training/libero_processed"
if [ -d "$PRETRAIN" ] && [ -f "$PRETRAIN/stat.json" ]; then
    ok "$PRETRAIN"
else
    miss "$PRETRAIN — run finetune_wm.py only after this exists"
fi

# ---- 8. wandb (offline mode is fine) ----
note "8) wandb"
if command -v wandb >/dev/null 2>&1; then
    ok "wandb CLI present (offline mode will buffer locally)"
else
    note "wandb CLI not on PATH; runs will set WANDB_MODE=offline"
fi

echo
note "all caches present. you are ready to submit run_wm_loop.sh"
