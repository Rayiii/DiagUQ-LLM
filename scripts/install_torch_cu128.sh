#!/usr/bin/env bash
# Install PyTorch 2.8.0 + torchvision 0.23.0 + torchaudio 2.8.0 for CUDA 12.8
# from the official PyTorch wheel index.
#
# Prerequisites:
#   - the `diaguq` conda env created from environment.yaml is ACTIVATED
#   - the host has a CUDA 12.8 capable driver (AutoDL CUDA 12.8 image is fine)
#
# Usage:
#   conda activate diaguq
#   bash scripts/install_torch_cu128.sh
#
# We avoid wiring this into environment.yaml because `conda env create`
# routes pip through a generic mirror which cannot resolve the
# `+cu128` wheels reliably.

set -euo pipefail

TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"
INDEX_URL="https://download.pytorch.org/whl/cu128"

echo "[install_torch_cu128] installing torch==${TORCH_VERSION} \
torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} from ${INDEX_URL}"

python -m pip install --upgrade pip
python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${INDEX_URL}"

echo
echo "[install_torch_cu128] verification:"
python - <<'PY'
import sys
print(f"  python              : {sys.version.split()[0]}")
try:
    import torch
    print(f"  torch               : {torch.__version__}")
    print(f"  torch.version.cuda  : {torch.version.cuda}")
    print(f"  cuda.is_available() : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda device count   : {torch.cuda.device_count()}")
        print(f"  cuda device name    : {torch.cuda.get_device_name(0)}")
except Exception as exc:  # noqa: BLE001
    print(f"  torch import FAILED : {exc}")
    sys.exit(1)
PY

echo "[install_torch_cu128] done."
