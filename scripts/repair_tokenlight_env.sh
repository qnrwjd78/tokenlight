#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

CONSTRAINTS="$(mktemp)"
trap 'rm -f "${CONSTRAINTS}"' EXIT

cat > "${CONSTRAINTS}" <<'EOF'
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
torchcodec==0.2.1
triton==3.2.0
transformers==4.56.2
peft==0.17.1
accelerate==1.13.0
datasets==5.0.0
huggingface_hub>=0.34.0,<1.0
fsspec>=2023.1.0,<=2026.4.0
tensorboard>=2.16,<3
EOF

echo "[1/4] Repairing pinned runtime packages"
python -m pip install --no-cache-dir --force-reinstall \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  -c "${CONSTRAINTS}" \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  torchcodec==0.2.1 \
  triton==3.2.0 \
  transformers==4.56.2 \
  peft==0.17.1 \
  accelerate==1.13.0 \
  datasets==5.0.0 \
  "huggingface_hub>=0.34.0,<1.0" \
  "fsspec[http]>=2023.1.0,<=2026.4.0" \
  "tensorboard>=2.16,<3" \
  deepspeed

echo "[2/4] Clearing torch extension cache"
rm -rf "${HOME}/.cache/torch_extensions"

echo "[3/4] Checking Python imports and versions"
python - <<'PY'
import importlib.metadata as md

import torch
import torchvision
import torchaudio
import transformers
import peft
import accelerate
import deepspeed
import huggingface_hub
import fsspec
from torch.utils.tensorboard import SummaryWriter

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("fsspec", fsspec.__version__)
print("tensorboard SummaryWriter OK", SummaryWriter)

assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torchvision.__version__.startswith("0.21.0"), torchvision.__version__
assert torchaudio.__version__.startswith("2.6.0"), torchaudio.__version__
assert transformers.__version__ == "4.56.2", transformers.__version__
assert peft.__version__ == "0.17.1", peft.__version__
assert accelerate.__version__ == "1.13.0", accelerate.__version__

from transformers import BloomPreTrainedModel  # noqa: F401
from diffsynth.diffusion import DiffusionTrainingModule  # noqa: F401

print("import checks OK")
PY

echo "[4/4] Checking sample dataset and weights"
python - <<'PY'
from pathlib import Path

required = [
    Path("/workspace/outputs/sample_exr_png/metadata.jsonl"),
    Path("weights/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"),
    Path("weights/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"),
    Path("weights/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"),
    Path("weights/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
    Path("weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    Path("weights/Wan2.2-TI2V-5B/google/umt5-xxl"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))

metadata = required[0]
with metadata.open("r", encoding="utf-8") as f:
    rows = sum(1 for _ in f)
print(f"metadata rows: {rows}")
if rows <= 0:
    raise SystemExit(f"No rows in {metadata}")
PY

echo "TokenLight environment repair/check OK"
