#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_tokenlight_pbr_single.json}"

json_value() {
  python3 - "$TRAIN_CONFIG" "$1" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
key_path = sys.argv[2].split(".")
if not config_path.exists():
    sys.exit(0)
with config_path.open("r", encoding="utf-8") as f:
    value = json.load(f)
for key in key_path:
    if not isinstance(value, dict) or key not in value:
        sys.exit(0)
    value = value[key]
if isinstance(value, bool):
    print("1" if value else "0")
elif value is not None:
    print(value)
PY
}

count_devices() {
  python3 - "$1" <<'PY'
import sys

value = sys.argv[1].strip()
items = [item.strip() for item in value.split(",") if item.strip()]
if not items:
    sys.exit(0)
print(1 if items == ["cpu"] else len(items))
PY
}

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(json_value launch.accelerate_config)}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_single_gpu.yaml}"
GPU_DEVICES="${GPU_DEVICES:-$(json_value launch.gpu_devices)}"

if [[ -n "${GPU_DEVICES}" && "${GPU_DEVICES}" != "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

NUM_PROCESSES="${NUM_PROCESSES:-$(count_devices "${GPU_DEVICES:-}")}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export TOKENLIGHT_RUN_TIMESTAMP="${RUN_TIMESTAMP}"
export TOKENLIGHT_TRAIN_CONFIG="${TRAIN_CONFIG}"

accelerate_args=()
if [[ -f "${ACCELERATE_CONFIG}" ]]; then
  accelerate_args=(--config_file "${ACCELERATE_CONFIG}")
fi
accelerate_args+=(--num_processes "${NUM_PROCESSES}")

echo "TokenLight PBR single train config: ${TRAIN_CONFIG}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "GPU devices: ${GPU_DEVICES:-accelerate default}"
echo "Num processes: ${NUM_PROCESSES}"

accelerate launch "${accelerate_args[@]}" model/train_tokenlight_pbr_single.py \
  --config "${TRAIN_CONFIG}" \
  "$@"
