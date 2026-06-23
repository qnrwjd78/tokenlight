#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

INFER_CONFIG="${INFER_CONFIG:-configs/infer_config.json}"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights/Wan2.2-TI2V-5B}"

json_value() {
  python3 - "$INFER_CONFIG" "$1" <<'PY'
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

GPU_DEVICES="${GPU_DEVICES:-$(json_value launch.gpu_devices)}"
if [[ -n "${GPU_DEVICES}" && "${GPU_DEVICES}" != "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

python3 model/infer_tokenlight.py \
  --weights_dir "${WEIGHTS_DIR}" \
  "$@"
