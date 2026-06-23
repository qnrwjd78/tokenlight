#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

WEIGHTS_DIR="${WEIGHTS_DIR:-weights/Wan2.2-TI2V-5B}"
OUTPUT="${OUTPUT:-outputs/wan22_ti2v_5b.mp4}"

python3 model/infer.py \
  --weights_dir "${WEIGHTS_DIR}" \
  --output "${OUTPUT}" \
  "$@"
