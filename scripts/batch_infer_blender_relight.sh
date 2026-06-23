#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

INFER_CONFIG="${INFER_CONFIG:-configs/infer_config.json}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export TOKENLIGHT_RUN_TIMESTAMP="${RUN_TIMESTAMP}"

python3 model/batch_infer_blender_relight.py \
  --config "${INFER_CONFIG}" \
  "$@"
