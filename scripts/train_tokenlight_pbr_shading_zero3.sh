#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_tokenlight_pbr_shading_zero3.json}" \
  scripts/train_tokenlight_pbr_wan22_lora_zero3.sh "$@"
