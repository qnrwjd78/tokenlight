#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

python3 scripts/build_objaverse_pbr_dataset.py \
  --aux-type depth \
  --raw-root "${RAW_ROOT:-data/objaverse_sample_completed_20260624_144354}" \
  --output-root "${OUTPUT_ROOT:-data/objaverse_pbr_depth}" \
  "$@"
