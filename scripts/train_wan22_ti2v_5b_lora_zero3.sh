#!/usr/bin/env bash
set -euo pipefail

export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_zero3_cpuoffload.yaml}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_wan22_ti2v_5b_lora.sh" \
  --use_gradient_checkpointing \
  --initialize_model_on_cpu \
  "$@"
