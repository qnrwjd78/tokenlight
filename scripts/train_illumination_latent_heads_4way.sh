#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

DATA_ROOT="${DATA_ROOT:-data/objaverse_ratio3p5_cube1p6_direct_scene0000_1999_640_png}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_ITEMS="${MAX_ITEMS:-}"
MAX_STEPS="${MAX_STEPS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-model/train/illum_head_4way}"

common_args=(
  --data_root "${DATA_ROOT}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
)

if [[ -n "${MAX_ITEMS}" ]]; then
  common_args+=(--max_items "${MAX_ITEMS}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  common_args+=(--max_steps "${MAX_STEPS}")
fi

for target in luminance log_luminance; do
  for arch in lite resunet; do
    echo "Training illumination latent head: target=${target}, arch=${arch}"
    python3 scripts/train_illumination_latent_head.py \
      "${common_args[@]}" \
      --target "${target}" \
      --arch "${arch}" \
      --output_dir "${OUTPUT_ROOT}_${target}_${arch}"
  done
done
