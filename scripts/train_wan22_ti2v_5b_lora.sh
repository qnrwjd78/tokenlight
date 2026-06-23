#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

DATASET_BASE_PATH="${DATASET_BASE_PATH:-data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-model/train/Wan2.2-TI2V-5B_lora}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_single_gpu.yaml}"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights/Wan2.2-TI2V-5B}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-49}"
DATASET_REPEAT="${DATASET_REPEAT:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"
DATA_FILE_KEYS="${DATA_FILE_KEYS:-video}"
EXTRA_INPUTS="${EXTRA_INPUTS:-input_image}"

DIFFUSION_SHARDS="[\"${WEIGHTS_DIR}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${WEIGHTS_DIR}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${WEIGHTS_DIR}/diffusion_pytorch_model-00003-of-00003.safetensors\"]"
LOCAL_MODEL_PATHS="[${DIFFUSION_SHARDS},\"${WEIGHTS_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${WEIGHTS_DIR}/Wan2.2_VAE.pth\"]"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WEIGHTS_DIR}/google/umt5-xxl}"

accelerate_args=()
if [[ -f "${ACCELERATE_CONFIG}" ]]; then
  accelerate_args=(--config_file "${ACCELERATE_CONFIG}")
fi
if [[ -n "${NUM_PROCESSES:-}" ]]; then
  accelerate_args+=(--num_processes "${NUM_PROCESSES}")
fi

model_args=(--model_paths "${MODEL_PATHS:-${LOCAL_MODEL_PATHS}}" --tokenizer_path "${TOKENIZER_PATH}")

accelerate launch "${accelerate_args[@]}" model/train.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "${DATA_FILE_KEYS}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  "${model_args[@]}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_rank "${LORA_RANK}" \
  --extra_inputs "${EXTRA_INPUTS}" \
  "$@"
