#!/usr/bin/env bash
set -e

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-tokenlight-wan}"
fi

if [ -d /workspace/src ]; then
  export PYTHONPATH="/workspace/src:/workspace/repos/relighting_dataset:${PYTHONPATH}"
elif [ -d /workspace/code/tokenlight/src ]; then
  export PYTHONPATH="/workspace/code/tokenlight/src:/workspace/code/tokenlight/repos/relighting_dataset:/workspace/code/repos/relighting_dataset:${PYTHONPATH}"
fi

mkdir -p \
  /workspace/data \
  /workspace/models \
  /workspace/runs \
  "${HF_HOME:-/workspace/data/.cache/huggingface}" \
  "${MODELSCOPE_CACHE:-/workspace/data/.cache/modelscope}" \
  "${DIFFSYNTH_MODEL_BASE_PATH:-/workspace/models}" \
  "${TORCH_HOME:-/workspace/data/.cache/torch}" \
  "${PIP_CACHE_DIR:-/workspace/data/.cache/pip}"

exec "$@"
