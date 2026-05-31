#!/usr/bin/env bash
set -e

if [ -d /workspace/code/tokenlight/src ]; then
  export PYTHONPATH="/workspace/code/tokenlight/src:/workspace/code/repos/unirelight:${PYTHONPATH}"
elif [ -d /workspace/src ]; then
  export PYTHONPATH="/workspace/src:/workspace/code/repos/unirelight:${PYTHONPATH}"
else
  export PYTHONPATH="/workspace/code/repos/unirelight:${PYTHONPATH}"
fi

mkdir -p /workspace/data /workspace/runs /workspace/.cache/huggingface /workspace/.cache/torch

exec "$@"
