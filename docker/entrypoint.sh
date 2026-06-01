#!/usr/bin/env bash
set -e

if [ -d /workspace/code/tokenlight/src ]; then
  export PYTHONPATH="/workspace/code/tokenlight/src:/workspace/code/tokenlight/repos/unirelight:/workspace/code/tokenlight/repos/relighting_dataset:/workspace/code/repos/unirelight:/workspace/code/repos/relighting_dataset:${PYTHONPATH}"
elif [ -d /workspace/src ]; then
  export PYTHONPATH="/workspace/src:/workspace/repos/unirelight:/workspace/repos/relighting_dataset:/workspace/code/repos/unirelight:/workspace/code/repos/relighting_dataset:${PYTHONPATH}"
else
  export PYTHONPATH="/workspace/code/tokenlight/repos/unirelight:/workspace/code/tokenlight/repos/relighting_dataset:/workspace/code/repos/unirelight:/workspace/code/repos/relighting_dataset:${PYTHONPATH}"
fi

mkdir -p /workspace/data /workspace/runs /workspace/.cache/huggingface /workspace/.cache/torch

exec "$@"
