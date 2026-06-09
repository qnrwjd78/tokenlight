# Docker Workspace

The container expects this repo mounted at `/workspace`.

```bash
docker build -f docker/Dockerfile -t tokenlight-wan .

docker run -it --ipc=host --gpus all \
  -v "$PWD":/workspace \
  tokenlight-wan bash
```

Runtime paths:

```text
/workspace/src
/workspace/scripts
/workspace/configs
/workspace/repos/relighting_dataset
/workspace/data
/workspace/models
/workspace/runs
```

The entrypoint sets:

```text
PYTHONPATH=/workspace/src:/workspace/repos/relighting_dataset
DIFFSYNTH_MODEL_BASE_PATH=/workspace/models
HF_HOME=/workspace/data/.cache/huggingface
MODELSCOPE_CACHE=/workspace/data/.cache/modelscope
```
