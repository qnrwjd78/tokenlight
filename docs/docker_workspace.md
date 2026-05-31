# Docker Workspace

The container image is intentionally small: it contains CUDA/PyTorch and Python
runtime dependencies only. Code, checkpoints, data, caches, and outputs are
mounted from the host under `/workspace`.

## Host Layout

Use this layout on the host:

```text
relighting/
  code/
    tokenlight/
      configs/
      docker/
      scripts/
      src/
      README.md
    repos/
      unirelight/
  data/
  runs/
```

`relighting` is the directory you mount into the container:

```bash
docker run -it --name <container-name> -v ${PWD}:/workspace --gpus all <image-name> bash
```

Run that command from the `relighting` directory.

## Build Image

From the host `relighting` directory:

```bash
docker build -f code/tokenlight/docker/Dockerfile -t tokenlight-cosmos code/tokenlight
```

The build context excludes source/data/checkpoints. The image only installs the
runtime environment.

## Clone Repos

Clone external repos under `relighting/code/repos`:

```bash
mkdir -p code/repos data runs
git clone <unirelight-repo-url> code/repos/unirelight
```

This project should live at:

```text
relighting/code/tokenlight
```

## Start Container

From `relighting`:

```bash
docker run -it --ipc=host --name tokenlight-dev -v ${PWD}:/workspace --gpus all tokenlight-cosmos bash
```

Inside the container, `/workspace` is the host `relighting` directory.

The entrypoint sets:

```text
PYTHONPATH=/workspace/code/tokenlight/src:/workspace/code/repos/unirelight
HF_HOME=/workspace/.cache/huggingface
TORCH_HOME=/workspace/.cache/torch
```

## Download Base Weights

Inside the container:

```bash
cd /workspace/code/repos/unirelight
huggingface-cli login
python scripts/download_unirelight_checkpoints.py --checkpoint_dir checkpoints
```

Expected files:

```text
/workspace/code/repos/unirelight/checkpoints/UniRelight/model.pt
/workspace/code/repos/unirelight/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit
/workspace/code/repos/unirelight/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit
/workspace/code/repos/unirelight/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/image_mean_std.pt
```

## Check Base

From `/workspace`:

```bash
python code/tokenlight/scripts/inspect_base.py --config code/tokenlight/configs/tokenlight_cosmos.toml
```

## Train

Place data and manifests under:

```text
/workspace/data
```

Then run from `/workspace`:

```bash
python code/tokenlight/scripts/train.py \
  --config code/tokenlight/configs/tokenlight_cosmos.toml \
  --manifest data/train.jsonl \
  --data-root data \
  --output runs/tokenlight
```

Distributed example:

```bash
torchrun --nproc_per_node=8 code/tokenlight/scripts/train.py \
  --config code/tokenlight/configs/tokenlight_cosmos.toml \
  --manifest data/train.jsonl \
  --data-root data \
  --output runs/tokenlight \
  --fsdp
```

## Notes

The image does not copy repository files into the container. If you edit code on
the host, the container sees the changes immediately through the bind mount.

Optional heavy dependencies such as Apex and nvdiffrast are not installed in the
base image. Add them only if a specific UniRelight/Cosmos code path requires
them.
