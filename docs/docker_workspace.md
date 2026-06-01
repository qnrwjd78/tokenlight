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
      repos/
        unirelight/
        relighting_dataset/
      scripts/
      src/
      README.md
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

## Initialize Submodules

External repos are tracked as submodules under `relighting/code/tokenlight/repos`:

```bash
mkdir -p data runs
git -C code/tokenlight submodule update --init --recursive
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
PYTHONPATH=/workspace/code/tokenlight/src:/workspace/code/tokenlight/repos/unirelight:/workspace/code/tokenlight/repos/relighting_dataset
HF_HOME=/workspace/.cache/huggingface
TORCH_HOME=/workspace/.cache/torch
```

## Generate Synthetic Components

`repos/relighting_dataset` writes EXR components and metadata. Keep that repo
external and use this project's config so outputs land under `/workspace/data`:

```bash
cd /workspace/code/tokenlight/repos/relighting_dataset
python scripts/run_blender_batch.py \
  --config /workspace/code/tokenlight/configs/relighting_dataset_960.json \
  --max-scenes 1 \
  --resolution 256 \
  --samples 32
```

That debug command should write:

```text
/workspace/data/tokenlight_synthetic/dataset_manifest.json
/workspace/data/tokenlight_synthetic/scenes/scene_000000/meta.json
```

For the full 960 x 960 component set, remove the debug overrides:

```bash
python scripts/run_blender_batch.py \
  --config /workspace/code/tokenlight/configs/relighting_dataset_960.json
```

The renderer needs Blender. The minimal TokenLight training image does not
install Blender by default. Use a Blender-capable image, install Blender in the
container, or run the render step on the host and keep the output under the
mounted `relighting/data` directory.

Before full generation, populate these files inside
`/workspace/code/tokenlight/repos/relighting_dataset`:

```text
manifests/objects.txt
manifests/hdris.txt
manifests/fixture_scenes.jsonl
```

If `objects.txt` or `hdris.txt` are empty, the renderer falls back to primitives
and constant environment lighting. That is acceptable for a smoke test, not for a
paper-quality dataset. Fixture samples require valid `.blend` scenes in
`fixture_scenes.jsonl`; otherwise fixture rendering is skipped.

## Download Base Weights

Inside the container:

```bash
cd /workspace/code/tokenlight/repos/unirelight
huggingface-cli login
python scripts/download_unirelight_checkpoints.py --checkpoint_dir checkpoints
```

Expected files:

```text
/workspace/code/tokenlight/repos/unirelight/checkpoints/UniRelight/model.pt
/workspace/code/tokenlight/repos/unirelight/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit
/workspace/code/tokenlight/repos/unirelight/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit
/workspace/code/tokenlight/repos/unirelight/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/image_mean_std.pt
```

## Check Base

From `/workspace/code/tokenlight`:

```bash
python scripts/inspect_base.py --config configs/tokenlight_cosmos.toml
```

## Train

Place data and manifests under:

```text
/workspace/data
```

Then run from `/workspace/code/tokenlight`:

```bash
python scripts/train.py \
  --config configs/tokenlight_cosmos.toml \
  --manifest /workspace/data/train.jsonl \
  --data-root /workspace/data \
  --output /workspace/runs/tokenlight
```

To train directly from `repos/relighting_dataset` component outputs, skip a JSONL
manifest and use the adapter:

```bash
python scripts/train.py \
  --config configs/tokenlight_cosmos.toml \
  --dataset-type relighting-components \
  --component-repo repos/relighting_dataset \
  --component-root /workspace/data/tokenlight_synthetic \
  --component-modes spatial ambient diffuse \
  --component-length 100000 \
  --max-lights 1 \
  --image-range minus_one_one \
  --output runs/tokenlight
```

Use `--component-modes spatial ambient diffuse fixture` only after fixture
scenes have been rendered. `--max-lights` must stay `1` until the model has a
multi-light token packing scheme.

Distributed example:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/tokenlight_cosmos.toml \
  --dataset-type relighting-components \
  --component-repo repos/relighting_dataset \
  --component-root /workspace/data/tokenlight_synthetic \
  --component-modes spatial ambient diffuse \
  --output /workspace/runs/tokenlight \
  --fsdp
```

## Notes

The image does not copy repository files into the container. If you edit code on
the host, the container sees the changes immediately through the bind mount.

Optional heavy dependencies such as Apex and nvdiffrast are not installed in the
base image. Add them only if a specific UniRelight/Cosmos code path requires
them.
