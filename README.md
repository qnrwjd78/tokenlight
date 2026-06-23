# TokenLight Reprod Wan2.2 Baseline

This folder keeps the official DiffSynth-Studio Wan2.2-TI2V-5B baseline and
adds separate TokenLight conditioning entrypoints.

## Layout

```text
data/      dataset files
scripts/   runnable shell commands
model/     official baseline code and TokenLight extension code
configs/   accelerate configs
docker/    minimal Docker image
repos/     external repos, if needed later
weights/   local Wan2.2-TI2V-5B weights
```

## Docker

Build from this folder:

```bash
docker build -f docker/Dockerfile -t tokenlight-reprod .
```

Run with the mounted workspace:

```bash
docker run -it --ipc=host --name tokenlight-reprod \
  -v ${PWD}:/workspace --gpus all tokenlight-reprod bash
```

## LoRA Train

The default command follows the official DiffSynth Wan2.2-TI2V-5B LoRA example,
but loads local weights from:

```text
weights/Wan2.2-TI2V-5B
```

Put the DiffSynth example dataset under:

```text
data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B
```

Then run:

```bash
bash scripts/train_wan22_ti2v_5b_lora.sh
```

The official training entrypoint is preserved as:

```text
model/train.py
```

The official DiffSynth inference example is preserved as:

```text
model/infer_official.py
```

## LoRA Train With ZeRO-3

The ZeRO-3 launcher uses `configs/accelerate_zero3_cpuoffload.yaml` and
`configs/ds_z3_cpuoffload.json`.

```bash
bash scripts/train_wan22_ti2v_5b_lora_zero3.sh
```

For multi-GPU:

```bash
NUM_PROCESSES=4 bash scripts/train_wan22_ti2v_5b_lora_zero3.sh
```

Useful overrides:

```bash
DATASET_BASE_PATH=data/my_dataset \
DATASET_METADATA_PATH=data/my_dataset/metadata.csv \
OUTPUT_PATH=model/train/my_wan22_lora \
bash scripts/train_wan22_ti2v_5b_lora.sh
```

## Inference

Text-to-video:

```bash
bash scripts/infer_wan22_ti2v_5b.sh \
  --prompt "Two cute cats wearing boxing gloves fight on a boxing ring."
```

Image-to-video:

```bash
bash scripts/infer_wan22_ti2v_5b.sh \
  --input_image data/input.png \
  --prompt "Two cute cats wearing boxing gloves fight on a boxing ring."
```

## TokenLight Metadata

TokenLight training metadata should contain these columns:

```text
video,input_image,mask,prompt,attrs_json
```

Meanings:

```text
video       target relit image/video
input_image source image
mask        optional relighting/object/fixture mask
prompt      fixed generic text prompt
attrs_json  numeric light condition JSON
```

Example `attrs_json`:

```json
{"a":0.014,"x":0.2,"y":-0.4,"z":0.8,"r":1.0,"g":1.0,"b":1.0,"lambda":1.2,"d":0.06}
```

## TokenLight Train

TokenLight training uses source/mask/light prefix tokens before the noisy Wan
target tokens. Text prompt stays fixed; CFG/dropout is applied only to light
tokens.

Single-GPU, no ZeRO-3:

```bash
bash scripts/train_tokenlight_wan22_lora_single.sh
```

Multi-GPU with DeepSpeed ZeRO-3:

```bash
NUM_PROCESSES=4 bash scripts/train_tokenlight_wan22_lora_zero3.sh
```

By default this uses `configs/accelerate_zero3.yaml` and `configs/ds_z3.json`
without CPU offload. To use CPU offload instead:

```bash
ACCELERATE_CONFIG=configs/accelerate_zero3_cpuoffload.yaml \
bash scripts/train_tokenlight_wan22_lora_zero3.sh
```

The two TokenLight train entrypoints are intentionally separate:

```text
model/train_tokenlight_single.py   configs/train_tokenlight_single.json
model/train_tokenlight_zero3.py    configs/train_tokenlight_zero3.json
```

`scripts/train_tokenlight_wan22_lora.sh` is kept only as a backwards-compatible
alias for the single-GPU script.

## TokenLight Inference

```bash
bash scripts/infer_tokenlight_wan22.sh \
  --source data/source.png \
  --attrs '{"a":0.014,"x":0.2,"y":-0.4,"z":0.8,"r":1.0,"g":1.0,"b":1.0,"lambda":1.2,"d":0.06}' \
  --checkpoint model/train/tokenlight_wan22_lora/step-100.safetensors \
  --cfg_scale 2.0 \
  --output outputs/tokenlight.png
```

## Lightoken Encoder

`model/lightoken_encoder.py` contains a standalone TokenLight-style numeric
light encoder using Gaussian Fourier features plus one projection layer per
attribute.
