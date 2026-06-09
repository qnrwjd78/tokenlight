# TokenLight Wan2.2-TI2V-5B Base

This is the only active model path in the repo.

## What Changed

Removed:

- old non-Wan config and inspection path
- native TokenLight DiT, VAE wrapper, flow sampler, and FSDP trainer
- 960 square config

Kept:

- `repos/relighting_dataset` component synthesis
- source/target relighting pair export
- TokenLight numeric light attributes

Added:

- Wan2.2-TI2V-5B DiffSynth trainer in `scripts/train.py`
- TokenLight source/mask/light token injection in `src/tokenlight/wan.py`
- Wan inference with paper-style TokenLight DiT prefix tokens in `scripts/infer.py`

## Numeric Conditioning

Training does not depend on numeric values in prompt text by default.

The exported prompt is generic:

```text
photorealistic object relighting... apply a localized point-light edit.
```

The numeric values live in:

```text
attrs_json
```

At training time, the custom Wan model function follows the TokenLight paper
conditioning layout inside DiT self-attention:

```text
[source latent tokens] + [mask latent tokens] + [a, dg, x, y, z, r, g, b, lambda, d, t] + [noisy target latent tokens]
```

Each scalar uses Gaussian Fourier features with `sigma=5`, matching the
TokenLight-style numeric encoding. During CFG/dropout, source and mask tokens
stay present while light tokens become null `-1` tokens.

## Download Base Weights

The train command can download automatically:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/models
```

and pass:

```bash
--model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
```

Manual download:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /workspace/models/Wan2.2-TI2V-5B \
  --include "diffusion_pytorch_model*.safetensors" \
  --include "models_t5_umt5-xxl-enc-bf16.pth" \
  --include "Wan2.2_VAE.pth"
```

## Render

```bash
cd /workspace/repos/relighting_dataset
python scripts/run_blender_batch.py \
  --config /workspace/configs/relighting_dataset_wan22_1280x704.json \
  --max-scenes 10 \
  --width 1280 \
  --height 704 \
  --samples 128
```

## Export Pre-Rendered Point-Light PNGs

```bash
cd /workspace
python scripts/export_wan22_tokenlight_dataset.py \
  --dataset-kind point-light-png \
  --data-root /workspace/data/sample \
  --output /workspace/data/tokenlight_wan22_train \
  --modes spatial ambient diffuse \
  --pairing all-targets \
  --count 0 \
  --width 1280 \
  --height 704 \
  --num-frames 1 \
  --target-format png \
  --include-object-masks \
  --prompt-mode generic \
  --overwrite
```

This path reads `scene_*/spatial/point_lights/light_*.png` directly for the
spatial task. Ambient and diffuse tasks are synthesized from EXR components
using the same scene `meta.json`, then exported as ordinary PNG metadata rows.
By default spatial skips lights marked invalid in `meta.json`, because those
PNGs may be copied from another light. Add `--include-invalid-lights` only if
you want all 64 spatial light IDs per scene.

## LoRA Training

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --config_file configs/accelerate_wan22_6x40gb_zero3.yaml \
  scripts/train.py \
  --dataset_base_path /workspace/data/tokenlight_wan22_train \
  --dataset_metadata_path /workspace/data/tokenlight_wan22_train/metadata.csv \
  --data_file_keys video,input_image,mask \
  --height 704 \
  --width 1280 \
  --num_frames 1 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path /workspace/runs/tokenlight_wan22_lora \
  --lora_base_model dit \
  --lora_target_modules "q,k,v" \
  --lora_rank 32 \
  --tokenlight_light_tokens \
  --tokenlight_source_tokens \
  --tokenlight_mask_tokens \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --initialize_model_on_cpu
```

## Full DiT Training

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --config_file configs/accelerate_wan22_6x40gb_zero3.yaml \
  scripts/train.py \
  --dataset_base_path /workspace/data/tokenlight_wan22_train \
  --dataset_metadata_path /workspace/data/tokenlight_wan22_train/metadata.csv \
  --data_file_keys video,input_image,mask \
  --height 704 \
  --width 1280 \
  --num_frames 1 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path /workspace/runs/tokenlight_wan22_full \
  --trainable_models dit \
  --tokenlight_light_tokens \
  --tokenlight_source_tokens \
  --tokenlight_mask_tokens \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --initialize_model_on_cpu
```

## References

- DiffSynth Wan training entrypoint:
  https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/train.py
- DiffSynth Wan2.2-TI2V-5B full training example:
  https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B.sh
- Wan2.2-TI2V-5B Hugging Face:
  https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
