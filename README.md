# TokenLight Wan2.2 Reproduction

This repo now targets TokenLight reproduction on top of the public
Wan2.2-TI2V-5B base model through DiffSynth-Studio.

The active method is:

```text
source image I
target relit image/video Ir
TokenLight numeric light attrs DeltaL

Wan TI2V:
  source     = I  (loaded from input_image column)
  video       = Ir
  prompt      = generic task text
  DiT tokens  = [source latent] + [mask latent] + [light attrs] + [noisy target latent]
```

The old non-Wan training path has been removed.

## Layout

```text
configs/relighting_dataset_wan22_1280x704.json  1280x704 component render config
configs/accelerate_wan22_6x40gb_zero3.yaml      6 GPU Accelerate/DeepSpeed ZeRO-3 config
configs/tokenlight_wan22_ti2v_5b.toml           baseline settings record
docs/tokenlight_wan22_base.md                   full command reference
docker/Dockerfile                               Wan/DiffSynth runtime image
scripts/export_wan22_tokenlight_dataset.py      component pairs -> Wan metadata
scripts/train.py                                Wan2.2 TokenLight trainer
scripts/infer.py                                Wan2.2 TokenLight inference
scripts/evaluate.py                             image metric helper
src/tokenlight/data.py                          relighting_dataset adapter
src/tokenlight/wan.py                           paper-style Wan token injection
```

## Render Components

```bash
cd /workspace/repos/relighting_dataset
python scripts/run_blender_batch.py \
  --config /workspace/configs/relighting_dataset_wan22_1280x704.json \
  --max-scenes 10 \
  --width 1280 \
  --height 704 \
  --samples 128
```

## Export Wan Dataset From Point-Light PNGs

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

`spatial` reads the 64 point-light PNGs directly. `ambient` and `diffuse`
synthesize PNG pairs from EXR components during export, then write normal Wan
metadata rows. `--pairing all-targets --count 0` exports every deterministic
target row once. If you only want the 64-light task, pass `--modes spatial`.
If you really want all 64 spatial IDs even when the renderer marked some as
copied/invalid, add `--include-invalid-lights`.

The CSV contains:

```text
video,input_image,mask,prompt,task,attrs_json
```

`attrs_json` is encoded as DiT self-attention light tokens. The prompt stays
generic unless `--prompt-mode attrs` is selected.

## Train LoRA

```bash
cd /workspace
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/models

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

## Train Full DiT

```bash
cd /workspace
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/models

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
