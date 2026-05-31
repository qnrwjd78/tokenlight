# TokenLight Reproduction

This repository targets the TokenLight paper. UniRelight is not the method being
reproduced. The only UniRelight/Cosmos role is to provide a public substitute for
TokenLight's unpublished pretrained text-to-video base checkpoint.

Implemented TokenLight paper logic:

- image, mask, and target latents are tokenized into one full self-attention sequence
- scalar lighting attributes use Gaussian Fourier features with `sigma = 5`
- vector lighting attributes are represented as component tokens
- fixture masks are encoded through the same VAE path as images
- training uses the linear-interpolant flow-matching target `X - eps`
- inference keeps source image and mask conditioning and drops only light tokens for CFG
- renderer utilities preserve the paper's linear RGB component-composition equations
- camera-light canonicalization follows the paper's Sim(3) scaling rules

The paper does not publicly identify the exact pretrained text-to-video checkpoint,
VAE weights, DiT block config, token MLP dimensions, sampler update, asset list, or
evaluation split. In this repo, the base-model gap is filled by the Cosmos/UniRelight
checkpoint family, while the TokenLight task/model/data logic stays separate.

See [docs/tokenlight_cosmos_base.md](docs/tokenlight_cosmos_base.md) for the
decision boundary.

## Layout

```text
configs/tokenlight_cosmos.toml    TokenLight config with Cosmos/UniRelight base
docs/tokenlight_cosmos_base.md    TokenLight-first base-model decision
docs/docker_workspace.md          Docker workspace and mount layout
docker/Dockerfile                 Minimal CUDA/PyTorch runtime image
src/tokenlight/model.py           DiT sequence model
src/tokenlight/tokenizer.py       Lighting attribute tokenizer
src/tokenlight/flow.py            Flow-matching training loss
src/tokenlight/sampler.py         Deterministic flow/DDIM-style sampler
src/tokenlight/data.py            Manifest and component-render datasets
src/tokenlight/color.py           Linear RGB composition and Reinhard tone mapping
src/tokenlight/canonical.py       Sim(3) camera-light canonical transform
scripts/train.py                  Training entrypoint
scripts/infer.py                  Inference entrypoint
scripts/evaluate.py               PSNR/SSIM/optional LPIPS evaluation
scripts/inspect_base.py           Cosmos base-file readiness check
scripts/blender_render_components.py  Blender component-render helper
```

## Paper-scale Config

`configs/tokenlight_cosmos.toml` uses the public TokenLight settings plus the
Cosmos base substitution:

- input resolution: `960`
- Cosmos latent: `16 x 120 x 120`
- patch size: `2`
- token grid: `60 x 60`
- hidden dim: `4096`
- depth: `28`, heads: `32`, matching Cosmos FADITV2 config
- transformer source: `cosmos_faditv2_tokenlight`
- bfloat16, AdamW, LR `1e-5`, WD `0.01`, betas `(0.9, 0.95)`
- global batch `160`, steps `15000`
- sampler steps `50`, light-token CFG scale `2`

This intentionally does not use UniRelight's `env_ldr/env_log/env_nrm` relighting
conditioner as the TokenLight condition interface. TokenLight light attributes
remain numeric tokens.

The Cosmos-backed model keeps TokenLight's sequence:

```text
[source image tokens] + [mask tokens] + [light tokens] + [noisy target tokens]
```

but replaces the native toy transformer with Cosmos/FADITV2-style modules:

```text
x_embedder
t_embedder
blocks.block*
final_layer
```

This naming is intentional so compatible tensors from Cosmos checkpoints can be
loaded into the TokenLight model.

## Manifest Format

Training manifests are JSONL files. Paths are relative to `--data-root`.

```json
{"source":"source/0001.png","target":"target/0001.png","mask":"mask/0001.png","attrs":{"a":0.8,"x":0.1,"y":0.4,"z":0.7,"r":1.0,"g":0.9,"b":0.7,"lambda":0.6,"d":0.3}}
```

Missing attributes are represented as null light tokens. This is used for real
capture pairs where color, precise light position, or diffuse controls are unknown.

Component-render manifests can be loaded through `ComponentRelightDataset` for
spatial/fixture supervision and `DiffuseSpreadDataset` for spread-control pairs.

## Base Check

Before training, check whether the Cosmos/UniRelight base assets referenced by
the config are present:

```powershell
python scripts/inspect_base.py --config configs/tokenlight_cosmos.toml
```

If files are missing, download the UniRelight/Cosmos checkpoints first. These
files are used as base VAE/backbone assets only; they do not change the
TokenLight condition interface.

## Train

```powershell
pip install -e .
python scripts/train.py --config configs/tokenlight_cosmos.toml --manifest data/train.jsonl --data-root data --output runs/tokenlight
```

For a real paper-scale run, download the Cosmos/UniRelight tokenizer/checkpoint
files referenced by the config and launch with distributed FSDP on the target
hardware.

## Inference

```powershell
python scripts/infer.py --config configs/tokenlight_cosmos.toml --checkpoint runs/tokenlight/latest.pt --source input.png --attrs attrs.json --output relit.png
```

The sampler implements the paper-compatible deterministic flow update:

```text
v_cond   = model(z, tau, I, DeltaL)
v_uncond = model(z, tau, I, drop(DeltaL))
v        = v_uncond + cfg_scale * (v_cond - v_uncond)
z_next   = z + dt * v
```

If the official sampler becomes available, replace `TokenLightSampler.step`.
