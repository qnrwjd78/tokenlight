# TokenLight Reproduction With Cosmos Base

This repository is for reproducing the TokenLight paper. UniRelight is not the
method, data format, conditioner, objective, or inference API we are trying to
reproduce.

The only role of `repos/unirelight` is to provide a concrete public
Cosmos-based text/video diffusion base model in place of TokenLight's
unpublished "pretrained text-to-video checkpoint".

`repos/unirelight` is read-only. Do not edit files under that directory.

## Non-Negotiable Goal

Implement TokenLight as described in the paper:

```text
input image I
lighting edit DeltaL
target relit image Ir

VAE latent:
  X = VAE(Ir)
  eps ~ N(0, I)
  z_tau = (1 - tau) eps + tau X

network:
  u_theta(z_tau, tau, I, DeltaL)

target:
  X - eps

loss:
  || u_theta(...) - (X - eps) ||^2
```

The model sequence stays TokenLight-style:

```text
[source image tokens]
+ [mask latent tokens if present]
+ [light attribute tokens]
+ [noisy target tokens]
```

with full self-attention and output only on target-token positions.

## What Cosmos/UniRelight Is Used For

Use Cosmos/UniRelight only for the hidden base-model components that TokenLight
does not publicly identify:

```text
pretrained text/video diffusion prior
VAE/tokenizer
DiT config and pretrained weights
sampler conventions if needed
```

Concrete source:

```text
repo path: repos/unirelight
base config name: unirelight_cosmos_f57_480p
Cosmos net family: faditv2_7b
VAE/tokenizer: cosmos_diffusion_tokenizer_comp8x8x8
latent channels: 16
spatial compression: 8x
model width: 4096
blocks: 28
heads: 32
```

Relevant reference files, read-only:

```text
repos/unirelight/cosmos_predict1/diffusion/config/base/net.py
repos/unirelight/cosmos_predict1/diffusion/training/config/base/vae.py
repos/unirelight/cosmos_predict1/diffusion/training/config/video2world_relight/exp_unirelight.py
repos/unirelight/cosmos_predict1/diffusion/training/networks/general_dit.py
```

## What We Do Not Use From UniRelight

Do not base the TokenLight implementation on UniRelight's relighting method:

```text
do not use env_ldr/env_log/env_nrm as the TokenLight condition interface
do not use basecolor decomposition as the TokenLight target formulation
do not use UniRelight's rgb_ref/input/basecolor state layout as the paper model
do not replace TokenLight light tokens with environment-map-only controls
do not replace TokenLight flow-matching target with UniRelight's training objective
```

UniRelight can supply weights and modules. It should not define the task.

## TokenLight Paper Mapping

| TokenLight component | This repo implementation | Cosmos/UniRelight role |
|---|---|---|
| source image tokens | `src/tokenlight/model.py` | VAE can come from Cosmos |
| mask latent tokens | `src/tokenlight/model.py` | VAE can come from Cosmos |
| scalar/vector light tokens | `src/tokenlight/tokenizer.py` | none |
| full self-attention sequence | `src/tokenlight/cosmos_model.py` | Cosmos/FADITV2 blocks are used as the base transformer |
| flow matching `X - eps` | `src/tokenlight/flow.py` | none |
| light-only CFG | `src/tokenlight/sampler.py` | sampler schedule may be adjusted later |
| component render equations | `src/tokenlight/color.py`, `src/tokenlight/data.py` | none |
| canonical light coordinates | `src/tokenlight/canonical.py` | none |

## Base Substitution Caveat

TokenLight's paper hints at a 960px latent around `12 x 120 x 120`, but the
available Cosmos tokenizer from UniRelight uses `16` latent channels and 8x
spatial compression. For a 960px input this gives:

```text
16 x 120 x 120
```

This is an intentional base substitution because the official TokenLight VAE is
not public.

## Implementation Rule

Every new feature should answer this question:

```text
Is this TokenLight paper logic, or is this only a way to initialize the hidden base?
```

If it is TokenLight paper logic, implement it in this repo.

If it is only base-model initialization, read from `repos/unirelight` without
modifying it.

## Current Cosmos-Backed Architecture

The active config uses:

```text
model.source = "cosmos_faditv2_tokenlight"
```

This constructs `TokenLightCosmosDiT`, which keeps the TokenLight paper sequence
but backs the transformer with Cosmos module names:

```text
x_embedder      Cosmos PatchEmbed
t_embedder      Cosmos SDXL timestep embedding
blocks.block*   Cosmos GeneralDITTransformerBlock
final_layer     Cosmos FinalLayer
```

The flat TokenLight token sequence is exposed to Cosmos blocks as a degenerate
`B x 1 x 1 x N x D` grid. Full attention therefore still spans source, mask,
light, and target tokens together.

Cross-attention exists in the Cosmos `FA-CA-MLP` block, but TokenLight does not
use text or UniRelight environment conditioning. The model provides learned null
cross-attention tokens so the architecture remains checkpoint-compatible without
changing the TokenLight light-token interface.
