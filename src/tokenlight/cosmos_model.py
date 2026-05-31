from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .config import BaseConfig, ModelConfig, TokenizerConfig, VAEConfig
from .model import patchify, unpatchify
from .tokenizer import LightTokenizer


def _load_cosmos_modules(repo_path: str | Path):
    repo = Path(repo_path)
    if not repo.exists():
        raise FileNotFoundError(f"Cosmos/UniRelight repo path does not exist: {repo}")
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        from cosmos_predict1.diffusion.training.module.blocks import (
            FinalLayer,
            GeneralDITTransformerBlock,
            PatchEmbed,
            SDXLTimesteps,
            SDXLTimestepEmbedding,
        )
    except Exception as exc:  # pragma: no cover - requires UniRelight/Cosmos env
        raise RuntimeError(
            "Could not import Cosmos FADITV2 building blocks. Install the "
            "UniRelight/Cosmos environment, including transformer-engine and "
            "megatron-core, before constructing the Cosmos-backed TokenLight model."
        ) from exc
    return FinalLayer, GeneralDITTransformerBlock, PatchEmbed, SDXLTimesteps, SDXLTimestepEmbedding


class TokenLightCosmosDiT(nn.Module):
    """TokenLight sequence model backed by Cosmos/FADITV2 DiT blocks.

    This class keeps TokenLight's paper interface:

      [source tokens] + [mask tokens] + [light tokens] + [noisy target tokens]

    and uses Cosmos modules only for the hidden base-model pieces:

      x_embedder, t_embedder, transformer blocks, final layer

    UniRelight's relighting conditioner, env_ldr/env_log/env_nrm interface, and
    basecolor formulation are not used.
    """

    def __init__(
        self,
        base_config: BaseConfig,
        vae_config: VAEConfig,
        model_config: ModelConfig,
        tokenizer_config: TokenizerConfig,
    ):
        super().__init__()
        if base_config.provider != "cosmos_unirelight":
            raise ValueError("TokenLightCosmosDiT requires base.provider='cosmos_unirelight'")
        (
            FinalLayer,
            GeneralDITTransformerBlock,
            PatchEmbed,
            SDXLTimesteps,
            SDXLTimestepEmbedding,
        ) = _load_cosmos_modules(base_config.repo_path)

        self.latent_channels = vae_config.latent_channels
        self.latent_size = vae_config.latent_size
        self.patch_size = model_config.patch_size
        self.hidden_dim = model_config.hidden_dim
        if self.latent_size % self.patch_size:
            raise ValueError("latent_size must be divisible by patch_size")
        self.grid_size = self.latent_size // self.patch_size

        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_size,
            temporal_patch_size=1,
            in_channels=self.latent_channels,
            out_channels=self.hidden_dim,
            bias=False,
            keep_spatio=True,
            legacy_patch_emb=True,
        )
        self.t_embedder = nn.Sequential(
            SDXLTimesteps(self.hidden_dim),
            SDXLTimestepEmbedding(
                self.hidden_dim,
                self.hidden_dim,
                use_adaln_lora=model_config.use_adaln_lora,
            ),
        )
        self.blocks = nn.ModuleDict(
            {
                f"block{idx}": GeneralDITTransformerBlock(
                    x_dim=self.hidden_dim,
                    context_dim=model_config.crossattn_emb_channels,
                    num_heads=model_config.num_heads,
                    block_config=model_config.block_config,
                    mlp_ratio=model_config.mlp_ratio,
                    window_sizes=[],
                    spatial_attn_win_size=1,
                    temporal_attn_win_size=1,
                    use_checkpoint=False,
                    x_format="BTHWD",
                    use_adaln_lora=model_config.use_adaln_lora,
                    adaln_lora_dim=model_config.adaln_lora_dim,
                    n_views=1,
                )
                for idx in range(model_config.depth)
            }
        )
        self.final_layer = FinalLayer(
            hidden_size=self.hidden_dim,
            spatial_patch_size=self.patch_size,
            temporal_patch_size=1,
            out_channels=self.latent_channels,
            use_adaln_lora=model_config.use_adaln_lora,
            adaln_lora_dim=model_config.adaln_lora_dim,
        )

        self.pos_embed = nn.Parameter(torch.randn(1, self.grid_size * self.grid_size, self.hidden_dim) * 0.02)
        self.source_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.mask_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.target_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.null_crossattn = nn.Parameter(
            torch.zeros(1, model_config.null_crossattn_tokens, model_config.crossattn_emb_channels)
        )
        self.timestep_scale = model_config.timestep_scale

        self.light_tokenizer = LightTokenizer(
            hidden_dim=self.hidden_dim,
            fourier_features=tokenizer_config.fourier_features,
            fourier_sigma=tokenizer_config.fourier_sigma,
            mlp_hidden_dim=tokenizer_config.mlp_hidden_dim,
        )

    def _pos(self, grid: tuple[int, int], dtype, device) -> torch.Tensor:
        grid_h, grid_w = grid
        if grid_h == self.grid_size and grid_w == self.grid_size:
            return self.pos_embed.to(device=device, dtype=dtype)
        pos = self.pos_embed.reshape(1, self.grid_size, self.grid_size, self.hidden_dim)
        pos = pos.permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.hidden_dim).to(dtype=dtype)

    def _image_tokens(self, latent: torch.Tensor, type_embed: torch.Tensor):
        if latent.ndim != 4:
            raise ValueError(f"Expected image latent [B,C,H,W], got {tuple(latent.shape)}")
        embedded = self.x_embedder(latent.unsqueeze(2))
        batch, _, grid_h, grid_w, hidden = embedded.shape
        tokens = embedded.reshape(batch, grid_h * grid_w, hidden)
        pos = self._pos((grid_h, grid_w), tokens.dtype, tokens.device)
        return tokens + pos + type_embed.to(device=tokens.device, dtype=tokens.dtype), (grid_h, grid_w)

    def _time_embedding(self, tau: torch.Tensor):
        time_input = tau.to(dtype=torch.float32, device=tau.device) * self.timestep_scale
        emb_B_D, adaln_lora_B_3D = self.t_embedder(time_input)
        return emb_B_D, adaln_lora_B_3D

    def forward(
        self,
        source_latent: torch.Tensor,
        noisy_target_latent: torch.Tensor,
        tau: torch.Tensor,
        light_attrs: Mapping[str, torch.Tensor | float | int] | None,
        mask_latent: torch.Tensor | None = None,
        drop_light: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        batch = source_latent.shape[0]
        source_tokens, grid = self._image_tokens(source_latent, self.source_type)
        target_tokens, target_grid = self._image_tokens(noisy_target_latent, self.target_type)
        if target_grid != grid:
            raise ValueError(f"Source grid {grid} and target grid {target_grid} must match")

        sequence = [source_tokens]
        if mask_latent is not None:
            mask_tokens, mask_grid = self._image_tokens(mask_latent, self.mask_type)
            if mask_grid != grid:
                raise ValueError(f"Mask grid {mask_grid} and source grid {grid} must match")
            sequence.append(mask_tokens)

        light_tokens, _ = self.light_tokenizer(
            light_attrs,
            batch_size=batch,
            device=source_latent.device,
            drop_light=drop_light,
        )
        sequence.extend([light_tokens.to(dtype=source_tokens.dtype), target_tokens])
        target_start = sum(part.shape[1] for part in sequence[:-1])
        x = torch.cat(sequence, dim=1)

        emb_B_D, adaln_lora_B_3D = self._time_embedding(tau.to(device=x.device))
        emb_B_D = emb_B_D.to(dtype=x.dtype)
        if adaln_lora_B_3D is not None:
            adaln_lora_B_3D = adaln_lora_B_3D.to(dtype=x.dtype)
        crossattn = self.null_crossattn.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1)

        # Cosmos blocks expect B,T,H,W,D. We keep TokenLight's flat sequence and
        # expose it as a degenerate 1 x 1 x N grid so full attention spans every
        # source/mask/light/target token.
        x_5d = x.unsqueeze(1).unsqueeze(1)
        for block in self.blocks.values():
            x_5d = block(
                x_5d,
                emb_B_D=emb_B_D,
                crossattn_emb=crossattn,
                crossattn_mask=None,
                rope_emb_L_1_1_D=None,
                adaln_lora_B_3D=adaln_lora_B_3D,
            )
        x = x_5d.squeeze(1).squeeze(1)
        target_features = x[:, target_start:]
        patch_tokens = self.final_layer(target_features, emb_B_D, adaln_lora_B_3D)
        return unpatchify(patch_tokens, self.latent_channels, grid, self.patch_size)


def is_cosmos_source(model_config: ModelConfig) -> bool:
    return model_config.source.lower() in {"cosmos_faditv2_tokenlight", "cosmos", "faditv2"}
