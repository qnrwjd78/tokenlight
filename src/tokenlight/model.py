from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig, TokenizerConfig, VAEConfig
from .tokenizer import LightTokenizer


def patchify(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, channels, height, width = x.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"Latent shape {(height, width)} is not divisible by patch size {patch_size}")
    grid_h = height // patch_size
    grid_w = width // patch_size
    x = x.reshape(batch, channels, grid_h, patch_size, grid_w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(batch, grid_h * grid_w, channels * patch_size * patch_size), (grid_h, grid_w)


def unpatchify(tokens: torch.Tensor, channels: int, grid: tuple[int, int], patch_size: int) -> torch.Tensor:
    batch, _, patch_dim = tokens.shape
    expected = channels * patch_size * patch_size
    if patch_dim != expected:
        raise ValueError(f"Patch dim {patch_dim} does not match expected {expected}")
    grid_h, grid_w = grid
    x = tokens.reshape(batch, grid_h, grid_w, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(batch, channels, grid_h * patch_size, grid_w * patch_size)


def timestep_embedding(tau: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    tau = tau.float().view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=tau.device) / max(half, 1)
    )
    args = tau[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.net(timestep_embedding(tau, self.hidden_dim))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 4))

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.ada(time_cond).chunk(4, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = modulate(self.norm2(x), shift2, scale2)
        return x + self.mlp(h)


class TokenLightDiT(nn.Module):
    """TokenLight DiT sequence model.

    Sequence order:
      [source image tokens] + [mask tokens if present] + [light tokens] + [noisy target tokens]

    Source, mask, and target patches use the same 2D positional embedding at
    corresponding patch coordinates, matching the paper description.
    """

    def __init__(
        self,
        vae_config: VAEConfig,
        model_config: ModelConfig,
        tokenizer_config: TokenizerConfig,
    ):
        super().__init__()
        self.latent_channels = vae_config.latent_channels
        self.latent_size = vae_config.latent_size
        self.patch_size = model_config.patch_size
        self.hidden_dim = model_config.hidden_dim
        if self.latent_size % self.patch_size:
            raise ValueError("latent_size must be divisible by patch_size")
        self.grid_size = self.latent_size // self.patch_size
        patch_dim = self.latent_channels * self.patch_size * self.patch_size

        self.source_proj = nn.Linear(patch_dim, self.hidden_dim)
        self.target_proj = nn.Linear(patch_dim, self.hidden_dim)
        self.mask_proj = nn.Linear(patch_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, patch_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, self.grid_size * self.grid_size, self.hidden_dim) * 0.02)
        self.source_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.mask_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.target_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.light_tokenizer = LightTokenizer(
            hidden_dim=self.hidden_dim,
            fourier_features=tokenizer_config.fourier_features,
            fourier_sigma=tokenizer_config.fourier_sigma,
            mlp_hidden_dim=tokenizer_config.mlp_hidden_dim,
        )
        self.time_embed = TimestepMLP(self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=model_config.num_heads,
                    mlp_ratio=model_config.mlp_ratio,
                    dropout=model_config.dropout,
                )
                for _ in range(model_config.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def _pos(self, grid: tuple[int, int], dtype, device) -> torch.Tensor:
        grid_h, grid_w = grid
        if grid_h == self.grid_size and grid_w == self.grid_size:
            return self.pos_embed.to(device=device, dtype=dtype)
        pos = self.pos_embed.reshape(1, self.grid_size, self.grid_size, self.hidden_dim)
        pos = pos.permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.hidden_dim).to(dtype=dtype)

    def _image_tokens(self, latent: torch.Tensor, projection: nn.Linear, type_embed: torch.Tensor):
        patches, grid = patchify(latent, self.patch_size)
        pos = self._pos(grid, patches.dtype, patches.device)
        return projection(patches) + pos + type_embed.to(device=patches.device, dtype=patches.dtype), grid

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
        source_tokens, grid = self._image_tokens(source_latent, self.source_proj, self.source_type)
        target_tokens, target_grid = self._image_tokens(noisy_target_latent, self.target_proj, self.target_type)
        if target_grid != grid:
            raise ValueError(f"Source grid {grid} and target grid {target_grid} must match")
        sequence = [source_tokens]
        if mask_latent is not None:
            mask_tokens, mask_grid = self._image_tokens(mask_latent, self.mask_proj, self.mask_type)
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
        time_cond = self.time_embed(tau.to(device=x.device)).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, time_cond)
        x = self.norm(x)
        target_out = self.out_proj(x[:, target_start:])
        return unpatchify(target_out, self.latent_channels, grid, self.patch_size)
