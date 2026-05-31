from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


LIGHT_TOKEN_NAMES = ("a", "dg", "x", "y", "z", "r", "g", "b", "lambda", "d", "t")

ALIASES = {
    "ambient": "a",
    "ambient_scale": "a",
    "diffuse": "dg",
    "diffuse_delta": "dg",
    "px": "x",
    "py": "y",
    "pz": "z",
    "red": "r",
    "green": "g",
    "blue": "b",
    "intensity": "lambda",
    "lam": "lambda",
    "lambda_": "lambda",
    "radius": "d",
    "spread": "d",
    "transition": "t",
}


class GaussianFourierFeatures(nn.Module):
    def __init__(self, features: int, sigma: float):
        super().__init__()
        if features % 2 != 0:
            raise ValueError("Fourier feature dimension must be even.")
        weight = torch.randn(features // 2) * sigma
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.view(value.shape[0], 1)
        projected = 2.0 * math.pi * value * self.weight.view(1, -1)
        return torch.cat([projected.sin(), projected.cos()], dim=-1)


class ScalarAttributeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, fourier_features: int, sigma: float, mlp_hidden_dim: int):
        super().__init__()
        self.fourier = GaussianFourierFeatures(fourier_features, sigma)
        self.net = nn.Sequential(
            nn.Linear(fourier_features, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(self.fourier(value))


class LightTokenizer(nn.Module):
    """Tokenize TokenLight lighting controls as numeric attribute tokens."""

    token_names = LIGHT_TOKEN_NAMES

    def __init__(
        self,
        hidden_dim: int,
        fourier_features: int = 512,
        fourier_sigma: float = 5.0,
        mlp_hidden_dim: int | None = None,
    ):
        super().__init__()
        mlp_hidden_dim = mlp_hidden_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleDict(
            {
                name: ScalarAttributeEncoder(
                    hidden_dim=hidden_dim,
                    fourier_features=fourier_features,
                    sigma=fourier_sigma,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for name in self.token_names
            }
        )
        self.type_embeddings = nn.Parameter(torch.zeros(len(self.token_names), hidden_dim))
        self.null_tokens = nn.Parameter(torch.randn(len(self.token_names), hidden_dim) * 0.02)

    def _canonicalize(self, attrs: Mapping[str, torch.Tensor | float | int] | None) -> dict[str, torch.Tensor]:
        if attrs is None:
            return {}
        canonical: dict[str, torch.Tensor] = {}
        for key, value in attrs.items():
            canonical[ALIASES.get(key, key)] = value
        return canonical

    def _batch_size(self, attrs: Mapping[str, torch.Tensor], batch_size: int | None) -> int:
        if batch_size is not None:
            return batch_size
        for value in attrs.values():
            value = torch.as_tensor(value)
            if value.ndim == 0:
                continue
            return value.shape[0]
        raise ValueError("batch_size is required when no batched attributes are provided.")

    def _drop_mask(self, drop_light, batch_size: int, device) -> torch.Tensor:
        if isinstance(drop_light, bool):
            return torch.full((batch_size,), drop_light, dtype=torch.bool, device=device)
        mask = torch.as_tensor(drop_light, dtype=torch.bool, device=device)
        if mask.ndim == 0:
            mask = mask.expand(batch_size)
        if mask.shape[0] != batch_size:
            raise ValueError(f"drop_light must have batch {batch_size}, got {tuple(mask.shape)}")
        return mask

    def _value(self, raw, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(raw, dtype=torch.float32, device=device)
        if value.ndim == 0:
            value = value.expand(batch_size)
        elif value.ndim > 1:
            value = value.view(value.shape[0], -1)[:, 0]
        if value.shape[0] != batch_size:
            raise ValueError(f"Attribute batch must be {batch_size}, got {tuple(value.shape)}")
        missing = torch.isnan(value)
        value = torch.where(missing, torch.zeros_like(value), value)
        return value, missing

    def forward(
        self,
        attrs: Mapping[str, torch.Tensor | float | int] | None,
        batch_size: int | None = None,
        device=None,
        drop_light: bool | torch.Tensor = False,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        attrs = self._canonicalize(attrs)
        if device is None:
            device = self.type_embeddings.device
        batch_size = self._batch_size(attrs, batch_size)
        drop_mask = self._drop_mask(drop_light, batch_size, device)
        outputs = []
        for index, name in enumerate(self.token_names):
            null = self.null_tokens[index].view(1, self.hidden_dim).expand(batch_size, -1)
            if name not in attrs:
                token = null
            else:
                value, missing = self._value(attrs[name], batch_size, device)
                token = self.encoders[name](value) + self.type_embeddings[index].view(1, -1)
                replace = missing | drop_mask
                token = torch.where(replace.view(batch_size, 1), null, token)
            outputs.append(token.unsqueeze(1))
        return torch.cat(outputs, dim=1), self.token_names
