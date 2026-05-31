from __future__ import annotations

import torch


def reinhard_tonemap(linear_rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    linear_rgb = torch.clamp(linear_rgb, min=0.0)
    return linear_rgb / (1.0 + linear_rgb + eps)


def _broadcast_scalar(value, target: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=target.dtype, device=target.device)
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def _broadcast_color(color, target: torch.Tensor) -> torch.Tensor:
    color = torch.as_tensor(color, dtype=target.dtype, device=target.device)
    if color.ndim == 1:
        color = color.view(3, 1, 1)
    elif color.ndim == 2:
        color = color.view(color.shape[0], color.shape[1], 1, 1)
    while color.ndim < target.ndim:
        color = color.unsqueeze(0)
    return color


def compose_relight(
    ambient: torch.Tensor,
    contribution: torch.Tensor,
    ambient_scale,
    intensity,
    color,
    tone_map: bool = True,
) -> torch.Tensor:
    """Compose Ir = T(a I + lambda c O) in linear RGB."""
    a = _broadcast_scalar(ambient_scale, ambient)
    lam = _broadcast_scalar(intensity, ambient)
    c = _broadcast_color(color, ambient)
    linear = a * ambient + lam * c * contribution
    if tone_map:
        return reinhard_tonemap(linear).clamp(0.0, 1.0)
    return linear


def compose_diffuse_pair(
    ambient: torch.Tensor,
    source_spread: torch.Tensor,
    target_spread: torch.Tensor,
    ambient_scale,
    ambient_color,
    intensity,
    light_color,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose the paper's spread-control pair.

    I  = T(a c1 A + lambda c2 O1)
    Ir = T(a c1 A + lambda c2 O2)
    """
    a = _broadcast_scalar(ambient_scale, ambient)
    lam = _broadcast_scalar(intensity, ambient)
    c1 = _broadcast_color(ambient_color, ambient)
    c2 = _broadcast_color(light_color, ambient)
    source = reinhard_tonemap(a * c1 * ambient + lam * c2 * source_spread).clamp(0.0, 1.0)
    target = reinhard_tonemap(a * c1 * ambient + lam * c2 * target_spread).clamp(0.0, 1.0)
    return source, target
