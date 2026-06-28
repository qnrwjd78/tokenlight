from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


IlluminationTarget = Literal["luminance", "log_luminance"]
IlluminationHeadArch = Literal["lite", "resunet"]
IlluminationLossType = Literal["mse", "smooth_l1"]


@dataclass
class IlluminationLatentHeadConfig:
    latent_channels: int = 48
    target: IlluminationTarget = "luminance"
    arch: IlluminationHeadArch = "lite"
    hidden_channels: int = 128
    mid_channels: int = 192
    bottleneck_channels: int = 256
    lite_blocks: int = 4
    eps: float = 1e-3
    projected_residual: bool = True
    normalize_loss: bool = True
    loss_type: IlluminationLossType = "mse"
    cosine_weight: float = 0.1
    multiscale_weights: tuple[float, ...] = (1.0, 0.5, 0.25)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IlluminationLatentHeadConfig":
        values = dict(data)
        if "multiscale_weights" in values:
            values["multiscale_weights"] = tuple(float(item) for item in values["multiscale_weights"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["multiscale_weights"] = list(self.multiscale_weights)
        return data


def _check_target(target: str) -> IlluminationTarget:
    if target not in {"luminance", "log_luminance"}:
        raise ValueError(f"Unknown illumination target `{target}`")
    return target  # type: ignore[return-value]


def _check_arch(arch: str) -> IlluminationHeadArch:
    if arch not in {"lite", "resunet"}:
        raise ValueError(f"Unknown illumination head arch `{arch}`")
    return arch  # type: ignore[return-value]


def _check_loss_type(loss_type: str) -> IlluminationLossType:
    if loss_type not in {"mse", "smooth_l1"}:
        raise ValueError(f"Unknown illumination loss type `{loss_type}`")
    return loss_type  # type: ignore[return-value]


def unit_rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype)
    view_shape = [1] * rgb.ndim
    view_shape[1] = 3
    weights = weights.view(*view_shape)
    return (rgb[:, :3] * weights).sum(dim=1, keepdim=True)


def make_illumination_image_tensor(
    unit_rgb: torch.Tensor,
    *,
    target: IlluminationTarget = "luminance",
    eps: float = 1e-3,
) -> torch.Tensor:
    """Return a 3-channel [0, 1] illumination image/video tensor.

    `unit_rgb` is expected to be `[B, 3, ...]` in image range `[0, 1]`.
    The output has the same spatial/video dimensions as the input and 3 channels.
    """

    target = _check_target(target)
    luminance = unit_rgb_to_luminance(unit_rgb).clamp(0.0, 1.0)
    if target == "log_luminance":
        eps = float(eps)
        log_eps = torch.log(luminance.new_tensor(eps))
        illum = (torch.log(luminance.clamp_min(eps)) - log_eps) / (-log_eps)
        illum = illum.clamp(0.0, 1.0)
    else:
        illum = luminance
    return illum.expand(-1, 3, *illum.shape[2:]).contiguous()


def unit_to_vae_range(unit_tensor: torch.Tensor) -> torch.Tensor:
    return unit_tensor.mul(2.0).sub(1.0)


def vae_range_to_unit(vae_tensor: torch.Tensor) -> torch.Tensor:
    return vae_tensor.add(1.0).mul(0.5).clamp(0.0, 1.0)


def _norm_groups(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        groups = _norm_groups(channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LiteIlluminationLatentHead(nn.Module):
    def __init__(self, config: IlluminationLatentHeadConfig) -> None:
        super().__init__()
        c = int(config.latent_channels)
        h = int(config.hidden_channels)
        self.projected_residual = bool(config.projected_residual)
        self.base = nn.Conv2d(c, c, 1) if self.projected_residual else None
        blocks: list[nn.Module] = [
            nn.Conv2d(c, h, 3, padding=1),
            nn.SiLU(),
        ]
        blocks += [ResBlock2d(h) for _ in range(int(config.lite_blocks))]
        blocks += [
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(),
            nn.Conv2d(h, c, 3, padding=1),
        ]
        self.delta = nn.Sequential(*blocks)

    def forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.delta(x)
        if self.base is None:
            return delta
        return self.base(x) + delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_2d_head(self.forward_2d, x)


class ResUNetIlluminationLatentHead(nn.Module):
    def __init__(self, config: IlluminationLatentHeadConfig) -> None:
        super().__init__()
        c = int(config.latent_channels)
        h = int(config.hidden_channels)
        m = int(config.mid_channels)
        b = int(config.bottleneck_channels)
        self.projected_residual = bool(config.projected_residual)
        self.base = nn.Conv2d(c, c, 1) if self.projected_residual else None

        self.in_proj = nn.Conv2d(c, h, 3, padding=1)
        self.enc0 = nn.Sequential(ResBlock2d(h), ResBlock2d(h))
        self.down1 = nn.Conv2d(h, m, 3, stride=2, padding=1)
        self.enc1 = nn.Sequential(ResBlock2d(m), ResBlock2d(m))
        self.down2 = nn.Conv2d(m, b, 3, stride=2, padding=1)
        self.mid = nn.Sequential(
            ResBlock2d(b, dilation=1),
            ResBlock2d(b, dilation=2),
            ResBlock2d(b, dilation=4),
        )
        self.up1 = nn.Conv2d(b, m, 3, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(m + m, m, 3, padding=1), ResBlock2d(m))
        self.up0 = nn.Conv2d(m, h, 3, padding=1)
        self.dec0 = nn.Sequential(nn.Conv2d(h + h, h, 3, padding=1), ResBlock2d(h))
        self.out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(),
            nn.Conv2d(h, c, 3, padding=1),
        )

    def forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(self.in_proj(x))
        x1 = self.enc1(self.down1(x0))
        xm = self.mid(self.down2(x1))
        y1 = F.interpolate(xm, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(torch.cat([self.up1(y1), x1], dim=1))
        y0 = F.interpolate(y1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y0 = self.dec0(torch.cat([self.up0(y0), x0], dim=1))
        delta = self.out(y0)
        if self.base is None:
            return delta
        return self.base(x) + delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_2d_head(self.forward_2d, x)


def _apply_2d_head(forward_2d, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return forward_2d(x)
    if x.ndim != 5:
        raise ValueError(f"Expected latent tensor [B,C,H,W] or [B,C,F,H,W], got {tuple(x.shape)}")
    b, c, f, h, w = x.shape
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    y_2d = forward_2d(x_2d)
    return y_2d.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def build_illumination_latent_head(
    config: IlluminationLatentHeadConfig | dict[str, Any] | None = None,
    **overrides,
) -> nn.Module:
    if config is None:
        cfg = IlluminationLatentHeadConfig()
    elif isinstance(config, IlluminationLatentHeadConfig):
        cfg = IlluminationLatentHeadConfig.from_dict(config.to_dict())
    else:
        cfg = IlluminationLatentHeadConfig.from_dict(config)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.target = _check_target(cfg.target)
    cfg.arch = _check_arch(cfg.arch)
    cfg.loss_type = _check_loss_type(cfg.loss_type)
    if cfg.arch == "lite":
        return LiteIlluminationLatentHead(cfg)
    if cfg.arch == "resunet":
        return ResUNetIlluminationLatentHead(cfg)
    raise ValueError(f"Unknown illumination head arch `{cfg.arch}`")


def _pool_latent(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if x.ndim == 5:
        b, c, f, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        y = F.avg_pool2d(y, kernel_size=kernel_size, stride=kernel_size)
        return y.reshape(b, f, c, y.shape[-2], y.shape[-1]).permute(0, 2, 1, 3, 4).contiguous()
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)


def _pointwise_latent_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: IlluminationLossType) -> torch.Tensor:
    loss_type = _check_loss_type(loss_type)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred.float(), target.float())
    return F.mse_loss(pred.float(), target.float())


def _stat_view(stat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    stat = stat.to(device=ref.device, dtype=ref.dtype)
    if stat.ndim == 1:
        return stat.view(1, stat.shape[0], *([1] * (ref.ndim - 2)))
    while stat.ndim < ref.ndim:
        stat = stat.unsqueeze(-1)
    if stat.ndim != ref.ndim:
        raise ValueError(f"Cannot broadcast stat shape {tuple(stat.shape)} to latent shape {tuple(ref.shape)}")
    return stat


def normalize_illumination_latents(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mean is None or std is None:
        return pred.float(), target.float()
    mean_view = _stat_view(mean, target)
    std_view = _stat_view(std, target).clamp_min(float(eps))
    return (pred.float() - mean_view.float()) / std_view.float(), (target.float() - mean_view.float()) / std_view.float()


def latent_multiscale_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: tuple[float, ...] | list[float] = (1.0, 0.5, 0.25),
    *,
    loss_type: IlluminationLossType = "mse",
) -> torch.Tensor:
    total = pred.new_zeros(())
    for level, weight in enumerate(weights):
        kernel_size = 2**level
        if min(pred.shape[-2:]) < kernel_size:
            continue
        x = _pool_latent(pred.float(), kernel_size)
        y = _pool_latent(target.float(), kernel_size)
        total = total + float(weight) * _pointwise_latent_loss(x, y, loss_type)
    return total


def latent_multiscale_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: tuple[float, ...] | list[float] = (1.0, 0.5, 0.25),
) -> torch.Tensor:
    return latent_multiscale_loss(pred, target, weights, loss_type="mse")


def latent_cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(pred.float().flatten(1), dim=1, eps=eps)
    y = F.normalize(target.float().flatten(1), dim=1, eps=eps)
    return 1.0 - (x * y).sum(dim=1).mean()


def illumination_latent_head_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    multiscale_weights: tuple[float, ...] | list[float] = (1.0, 0.5, 0.25),
    cosine_weight: float = 0.1,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    normalize: bool = True,
    norm_eps: float = 1e-6,
    loss_type: IlluminationLossType = "mse",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw_mse = F.mse_loss(pred.float(), target.float())
    if normalize:
        pred_loss, target_loss = normalize_illumination_latents(pred, target, mean=mean, std=std, eps=norm_eps)
    else:
        pred_loss, target_loss = pred.float(), target.float()
    point = _pointwise_latent_loss(pred_loss, target_loss, _check_loss_type(loss_type))
    ms = latent_multiscale_loss(pred_loss, target_loss, multiscale_weights, loss_type=loss_type)
    cos = latent_cosine_loss(pred_loss, target_loss)
    loss = ms + float(cosine_weight) * cos
    return loss, {
        "loss": loss.detach(),
        "base_loss": point.detach(),
        "raw_mse": raw_mse.detach(),
        "multiscale_loss": ms.detach(),
        "cosine": cos.detach(),
    }


def save_illumination_head_checkpoint(
    path: str | Path,
    head: nn.Module,
    config: IlluminationLatentHeadConfig,
    *,
    latent_mean: torch.Tensor | None = None,
    latent_std: torch.Tensor | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    if latent_mean is not None:
        state["latent_mean"] = latent_mean.detach().cpu()
    if latent_std is not None:
        state["latent_std"] = latent_std.detach().cpu()
    metadata = {
        "illumination_head_config": json.dumps(config.to_dict(), sort_keys=True),
    }
    if extra:
        metadata["extra"] = json.dumps(extra, sort_keys=True, default=str)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state, str(path), metadata=metadata)
    else:
        torch.save({"state_dict": state, "config": config.to_dict(), "extra": extra or {}}, path)


def load_illumination_head_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, IlluminationLatentHeadConfig, dict[str, Any]]:
    path = Path(path)
    extra: dict[str, Any] = {}
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        state = load_file(str(path), device=str(map_location))
        with safe_open(str(path), framework="pt", device=str(map_location)) as f:
            metadata = f.metadata() or {}
        config_json = metadata.get("illumination_head_config")
        if not config_json:
            raise ValueError(f"{path} does not contain illumination_head_config metadata")
        config = IlluminationLatentHeadConfig.from_dict(json.loads(config_json))
        if metadata.get("extra"):
            extra = json.loads(metadata["extra"])
    else:
        loaded = torch.load(path, map_location=map_location)
        state = loaded.get("state_dict", loaded)
        config = IlluminationLatentHeadConfig.from_dict(loaded["config"])
        extra = dict(loaded.get("extra", {}))
    latent_mean = state.pop("latent_mean", None)
    latent_std = state.pop("latent_std", None)
    if latent_mean is not None:
        extra["latent_mean"] = latent_mean
    if latent_std is not None:
        extra["latent_std"] = latent_std
    head = build_illumination_latent_head(config)
    head.load_state_dict(state, strict=strict)
    return head, config, extra
