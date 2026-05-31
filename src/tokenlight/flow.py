from __future__ import annotations

import torch
from torch.nn import functional as F


def flow_interpolant(target_latent: torch.Tensor, tau: torch.Tensor, noise: torch.Tensor | None = None):
    if noise is None:
        noise = torch.randn_like(target_latent)
    tau_view = tau.view(tau.shape[0], *([1] * (target_latent.ndim - 1)))
    z_tau = (1.0 - tau_view) * noise + tau_view * target_latent
    velocity = target_latent - noise
    return z_tau, velocity, noise


def flow_matching_loss(
    model,
    vae,
    batch: dict,
    light_dropout_prob: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    source = batch["source"]
    target = batch["target"]
    mask = batch.get("mask")
    attrs = batch.get("attrs", {})

    source_latent = vae.encode(source)
    target_latent = vae.encode(target)
    mask_latent = vae.encode(mask) if mask is not None else None

    batch_size = target_latent.shape[0]
    tau = torch.rand(batch_size, device=target_latent.device, dtype=target_latent.dtype)
    z_tau, velocity, _ = flow_interpolant(target_latent, tau)

    if light_dropout_prob > 0:
        drop_light = torch.rand(batch_size, device=target_latent.device) < light_dropout_prob
    else:
        drop_light = False

    pred = model(
        source_latent=source_latent,
        noisy_target_latent=z_tau,
        tau=tau,
        light_attrs=attrs,
        mask_latent=mask_latent,
        drop_light=drop_light,
    )
    loss = F.mse_loss(pred.float(), velocity.float())
    return loss, {"mse": float(loss.detach().cpu())}
