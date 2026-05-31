from __future__ import annotations

from collections.abc import Mapping

import torch


class TokenLightSampler:
    """Deterministic flow/DDIM-style sampler for TokenLight.

    The official paper sampler is described as DDIM with 50 steps, but the exact
    update is unpublished. This class isolates the update so it can be replaced
    with the official sampler when available.
    """

    def __init__(self, model, vae, steps: int = 50, cfg_scale: float = 2.0):
        self.model = model
        self.vae = vae
        self.steps = steps
        self.cfg_scale = cfg_scale

    @torch.no_grad()
    def sample(
        self,
        source_image: torch.Tensor,
        light_attrs: Mapping[str, torch.Tensor | float | int] | None,
        mask: torch.Tensor | None = None,
        latent_shape: tuple[int, int, int, int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        source_latent = self.vae.encode(source_image)
        mask_latent = self.vae.encode(mask) if mask is not None else None
        if latent_shape is None:
            latent_shape = tuple(source_latent.shape)
        z = torch.randn(latent_shape, device=source_latent.device, dtype=source_latent.dtype, generator=generator)
        batch = z.shape[0]
        times = torch.linspace(0.0, 1.0, self.steps + 1, device=z.device, dtype=z.dtype)
        for index in range(self.steps):
            tau = torch.full((batch,), times[index].item(), device=z.device, dtype=z.dtype)
            dt = times[index + 1] - times[index]
            v_cond = self.model(source_latent, z, tau, light_attrs, mask_latent=mask_latent, drop_light=False)
            if self.cfg_scale == 1.0:
                v = v_cond
            else:
                v_uncond = self.model(source_latent, z, tau, light_attrs, mask_latent=mask_latent, drop_light=True)
                v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            z = z + dt * v
        return self.vae.decode(z).clamp(0.0, 1.0)
