from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

from .config import VAEConfig


class VAEAdapter(nn.Module):
    def encode(self, image: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class IdentityVAE(VAEAdapter):
    """Use tensors as already-encoded latents.

    This is useful for tests and for precomputed-latent manifests. It is not a
    substitute for the paper's unpublished VAE.
    """

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class TorchScriptVAE(VAEAdapter):
    def __init__(self, path: str | Path, scaling_factor: float = 1.0):
        super().__init__()
        self.module = torch.jit.load(str(path), map_location="cpu")
        self.scaling_factor = scaling_factor

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self.module, "encode"):
            latent = self.module.encode(image)
        else:
            latent = self.module(image)
        return latent * self.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        scaled = latent / self.scaling_factor
        if hasattr(self.module, "decode"):
            return self.module.decode(scaled)
        raise RuntimeError("TorchScript VAE does not expose a decode method.")


class DiffusersVAE(VAEAdapter):
    def __init__(self, model_id_or_path: str, scaling_factor: float | None = None):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install with `pip install -e .[diffusers]` to use a diffusers VAE.") from exc
        self.vae = AutoencoderKL.from_pretrained(model_id_or_path)
        self.scaling_factor = scaling_factor or getattr(self.vae.config, "scaling_factor", 1.0)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(image).latent_dist
        return posterior.sample() * self.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.scaling_factor).sample


class CosmosUniRelightImageVAE(VAEAdapter):
    """Cosmos tokenizer adapter sourced from `repos/unirelight`.

    This uses the Cosmos tokenizer files as a base VAE only. It does not use
    UniRelight's relighting conditioner, environment-map interface, or training
    objective.
    """

    def __init__(
        self,
        repo_path: str | Path,
        encoder_path: str | Path,
        decoder_path: str | Path,
        mean_std_path: str | Path,
        latent_channels: int = 16,
        is_bf16: bool = True,
    ):
        super().__init__()
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Cosmos/UniRelight repo path does not exist: {repo_path}")
        for required in (encoder_path, decoder_path, mean_std_path):
            if not Path(required).exists():
                raise FileNotFoundError(
                    f"Missing Cosmos tokenizer file: {required}. "
                    "Download UniRelight/Cosmos checkpoints before building the VAE."
                )
        repo_str = str(repo_path.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            from cosmos_predict1.diffusion.training.module.pretrained_vae_base import JITVAE
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            raise RuntimeError(
                "Could not import Cosmos JITVAE from repos/unirelight. Install the "
                "UniRelight/Cosmos environment first."
            ) from exc
        self.vae = JITVAE(
            enc_fp=str(encoder_path),
            dec_fp=str(decoder_path),
            name="cosmos_diffusion_tokenizer_comp8x8x8",
            mean_std_fp=str(mean_std_path),
            latent_ch=latent_channels,
            is_image=True,
            is_bf16=is_bf16,
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent)


def build_vae(config: VAEConfig) -> VAEAdapter:
    adapter = config.adapter.lower()
    if adapter == "identity":
        return IdentityVAE()
    if adapter == "torchscript":
        if not config.path:
            raise ValueError("vae.path is required for adapter='torchscript'")
        return TorchScriptVAE(config.path, scaling_factor=config.scaling_factor)
    if adapter == "diffusers":
        model_id = config.model_id or config.path
        if not model_id:
            raise ValueError("vae.model_id or vae.path is required for adapter='diffusers'")
        return DiffusersVAE(model_id, scaling_factor=config.scaling_factor)
    if adapter == "cosmos_unirelight":
        return CosmosUniRelightImageVAE(
            repo_path=config.repo_path,
            encoder_path=config.encoder_path,
            decoder_path=config.decoder_path,
            mean_std_path=config.mean_std_path,
            latent_channels=config.latent_channels,
            is_bf16=True,
        )
    raise ValueError(f"Unknown VAE adapter: {config.adapter}")
