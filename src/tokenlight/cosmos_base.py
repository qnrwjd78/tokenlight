from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import BaseConfig, VAEConfig


@dataclass(frozen=True)
class CosmosBaseReport:
    provider: str
    repo_path: Path
    checkpoint_path: Path
    config_name: str
    net_family: str
    tokenizer_encoder: Path
    tokenizer_decoder: Path
    tokenizer_mean_std: Path
    missing: tuple[Path, ...]

    @property
    def ready(self) -> bool:
        return len(self.missing) == 0


def inspect_cosmos_base(base: BaseConfig, vae: VAEConfig) -> CosmosBaseReport:
    """Inspect the Cosmos/UniRelight files used only as TokenLight base assets."""
    repo_path = Path(base.repo_path or vae.repo_path)
    checkpoint_path = Path(base.checkpoint_path)
    encoder_path = Path(vae.encoder_path)
    decoder_path = Path(vae.decoder_path)
    mean_std_path = Path(vae.mean_std_path)
    required = (repo_path, checkpoint_path, encoder_path, decoder_path, mean_std_path)
    missing = tuple(path for path in required if not path.exists())
    return CosmosBaseReport(
        provider=base.provider,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_name=base.config_name,
        net_family=base.net_family,
        tokenizer_encoder=encoder_path,
        tokenizer_decoder=decoder_path,
        tokenizer_mean_std=mean_std_path,
        missing=missing,
    )


def assert_tokenlight_first_base_config(base: BaseConfig) -> None:
    """Prevent accidentally switching to the UniRelight relighting conditioner."""
    if base.provider == "cosmos_unirelight" and base.use_unirelight_relighting_conditioner:
        raise ValueError(
            "Invalid configuration: UniRelight/Cosmos may be used only as the "
            "TokenLight base model. Set use_unirelight_relighting_conditioner=false."
        )
