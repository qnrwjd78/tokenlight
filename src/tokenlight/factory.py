from __future__ import annotations

from .config import TokenLightConfig
from .cosmos_base import assert_tokenlight_first_base_config
from .cosmos_model import TokenLightCosmosDiT, is_cosmos_source
from .model import TokenLightDiT


def build_model(config: TokenLightConfig):
    assert_tokenlight_first_base_config(config.base)
    if is_cosmos_source(config.model):
        return TokenLightCosmosDiT(config.base, config.vae, config.model, config.tokenizer)
    if config.model.source.lower() in {"tokenlight_sequence_dit", "native"}:
        return TokenLightDiT(config.vae, config.model, config.tokenizer)
    raise ValueError(f"Unknown model.source: {config.model.source}")
