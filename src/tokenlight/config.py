from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


@dataclass
class BaseConfig:
    provider: str = "native"
    repo_path: str = ""
    checkpoint_path: str = ""
    config_name: str = ""
    net_family: str = ""
    use_unirelight_relighting_conditioner: bool = False
    init_backbone: bool = True


@dataclass
class VAEConfig:
    adapter: str = "identity"
    image_size: int = 960
    latent_channels: int = 16
    latent_size: int = 120
    spatial_compression: int = 8
    scaling_factor: float = 1.0
    path: str = ""
    model_id: str = ""
    repo_path: str = ""
    encoder_path: str = ""
    decoder_path: str = ""
    mean_std_path: str = ""


@dataclass
class ModelConfig:
    hidden_dim: int = 4096
    depth: int = 28
    num_heads: int = 32
    mlp_ratio: float = 4.0
    patch_size: int = 2
    dropout: float = 0.0
    source: str = "cosmos_faditv2_tokenlight"
    block_config: str = "FA-CA-MLP"
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 256
    crossattn_emb_channels: int = 4096
    null_crossattn_tokens: int = 2
    timestep_scale: float = 1000.0


@dataclass
class TokenizerConfig:
    fourier_features: int = 512
    fourier_sigma: float = 5.0
    mlp_hidden_dim: int = 4096


@dataclass
class TrainingConfig:
    precision: str = "bf16"
    global_batch_size: int = 160
    steps: int = 15000
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    light_dropout_prob: float = 0.1
    grad_clip_norm: float = 1.0
    save_every: int = 1000


@dataclass
class SamplerConfig:
    steps: int = 50
    cfg_scale: float = 2.0


@dataclass
class DataConfig:
    linear_rgb: bool = True
    tone_mapper: str = "reinhard"


@dataclass
class TokenLightConfig:
    base: BaseConfig = field(default_factory=BaseConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _dataclass_from_section(cls, raw: dict | None):
    raw = raw or {}
    valid = {f.name for f in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in valid})


def load_config(path: str | Path) -> TokenLightConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return TokenLightConfig(
        base=_dataclass_from_section(BaseConfig, raw.get("base")),
        vae=_dataclass_from_section(VAEConfig, raw.get("vae")),
        model=_dataclass_from_section(ModelConfig, raw.get("model")),
        tokenizer=_dataclass_from_section(TokenizerConfig, raw.get("tokenizer")),
        training=_dataclass_from_section(TrainingConfig, raw.get("training")),
        sampler=_dataclass_from_section(SamplerConfig, raw.get("sampler")),
        data=_dataclass_from_section(DataConfig, raw.get("data")),
    )
