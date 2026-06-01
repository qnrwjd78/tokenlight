import torch

from tokenlight.canonical import transform_camera_light
from tokenlight.color import compose_relight
from tokenlight.config import ModelConfig, TokenizerConfig, VAEConfig
from tokenlight.data import RelightingComponentAdapterDataset
from tokenlight.flow import flow_matching_loss
from tokenlight.model import TokenLightDiT
from tokenlight.sampler import TokenLightSampler
from tokenlight.vae import IdentityVAE


def tiny_model():
    vae = VAEConfig(adapter="identity", image_size=8, latent_channels=3, latent_size=8)
    model = ModelConfig(hidden_dim=32, depth=1, num_heads=4, mlp_ratio=2.0, patch_size=2)
    tokenizer = TokenizerConfig(fourier_features=16, fourier_sigma=5.0, mlp_hidden_dim=32)
    return TokenLightDiT(vae, model, tokenizer), IdentityVAE()


def test_canonical_scale_rules():
    out = transform_camera_light(
        camera=[0, 0, 1],
        light=[1, 0, 0],
        intensity=2.0,
        radius=0.5,
        canonical_center=[0, 0, 0],
        target_center=[10, 0, 0],
        scale=3.0,
    )
    assert torch.allclose(out.light, torch.tensor([13.0, 0.0, 0.0]))
    assert torch.allclose(out.intensity, torch.tensor(18.0))
    assert torch.allclose(out.radius, torch.tensor(1.5))


def test_compose_relight_shape():
    ambient = torch.ones(3, 4, 4)
    contrib = torch.ones(3, 4, 4) * 0.5
    out = compose_relight(ambient, contrib, 0.8, 0.5, torch.tensor([1.0, 0.5, 0.25]))
    assert out.shape == ambient.shape
    assert out.min() >= 0 and out.max() <= 1


def test_relighting_component_condition_mapping():
    attrs = RelightingComponentAdapterDataset.attrs_from_condition(
        {
            "task": "spatial",
            "ambient_scale": 0.7,
            "lights": [
                {
                    "position": [0.1, -0.2, 0.8],
                    "color": [1.0, 0.9, 0.7],
                    "intensity": 0.5,
                    "radius": 0.06,
                }
            ],
        }
    )
    assert attrs == {
        "a": 0.7,
        "x": 0.1,
        "y": -0.2,
        "z": 0.8,
        "r": 1.0,
        "g": 0.9,
        "b": 0.7,
        "lambda": 0.5,
        "d": 0.06,
    }


def test_model_flow_and_sampler_smoke():
    model, vae = tiny_model()
    batch = {
        "source": torch.rand(2, 3, 8, 8),
        "target": torch.rand(2, 3, 8, 8),
        "attrs": {
            "a": torch.tensor([0.8, 0.9]),
            "x": torch.tensor([0.1, 0.2]),
            "y": torch.tensor([0.2, 0.3]),
            "z": torch.tensor([0.3, 0.4]),
            "lambda": torch.tensor([0.5, 0.6]),
        },
    }
    loss, metrics = flow_matching_loss(model, vae, batch, light_dropout_prob=0.5)
    assert loss.ndim == 0
    assert "mse" in metrics
    sampler = TokenLightSampler(model, vae, steps=2, cfg_scale=2.0)
    image = sampler.sample(batch["source"][:1], {key: value[:1] for key, value in batch["attrs"].items()})
    assert image.shape == (1, 3, 8, 8)
