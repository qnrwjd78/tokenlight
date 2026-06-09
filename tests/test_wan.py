import torch

from tokenlight.wan import (
    TokenLightAttributeTokenEncoder,
    attrs_json,
    gradient_checkpoint_forward_compatible,
    light_attrs_to_prompt,
    parse_attrs_json,
)


def test_prompt_hides_numeric_values_by_default():
    attrs = {"a": 0.7, "x": 0.1, "y": -0.2, "z": 0.5, "lambda": 1.1}
    prompt = light_attrs_to_prompt(attrs, task="spatial")
    assert "0.700" not in prompt
    assert "point-light edit" in prompt


def test_attrs_json_roundtrip():
    encoded = attrs_json({"ambient_scale": 0.5, "intensity": 1.2})
    decoded = parse_attrs_json(encoded)
    assert decoded["a"] == 0.5
    assert decoded["lambda"] == 1.2


def test_light_tokens_are_dit_tokens_with_null_dropout():
    encoder = TokenLightAttributeTokenEncoder(token_dim=32, fourier_features=8, hidden_dim=16)
    assert encoder.fourier.weight.shape == (len(encoder.token_names), 8)
    assert encoder.fourier.out_dim == 16
    assert not torch.equal(encoder.fourier.weight[0], encoder.fourier.weight[1])

    attrs = [{"a": 0.5, "x": 0.2}, {"dg": -0.3, "r": 1.0}]
    out = encoder(attrs)
    assert out.shape == (2, len(encoder.token_names), 32)
    assert not torch.all(out[0, 0] == -1)
    assert torch.all(out[0, 1] == -1)

    dropped = encoder(attrs, drop_light=True)
    assert torch.all(dropped == -1)


def test_zero3_checkpoint_path_handles_inputs_without_grad():
    block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU(), torch.nn.Linear(4, 4))
    for param in block.parameters():
        param.ds_id = 0

    x = torch.randn(2, 4)
    out = gradient_checkpoint_forward_compatible(block, True, True, x)
    out.square().mean().backward()

    assert x.requires_grad is False
    assert block[0].weight.grad is not None
    assert block[2].weight.grad is not None
