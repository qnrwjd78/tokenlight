from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from tokenlight.checkpoint import load_checkpoint
from tokenlight.config import load_config
from tokenlight.cosmos_base import assert_tokenlight_first_base_config, inspect_cosmos_base
from tokenlight.data import load_tensor_image
from tokenlight.factory import build_model
from tokenlight.sampler import TokenLightSampler
from tokenlight.vae import build_vae


def save_image(tensor: torch.Tensor, path: str | Path):
    image = tensor.detach().cpu().clamp(0, 1)
    if image.ndim == 4:
        image = image[0]
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--attrs", required=True, help="JSON file containing lighting attributes")
    parser.add_argument("--mask", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    assert_tokenlight_first_base_config(cfg.base)
    if cfg.base.provider == "cosmos_unirelight":
        report = inspect_cosmos_base(cfg.base, cfg.vae)
        if not report.ready:
            missing = "\n".join(f"  - {path}" for path in report.missing)
            raise FileNotFoundError(f"Missing Cosmos/UniRelight base files:\n{missing}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, strict=True)
    vae = build_vae(cfg.vae).to(device).eval()

    source = load_tensor_image(args.source).unsqueeze(0).to(device)
    mask = load_tensor_image(args.mask).unsqueeze(0).to(device) if args.mask else None
    with Path(args.attrs).open("r", encoding="utf-8") as handle:
        attrs = json.load(handle)
    attrs = {key: torch.tensor([float(value)], device=device) for key, value in attrs.items() if value is not None}
    sampler = TokenLightSampler(
        model,
        vae,
        steps=args.steps or cfg.sampler.steps,
        cfg_scale=args.cfg_scale or cfg.sampler.cfg_scale,
    )
    with torch.no_grad():
        output = sampler.sample(source, attrs, mask=mask)
    save_image(output, args.output)


if __name__ == "__main__":
    main()
