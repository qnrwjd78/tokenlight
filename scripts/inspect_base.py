from __future__ import annotations

import argparse

from tokenlight.config import load_config
from tokenlight.cosmos_base import assert_tokenlight_first_base_config, inspect_cosmos_base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenlight_cosmos.toml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    assert_tokenlight_first_base_config(cfg.base)
    report = inspect_cosmos_base(cfg.base, cfg.vae)
    print(f"provider: {report.provider}")
    print(f"repo: {report.repo_path}")
    print(f"checkpoint: {report.checkpoint_path}")
    print(f"config_name: {report.config_name}")
    print(f"net_family: {report.net_family}")
    print(f"tokenizer_encoder: {report.tokenizer_encoder}")
    print(f"tokenizer_decoder: {report.tokenizer_decoder}")
    print(f"tokenizer_mean_std: {report.tokenizer_mean_std}")
    if report.ready:
        print("status: ready")
    else:
        print("status: missing files")
        for path in report.missing:
            print(f"missing: {path}")


if __name__ == "__main__":
    main()
