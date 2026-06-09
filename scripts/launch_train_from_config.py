from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

BOOLEAN_OPTIONAL_KEYS = {
    "tokenlight_light_tokens",
    "tokenlight_source_tokens",
    "tokenlight_mask_tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch TokenLight Wan training from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to a JSON launch/training config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument(
        "extra_train_args",
        nargs=argparse.REMAINDER,
        help="Optional args appended after the JSON train_args. Use `--` before them.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected JSON object in {config_path}")
    return config


def append_cli_args(command: list[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        option = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(option)
            elif key in BOOLEAN_OPTIONAL_KEYS:
                command.append(f"--no-{key}")
            continue
        if isinstance(value, list):
            command.append(option)
            command.extend(str(item) for item in value)
            continue
        command.extend([option, str(value)])


def build_command(config: dict[str, Any]) -> tuple[list[str], dict[str, str], Path]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in config.get("env", {}).items()})
    if config.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])

    working_dir = Path(config.get("working_dir", REPO_ROOT))
    if not working_dir.is_absolute():
        working_dir = REPO_ROOT / working_dir

    command = [str(config.get("accelerate_bin", "accelerate")), "launch"]
    append_cli_args(command, dict(config.get("accelerate", {})))
    command.append(str(config.get("train_script", "scripts/train.py")))
    append_cli_args(command, dict(config.get("train_args", {})))
    return command, env, working_dir


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    command, env, working_dir = build_command(config)
    extra = list(args.extra_train_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)

    visible_env = {
        key: env[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "DIFFSYNTH_MODEL_BASE_PATH",
            "DIFFSYNTH_SKIP_DOWNLOAD",
            "TOKENIZERS_PARALLELISM",
        )
        if key in env
    }
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in visible_env.items())
    printable = shlex.join(command)
    print(f"cd {shlex.quote(str(working_dir))}")
    print(f"{env_prefix} {printable}".strip())
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=working_dir, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
