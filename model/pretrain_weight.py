from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def wan22_weight_dir(path: str | Path | None = None) -> Path:
    return (Path(path) if path else project_root() / "weights" / "Wan2.2-TI2V-5B").resolve()


def wan22_model_paths(path: str | Path | None = None) -> list[object]:
    root = wan22_weight_dir(path)
    shards = sorted(root.glob("diffusion_pytorch_model-*.safetensors"))
    return [
        [str(item) for item in shards],
        str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        str(root / "Wan2.2_VAE.pth"),
    ]


def wan22_tokenizer_path(path: str | Path | None = None) -> str:
    return str(wan22_weight_dir(path) / "google" / "umt5-xxl")


def validate_wan22_weights(path: str | Path | None = None) -> None:
    root = wan22_weight_dir(path)
    required = [
        root / "diffusion_pytorch_model-00001-of-00003.safetensors",
        root / "diffusion_pytorch_model-00002-of-00003.safetensors",
        root / "diffusion_pytorch_model-00003-of-00003.safetensors",
        root / "diffusion_pytorch_model.safetensors.index.json",
        root / "models_t5_umt5-xxl-enc-bf16.pth",
        root / "Wan2.2_VAE.pth",
        root / "google" / "umt5-xxl" / "tokenizer.json",
        root / "google" / "umt5-xxl" / "spiece.model",
    ]
    missing = [item for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError("Missing Wan2.2-TI2V-5B weight file(s): " + ", ".join(str(item) for item in missing))
