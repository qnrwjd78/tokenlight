#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATA_ROOT = "data/objaverse_ratio3p5_cube1p6_direct_scene0000_1999_640_png"
DEFAULT_IMAGE_KEYS = "video,input_image,pbr_depth_image,pbr_normal_image"
CACHE_VERSION = 1

torch = None
Image = None
DataLoader = None
tqdm = None
ModelConfig = None
WanVideoPipeline = None


def ensure_runtime_imports(*, include_model: bool) -> None:
    global torch, Image, DataLoader, tqdm, ModelConfig, WanVideoPipeline

    if torch is None or Image is None or DataLoader is None or tqdm is None:
        import torch as _torch
        from PIL import Image as _Image
        from torch.utils.data import DataLoader as _DataLoader
        from tqdm import tqdm as _tqdm

        torch = _torch
        Image = _Image
        DataLoader = _DataLoader
        tqdm = _tqdm

    if include_model and WanVideoPipeline is None:
        from diffsynth.pipelines.wan_video import ModelConfig as _ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline as _WanVideoPipeline

        ModelConfig = _ModelConfig
        WanVideoPipeline = _WanVideoPipeline


@dataclass(frozen=True)
class CacheAsset:
    path: str
    keys: tuple[str, ...]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def default_metadata_path(data_root: Path) -> Path:
    return REPO_ROOT / "data_train" / data_root.name / "metadata.jsonl"


def default_output_dir(data_root: Path) -> Path:
    return REPO_ROOT / "data_train" / data_root.name / "vae_latent_cache"


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_metadata_rows(path: Path, *, limit_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit_rows is not None and len(rows) >= limit_rows:
                break
    return rows


def canonical_asset_path(value: str, data_root: Path) -> tuple[str, Path]:
    raw = Path(value)
    abs_path = raw if raw.is_absolute() else data_root / raw
    if not raw.is_absolute():
        return raw.as_posix(), abs_path
    root = data_root.as_posix().rstrip("/")
    text = raw.as_posix()
    prefix = f"{root}/"
    if text.startswith(prefix):
        return text[len(prefix) :], abs_path
    return text, abs_path


def discover_assets(
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    image_keys: list[str],
    max_items: int | None,
) -> list[CacheAsset]:
    key_map: OrderedDict[str, set[str]] = OrderedDict()
    for row in rows:
        if row.get("valid") is False:
            continue
        for key in image_keys:
            value = row.get(key)
            if value in (None, ""):
                continue
            path, _ = canonical_asset_path(str(value), data_root)
            key_map.setdefault(path, set()).add(key)
            if max_items is not None and len(key_map) >= max_items:
                break
        if max_items is not None and len(key_map) >= max_items:
            break
    return [CacheAsset(path=path, keys=tuple(sorted(keys))) for path, keys in key_map.items()]


def cache_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:20]


class ImageAssetDataset:
    def __init__(self, assets: list[CacheAsset], *, data_root: Path, width: int, height: int) -> None:
        self.assets = assets
        self.data_root = data_root
        self.width = int(width)
        self.height = int(height)

    def __len__(self) -> int:
        return len(self.assets)

    def __getitem__(self, index: int) -> tuple[int, str, Any]:
        ensure_runtime_imports(include_model=False)
        asset = self.assets[index]
        path = Path(asset.path)
        path = path if path.is_absolute() else self.data_root / path
        image = Image.open(path).convert("RGB")
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        return index, asset.path, image


def collate_assets(items: list[tuple[int, str, Any]]) -> tuple[list[int], list[str], list[Any]]:
    indices, paths, images = zip(*items)
    return list(indices), list(paths), list(images)


def load_vae_pipe(args: argparse.Namespace) -> Any:
    ensure_runtime_imports(include_model=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    vae_path = repo_path(args.vae_path or Path(args.weights_dir) / "Wan2.2_VAE.pth")
    if not vae_path.exists():
        raise FileNotFoundError(f"Missing VAE checkpoint: {vae_path}")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16 if args.vae_dtype == "bf16" else torch.float32,
        device=device,
        model_configs=[ModelConfig(str(vae_path))],
        tokenizer_config=None,
    )
    pipe.load_models_to_device(["vae"])
    pipe.vae.eval()
    for param in pipe.vae.parameters():
        param.requires_grad_(False)
    return pipe


def preprocess_batch(pipe: Any, images: list[Any]) -> Any:
    videos = [
        pipe.preprocess_video(
            [image],
            torch_dtype=torch.float32,
            device=pipe.device,
        )
        for image in images
    ]
    return torch.cat(videos, dim=0)


def save_dtype(value: str) -> Any:
    ensure_runtime_imports(include_model=False)
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported save dtype: {value}")


def encode_batch(pipe: Any, images: list[Any], args: argparse.Namespace) -> Any:
    ensure_runtime_imports(include_model=False)
    with torch.no_grad():
        pixels = preprocess_batch(pipe, images).to(dtype=pipe.torch_dtype, device=pipe.device)
        latents = pipe.vae.encode(
            pixels,
            device=pipe.device,
            tiled=bool(args.vae_tiled),
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
        )
        return latents.to(dtype=save_dtype(args.save_dtype), device="cpu").contiguous()


def atomic_save_safetensors(tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    save_file(tensors, str(tmp_path), metadata=metadata)
    os.replace(tmp_path, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ("cache_config.json", "cache_summary.json", "index.jsonl"):
            try:
                (output_dir / name).unlink()
            except FileNotFoundError:
                pass
        for pattern in ("shard_*.safetensors", "*.tmp.*"):
            for path in (output_dir / "shards").glob(pattern):
                if path.is_file():
                    path.unlink()


def asset_counts(assets: list[CacheAsset]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for asset in assets:
        for key in asset.keys:
            counts[key] += 1
    return dict(sorted(counts.items()))


def dry_run_summary(args: argparse.Namespace, assets: list[CacheAsset], rows: list[dict[str, Any]]) -> dict[str, Any]:
    latent_h = (int(args.height) + int(args.latent_downsample) - 1) // int(args.latent_downsample)
    latent_w = (int(args.width) + int(args.latent_downsample) - 1) // int(args.latent_downsample)
    bytes_per_item = int(args.latent_channels) * latent_h * latent_w * int(args.dry_run_dtype_bytes)
    return {
        "cache_version": CACHE_VERSION,
        "dry_run": True,
        "row_count": len(rows),
        "asset_count": len(assets),
        "asset_counts_by_key": asset_counts(assets),
        "estimated_latent_shape_per_asset": [
            int(args.latent_channels),
            1,
            latent_h,
            latent_w,
        ],
        "estimated_bytes_per_asset": bytes_per_item,
        "estimated_total_bytes": bytes_per_item * len(assets),
        "image_keys": csv_items(args.image_keys),
    }


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_devices)

    data_root = repo_path(args.data_root)
    metadata_path = repo_path(args.metadata_path) if args.metadata_path else default_metadata_path(data_root)
    output_dir = repo_path(args.output_dir) if args.output_dir else default_output_dir(data_root)
    image_keys = csv_items(args.image_keys)

    rows = load_metadata_rows(metadata_path, limit_rows=args.limit_rows)
    assets = discover_assets(rows, data_root=data_root, image_keys=image_keys, max_items=args.max_items)
    if not assets:
        raise ValueError(f"No image assets found for keys {image_keys} in {metadata_path}")

    if args.dry_run:
        summary = dry_run_summary(args, assets, rows)
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
        return summary

    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    config = {
        "cache_version": CACHE_VERSION,
        "data_root": data_root.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "image_keys": image_keys,
        "height": int(args.height),
        "width": int(args.width),
        "weights_dir": repo_path(args.weights_dir).as_posix(),
        "vae_path": repo_path(args.vae_path or Path(args.weights_dir) / "Wan2.2_VAE.pth").as_posix(),
        "gpu_devices": args.gpu_devices,
        "device": args.device,
        "vae_dtype": args.vae_dtype,
        "save_dtype": args.save_dtype,
        "vae_tiled": bool(args.vae_tiled),
        "tile_size": list(args.tile_size),
        "tile_stride": list(args.tile_stride),
        "batch_size": int(args.batch_size),
        "shard_size": int(args.shard_size),
        "row_count": len(rows),
        "asset_count": len(assets),
        "asset_counts_by_key": asset_counts(assets),
    }
    write_json(output_dir / "cache_config.json", config)

    pipe = load_vae_pipe(args)
    dataset = ImageAssetDataset(assets, data_root=data_root, width=args.width, height=args.height)
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_assets,
        pin_memory=False,
        drop_last=False,
    )

    index_rows: list[dict[str, Any]] = []
    shard_tensors: dict[str, torch.Tensor] = {}
    shard_rows: list[dict[str, Any]] = []
    shard_index = 0

    def flush_shard() -> None:
        nonlocal shard_index, shard_tensors, shard_rows
        if not shard_tensors:
            return
        shard_rel = Path("shards") / f"shard_{shard_index:06d}.safetensors"
        shard_path = output_dir / shard_rel
        atomic_save_safetensors(
            shard_tensors,
            shard_path,
            metadata={
                "cache_version": str(CACHE_VERSION),
                "save_dtype": args.save_dtype,
                "height": str(args.height),
                "width": str(args.width),
            },
        )
        for row in shard_rows:
            row["shard"] = shard_rel.as_posix()
        index_rows.extend(shard_rows)
        shard_tensors = {}
        shard_rows = []
        shard_index += 1

    iterator = tqdm(dataloader, desc="encode vae latents")
    for batch_indices, batch_paths, batch_images in iterator:
        latents = encode_batch(pipe, batch_images, args)
        for item_offset, (asset_index, path) in enumerate(zip(batch_indices, batch_paths)):
            asset = assets[int(asset_index)]
            tensor_name = f"latent_{cache_id(asset.path)}"
            latent = latents[item_offset].contiguous()
            shard_tensors[tensor_name] = latent
            shard_rows.append(
                {
                    "asset_index": int(asset_index),
                    "path": path,
                    "keys": list(asset.keys),
                    "tensor": tensor_name,
                    "shape": list(latent.shape),
                    "dtype": str(latent.dtype).replace("torch.", ""),
                }
            )
            if len(shard_tensors) >= int(args.shard_size):
                flush_shard()
        iterator.set_postfix({"assets": len(index_rows) + len(shard_rows)})
    flush_shard()

    write_index(output_dir / "index.jsonl", index_rows)
    summary = {
        **config,
        "shard_count": shard_index,
        "index_path": (output_dir / "index.jsonl").as_posix(),
    }
    write_json(output_dir / "cache_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a sharded VAE latent cache from TokenLight metadata.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--metadata-path", default="")
    p.add_argument("--output-dir", "--output-path", "--output", default="")
    p.add_argument("--image-keys", default=DEFAULT_IMAGE_KEYS)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--weights-dir", default="weights/Wan2.2-TI2V-5B")
    p.add_argument("--vae-path", default="")
    p.add_argument(
        "--gpu-devices",
        "--gpu_devices",
        default="",
        help="Comma-separated GPU ids to expose through CUDA_VISIBLE_DEVICES, e.g. 0 or 2,3.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument("--vae-dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--save-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--vae-tiled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    p.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--latent-channels", type=int, default=48, help="Only used for --dry-run size estimates.")
    p.add_argument("--latent-downsample", type=int, default=8, help="Only used for --dry-run size estimates.")
    p.add_argument("--dry-run-dtype-bytes", type=int, default=2, help="Only used for --dry-run size estimates.")
    return p


def main() -> int:
    build_cache(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
