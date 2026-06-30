#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.build_vae_latent_cache as base
from model.illumination_latent_head import make_illumination_image_tensor, unit_to_vae_range


CACHE_VERSION = 1


def default_output_dir(data_root: Path, target: str) -> Path:
    return REPO_ROOT / "data_train" / data_root.name / "illum_latent_cache" / target


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for pattern in (
        "cache_config.json",
        "cache_summary.json",
        "cache_summary_part_*.json",
        "index.jsonl",
        "index_part_*.jsonl",
    ):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    for path in (output_dir / "shards").glob("*.safetensors"):
        if path.is_file():
            path.unlink()
    for path in (output_dir / "shards").glob("*.tmp.*"):
        if path.is_file():
            path.unlink()


def preprocess_unit_batch(pipe: Any, images: list[Any]) -> Any:
    base.ensure_runtime_imports(include_model=False)
    videos = [
        pipe.preprocess_video(
            [image],
            torch_dtype=base.torch.float32,
            device=pipe.device,
            min_value=0,
            max_value=1,
        )
        for image in images
    ]
    return base.torch.cat(videos, dim=0)


def encode_illumination_batch(pipe: Any, images: list[Any], args: argparse.Namespace) -> Any:
    base.ensure_runtime_imports(include_model=False)
    with base.torch.no_grad():
        rgb_unit = preprocess_unit_batch(pipe, images)
        illum_unit = make_illumination_image_tensor(rgb_unit, target=args.target, eps=float(args.eps))
        illum_vae = unit_to_vae_range(illum_unit)
        latents = pipe.vae.encode(
            illum_vae.to(dtype=pipe.torch_dtype, device=pipe.device),
            device=pipe.device,
            tiled=bool(args.vae_tiled),
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
        )
        return latents.to(dtype=base.save_dtype(args.save_dtype), device="cpu").contiguous()


def cache_config(
    args: argparse.Namespace,
    *,
    data_root: Path,
    metadata_path: Path,
    output_dir: Path,
    image_keys: list[str],
    rows: list[dict[str, Any]],
    all_assets: list[base.CacheAsset],
    assets: list[base.CacheAsset],
) -> dict[str, Any]:
    config = base.cache_config(
        args,
        data_root=data_root,
        metadata_path=metadata_path,
        output_dir=output_dir,
        image_keys=image_keys,
        rows=rows,
        all_assets=all_assets,
        assets=assets,
    )
    config.update(
        {
            "cache_kind": "illumination_latent_cache",
            "cache_version": CACHE_VERSION,
            "target": args.target,
            "eps": float(args.eps),
        }
    )
    return config


def dry_run_summary(args: argparse.Namespace, assets: list[base.CacheAsset], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = base.dry_run_summary(args, assets, rows)
    summary.update(
        {
            "cache_kind": "illumination_latent_cache",
            "target": args.target,
            "eps": float(args.eps),
        }
    )
    return summary


def tensor_name_for(target: str, path: str) -> str:
    return f"latent_{target}_{base.cache_id(f'{target}:{path}')}"


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_devices)

    data_root = base.repo_path(args.data_root)
    metadata_path = base.repo_path(args.metadata_path) if args.metadata_path else base.default_metadata_path(data_root)
    output_dir = base.repo_path(args.output_dir) if args.output_dir else default_output_dir(data_root, args.target)
    image_keys = base.csv_items(args.image_keys)

    rows = base.load_metadata_rows(metadata_path, limit_rows=args.limit_rows)
    all_assets = base.discover_assets(rows, data_root=data_root, image_keys=image_keys, max_items=args.max_items)
    assets = base.partition_assets(
        all_assets,
        partition_count=int(args.partition_count),
        partition_index=int(args.partition_index),
    )
    if not all_assets:
        raise ValueError(f"No image assets found for keys {image_keys} in {metadata_path}")

    if args.dry_run:
        summary = dry_run_summary(args, assets, rows)
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
        return summary

    if not args.skip_output_prepare:
        prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    config = cache_config(
        args,
        data_root=data_root,
        metadata_path=metadata_path,
        output_dir=output_dir,
        image_keys=image_keys,
        rows=rows,
        all_assets=all_assets,
        assets=assets,
    )
    if not args.skip_output_prepare:
        base.write_json(output_dir / "cache_config.json", config)

    base.ensure_runtime_imports(include_model=False)
    pipe = base.load_vae_pipe(args)
    dataset = base.ImageAssetDataset(assets, data_root=data_root, width=args.width, height=args.height)
    dataloader = base.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=base.collate_assets,
        pin_memory=False,
        drop_last=False,
    )

    index_rows: list[dict[str, Any]] = []
    shard_tensors: dict[str, Any] = {}
    shard_rows: list[dict[str, Any]] = []
    shard_index = 0

    def flush_shard() -> None:
        nonlocal shard_index, shard_tensors, shard_rows
        if not shard_tensors:
            return
        shard_rel = Path("shards") / f"{args.shard_prefix}shard_{shard_index:06d}.safetensors"
        shard_path = output_dir / shard_rel
        base.atomic_save_safetensors(
            shard_tensors,
            shard_path,
            metadata={
                "cache_kind": "illumination_latent_cache",
                "cache_version": str(CACHE_VERSION),
                "target": str(args.target),
                "eps": str(args.eps),
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

    iterator = base.tqdm(dataloader, desc=f"encode {args.target} vae latents")
    for batch_indices, batch_paths, batch_images in iterator:
        latents = encode_illumination_batch(pipe, batch_images, args)
        for item_offset, (asset_index, path) in enumerate(zip(batch_indices, batch_paths)):
            asset = assets[int(asset_index)]
            tensor_name = tensor_name_for(args.target, asset.path)
            latent = latents[item_offset].contiguous()
            shard_tensors[tensor_name] = latent
            shard_rows.append(
                {
                    "asset_index": int(asset.asset_index),
                    "partition_index": int(args.partition_index),
                    "path": path,
                    "keys": list(asset.keys),
                    "target": args.target,
                    "eps": float(args.eps),
                    "tensor": tensor_name,
                    "shape": list(latent.shape),
                    "dtype": str(latent.dtype).replace("torch.", ""),
                }
            )
            if len(shard_tensors) >= int(args.shard_size):
                flush_shard()
        iterator.set_postfix({"assets": len(index_rows) + len(shard_rows)})
    flush_shard()

    index_name = (
        f"index_part_{int(args.partition_index):02d}.jsonl"
        if int(args.partition_count) > 1
        else "index.jsonl"
    )
    base.write_index(output_dir / index_name, index_rows)
    summary = {
        **config,
        "shard_count": shard_index,
        "index_path": (output_dir / index_name).as_posix(),
    }
    summary_name = (
        f"cache_summary_part_{int(args.partition_index):02d}.json"
        if int(args.partition_count) > 1
        else "cache_summary.json"
    )
    base.write_json(output_dir / summary_name, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return summary


def gpu_device_items(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def should_launch_multi_gpu(args: argparse.Namespace) -> bool:
    return (
        not bool(args.multi_gpu_worker)
        and not bool(args.dry_run)
        and int(args.partition_count) == 1
        and len(gpu_device_items(args.gpu_devices)) > 1
    )


def build_worker_command(args: argparse.Namespace, *, gpu: str, partition_count: int, partition_index: int) -> list[str]:
    script = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script),
        "--data-root",
        str(args.data_root),
        "--metadata-path",
        str(args.metadata_path),
        "--output-dir",
        str(args.output_dir),
        "--image-keys",
        str(args.image_keys),
        "--target",
        str(args.target),
        "--eps",
        str(args.eps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--weights-dir",
        str(args.weights_dir),
        "--vae-path",
        str(args.vae_path),
        "--gpu-devices",
        str(gpu),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--shard-size",
        str(args.shard_size),
        "--vae-dtype",
        str(args.vae_dtype),
        "--save-dtype",
        str(args.save_dtype),
        "--tile-size",
        str(args.tile_size[0]),
        str(args.tile_size[1]),
        "--tile-stride",
        str(args.tile_stride[0]),
        str(args.tile_stride[1]),
        "--partition-count",
        str(partition_count),
        "--partition-index",
        str(partition_index),
        "--shard-prefix",
        f"part_{partition_index:02d}_",
        "--skip-output-prepare",
        "--multi-gpu-worker",
    ]
    cmd.append("--vae-tiled" if args.vae_tiled else "--no-vae-tiled")
    if args.limit_rows is not None:
        cmd.extend(["--limit-rows", str(args.limit_rows)])
    if args.max_items is not None:
        cmd.extend(["--max-items", str(args.max_items)])
    return cmd


def launch_multi_gpu(args: argparse.Namespace) -> dict[str, Any]:
    devices = gpu_device_items(args.gpu_devices)
    if len(devices) <= 1:
        return build_cache(args)

    data_root = base.repo_path(args.data_root)
    metadata_path = base.repo_path(args.metadata_path) if args.metadata_path else base.default_metadata_path(data_root)
    output_dir = base.repo_path(args.output_dir) if args.output_dir else default_output_dir(data_root, args.target)
    image_keys = base.csv_items(args.image_keys)

    rows = base.load_metadata_rows(metadata_path, limit_rows=args.limit_rows)
    assets = base.discover_assets(rows, data_root=data_root, image_keys=image_keys, max_items=args.max_items)
    if not assets:
        raise ValueError(f"No image assets found for keys {image_keys} in {metadata_path}")

    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    config = cache_config(
        args,
        data_root=data_root,
        metadata_path=metadata_path,
        output_dir=output_dir,
        image_keys=image_keys,
        rows=rows,
        all_assets=assets,
        assets=assets,
    )
    config["gpu_devices"] = ",".join(devices)
    config["multi_gpu_worker_count"] = len(devices)
    base.write_json(output_dir / "cache_config.json", config)

    print(
        f"[illum-cache] launching {len(devices)} GPU workers: {','.join(devices)} "
        f"target={args.target} assets={len(assets)} output={output_dir}",
        flush=True,
    )
    processes = []
    for partition_index, gpu in enumerate(devices):
        cmd = build_worker_command(
            args,
            gpu=gpu,
            partition_count=len(devices),
            partition_index=partition_index,
        )
        processes.append((gpu, subprocess.Popen(cmd, cwd=REPO_ROOT)))

    failed: list[tuple[str, int]] = []
    for gpu, process in processes:
        code = process.wait()
        if code != 0:
            failed.append((gpu, code))
    if failed:
        detail = ", ".join(f"gpu {gpu}: exit {code}" for gpu, code in failed)
        raise RuntimeError(f"Illumination latent cache worker failed ({detail})")

    merged_rows = base.merge_partition_indexes(output_dir, partition_count=len(devices))
    shard_count = len(list((output_dir / "shards").glob("*.safetensors")))
    summary = {
        **config,
        "shard_count": shard_count,
        "index_path": (output_dir / "index.jsonl").as_posix(),
        "merged_index_rows": len(merged_rows),
    }
    base.write_json(output_dir / "cache_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build sharded luminance/log-luminance VAE latent cache from TokenLight metadata.")
    p.add_argument("--data-root", default=base.DEFAULT_DATA_ROOT)
    p.add_argument("--metadata-path", default="")
    p.add_argument("--output-dir", "--output-path", "--output", default="")
    p.add_argument("--image-keys", default="video")
    p.add_argument("--target", choices=("luminance", "log_luminance"), required=True)
    p.add_argument("--eps", type=float, default=1e-3)
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
    p.add_argument("--partition-count", type=int, default=1)
    p.add_argument("--partition-index", type=int, default=0)
    p.add_argument("--shard-prefix", default="")
    p.add_argument("--skip-output-prepare", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--multi-gpu-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--latent-channels", type=int, default=48, help="Only used for --dry-run size estimates.")
    p.add_argument("--latent-downsample", type=int, default=8, help="Only used for --dry-run size estimates.")
    p.add_argument("--dry-run-dtype-bytes", type=int, default=2, help="Only used for --dry-run size estimates.")
    return p


def main() -> int:
    args = parser().parse_args()
    if should_launch_multi_gpu(args):
        launch_multi_gpu(args)
    else:
        build_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
