#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

from model.illumination_latent_head import (
    IlluminationLatentHeadConfig,
    build_illumination_latent_head,
    illumination_latent_head_loss,
    make_illumination_image_tensor,
    save_illumination_head_checkpoint,
    unit_to_vae_range,
)


DEFAULT_DATA_ROOT = "data/objaverse_ratio3p5_cube1p6_direct_scene0000_1999_640_png"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def default_metadata_path(data_root: Path) -> Path:
    return REPO_ROOT / "data_train" / data_root.name / "metadata.jsonl"


def default_latent_stats_path(data_root: Path, args: argparse.Namespace) -> Path:
    image_key = str(args.image_key).replace("/", "_")
    task = str(args.task).replace(",", "-") if args.task else "all"
    stats_items = args.stats_max_items if args.stats_max_items is not None else args.max_items
    item_count = f"n{stats_items}" if stats_items is not None else "all"
    seed = f"seed{args.shuffle_seed}" if stats_items is not None and args.shuffle_seed is not None else "seedall"
    name = f"{args.target}_{image_key}_{task}_{args.width}x{args.height}_{item_count}_{seed}.pt"
    return REPO_ROOT / "data_train" / data_root.name / "illum_latent_stats" / name


def load_metadata_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_image_rows(data_root: Path) -> list[dict[str, str]]:
    rows = []
    for scene_dir in sorted((data_root / "scenes").glob("scene_*")):
        for path in sorted((scene_dir / "samples").glob("*.png")):
            rows.append({"video": path.relative_to(data_root).as_posix()})
    return rows


def select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_root = repo_path(args.data_root)
    metadata_path = repo_path(args.metadata_path) if args.metadata_path else default_metadata_path(data_root)
    rows = load_metadata_rows(metadata_path) if metadata_path.exists() else discover_image_rows(data_root)
    if args.task:
        allowed = {item.strip() for item in args.task.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("task", "")) in allowed]
    rows = [row for row in rows if row.get(args.image_key)]
    if args.shuffle_seed is not None:
        rng = random.Random(int(args.shuffle_seed))
        rng.shuffle(rows)
    if args.max_items is not None:
        rows = rows[: int(args.max_items)]
    if not rows:
        raise ValueError(f"No rows with image key `{args.image_key}` under {data_root}")
    return rows


class IlluminationImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], *, data_root: Path, image_key: str, width: int, height: int) -> None:
        self.rows = rows
        self.data_root = data_root
        self.image_key = image_key
        self.width = int(width)
        self.height = int(height)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Image.Image:
        row = self.rows[index]
        path = Path(str(row[self.image_key]))
        path = path if path.is_absolute() else self.data_root / path
        image = Image.open(path).convert("RGB")
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        return image


def collate_images(images: list[Image.Image]) -> list[Image.Image]:
    return images


def load_vae_pipe(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    vae_path = repo_path(args.vae_path or Path(args.weights_dir) / "Wan2.2_VAE.pth")
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


def preprocess_batch(pipe, images: list[Image.Image], *, min_value: float, max_value: float) -> torch.Tensor:
    videos = [
        pipe.preprocess_video(
            [image],
            torch_dtype=torch.float32,
            device=pipe.device,
            min_value=min_value,
            max_value=max_value,
        )
        for image in images
    ]
    return torch.cat(videos, dim=0)


@torch.no_grad()
def encode_latents(pipe, video_tensor: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    latents = pipe.vae.encode(
        video_tensor.to(dtype=pipe.torch_dtype, device=pipe.device),
        device=pipe.device,
        tiled=bool(args.vae_tiled),
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
    )
    return latents.float()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def acquire_file_lock(lock_path: Path, *, poll_seconds: float = 5.0) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd
        except FileExistsError:
            print(f"Waiting for illumination latent stats lock: {lock_path}")
            time.sleep(poll_seconds)


def release_file_lock(fd: int, lock_path: Path) -> None:
    os.close(fd)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


@torch.no_grad()
def compute_illumination_latent_stats(
    pipe,
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    config: IlluminationLatentHeadConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats_rows = rows[: int(args.stats_max_items)] if args.stats_max_items is not None else rows
    dataset = IlluminationImageDataset(
        stats_rows,
        data_root=data_root,
        image_key=args.image_key,
        width=args.width,
        height=args.height,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.stats_batch_size or args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_images,
        drop_last=False,
    )
    channel_sum = torch.zeros(config.latent_channels, dtype=torch.float64)
    channel_sq_sum = torch.zeros(config.latent_channels, dtype=torch.float64)
    count = 0
    iterator = tqdm(dataloader, desc=f"stats {config.target}", leave=False)
    for images in iterator:
        rgb_unit = preprocess_batch(pipe, images, min_value=0, max_value=1)
        illum_unit = make_illumination_image_tensor(rgb_unit, target=config.target, eps=config.eps)
        illum_vae = unit_to_vae_range(illum_unit)
        z_illum = encode_latents(pipe, illum_vae, args).cpu().double()
        reduce_dims = tuple(dim for dim in range(z_illum.ndim) if dim != 1)
        channel_sum += z_illum.sum(dim=reduce_dims)
        channel_sq_sum += z_illum.square().sum(dim=reduce_dims)
        count += z_illum.numel() // z_illum.shape[1]
    if count <= 0:
        raise ValueError("Could not compute illumination latent stats from an empty dataset")
    mean = channel_sum / count
    var = (channel_sq_sum / count - mean.square()).clamp_min(float(args.latent_std_min) ** 2)
    std = var.sqrt().clamp_min(float(args.latent_std_min))
    return {
        "mean": mean.float(),
        "std": std.float(),
        "count": count,
        "target": config.target,
        "image_key": args.image_key,
        "width": args.width,
        "height": args.height,
        "row_count": len(stats_rows),
        "stats_max_items": args.stats_max_items,
    }


def load_or_compute_latent_stats(
    pipe,
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    config: IlluminationLatentHeadConfig,
    args: argparse.Namespace,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
    if not args.normalize_latent_loss:
        return None, None, None
    stats_path = repo_path(args.latent_stats_path) if args.latent_stats_path else default_latent_stats_path(data_root, args)
    if stats_path.exists() and not args.force_recompute_latent_stats:
        payload = torch.load(stats_path, map_location="cpu")
        print(f"Loaded illumination latent stats from {stats_path}")
    else:
        lock_path = stats_path.with_name(f"{stats_path.name}.lock")
        lock_fd = acquire_file_lock(lock_path)
        try:
            if stats_path.exists() and not args.force_recompute_latent_stats:
                payload = torch.load(stats_path, map_location="cpu")
                print(f"Loaded illumination latent stats from {stats_path}")
            else:
                print(f"Computing illumination latent stats: {stats_path}")
                payload = compute_illumination_latent_stats(pipe, rows, data_root=data_root, config=config, args=args)
                atomic_torch_save(payload, stats_path)
                print(f"Saved illumination latent stats to {stats_path}")
        finally:
            release_file_lock(lock_fd, lock_path)
    mean = payload["mean"].float()
    std = payload["std"].float()
    if mean.numel() != config.latent_channels or std.numel() != config.latent_channels:
        raise ValueError(
            f"Latent stats channel count mismatch: mean={mean.numel()}, std={std.numel()}, "
            f"expected={config.latent_channels}"
        )
    payload = dict(payload)
    payload["path"] = stats_path.as_posix()
    return mean.to(pipe.device), std.to(pipe.device), payload


def latent_stats_metadata(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {key: value for key, value in payload.items() if key not in {"mean", "std"}}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def maybe_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"TensorBoard disabled: {exc}")
        return None
    return SummaryWriter(log_dir=str(output_dir / "tensorboard_log"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain latent-to-latent illumination translator heads.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata_path", default="")
    parser.add_argument("--image_key", default="video")
    parser.add_argument("--task", default="", help="Optional comma list: ambient_only,single_light,double_light")
    parser.add_argument("--weights_dir", default="weights/Wan2.2-TI2V-5B")
    parser.add_argument("--vae_path", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--append_timestamp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target", choices=("luminance", "log_luminance"), default="luminance")
    parser.add_argument("--arch", choices=("lite", "resunet"), default="lite")
    parser.add_argument("--latent_channels", type=int, default=48)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--mid_channels", type=int, default=192)
    parser.add_argument("--bottleneck_channels", type=int, default=256)
    parser.add_argument("--lite_blocks", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--cosine_weight", type=float, default=0.1)
    parser.add_argument("--multiscale_weights", default="1.0,0.5,0.25")
    parser.add_argument("--normalize_latent_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss_type", choices=("mse", "smooth_l1"), default="mse")
    parser.add_argument("--latent_norm_eps", type=float, default=1e-6)
    parser.add_argument("--latent_std_min", type=float, default=1e-6)
    parser.add_argument("--latent_stats_path", default="")
    parser.add_argument("--force_recompute_latent_stats", action="store_true")
    parser.add_argument("--stats_batch_size", type=int, default=0)
    parser.add_argument("--stats_max_items", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae_dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--vae_tiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--enable_tensorboard", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = repo_path(args.data_root)
    rows = select_rows(args)
    output_dir = Path(args.output_dir or f"model/train/illum_head_{args.target}_{args.arch}")
    output_dir = repo_path(output_dir)
    if args.append_timestamp:
        output_dir = Path(f"{str(output_dir).rstrip('/')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = tuple(float(item) for item in str(args.multiscale_weights).split(",") if item.strip())
    config = IlluminationLatentHeadConfig(
        latent_channels=args.latent_channels,
        target=args.target,
        arch=args.arch,
        hidden_channels=args.hidden_channels,
        mid_channels=args.mid_channels,
        bottleneck_channels=args.bottleneck_channels,
        lite_blocks=args.lite_blocks,
        eps=args.eps,
        normalize_loss=args.normalize_latent_loss,
        loss_type=args.loss_type,
        cosine_weight=args.cosine_weight,
        multiscale_weights=weights,
    )
    latent_stats_path = None
    if args.normalize_latent_loss:
        latent_stats_path = repo_path(args.latent_stats_path) if args.latent_stats_path else default_latent_stats_path(data_root, args)
    save_json(
        output_dir / "train_config.json",
        {
            "args": vars(args),
            "data_root": data_root.as_posix(),
            "row_count": len(rows),
            "head_config": config.to_dict(),
            "latent_stats_path": latent_stats_path.as_posix() if latent_stats_path else None,
        },
    )

    dataset = IlluminationImageDataset(
        rows,
        data_root=data_root,
        image_key=args.image_key,
        width=args.width,
        height=args.height,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_images,
        drop_last=True,
    )
    pipe = load_vae_pipe(args)
    latent_mean, latent_std, latent_stats_info = load_or_compute_latent_stats(
        pipe,
        rows,
        data_root=data_root,
        config=config,
        args=args,
    )
    head = build_illumination_latent_head(config).to(device=pipe.device, dtype=torch.float32)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = maybe_writer(output_dir, args.enable_tensorboard)

    global_step = 0
    try:
        for epoch in range(int(args.num_epochs)):
            iterator = tqdm(dataloader, desc=f"illum {args.target}/{args.arch} epoch {epoch}")
            for images in iterator:
                rgb_vae = preprocess_batch(pipe, images, min_value=-1, max_value=1)
                rgb_unit = preprocess_batch(pipe, images, min_value=0, max_value=1)
                illum_unit = make_illumination_image_tensor(rgb_unit, target=config.target, eps=config.eps)
                illum_vae = unit_to_vae_range(illum_unit)
                z_rgb = encode_latents(pipe, rgb_vae, args)
                z_illum = encode_latents(pipe, illum_vae, args)

                pred = head(z_rgb)
                loss, metrics = illumination_latent_head_loss(
                    pred,
                    z_illum,
                    multiscale_weights=config.multiscale_weights,
                    cosine_weight=config.cosine_weight,
                    mean=latent_mean,
                    std=latent_std,
                    normalize=config.normalize_loss,
                    norm_eps=args.latent_norm_eps,
                    loss_type=config.loss_type,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                global_step += 1
                loss_value = float(loss.detach().cpu())
                iterator.set_postfix(loss=f"{loss_value:.5f}", step=global_step)
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, global_step)
                    for key, value in metrics.items():
                        writer.add_scalar(f"train/{key}", float(value.cpu()), global_step)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                if args.save_steps and global_step % int(args.save_steps) == 0:
                    save_illumination_head_checkpoint(
                        output_dir / f"step-{global_step}.safetensors",
                        head,
                        config,
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                        extra={
                            "global_step": global_step,
                            "epoch": epoch,
                            "latent_stats": latent_stats_metadata(latent_stats_info),
                        },
                    )
                if args.max_steps is not None and global_step >= int(args.max_steps):
                    break
            save_illumination_head_checkpoint(
                output_dir / f"epoch-{epoch}.safetensors",
                head,
                config,
                latent_mean=latent_mean,
                latent_std=latent_std,
                extra={
                    "global_step": global_step,
                    "epoch": epoch,
                    "latent_stats": latent_stats_metadata(latent_stats_info),
                },
            )
            if args.max_steps is not None and global_step >= int(args.max_steps):
                break
    finally:
        if writer is not None:
            writer.close()

    save_illumination_head_checkpoint(
        output_dir / "final.safetensors",
        head,
        config,
        latent_mean=latent_mean,
        latent_std=latent_std,
        extra={"global_step": global_step, "latent_stats": latent_stats_metadata(latent_stats_info)},
    )
    print(f"Saved illumination head to {output_dir / 'final.safetensors'}")


if __name__ == "__main__":
    main()
