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
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
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
DEFAULT_STATS_MAX_ITEMS = 20000


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_world_size() -> int:
    return dist.get_world_size() if distributed_is_initialized() else 1


def distributed_rank() -> int:
    return dist.get_rank() if distributed_is_initialized() else 0


def distributed_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return distributed_rank() == 0


def log(message: str, *, main_only: bool = True) -> None:
    if not main_only or is_main_process():
        print(message, flush=True)


def setup_distributed(args: argparse.Namespace) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested_device = str(args.device)
    args.requested_device = requested_device
    if world_size <= 1:
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "backend": None,
            "requested_device": requested_device,
            "device": str(args.device),
        }

    use_cuda = torch.cuda.is_available() and requested_device.startswith("cuda")
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    elif requested_device.startswith("cuda"):
        args.device = "cpu"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return {
        "distributed": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "backend": backend,
        "requested_device": requested_device,
        "device": str(args.device),
    }


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.destroy_process_group()


def broadcast_run_timestamp() -> str:
    value = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main_process() else ""
    if distributed_is_initialized():
        values = [value]
        dist.broadcast_object_list(values, src=0)
        value = str(values[0])
    return value


def unwrap_head(head: torch.nn.Module) -> torch.nn.Module:
    return head.module if isinstance(head, DistributedDataParallel) else head


def average_metric_tensors(metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not distributed_is_initialized():
        return metrics
    averaged: dict[str, torch.Tensor] = {}
    world_size = float(distributed_world_size())
    for key, value in metrics.items():
        tensor = value.detach().float()
        if tensor.ndim != 0:
            tensor = tensor.mean()
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        averaged[key] = tensor / world_size
    return averaged


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def default_metadata_path(data_root: Path) -> Path:
    return REPO_ROOT / "data_train" / data_root.name / "metadata.jsonl"


def default_latent_stats_path(data_root: Path, args: argparse.Namespace) -> Path:
    image_key = str(args.image_key).replace("/", "_")
    task = str(args.task).replace(",", "-") if args.task else "all"
    stats_items = args.stats_max_items if args.stats_max_items and args.stats_max_items > 0 else args.max_items
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


def lock_owner_pid(lock_path: Path) -> int | None:
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_looks_like_stats_owner(pid: int) -> bool:
    cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        text = cmdline.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return pid_is_running(pid)
    return "train_illumination_latent_head.py" in text


def acquire_file_lock(lock_path: Path, *, poll_seconds: float = 5.0, report_seconds: float = 60.0) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    last_report = 0.0
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd
        except FileExistsError:
            owner_pid = lock_owner_pid(lock_path)
            if owner_pid is not None and (not pid_is_running(owner_pid) or not pid_looks_like_stats_owner(owner_pid)):
                print(f"Removing stale illumination latent stats lock: {lock_path} pid={owner_pid}")
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            now = time.time()
            if now - last_report >= report_seconds:
                suffix = f" pid={owner_pid}" if owner_pid is not None else ""
                print(f"Waiting for illumination latent stats lock: {lock_path}{suffix}")
                last_report = now
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
    writer=None,
) -> dict[str, Any]:
    stats_rows = rows[: int(args.stats_max_items)] if args.stats_max_items and args.stats_max_items > 0 else rows
    total_stats_rows = len(stats_rows)
    if distributed_is_initialized():
        rank = distributed_rank()
        world_size = distributed_world_size()
        stats_rows = stats_rows[rank::world_size]
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
    stats_device = pipe.device if distributed_is_initialized() else torch.device("cpu")
    channel_sum = torch.zeros(config.latent_channels, dtype=torch.float64, device=stats_device)
    channel_sq_sum = torch.zeros(config.latent_channels, dtype=torch.float64, device=stats_device)
    count = torch.zeros((), dtype=torch.float64, device=stats_device)
    processed_images = 0
    iterator = tqdm(dataloader, desc=f"stats {config.target}", leave=False, disable=not is_main_process())
    for images in iterator:
        rgb_unit = preprocess_batch(pipe, images, min_value=0, max_value=1)
        illum_unit = make_illumination_image_tensor(rgb_unit, target=config.target, eps=config.eps)
        illum_vae = unit_to_vae_range(illum_unit)
        z_illum = encode_latents(pipe, illum_vae, args).to(device=stats_device, dtype=torch.float64)
        reduce_dims = tuple(dim for dim in range(z_illum.ndim) if dim != 1)
        channel_sum += z_illum.sum(dim=reduce_dims)
        channel_sq_sum += z_illum.square().sum(dim=reduce_dims)
        count += z_illum.new_tensor(z_illum.numel() // z_illum.shape[1], dtype=torch.float64)
        processed_images += len(images)
        if is_main_process():
            iterator.set_postfix(images=processed_images)
        if writer is not None and is_main_process():
            writer.add_scalar("stats/images", processed_images, processed_images)
            writer.flush()

    if distributed_is_initialized():
        dist.all_reduce(channel_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(channel_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    count_value = int(count.item())
    if count_value <= 0:
        raise ValueError("Could not compute illumination latent stats from an empty dataset")
    mean = channel_sum / count
    var = (channel_sq_sum / count - mean.square()).clamp_min(float(args.latent_std_min) ** 2)
    std = var.sqrt().clamp_min(float(args.latent_std_min))
    return {
        "mean": mean.detach().cpu().float(),
        "std": std.detach().cpu().float(),
        "count": count_value,
        "target": config.target,
        "image_key": args.image_key,
        "width": args.width,
        "height": args.height,
        "row_count": total_stats_rows,
        "stats_max_items": args.stats_max_items,
        "distributed_world_size": distributed_world_size(),
    }


def load_or_compute_latent_stats(
    pipe,
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    config: IlluminationLatentHeadConfig,
    args: argparse.Namespace,
    writer=None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
    if not args.normalize_latent_loss:
        return None, None, None
    stats_path = repo_path(args.latent_stats_path) if args.latent_stats_path else default_latent_stats_path(data_root, args)

    if distributed_is_initialized() and (args.force_recompute_latent_stats or not stats_path.exists()):
        log(f"Computing distributed illumination latent stats: {stats_path}")
        payload = compute_illumination_latent_stats(
            pipe,
            rows,
            data_root=data_root,
            config=config,
            args=args,
            writer=writer,
        )
        if is_main_process():
            atomic_torch_save(payload, stats_path)
            log(f"Saved illumination latent stats to {stats_path}")
    elif stats_path.exists() and not args.force_recompute_latent_stats:
        payload = torch.load(stats_path, map_location="cpu")
        log(f"Loaded illumination latent stats from {stats_path}")
    else:
        lock_path = stats_path.with_name(f"{stats_path.name}.lock")
        lock_fd = acquire_file_lock(lock_path)
        try:
            if stats_path.exists() and not args.force_recompute_latent_stats:
                payload = torch.load(stats_path, map_location="cpu")
                log(f"Loaded illumination latent stats from {stats_path}")
            else:
                log(f"Computing illumination latent stats: {stats_path}")
                payload = compute_illumination_latent_stats(
                    pipe,
                    rows,
                    data_root=data_root,
                    config=config,
                    args=args,
                    writer=writer,
                )
                atomic_torch_save(payload, stats_path)
                log(f"Saved illumination latent stats to {stats_path}")
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
    parser.add_argument(
        "--stats_max_items",
        type=int,
        default=DEFAULT_STATS_MAX_ITEMS,
        help="Number of images used for latent mean/std stats. Use 0 for the full selected dataset.",
    )
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
    dist_info = setup_distributed(args)
    writer = None
    try:
        data_root = repo_path(args.data_root)
        rows = select_rows(args)
        output_dir = Path(args.output_dir or f"model/train/illum_head_{args.target}_{args.arch}")
        output_dir = repo_path(output_dir)
        if args.append_timestamp:
            output_dir = Path(f"{str(output_dir).rstrip('/')}_{broadcast_run_timestamp()}")
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
            latent_stats_path = (
                repo_path(args.latent_stats_path)
                if args.latent_stats_path
                else default_latent_stats_path(data_root, args)
            )
        if is_main_process():
            save_json(
                output_dir / "train_config.json",
                {
                    "args": vars(args),
                    "data_root": data_root.as_posix(),
                    "row_count": len(rows),
                    "head_config": config.to_dict(),
                    "latent_stats_path": latent_stats_path.as_posix() if latent_stats_path else None,
                    "distributed": dist_info,
                    "per_gpu_batch_size": int(args.batch_size),
                    "effective_global_batch_size": int(args.batch_size) * distributed_world_size(),
                },
            )

        dataset = IlluminationImageDataset(
            rows,
            data_root=data_root,
            image_key=args.image_key,
            width=args.width,
            height=args.height,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=distributed_world_size(),
                rank=distributed_rank(),
                shuffle=True,
                seed=int(args.shuffle_seed or 0),
                drop_last=True,
            )
            if distributed_is_initialized()
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_images,
            drop_last=True,
        )
        pipe = load_vae_pipe(args)
        writer = maybe_writer(output_dir, args.enable_tensorboard and is_main_process())
        latent_mean, latent_std, latent_stats_info = load_or_compute_latent_stats(
            pipe,
            rows,
            data_root=data_root,
            config=config,
            args=args,
            writer=writer,
        )
        head = build_illumination_latent_head(config).to(device=pipe.device, dtype=torch.float32)
        head.train()
        if distributed_is_initialized():
            ddp_kwargs: dict[str, Any] = {}
            if pipe.device.type == "cuda":
                ddp_kwargs = {
                    "device_ids": [distributed_local_rank()],
                    "output_device": distributed_local_rank(),
                }
            head = DistributedDataParallel(head, **ddp_kwargs)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        log(
            "Illumination head training: "
            f"arch={args.arch}, target={args.target}, size={args.width}x{args.height}, "
            f"device={args.device}, world_size={distributed_world_size()}, "
            f"per_gpu_batch_size={args.batch_size}, "
            f"effective_global_batch_size={int(args.batch_size) * distributed_world_size()}"
        )

        global_step = 0
        for epoch in range(int(args.num_epochs)):
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = tqdm(
                dataloader,
                desc=f"illum {args.target}/{args.arch} epoch {epoch}",
                disable=not is_main_process(),
            )
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
                averaged_metrics = average_metric_tensors(metrics)
                loss_value = float(averaged_metrics["loss"].detach().cpu())
                if is_main_process():
                    iterator.set_postfix(loss=f"{loss_value:.5f}", step=global_step)
                if writer is not None and is_main_process():
                    writer.add_scalar("train/loss", loss_value, global_step)
                    for key, value in averaged_metrics.items():
                        writer.add_scalar(f"train/{key}", float(value.cpu()), global_step)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                if args.save_steps and global_step % int(args.save_steps) == 0 and is_main_process():
                    save_illumination_head_checkpoint(
                        output_dir / f"step-{global_step}.safetensors",
                        unwrap_head(head),
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
            if is_main_process():
                save_illumination_head_checkpoint(
                    output_dir / f"epoch-{epoch}.safetensors",
                    unwrap_head(head),
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
        if is_main_process():
            save_illumination_head_checkpoint(
                output_dir / "final.safetensors",
                unwrap_head(head),
                config,
                latent_mean=latent_mean,
                latent_std=latent_std,
                extra={"global_step": global_step, "latent_stats": latent_stats_metadata(latent_stats_info)},
            )
            log(f"Saved illumination head to {output_dir / 'final.safetensors'}")
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
