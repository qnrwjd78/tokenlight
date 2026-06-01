from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenlight.checkpoint import load_checkpoint, load_compatible_checkpoint, save_checkpoint
from tokenlight.config import load_config
from tokenlight.cosmos_base import assert_tokenlight_first_base_config, inspect_cosmos_base
from tokenlight.data import (
    RelightingComponentAdapterDataset,
    TokenLightManifestDataset,
    collate_tokenlight,
    move_batch_to_device,
)
from tokenlight.factory import build_model
from tokenlight.flow import flow_matching_loss
from tokenlight.vae import build_vae


def autocast_context(precision: str, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def maybe_wrap_fsdp(model, enabled: bool):
    if not enabled:
        return model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return FSDP(model)


def save_train_checkpoint(path: Path, model, optimizer, step: int, rank: int, fsdp_enabled: bool):
    if not fsdp_enabled:
        if rank == 0:
            save_checkpoint(path, model, optimizer=optimizer, step=step)
        return
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_cfg, optim_cfg):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model_state, "optimizer": optim_state, "step": step}, path)


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-type", choices=["manifest", "relighting-components"], default="manifest")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--component-root", default="")
    parser.add_argument("--component-repo", default="repos/relighting_dataset")
    parser.add_argument("--component-modes", nargs="+", default=["spatial", "ambient", "diffuse", "fixture"])
    parser.add_argument("--component-length", type=int, default=100_000)
    parser.add_argument("--component-seed", type=int, default=1234)
    parser.add_argument("--max-lights", type=int, default=1)
    parser.add_argument("--image-range", choices=["minus_one_one", "zero_one"], default="minus_one_one")
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--strict-init", action="store_true")
    parser.add_argument("--skip-base-init", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--fsdp", action="store_true")
    return parser.parse_args()


def build_dataset(args):
    if args.dataset_type == "manifest":
        if not args.manifest:
            raise ValueError("--manifest is required when --dataset-type=manifest")
        return TokenLightManifestDataset(args.manifest, root=args.data_root)
    component_root = args.component_root or args.data_root
    return RelightingComponentAdapterDataset(
        component_root=component_root,
        repo_path=args.component_repo,
        length=args.component_length,
        modes=tuple(args.component_modes),
        seed=args.component_seed,
        max_lights=args.max_lights,
        image_range=args.image_range,
        include_masks=not args.no_masks,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    assert_tokenlight_first_base_config(cfg.base)
    if cfg.base.provider == "cosmos_unirelight":
        report = inspect_cosmos_base(cfg.base, cfg.vae)
        if not report.ready:
            missing = "\n".join(f"  - {path}" for path in report.missing)
            raise FileNotFoundError(
                "Cosmos/UniRelight base files are missing. This base is used only "
                "for TokenLight VAE/backbone initialization, not for UniRelight "
                f"conditioning.\nMissing:\n{missing}"
            )
    rank, world_size, local_rank = setup_distributed(args)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = build_model(cfg).to(device)
    vae = build_vae(cfg.vae).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    init_checkpoint = args.init_checkpoint
    base_init_required = False
    if not init_checkpoint and cfg.base.provider == "cosmos_unirelight" and cfg.base.init_backbone and not args.skip_base_init:
        init_checkpoint = cfg.base.checkpoint_path
        base_init_required = True
    if init_checkpoint:
        if args.strict_init:
            load_checkpoint(init_checkpoint, model, strict=True)
        else:
            report = load_compatible_checkpoint(init_checkpoint, model)
            if rank == 0:
                print(
                    "base init:",
                    f"loaded={report.loaded_tensors}",
                    f"skipped={report.skipped_tensors}",
                    f"missing={report.missing_tensors}",
                    f"path={report.source_path}",
                )
            if base_init_required and report.loaded_tensors == 0:
                raise RuntimeError(
                    "Cosmos/UniRelight checkpoint was configured as the TokenLight base, "
                    "but no compatible tensors loaded into TokenLightDiT. Do not proceed "
                    "with random initialization for a paper-scale run; implement the "
                    "Cosmos-to-TokenLight weight mapping or use a compatible checkpoint."
                )
    model = maybe_wrap_fsdp(model, args.fsdp)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
    )
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer=optimizer, strict=True)

    dataset = build_dataset(args)
    if cfg.training.global_batch_size % world_size != 0:
        raise ValueError(
            f"global_batch_size={cfg.training.global_batch_size} must be divisible by world_size={world_size}"
        )
    per_rank_batch = cfg.training.global_batch_size // world_size
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=per_rank_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_tokenlight,
        drop_last=True,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    step = start_step
    progress = tqdm(total=cfg.training.steps, initial=start_step, disable=rank != 0)
    model.train()
    epoch = 0
    while step < cfg.training.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= cfg.training.steps:
                break
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(cfg.training.precision, device):
                loss, metrics = flow_matching_loss(
                    model,
                    vae,
                    batch,
                    light_dropout_prob=cfg.training.light_dropout_prob,
                )
            loss.backward()
            if cfg.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()
            step += 1
            progress.update(1)
            progress.set_postfix(metrics)
            if step % cfg.training.save_every == 0 or step == cfg.training.steps:
                target = output / f"step_{step:06d}.pt"
                save_train_checkpoint(target, model, optimizer, step, rank, args.fsdp)
                save_train_checkpoint(output / "latest.pt", model, optimizer, step, rank, args.fsdp)
        epoch += 1
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
