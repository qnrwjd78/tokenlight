from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import accelerate
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from tqdm import tqdm

from model.lightoken_encoder import LightokenEncoder, attrs_from_batch
from model.tokenlight_wan_pbr import TokenLightPBRTypeEmbedding, model_fn_wan_video_tokenlight_pbr
from model.train_tokenlight import (
    DEFAULT_TRAIN_CONFIG_PATH,
    TRAIN_CONFIG_BY_MODE,
    TOKENLIGHT_DEFAULT_PROMPT,
    TokenLightWanTrainingModule,
    _append_timestamp_to_output_path,
    _apply_config_defaults,
    _as_frames,
    _call_compatible_method,
    _collect_train_metrics,
    _coerce_trainable_parameter_dtype,
    _csv_value,
    _load_checkpoint_state_dict,
    _load_json_config,
    _mean_across_processes,
    _make_model_logger,
    _normalize_tokenlight_metadata_rows,
    _preferred_trainable_dtype,
    _resolve_weight_paths,
    _require_nonempty_light_attrs,
    _trainable_dtype_counts,
    _trainable_parameters,
    _collate_tokenlight_batch,
    build_dataset as build_tokenlight_dataset,
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    launch_data_process_task,
    save_training_config_snapshot,
    save_training_runtime_snapshot,
    wan_parser,
    OffloadTrainingManager,
    TokenLightTensorBoardMetrics,
)


PBR_CONFIG_BY_MODE = {
    "single": "configs/train_tokenlight_pbr_single.json",
    "zero3": "configs/train_tokenlight_pbr_zero3.json",
}


def _bool_mask(value: Any, *, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        mask = value.to(device=device, dtype=torch.bool).flatten()
        return mask.expand(batch) if mask.numel() == 1 else mask
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return torch.tensor(list(value), device=device, dtype=torch.bool).flatten()
    return torch.full((batch,), bool(value), device=device, dtype=torch.bool)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
    error = (pred.float() - target.float()).pow(2) * mask
    denom = mask.sum() * math.prod(pred.shape[1:])
    return error.sum() / denom.clamp_min(1.0)


def PBRFlowMatchSFTLoss(pipe, **inputs):
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    rgb_input = inputs["input_latents"]
    rgb_noise = torch.randn_like(rgb_input) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(rgb_input, rgb_noise, timestep)
    rgb_target = pipe.scheduler.training_target(rgb_input, rgb_noise, timestep)

    pbr_target = None
    pbr_mask = None
    pbr_input = inputs.get("tokenlight_pbr_input_latents")
    if pbr_input is not None:
        batch = int(pbr_input.shape[0])
        pbr_mask = _bool_mask(inputs.get("tokenlight_pbr_is_target", True), batch=batch, device=pbr_input.device)
        pbr_noise = torch.randn_like(pbr_input) * inputs.get("noise_scale", 1.0)
        noisy_pbr = pipe.scheduler.add_noise(pbr_input, pbr_noise, timestep)
        pbr_target = pipe.scheduler.training_target(pbr_input, pbr_noise, timestep)
        mask = pbr_mask.view(-1, 1, 1, 1, 1)
        inputs["tokenlight_pbr_latents"] = torch.where(mask, noisy_pbr, pbr_input)
        inputs["tokenlight_pbr_is_target"] = pbr_mask

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    if isinstance(noise_pred, tuple):
        rgb_pred, pbr_pred = noise_pred
    else:
        rgb_pred, pbr_pred = noise_pred, None

    if "first_frame_latents" in inputs:
        rgb_pred = rgb_pred[:, :, 1:]
        rgb_target = rgb_target[:, :, 1:]

    loss = F.mse_loss(rgb_pred.float(), rgb_target.float())
    if pbr_input is not None and pbr_pred is not None and pbr_target is not None and pbr_mask is not None:
        pbr_weight = float(inputs.get("tokenlight_pbr_loss_weight", 1.0))
        loss = loss + pbr_weight * _masked_mse(pbr_pred, pbr_target, pbr_mask)

    return loss * pipe.scheduler.training_weight(timestep)


def _load_matching_module_state(module: torch.nn.Module | None, state_dict: dict[str, torch.Tensor], prefix: str) -> None:
    if module is None or not state_dict:
        return
    module_state = module.state_dict()
    loaded = {}
    for key, value in state_dict.items():
        candidates = (f"{prefix}.", f"module.{prefix}.")
        for candidate in candidates:
            if not key.startswith(candidate):
                continue
            name = key[len(candidate) :]
            if name in module_state and tuple(module_state[name].shape) == tuple(value.shape):
                loaded[name] = value
            break
    if loaded:
        missing, unexpected = module.load_state_dict(loaded, strict=False)
        print(f"Loaded matching {prefix}: tensors={len(loaded)}, missing={len(missing)}, unexpected={len(unexpected)}")


class TokenLightPBRWanTrainingModule(TokenLightWanTrainingModule):
    def __init__(
        self,
        *args,
        tokenlight_pbr_image_key: str = "pbr_image",
        tokenlight_pbr_mode_key: str = "pbr_mode",
        tokenlight_pbr_aux_type: str = "shading",
        tokenlight_pbr_loss_weight: float = 1.0,
        tokenlight_pbr_default_mode: str = "target",
        **kwargs,
    ) -> None:
        super_kwargs = dict(kwargs)
        super_kwargs["tokenlight_light_tokens"] = False
        super_kwargs["tokenlight_source_tokens"] = False
        super_kwargs["tokenlight_mask_tokens"] = False
        super().__init__(*args, **super_kwargs)
        self.tokenlight_pbr_image_key = tokenlight_pbr_image_key
        self.tokenlight_pbr_mode_key = tokenlight_pbr_mode_key
        self.tokenlight_pbr_aux_type = tokenlight_pbr_aux_type
        self.tokenlight_pbr_loss_weight = float(tokenlight_pbr_loss_weight)
        self.tokenlight_pbr_default_mode = tokenlight_pbr_default_mode
        self.tokenlight_attrs_key = kwargs.get("tokenlight_attrs_key", "attrs_json")
        self.tokenlight_light_tokens = bool(kwargs.get("tokenlight_light_tokens", True))
        self.tokenlight_source_tokens = bool(kwargs.get("tokenlight_source_tokens", True))
        self.tokenlight_mask_tokens = False
        self.tokenlight_cfg_drop_prob = float(kwargs.get("tokenlight_cfg_drop_prob", 0.1))

        token_dim = int(kwargs.get("tokenlight_token_dim", 0) or 0)
        if token_dim <= 0:
            token_dim = int(self.pipe.dit.dim)
        self.light_encoder = (
            LightokenEncoder(
                token_dim=token_dim,
                fourier_features=int(kwargs.get("tokenlight_fourier_features", 512)),
                fourier_sigma=float(kwargs.get("tokenlight_fourier_sigma", 5.0)),
                max_lights=int(kwargs.get("tokenlight_max_lights", 2)),
                dropout=float(kwargs.get("tokenlight_light_dropout", 0.0)),
            )
            if self.tokenlight_light_tokens
            else None
        )
        self.tokenlight_type_embedding = TokenLightPBRTypeEmbedding(token_dim)
        checkpoint_state = _load_checkpoint_state_dict(kwargs.get("lora_checkpoint") or kwargs.get("resume_from_checkpoint"))
        _load_matching_module_state(self.light_encoder, checkpoint_state, "light_encoder")
        _load_matching_module_state(self.tokenlight_type_embedding, checkpoint_state, "tokenlight_type_embedding")

        self.pipe.model_fn = self._tokenlight_model_fn
        self.task_to_loss["sft"] = lambda pipe, inputs_shared, inputs_posi, inputs_nega: PBRFlowMatchSFTLoss(
            pipe, **inputs_shared, **inputs_posi
        )
        self.task_to_loss["sft:train"] = self.task_to_loss["sft"]

    def _tokenlight_model_fn(self, **kwargs):
        return model_fn_wan_video_tokenlight_pbr(
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            **kwargs,
        )

    def _pbr_target_mask_from_data(self, data, batch: int, device: torch.device) -> torch.Tensor:
        value = data.get(self.tokenlight_pbr_mode_key, data.get("depth_mode", self.tokenlight_pbr_default_mode))
        if isinstance(value, list):
            modes = value
        else:
            modes = [value] * batch
        target_values = []
        for item in modes:
            text = str(item or self.tokenlight_pbr_default_mode).lower()
            target_values.append(text not in {"condition", "cond", "input", "clean", "provided"})
        return torch.tensor(target_values, device=device, dtype=torch.bool)

    def _pbr_videos_from_data(self, data, batch: int):
        pbr = data.get(self.tokenlight_pbr_image_key)
        if not isinstance(pbr, list) or len(pbr) != batch:
            raise ValueError(f"Expected {batch} `{self.tokenlight_pbr_image_key}` images for PBR training")
        return [_as_frames(item) for item in pbr]

    def _batched_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super()._batched_inputs(data)
        batch = self._batch_size_from_data(data)
        inputs_shared["tokenlight_pbr_input_latents"] = self._encode_video_latents_batched(
            self._pbr_videos_from_data(data, batch),
            inputs_shared,
        )
        inputs_shared["tokenlight_pbr_is_target"] = self._pbr_target_mask_from_data(
            data, batch, inputs_shared["tokenlight_pbr_input_latents"].device
        )
        inputs_shared["tokenlight_pbr_loss_weight"] = self.tokenlight_pbr_loss_weight
        return inputs_shared, inputs_posi, inputs_nega

    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        if self.tokenlight_pbr_image_key in data:
            inputs_shared["tokenlight_pbr_image"] = _as_frames(data[self.tokenlight_pbr_image_key])[0]
            inputs_shared["tokenlight_pbr_is_target"] = str(
                data.get(self.tokenlight_pbr_mode_key, self.tokenlight_pbr_default_mode)
            ).lower() not in {"condition", "cond", "input", "clean", "provided"}
        return inputs_shared, inputs_posi, inputs_nega

    def _prepare_tokenlight_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = super()._prepare_tokenlight_inputs(inputs, data)
        if "tokenlight_pbr_image" in inputs_shared:
            inputs_shared["tokenlight_pbr_input_latents"] = self._encode_image_latents(
                inputs_shared["tokenlight_pbr_image"],
                inputs_shared,
            )
            inputs_shared["tokenlight_pbr_loss_weight"] = self.tokenlight_pbr_loss_weight
        inputs_shared.pop("tokenlight_pbr_image", None)
        return inputs_shared, inputs_posi, inputs_nega


def _parse_task_batch(value: str | dict[str, int] | None) -> dict[str, int]:
    if value is None:
        return {"single_light": 2, "double_light": 2, "ambient_only": 1}
    if isinstance(value, dict):
        return {str(key): int(item) for key, item in value.items() if int(item) > 0}
    result = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        key, count = item.split(":", 1)
        result[key.strip()] = int(count)
    return {key: count for key, count in result.items() if count > 0}


class BalancedTaskBatchSampler(Sampler[list[int]]):
    def __init__(self, rows: Sequence[dict[str, Any]], task_batch: dict[str, int], *, seed: int = 0) -> None:
        self.rows = rows
        self.task_batch = task_batch
        self.seed = int(seed)
        self.batch_size = int(sum(task_batch.values()))
        self.drop_last = False
        self.pools: dict[str, list[int]] = {task: [] for task in task_batch}
        for index, row in enumerate(rows):
            task = str(row.get("task", ""))
            if task in self.pools:
                self.pools[task].append(index)
        missing = [task for task, pool in self.pools.items() if not pool]
        if missing:
            raise ValueError(f"Cannot build balanced batches; missing task rows: {missing}")
        self._length = max(math.ceil(len(self.pools[task]) / count) for task, count in self.task_batch.items())

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed)
        pools = {task: list(indices) for task, indices in self.pools.items()}
        cursors = {task: 0 for task in self.task_batch}
        for indices in pools.values():
            rng.shuffle(indices)
        for _ in range(len(self)):
            batch = []
            for task, count in self.task_batch.items():
                pool = pools[task]
                for _ in range(count):
                    if cursors[task] >= len(pool):
                        rng.shuffle(pool)
                        cursors[task] = 0
                    batch.append(pool[cursors[task]])
                    cursors[task] += 1
            rng.shuffle(batch)
            yield batch


def _configure_deepspeed_batch_size(accelerator, args, batch_size: int) -> None:
    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    ds_config = getattr(plugin, "deepspeed_config", None)
    if not isinstance(ds_config, dict):
        return
    grad_accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    ds_config["train_micro_batch_size_per_gpu"] = int(batch_size)
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = int(batch_size) * grad_accum * num_processes
    print(
        "DeepSpeed batch setup: "
        f"train_micro_batch_size_per_gpu={ds_config['train_micro_batch_size_per_gpu']}, "
        f"gradient_accumulation_steps={ds_config['gradient_accumulation_steps']}, "
        f"train_batch_size={ds_config['train_batch_size']}"
    )


def wan_pbr_parser(train_mode: str = "single") -> argparse.ArgumentParser:
    parser = wan_parser(train_mode)
    parser.description = f"TokenLight PBR Wan2.2-TI2V-5B {train_mode} trainer."
    parser.set_defaults(
        output_path="model/train/tokenlight_pbr_wan22_lora",
        data_file_keys="video,input_image,pbr_image",
        tokenlight_mask_tokens=False,
        tokenlight_max_lights=2,
        batch_size=5,
    )
    parser.add_argument("--tokenlight_pbr_image_key", default="pbr_image")
    parser.add_argument("--tokenlight_pbr_mode_key", default="pbr_mode")
    parser.add_argument("--tokenlight_pbr_aux_type", choices=("shading", "depth"), default="shading")
    parser.add_argument("--tokenlight_pbr_loss_weight", type=float, default=1.0)
    parser.add_argument("--tokenlight_pbr_default_mode", choices=("target", "condition"), default="target")
    parser.add_argument("--balanced_task_batch", default="single_light:2,double_light:2,ambient_only:1")
    parser.add_argument("--balanced_batch_seed", type=int, default=0)
    return parser


def _default_config_for_mode(train_mode: str) -> str:
    return os.environ.get("TOKENLIGHT_TRAIN_CONFIG", PBR_CONFIG_BY_MODE.get(train_mode, DEFAULT_TRAIN_CONFIG_PATH))


def parse_tokenlight_pbr_args(train_mode: str = "single"):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=_default_config_for_mode(train_mode))
    pre_args, _ = pre_parser.parse_known_args()
    raw_config = _load_json_config(pre_args.config)

    parser = wan_pbr_parser(train_mode)
    _apply_config_defaults(parser, raw_config)
    parser.set_defaults(config=pre_args.config)
    args = parser.parse_args()
    args.train_mode = train_mode
    raw_config = _load_json_config(args.config)

    args.data_file_keys = _csv_value(args.data_file_keys)
    _resolve_weight_paths(args)
    _append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def build_dataset(args):
    dataset = build_tokenlight_dataset(args)
    if dataset.data:
        normalized = _normalize_tokenlight_metadata_rows(dataset.data)
        dataset.data = [row for row in normalized if row.get(args.tokenlight_pbr_image_key)]
    return dataset


def launch_tokenlight_pbr_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    args=None,
    **kwargs,
):
    del kwargs
    batch_size = int(args.batch_size)
    task_batch = _parse_task_batch(args.balanced_task_batch)
    if sum(task_batch.values()) != batch_size:
        raise ValueError(f"balanced_task_batch sums to {sum(task_batch.values())}, but batch_size={batch_size}")

    trainable_dtype = _preferred_trainable_dtype(model)
    before_dtype_counts = _coerce_trainable_parameter_dtype(model, trainable_dtype)
    after_dtype_counts = _trainable_dtype_counts(model)
    if accelerator.is_main_process:
        print(
            "Trainable parameter dtype setup: "
            f"target={trainable_dtype}, before={before_dtype_counts}, after={after_dtype_counts}"
        )

    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer = optimizer_class(_trainable_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    batch_sampler = BalancedTaskBatchSampler(dataset.data, task_batch, seed=int(args.balanced_batch_seed))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=_collate_tokenlight_batch,
        num_workers=args.dataset_num_workers,
    )

    enable_model_cpu_offload = getattr(args, "enable_model_cpu_offload", False)
    enable_optimizer_cpu_offload = getattr(args, "enable_optimizer_cpu_offload", False)
    cpu_offload_split_threshold = getattr(args, "cpu_offload_split_threshold", None)
    _configure_deepspeed_batch_size(accelerator, args, batch_size)
    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(
            model,
            accelerator.device,
            enable_optimizer_cpu_offload,
            cpu_offload_split_threshold,
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        offload_manager = None

    save_training_runtime_snapshot(args, accelerator, model)
    tb_metrics = TokenLightTensorBoardMetrics(args.output_path, enabled=args.enable_tensorboard_log)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    local_step = int(getattr(model_logger, "num_steps", 0))
    try:
        for epoch_id in range(int(args.start_epoch), int(args.start_epoch) + int(args.num_epochs)):
            iterator = tqdm(
                dataloader,
                desc=f"pbr epoch {epoch_id}",
                disable=not accelerator.is_local_main_process,
            )
            for data in iterator:
                with accelerator.accumulate(model):
                    loss = model(data)
                    accelerator.backward(loss)
                    if enable_model_cpu_offload:
                        offload_manager.after_backward()
                    optimizer.step()
                    scheduler.step()
                    metrics = _collect_train_metrics(accelerator, model, optimizer, loss)
                    optimizer.zero_grad()
                    _call_compatible_method(model_logger, "on_step_end", accelerator, model, args.save_steps, loss=loss)
                    logger_step = getattr(model_logger, "num_steps", None)
                    if logger_step is None or int(logger_step) <= local_step:
                        local_step += 1
                    else:
                        local_step = int(logger_step)
                    tb_metrics.log(accelerator, local_step, metrics)
                    if metrics and "train/loss" in metrics:
                        iterator.set_postfix(loss=f"{metrics['train/loss']:.4f}", step=local_step)
            if args.save_steps is None:
                _call_compatible_method(model_logger, "on_epoch_end", accelerator, model, epoch_id)
        _call_compatible_method(model_logger, "on_training_end", accelerator, model, args.save_steps)
    finally:
        tb_metrics.close(accelerator)


def main(train_mode: str = "single") -> None:
    args, raw_config, merged_config = parse_tokenlight_pbr_args(train_mode)
    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    }
    if "even_batches" in inspect.signature(accelerate.Accelerator).parameters:
        accelerator_kwargs["even_batches"] = False
    if train_mode == "single":
        accelerator_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    if hasattr(accelerator, "even_batches"):
        accelerator.even_batches = False
    save_training_config_snapshot(args, raw_config, merged_config, accelerator)
    dataset = build_dataset(args)
    model = TokenLightPBRWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=getattr(args, "use_gradient_checkpointing_offload", False),
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or getattr(args, "enable_model_cpu_offload", False)) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tokenlight_light_tokens=args.tokenlight_light_tokens,
        tokenlight_attrs_key=args.tokenlight_attrs_key,
        tokenlight_token_dim=args.tokenlight_token_dim,
        tokenlight_fourier_features=args.tokenlight_fourier_features,
        tokenlight_fourier_sigma=args.tokenlight_fourier_sigma,
        tokenlight_max_lights=args.tokenlight_max_lights,
        tokenlight_light_dropout=args.tokenlight_light_dropout,
        tokenlight_cfg_drop_prob=args.tokenlight_cfg_drop_prob,
        tokenlight_source_tokens=args.tokenlight_source_tokens,
        tokenlight_pbr_image_key=args.tokenlight_pbr_image_key,
        tokenlight_pbr_mode_key=args.tokenlight_pbr_mode_key,
        tokenlight_pbr_aux_type=args.tokenlight_pbr_aux_type,
        tokenlight_pbr_loss_weight=args.tokenlight_pbr_loss_weight,
        tokenlight_pbr_default_mode=args.tokenlight_pbr_default_mode,
    )
    model_logger = _make_model_logger(args)
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_tokenlight_pbr_training_task,
        "sft:train": launch_tokenlight_pbr_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
