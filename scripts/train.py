from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import accelerate
import torch

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize, LoadAudio, LoadVideo, ToAbsolutePath
from diffsynth.diffusion import *  # noqa: F403 - DiffSynth exposes trainer utilities here.
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

from tokenlight.wan import TokenLightAttributeTokenEncoder, attrs_from_batch, tokenlight_model_fn_wan_video


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def install_zero3_loader_compat() -> None:
    """Provide the private HF ZeRO-3 loader expected by DiffSynth when missing."""

    try:
        import transformers.integrations.deepspeed as ds_integration
    except Exception:
        return
    if hasattr(ds_integration, "_load_state_dict_into_zero3_model"):
        return

    def _load_state_dict_into_zero3_model(model_to_load, state_dict, load_config=None):
        del load_config
        import deepspeed

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        error_msgs = []
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        def load(module, prefix: str = "") -> None:
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            params = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [param for name, param in params.items() if name in state_dict]
            context = (
                deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0)
                if params_to_gather
                else nullcontext()
            )
            with context:
                if rank == 0:
                    module._load_from_state_dict(
                        state_dict,
                        prefix,
                        local_metadata,
                        True,
                        [],
                        [],
                        error_msgs,
                    )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model_to_load)
        for name, buffer in model_to_load.named_buffers():
            value = state_dict.get(name)
            if isinstance(value, torch.Tensor):
                buffer.data.copy_(value.to(device=buffer.device, dtype=buffer.dtype))
        return error_msgs

    ds_integration._load_state_dict_into_zero3_model = _load_state_dict_into_zero3_model


install_zero3_loader_compat()


def _as_frames(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_wan_lora_target_modules(value):
    """Map common Diffusers attention names to DiffSynth Wan module names."""
    if not isinstance(value, str) or not value:
        return value
    aliases = {
        "to_q": "q",
        "to_k": "k",
        "to_v": "v",
        "to_out.0": "o",
        "to_out": "o",
    }
    parts = [part.strip() for part in value.split(",") if part.strip()]
    mapped = [aliases.get(part, part) for part in parts]
    if mapped != parts:
        warnings.warn(
            "Mapped Diffusers LoRA target names to DiffSynth Wan names: "
            f"{','.join(parts)} -> {','.join(mapped)}",
            stacklevel=2,
        )
    return ",".join(mapped)


class TokenLightWanTrainingModule(DiffusionTrainingModule):  # noqa: F405
    """Wan2.2 TI2V trainer with TokenLight numeric light tokens."""

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        tokenlight_light_tokens=True,
        tokenlight_attrs_key="attrs_json",
        tokenlight_token_dim=0,
        tokenlight_fourier_features=512,
        tokenlight_fourier_sigma=5.0,
        tokenlight_mlp_hidden_dim=4096,
        tokenlight_light_dropout=0.0,
        tokenlight_cfg_drop_prob=0.0,
        tokenlight_source_tokens=True,
        tokenlight_mask_tokens=True,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        lora_target_modules = _normalize_wan_lora_target_modules(lora_target_modules)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        self.tokenlight_attrs_key = tokenlight_attrs_key
        self.tokenlight_light_tokens = bool(tokenlight_light_tokens)
        self.tokenlight_source_tokens = bool(tokenlight_source_tokens)
        self.tokenlight_mask_tokens = bool(tokenlight_mask_tokens)
        self.tokenlight_cfg_drop_prob = float(tokenlight_cfg_drop_prob)
        token_dim = int(tokenlight_token_dim) if int(tokenlight_token_dim) > 0 else int(self.pipe.dit.dim)
        self.light_encoder = (
            TokenLightAttributeTokenEncoder(
                token_dim=token_dim,
                fourier_features=int(tokenlight_fourier_features),
                fourier_sigma=float(tokenlight_fourier_sigma),
                hidden_dim=int(tokenlight_mlp_hidden_dim),
                dropout=float(tokenlight_light_dropout),
            )
            if self.tokenlight_light_tokens
            else None
        )
        if self.tokenlight_light_tokens or self.tokenlight_source_tokens or self.tokenlight_mask_tokens:
            self.pipe.model_fn = self._tokenlight_model_fn

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = [item for item in extra_inputs.split(",") if item] if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi
            ),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                image = _as_frames(data.get("input_image", data["video"]))[0]
                inputs_shared["input_image"] = image
                inputs_shared["tokenlight_source_image"] = image
            elif extra_input == "end_image":
                inputs_shared["end_image"] = _as_frames(data.get("end_image", data["video"]))[-1]
            elif extra_input in {"reference_image", "vace_reference_image"}:
                inputs_shared[extra_input] = _as_frames(data[extra_input])[0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            inputs_shared["num_frames"] = 4 * (len(_as_frames(data["video"])) - 1) + 1
        return inputs_shared

    def get_pipeline_inputs(self, data):
        video = _as_frames(data["video"])
        first_frame = video[0]
        tokenlight_source_image = _as_frames(data.get("input_image", video))[0]
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video,
            "height": first_frame.size[1],
            "width": first_frame.size[0],
            "num_frames": len(video),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "tokenlight_source_image": tokenlight_source_image,
        }
        if "mask" in data:
            inputs_shared["tokenlight_mask_image"] = _as_frames(data["mask"])[0]
        return self.parse_extra_inputs(data, self.extra_inputs, inputs_shared), inputs_posi, inputs_nega

    def _tokenlight_model_fn(self, **kwargs):
        return tokenlight_model_fn_wan_video(
            tokenlight_light_encoder=self.light_encoder,
            **kwargs,
        )

    def _encode_image_latents(self, image, inputs_shared):
        width = int(inputs_shared["width"])
        height = int(inputs_shared["height"])
        tiled = bool(inputs_shared.get("tiled", False))
        tile_size = inputs_shared.get("tile_size", (30, 52))
        tile_stride = inputs_shared.get("tile_stride", (15, 26))
        if hasattr(image, "resize"):
            image = image.resize((width, height))
        self.pipe.load_models_to_device(["vae"])
        pixel_values = self.pipe.preprocess_image(image).transpose(0, 1)
        latents = self.pipe.vae.encode(
            [pixel_values.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)],
            device=self.pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _prepare_tokenlight_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = inputs
        if not (self.tokenlight_light_tokens or self.tokenlight_source_tokens or self.tokenlight_mask_tokens):
            return inputs

        inputs_shared["tokenlight_attrs"] = attrs_from_batch(data, key=self.tokenlight_attrs_key)
        if self.tokenlight_source_tokens:
            if "first_frame_latents" in inputs_shared:
                inputs_shared["tokenlight_source_latents"] = inputs_shared["first_frame_latents"]
            elif "tokenlight_source_image" in inputs_shared:
                inputs_shared["tokenlight_source_latents"] = self._encode_image_latents(
                    inputs_shared["tokenlight_source_image"],
                    inputs_shared,
                )
        if self.tokenlight_mask_tokens and "tokenlight_mask_image" in inputs_shared:
            inputs_shared["tokenlight_mask_latents"] = self._encode_image_latents(
                inputs_shared["tokenlight_mask_image"],
                inputs_shared,
            )
        if self.light_encoder is not None:
            drop_light = False
            if self.training and self.tokenlight_cfg_drop_prob > 0 and "context" in inputs_posi:
                batch = inputs_posi["context"].shape[0]
                drop_light = torch.rand(batch, device=inputs_posi["context"].device) < self.tokenlight_cfg_drop_prob
            inputs_posi["tokenlight_drop_light"] = drop_light
            inputs_nega["tokenlight_drop_light"] = True
        inputs_shared.pop("tokenlight_source_image", None)
        inputs_shared.pop("tokenlight_mask_image", None)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs = self._prepare_tokenlight_inputs(inputs, data)
        return self.task_to_loss[self.task](self.pipe, *inputs)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        exported = super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        if self.light_encoder is not None:
            for key, value in state_dict.items():
                if key.startswith("light_encoder."):
                    exported[key] = value
        return exported


def wan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TokenLight Wan2.2-TI2V-5B trainer.")
    parser = add_general_config(parser)  # noqa: F405
    parser = add_video_size_config(parser)  # noqa: F405
    parser.set_defaults(use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_processor_path", type=str, default=None)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--framewise_decoding", default=False, action="store_true")
    parser.add_argument("--tokenlight_light_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenlight_attrs_key", default="attrs_json")
    parser.add_argument("--tokenlight_token_dim", type=int, default=0)
    parser.add_argument("--tokenlight_fourier_features", type=int, default=512)
    parser.add_argument("--tokenlight_fourier_sigma", type=float, default=5.0)
    parser.add_argument("--tokenlight_mlp_hidden_dim", type=int, default=4096)
    parser.add_argument("--tokenlight_light_dropout", type=float, default=0.0)
    parser.add_argument("--tokenlight_cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--tokenlight_source_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenlight_mask_tokens", action=argparse.BooleanOptionalAction, default=True)
    return parser


def build_dataset(args):
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )


def main() -> None:
    args = wan_parser().parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    dataset = build_dataset(args)
    model = TokenLightWanTrainingModule(
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
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tokenlight_light_tokens=args.tokenlight_light_tokens,
        tokenlight_attrs_key=args.tokenlight_attrs_key,
        tokenlight_token_dim=args.tokenlight_token_dim,
        tokenlight_fourier_features=args.tokenlight_fourier_features,
        tokenlight_fourier_sigma=args.tokenlight_fourier_sigma,
        tokenlight_mlp_hidden_dim=args.tokenlight_mlp_hidden_dim,
        tokenlight_light_dropout=args.tokenlight_light_dropout,
        tokenlight_cfg_drop_prob=args.tokenlight_cfg_drop_prob,
        tokenlight_source_tokens=args.tokenlight_source_tokens,
        tokenlight_mask_tokens=args.tokenlight_mask_tokens,
    )
    model_logger = ModelLogger(  # noqa: F405
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,  # noqa: F405
        "direct_distill:data_process": launch_data_process_task,  # noqa: F405
        "sft": launch_training_task,  # noqa: F405
        "sft:train": launch_training_task,  # noqa: F405
        "direct_distill": launch_training_task,  # noqa: F405
        "direct_distill:train": launch_training_task,  # noqa: F405
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
