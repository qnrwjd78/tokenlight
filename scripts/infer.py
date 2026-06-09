from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

from tokenlight.wan import (
    TokenLightAttributeTokenEncoder,
    light_attrs_to_prompt,
    parse_attrs_json,
    tokenlight_model_fn_wan_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenLight Wan2.2 TI2V inference with numeric light tokens.")
    parser.add_argument("--source", required=True, help="Source image I.")
    parser.add_argument("--attrs", default="", help="JSON file or inline JSON with TokenLight attrs.")
    parser.add_argument("--light_id", type=int, default=None, help="Spatial light number resolved from dataset metadata.")
    parser.add_argument(
        "--dataset_metadata_path",
        default="",
        help="Optional metadata.csv path used to resolve --light_id. Auto-detected from --source when omitted.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--task", choices=["spatial", "ambient", "diffuse", "fixture", "relighting"], default="relighting")
    parser.add_argument("--model_id_with_origin_paths", default="")
    parser.add_argument("--model_paths", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Combined training checkpoint containing LoRA and/or light_encoder weights.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        default="",
        help="Optional LoRA checkpoint. Defaults to --checkpoint when present.",
    )
    parser.add_argument("--tokenlight_light_checkpoint", default="")
    parser.add_argument("--mask", default="", help="Optional object/edit mask image for TokenLight mask latent tokens.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenlight_token_dim", type=int, default=0)
    parser.add_argument("--tokenlight_fourier_features", type=int, default=512)
    parser.add_argument("--tokenlight_fourier_sigma", type=float, default=5.0)
    parser.add_argument("--tokenlight_mlp_hidden_dim", type=int, default=4096)
    parser.add_argument("--no_tokenlight_light_tokens", action="store_true")
    parser.add_argument("--no_tokenlight_source_tokens", action="store_true")
    parser.add_argument("--no_tokenlight_mask_tokens", action="store_true")
    parser.add_argument("--wan_native_input_image", action="store_true")
    parser.add_argument("--preserve_first_frame", action="store_true")
    return parser.parse_args()


def parse_model_configs(args) -> list[ModelConfig]:
    if args.model_paths:
        payload = json.loads(args.model_paths)
        if isinstance(payload, str):
            payload = [payload]
        if isinstance(payload, list):
            configs = []
            for item in payload:
                if isinstance(item, str):
                    configs.append(ModelConfig(item))
                elif isinstance(item, list):
                    configs.append(ModelConfig(path=item))
                elif isinstance(item, dict):
                    configs.append(ModelConfig(**item))
                else:
                    raise ValueError(f"Unsupported --model_paths entry type: {type(item).__name__}")
            return configs
        raise ValueError("--model_paths must be a JSON string or list")
    value = args.model_id_with_origin_paths or (
        "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,"
        "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,"
        "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
    )
    configs = []
    for item in value.split(","):
        model_id, pattern = item.split(":", 1)
        configs.append(ModelConfig(model_id=model_id, origin_file_pattern=pattern))
    return configs


def load_attrs(value: str) -> dict[str, float]:
    path = Path(value)
    if path.exists():
        return parse_attrs_json(path.read_text(encoding="utf-8"))
    return parse_attrs_json(value)


def resolve_metadata_path(args) -> Path:
    if args.dataset_metadata_path:
        return Path(args.dataset_metadata_path)
    source_path = Path(args.source)
    for parent in (source_path.parent, *source_path.parents):
        candidate = parent / "metadata.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not auto-detect metadata.csv from --source. Pass --dataset_metadata_path explicitly."
    )


def load_attrs_from_light_id(args) -> dict[str, float]:
    if args.task != "spatial":
        raise ValueError("--light_id is currently only supported for --task spatial")
    metadata_path = resolve_metadata_path(args)
    source_name = Path(args.source).name
    target_names = {
        f"spatial_light_{int(args.light_id):03d}.png",
        f"spatial_light_{int(args.light_id):03d}.mp4",
    }
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("task") != "spatial":
                continue
            if Path(row.get("input_image", "")).name != source_name:
                continue
            if Path(row.get("video", "")).name not in target_names:
                continue
            attrs = parse_attrs_json(row["attrs_json"])
            print(
                f"Resolved light_id {int(args.light_id):03d} for {source_name} "
                f"from {Path(row['video']).name}: {json.dumps(attrs, sort_keys=True)}"
            )
            return attrs
    raise ValueError(
        f"Could not find spatial light {int(args.light_id):03d} for source {source_name} in {metadata_path}"
    )


def resolve_attrs(args) -> dict[str, float]:
    if args.light_id is not None:
        return load_attrs_from_light_id(args)
    if args.attrs:
        return load_attrs(args.attrs)
    raise ValueError("Provide either --attrs or --light_id")


def load_checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file

        state = load_file(path)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    return state


def resolve_light_checkpoint_path(args) -> str:
    return args.tokenlight_light_checkpoint or args.checkpoint


def resolve_lora_checkpoint_path(args) -> str:
    return args.lora_checkpoint or args.checkpoint


def _strip_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def extract_light_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    light_state = {}
    for key, value in state.items():
        key = _strip_prefix(key, "module.")
        if key.startswith("light_encoder."):
            light_state[key.removeprefix("light_encoder.")] = value
    return light_state


def extract_lora_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    lora_state = {}
    for key, value in state.items():
        key = _strip_prefix(key, "module.")
        key = _strip_prefix(key, "pipe.dit.")
        if ".lora_A." in key or ".lora_B." in key or key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            lora_state[key] = value
    return lora_state


def maybe_load_lora(args, pipe, state: dict[str, torch.Tensor] | None) -> bool:
    if state is None:
        return False
    lora_state = extract_lora_state(state)
    if not lora_state:
        return False
    pipe.load_lora(pipe.dit, state_dict=lora_state, alpha=1.0)
    print(f"Loaded {len(lora_state)} LoRA tensors from checkpoint.")
    return True


def load_light_encoder(args, pipe, state: dict[str, torch.Tensor] | None = None) -> TokenLightAttributeTokenEncoder:
    token_dim = args.tokenlight_token_dim if args.tokenlight_token_dim > 0 else int(pipe.dit.dim)
    encoder = TokenLightAttributeTokenEncoder(
        token_dim=token_dim,
        fourier_features=args.tokenlight_fourier_features,
        fourier_sigma=args.tokenlight_fourier_sigma,
        hidden_dim=args.tokenlight_mlp_hidden_dim,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    if state is not None:
        light_state = extract_light_state(state)
        if not light_state and any(
            key.startswith("fourier.") or key.startswith("value_mlp.") for key in state
        ):
            light_state = state
        if light_state:
            encoder.load_state_dict(light_state, strict=False)
            print(f"Loaded {len(light_state)} light encoder tensors from checkpoint.")
    encoder.eval()
    return encoder


def encode_image_latents(pipe, image, args):
    if hasattr(image, "resize"):
        image = image.resize((args.width, args.height))
    pipe.load_models_to_device(["vae"])
    pixel_values = pipe.preprocess_image(image).transpose(0, 1)
    latents = pipe.vae.encode(
        [pixel_values.to(dtype=pipe.torch_dtype, device=pipe.device)],
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )
    return latents.to(dtype=pipe.torch_dtype, device=pipe.device)


@torch.no_grad()
def tokenlight_wan_generate(pipe, light_encoder, attrs, prompt, input_image, mask_image, args):
    if light_encoder is not None or not args.no_tokenlight_source_tokens or not args.no_tokenlight_mask_tokens:
        pipe.model_fn = lambda **kwargs: tokenlight_model_fn_wan_video(
            tokenlight_light_encoder=light_encoder,
            **kwargs,
        )
    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": ""}
    inputs_shared = {
        "input_image": input_image if args.wan_native_input_image else None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1,
        "seed": args.seed,
        "rand_device": pipe.device,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "cfg_scale": args.cfg_scale,
        "cfg_merge": False,
        "sigma_shift": 5.0,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
        "vap_prompt": " ",
        "negative_vap_prompt": " ",
        "wantodance_music_path": None,
        "wantodance_reference_image": None,
        "wantodance_fps": 30,
        "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    inputs_shared["tokenlight_attrs"] = [attrs]
    if not args.no_tokenlight_source_tokens:
        if "first_frame_latents" in inputs_shared:
            inputs_shared["tokenlight_source_latents"] = inputs_shared["first_frame_latents"]
        else:
            inputs_shared["tokenlight_source_latents"] = encode_image_latents(pipe, input_image, args)
    if mask_image is not None and not args.no_tokenlight_mask_tokens:
        inputs_shared["tokenlight_mask_latents"] = encode_image_latents(pipe, mask_image, args)
    inputs_posi["tokenlight_drop_light"] = False
    inputs_nega["tokenlight_drop_light"] = True

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
        if args.cfg_scale != 1.0:
            noise_pred_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            noise_pred = noise_pred_nega + args.cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        inputs_shared["latents"] = pipe.scheduler.step(
            noise_pred,
            pipe.scheduler.timesteps[progress_id],
            inputs_shared["latents"],
        )
        if args.preserve_first_frame and "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        inputs_shared["latents"],
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )
    return pipe.vae_output_to_video(video)


def main() -> int:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer_config = (
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
        if args.tokenizer_path is None
        else ModelConfig(args.tokenizer_path)
    )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=parse_model_configs(args),
        tokenizer_config=tokenizer_config,
    )
    combined_checkpoint_state = None
    combined_checkpoint_path = args.checkpoint
    if combined_checkpoint_path:
        combined_checkpoint_state = load_checkpoint_state(combined_checkpoint_path)
    lora_checkpoint_path = resolve_lora_checkpoint_path(args)
    if lora_checkpoint_path:
        lora_state = combined_checkpoint_state if lora_checkpoint_path == combined_checkpoint_path else load_checkpoint_state(lora_checkpoint_path)
        maybe_load_lora(args, pipe, lora_state)
    attrs = resolve_attrs(args)
    prompt = args.prompt or light_attrs_to_prompt(attrs, task=args.task, include_values=False)
    image = Image.open(args.source).convert("RGB")
    mask = Image.open(args.mask).convert("RGB") if args.mask else None
    light_checkpoint_path = resolve_light_checkpoint_path(args)
    light_state = None
    if light_checkpoint_path:
        light_state = (
            combined_checkpoint_state
            if light_checkpoint_path == combined_checkpoint_path
            else load_checkpoint_state(light_checkpoint_path)
        )
    light_encoder = None if args.no_tokenlight_light_tokens else load_light_encoder(args, pipe, light_state)
    video = tokenlight_wan_generate(pipe, light_encoder, attrs, prompt, image, mask, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        video[0].save(output)
    else:
        save_video(video, str(output), fps=args.fps, quality=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
