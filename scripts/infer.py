from __future__ import annotations

import argparse
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
    parser.add_argument("--attrs", required=True, help="JSON file or inline JSON with TokenLight attrs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--task", choices=["spatial", "ambient", "diffuse", "fixture", "relighting"], default="relighting")
    parser.add_argument("--model_id_with_origin_paths", default="")
    parser.add_argument("--model_paths", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--tokenlight_light_checkpoint", default="")
    parser.add_argument("--mask", default="", help="Optional object/edit mask image for TokenLight mask latent tokens.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
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
            return [ModelConfig(item) if isinstance(item, str) else ModelConfig(**item) for item in payload]
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


def load_light_encoder(args, pipe) -> TokenLightAttributeTokenEncoder:
    token_dim = args.tokenlight_token_dim if args.tokenlight_token_dim > 0 else int(pipe.dit.dim)
    encoder = TokenLightAttributeTokenEncoder(
        token_dim=token_dim,
        fourier_features=args.tokenlight_fourier_features,
        fourier_sigma=args.tokenlight_fourier_sigma,
        hidden_dim=args.tokenlight_mlp_hidden_dim,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    if args.tokenlight_light_checkpoint:
        try:
            from safetensors.torch import load_file

            state = load_file(args.tokenlight_light_checkpoint)
        except Exception:
            state = torch.load(args.tokenlight_light_checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        light_state = {}
        for key, value in state.items():
            if key.startswith("light_encoder."):
                light_state[key.removeprefix("light_encoder.")] = value
        if not light_state:
            light_state = state
        encoder.load_state_dict(light_state, strict=False)
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
    attrs = load_attrs(args.attrs)
    prompt = args.prompt or light_attrs_to_prompt(attrs, task=args.task, include_values=False)
    image = Image.open(args.source).convert("RGB")
    mask = Image.open(args.mask).convert("RGB") if args.mask else None
    light_encoder = None if args.no_tokenlight_light_tokens else load_light_encoder(args, pipe)
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
