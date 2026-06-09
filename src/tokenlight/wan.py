from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


LIGHT_TOKEN_NAMES: tuple[str, ...] = ("a", "dg", "x", "y", "z", "r", "g", "b", "lambda", "d", "t")
ATTR_ALIASES = {
    "ambient": "a",
    "ambient_scale": "a",
    "ambient_scale_out": "a",
    "ambient_scale_delta": "a",
    "diffuse_gain": "dg",
    "spread_delta": "dg",
    "intensity": "lambda",
    "radius": "d",
    "spread": "d",
    "spread_out": "d",
    "transition": "t",
    "transition_on": "t",
    "red": "r",
    "green": "g",
    "blue": "b",
}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _triple(value: Any) -> tuple[float | None, float | None, float | None]:
    if value is None:
        return None, None, None
    values = list(value) if isinstance(value, Iterable) and not isinstance(value, (str, bytes)) else [value]
    values = (values + [None, None, None])[:3]
    return _finite_float(values[0]), _finite_float(values[1]), _finite_float(values[2])


def compact_attrs(attrs: Mapping[str, Any] | None) -> dict[str, float]:
    """Convert component-dataset condition names to TokenLight paper attr names."""

    if not attrs:
        return {}
    result: dict[str, float] = {}

    if "position" in attrs:
        x, y, z = _triple(attrs.get("position"))
        for name, value in zip(("x", "y", "z"), (x, y, z), strict=True):
            if value is not None:
                result[name] = value
    if "color" in attrs:
        r, g, b = _triple(attrs.get("color"))
        for name, value in zip(("r", "g", "b"), (r, g, b), strict=True):
            if value is not None:
                result[name] = value
    if "lights" in attrs:
        lights = list(attrs.get("lights") or [])
        if lights:
            result.update(compact_attrs(lights[0]))

    for key, value in attrs.items():
        name = ATTR_ALIASES.get(str(key), str(key))
        if name not in LIGHT_TOKEN_NAMES:
            continue
        number = _finite_float(value)
        if number is not None:
            result[name] = number
    return result


def attrs_json(attrs: Mapping[str, Any] | None) -> str:
    return json.dumps(compact_attrs(attrs), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parse_attrs_json(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return compact_attrs(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return compact_attrs(dict(value))
    text = value.strip()
    if not text:
        return {}
    return compact_attrs(json.loads(text))


def attrs_from_batch(data: Mapping[str, Any], key: str = "attrs_json") -> list[dict[str, float]]:
    value = data.get(key, data.get("attrs", None))
    if value is None:
        return [{}]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return [parse_attrs_json(value)]
    if isinstance(value, Sequence):
        return [parse_attrs_json(item) for item in value]
    return [parse_attrs_json(value)]


def light_attrs_to_prompt(
    attrs: Mapping[str, Any] | None,
    *,
    task: str = "relighting",
    prefix: str | None = None,
    include_values: bool = False,
) -> str:
    compact = compact_attrs(attrs)
    task_text = {
        "spatial": "apply a localized point-light edit",
        "ambient": "apply a global ambient illumination edit",
        "diffuse": "apply a diffuse lighting spread edit",
        "fixture": "apply a fixture lighting transition",
        "relighting": "apply the requested relighting edit",
    }.get(task, "apply the requested relighting edit")
    prompt = f"photorealistic object relighting, preserve geometry and materials, {task_text}"
    if prefix:
        prompt = f"{prefix.strip()} {prompt}"
    if include_values and compact:
        values = ", ".join(f"{name}={compact[name]:.3f}" for name in LIGHT_TOKEN_NAMES if name in compact)
        prompt = f"{prompt}; light attributes: {values}"
    return prompt


class GaussianFourierFeatures(nn.Module):
    def __init__(self, features: int = 512, sigma: float = 5.0, num_attributes: int = 1) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        if num_attributes <= 0:
            raise ValueError("num_attributes must be positive")
        self.register_buffer("weight", torch.randn(int(num_attributes), int(features)) * float(sigma))

    @property
    def out_dim(self) -> int:
        return int(self.weight.shape[-1]) * 2

    def forward(self, values: torch.Tensor, attribute_indices: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1, 1)
        attribute_indices = attribute_indices.reshape(-1).to(device=values.device, dtype=torch.long)
        if attribute_indices.numel() != values.shape[0]:
            raise ValueError(
                f"Got {attribute_indices.numel()} attribute indices for {values.shape[0]} scalar values"
            )
        weight = self.weight.to(device=values.device, dtype=values.dtype)[attribute_indices]
        phases = 2.0 * math.pi * values * weight
        return torch.cat([phases.sin(), phases.cos()], dim=-1)


class TokenLightAttributeTokenEncoder(nn.Module):
    """Encode TokenLight numeric light attributes as DiT self-attention tokens."""

    def __init__(
        self,
        token_dim: int,
        *,
        token_names: Sequence[str] = LIGHT_TOKEN_NAMES,
        fourier_features: int = 512,
        fourier_sigma: float = 5.0,
        hidden_dim: int | None = None,
        null_value: float = -1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_names = tuple(token_names)
        self.token_dim = int(token_dim)
        self.null_value = float(null_value)
        self.dropout = float(dropout)
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        hidden = int(hidden_dim or token_dim)
        self.fourier = GaussianFourierFeatures(
            fourier_features,
            fourier_sigma,
            num_attributes=len(self.token_names),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(self.fourier.out_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.token_dim),
        )

    def forward(
        self,
        attrs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        drop_light: bool | torch.Tensor | Sequence[bool] = False,
    ) -> torch.Tensor:
        attrs_list = self._attrs_list(attrs, batch_size=batch_size)
        batch = len(attrs_list)
        device = torch.device(device) if device is not None else self.value_mlp[0].weight.device
        dtype = dtype or self.value_mlp[0].weight.dtype
        values = torch.full((batch, len(self.token_names)), float("nan"), device=device, dtype=torch.float32)
        for row, item in enumerate(attrs_list):
            compact = compact_attrs(item)
            for col, name in enumerate(self.token_names):
                number = _finite_float(compact.get(name))
                if number is not None:
                    values[row, col] = number

        valid = torch.isfinite(values)
        drop_mask = self._drop_mask(drop_light, batch=batch, device=device)
        if self.training and self.dropout > 0:
            drop_mask = drop_mask | (torch.rand(batch, device=device) < self.dropout)
        valid = valid & ~drop_mask[:, None]

        # ZeRO-3 requires every rank to traverse the same trainable modules in
        # the same order. Encode all attribute slots, then mask invalid/dropped
        # tokens, so rank-local CFG dropout cannot skip this MLP on one rank.
        safe_values = torch.nan_to_num(values, nan=0.0).to(dtype=dtype)
        attribute_indices = torch.arange(len(self.token_names), device=device)
        attribute_indices = attribute_indices.expand(batch, -1).reshape(-1)
        encoded = self.fourier(safe_values.reshape(-1), attribute_indices)
        encoded = self.value_mlp(encoded).reshape(batch, len(self.token_names), self.token_dim)
        null_tokens = torch.full_like(encoded, self.null_value)
        return torch.where(valid[..., None], encoded, null_tokens)

    def _attrs_list(
        self,
        attrs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
        *,
        batch_size: int | None,
    ) -> list[dict[str, float]]:
        if attrs is None:
            attrs_list: list[dict[str, float]] = [{}]
        elif isinstance(attrs, Mapping) or isinstance(attrs, (str, bytes)):
            attrs_list = [parse_attrs_json(attrs)]
        else:
            attrs_list = [parse_attrs_json(item) for item in attrs]
        if batch_size is None:
            return attrs_list
        if len(attrs_list) == batch_size:
            return attrs_list
        if len(attrs_list) == 1:
            return attrs_list * int(batch_size)
        raise ValueError(f"Got {len(attrs_list)} attr records for batch size {batch_size}")

    @staticmethod
    def _drop_mask(drop_light: bool | torch.Tensor | Sequence[bool], *, batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(drop_light, torch.Tensor):
            mask = drop_light.to(device=device, dtype=torch.bool).flatten()
            if mask.numel() == 1:
                return mask.expand(batch)
            if mask.numel() != batch:
                raise ValueError(f"drop_light has {mask.numel()} values for batch size {batch}")
            return mask
        if isinstance(drop_light, Sequence) and not isinstance(drop_light, (str, bytes)):
            mask = torch.tensor(list(drop_light), device=device, dtype=torch.bool).flatten()
            if mask.numel() != batch:
                raise ValueError(f"drop_light has {mask.numel()} values for batch size {batch}")
            return mask
        return torch.full((batch,), bool(drop_light), device=device, dtype=torch.bool)


def _repeat_to_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"Cannot expand batch {tensor.shape[0]} to {batch}")
    return tensor.expand(batch, *tensor.shape[1:]).contiguous()


def _condition_latents_for_patchify(dit: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    in_channels = int(getattr(getattr(dit, "patch_embedding", None), "in_channels", latents.shape[1]))
    if latents.shape[1] == in_channels:
        return latents
    if latents.shape[1] > in_channels:
        return latents[:, :in_channels].contiguous()
    padding = torch.zeros(
        latents.shape[0],
        in_channels - latents.shape[1],
        *latents.shape[2:],
        device=latents.device,
        dtype=latents.dtype,
    )
    return torch.cat([latents, padding], dim=1)


def _patch_to_tokens(dit: nn.Module, latents: torch.Tensor, batch: int, control_camera_latents_input=None):
    from einops import rearrange

    latents = _repeat_to_batch(latents, batch)
    latents = _condition_latents_for_patchify(dit, latents)
    patches = dit.patchify(latents, control_camera_latents_input)
    f, h, w = patches.shape[2:]
    tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    return tokens, (f, h, w)


def _freqs_for_grid(dit: nn.Module, grid: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    f, h, w = grid
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    )
    return freqs.reshape(f * h * w, 1, -1).to(device)


def _assert_same_sequence_layout(prefix_len: int, sequence_len: int, device: torch.device) -> None:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    local = torch.tensor([prefix_len, sequence_len], device=device, dtype=torch.long)
    gathered = [torch.zeros_like(local) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, local)
    layouts = torch.stack(gathered, dim=0)
    if not torch.all(layouts == layouts[0]).item():
        rank = torch.distributed.get_rank()
        raise RuntimeError(
            f"Rank {rank} saw TokenLight sequence layout {local.tolist()}, "
            f"but all ranks saw {layouts.cpu().tolist()}"
        )


def _clean_prefix_t_mod(dit: nn.Module, prefix_len: int, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    timesteps = torch.zeros(prefix_len, dtype=dtype, device=device)
    clean = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timesteps).unsqueeze(0))
    clean = dit.time_projection(clean).unflatten(2, (6, dit.dim))
    return clean.expand(batch, -1, -1, -1).contiguous()


def _has_deepspeed_zero3_params(module: nn.Module) -> bool:
    return any(hasattr(param, "ds_id") for param in module.parameters(recurse=True))


def _with_checkpoint_input_grad(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    if any(isinstance(item, torch.Tensor) and item.requires_grad for item in inputs):
        return inputs
    patched = list(inputs)
    for index, item in enumerate(patched):
        if isinstance(item, torch.Tensor) and torch.is_floating_point(item):
            patched[index] = item.detach().requires_grad_(True)
            return tuple(patched)
    return inputs


def gradient_checkpoint_forward_compatible(
    module: nn.Module,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    *inputs: Any,
) -> Any:
    if not use_gradient_checkpointing:
        return module(*inputs)
    if _has_deepspeed_zero3_params(module):
        # DiffSynth's regular checkpoint path uses PyTorch non-reentrant checkpointing.
        # With ZeRO-3, recompute-time saved tensors can observe partitioned params as
        # shape [0], so use the reentrant path for ZeRO-managed modules.
        return checkpoint(module, *_with_checkpoint_input_grad(inputs), use_reentrant=True)

    from diffsynth.core.gradient import gradient_checkpoint_forward

    return gradient_checkpoint_forward(
        module,
        use_gradient_checkpointing,
        use_gradient_checkpointing_offload,
        *inputs,
    )


def tokenlight_model_fn_wan_video(
    *,
    dit: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    motion_controller: nn.Module | None = None,
    motion_bucket_id: torch.Tensor | None = None,
    tokenlight_light_encoder: TokenLightAttributeTokenEncoder | None = None,
    tokenlight_attrs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    tokenlight_drop_light: bool | torch.Tensor | Sequence[bool] = False,
    tokenlight_source_latents: torch.Tensor | None = None,
    tokenlight_mask_latents: torch.Tensor | None = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Wan model_fn with TokenLight paper-style self-attention prefix tokens.

    The DiT sequence is:
    [source latent tokens] + [mask latent tokens] + [lighting attr tokens] + [noisy target latent tokens].
    Wan native image conditioning tensors are still accepted so the public TI2V checkpoint keeps its expected channels.
    """

    del kwargs
    from einops import rearrange
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    if latents is None or timestep is None or context is None:
        raise ValueError("dit, latents, timestep, and context are required")

    if getattr(dit, "seperated_timestep", False) and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t_head = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t_head).unflatten(2, (6, dit.dim))
    else:
        t_head = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t_head).unflatten(1, (6, dit.dim))

    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)
    x = latents
    batch = context.shape[0]
    if x.shape[0] != batch:
        x = torch.cat([x] * batch, dim=0)
    if timestep.shape[0] != batch and timestep.ndim == 1 and timestep.numel() == 1:
        timestep = torch.cat([timestep] * batch, dim=0)

    if y is not None and getattr(dit, "require_vae_embedding", True):
        y = _repeat_to_batch(y, batch)
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and getattr(dit, "require_clip_embedding", True):
        clip_feature = _repeat_to_batch(clip_feature, batch)
        context = torch.cat([dit.img_emb(clip_feature), context], dim=1)

    patches = dit.patchify(x, control_camera_latents_input)
    target_grid = patches.shape[2:]
    target_tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    target_freqs = _freqs_for_grid(dit, target_grid, target_tokens.device)

    prefix_tokens: list[torch.Tensor] = []
    prefix_freqs: list[torch.Tensor] = []

    if tokenlight_source_latents is not None:
        source_tokens, source_grid = _patch_to_tokens(dit, tokenlight_source_latents, batch)
        prefix_tokens.append(source_tokens)
        prefix_freqs.append(_freqs_for_grid(dit, source_grid, target_tokens.device))

    if tokenlight_mask_latents is not None:
        mask_tokens, mask_grid = _patch_to_tokens(dit, tokenlight_mask_latents, batch)
        prefix_tokens.append(mask_tokens)
        prefix_freqs.append(_freqs_for_grid(dit, mask_grid, target_tokens.device))

    if tokenlight_light_encoder is not None:
        light_tokens_real = tokenlight_light_encoder(
            tokenlight_attrs,
            batch_size=batch,
            device=target_tokens.device,
            dtype=target_tokens.dtype,
            drop_light=False,
        )
        drop_mask = tokenlight_light_encoder._drop_mask(
            tokenlight_drop_light,
            batch=batch,
            device=target_tokens.device,
        )
        null_light = torch.full_like(light_tokens_real, tokenlight_light_encoder.null_value)
        light_tokens = torch.where(drop_mask[:, None, None], null_light, light_tokens_real)
        prefix_tokens.append(light_tokens)
        light_freqs = torch.ones(
            light_tokens.shape[1],
            1,
            target_freqs.shape[-1],
            device=target_tokens.device,
            dtype=target_freqs.dtype,
        )
        prefix_freqs.append(light_freqs)

    if prefix_tokens:
        prefix_len = sum(tokens.shape[1] for tokens in prefix_tokens)
        x = torch.cat([*prefix_tokens, target_tokens], dim=1)
        freqs = torch.cat([*prefix_freqs, target_freqs], dim=0)
        if t_mod.ndim == 4:
            clean_t_mod = _clean_prefix_t_mod(dit, prefix_len, t_mod.shape[0], t_mod.dtype, t_mod.device)
            t_mod = torch.cat([clean_t_mod, t_mod], dim=1)
    else:
        prefix_len = 0
        x = target_tokens
        freqs = target_freqs

    _assert_same_sequence_layout(prefix_len, x.shape[1], x.device)

    for block in dit.blocks:
        x = gradient_checkpoint_forward_compatible(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
            context,
            t_mod,
            freqs,
        )

    if prefix_len:
        x = x[:, prefix_len:]
    x = dit.head(x, t_head)
    x = dit.unpatchify(x, target_grid)
    return x
