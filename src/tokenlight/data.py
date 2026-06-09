from __future__ import annotations

from pathlib import Path
import json
import random
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def resolve_absolute_path(value: str | Path, base: str | Path = ".") -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def load_tensor_image(path: str | Path) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return _to_chw_tensor(torch.load(path, map_location="cpu"), path)
    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".exr", ".hdr"}:
        try:
            import imageio.v3 as iio
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install imageio with EXR/HDR support to read linear component images.") from exc
        array = iio.imread(path)
    else:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype("float32") / 255.0
    return _to_chw_tensor(array, path)


def _to_chw_tensor(value: Any, path: Path) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float()
    else:
        array = np.ascontiguousarray(np.asarray(value))
        tensor = torch.from_numpy(array).float()
    while tensor.ndim > 3 and 1 in tensor.shape:
        dim = next(index for index, size in enumerate(tensor.shape) if size == 1)
        tensor = tensor.squeeze(dim)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
        tensor = tensor[:3]
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    elif tensor.ndim != 3:
        raise ValueError(f"Could not convert {path} to CHW image tensor; got shape {tuple(tensor.shape)}")
    return tensor.contiguous()


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _triple(value: Any) -> tuple[float, float, float]:
    if value is None:
        return (float("nan"), float("nan"), float("nan"))
    values = list(value)
    values = (values + [float("nan"), float("nan"), float("nan")])[:3]
    return _float_or_nan(values[0]), _float_or_nan(values[1]), _float_or_nan(values[2])


def _scene_dirs(root: Path) -> list[Path]:
    if (root / "scenes").exists():
        root = root / "scenes"
    return sorted(p for p in root.glob("scene_*") if (p / "meta.json").exists())


def _reinhard_tensor(tensor: torch.Tensor, exposure: float = 1.0) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor.float() * float(exposure), nan=0.0, posinf=1.0, neginf=0.0)
    tensor = tensor.clamp_min(0.0)
    return (tensor / (1.0 + tensor)).clamp(0.0, 1.0).contiguous()


def _sample_color(rng: random.Random) -> tuple[float, float, float]:
    palette = [
        (1.0, 1.0, 1.0),
        (1.0, 0.86, 0.68),
        (0.68, 0.82, 1.0),
        (1.0, 0.34, 0.24),
        (0.25, 0.50, 1.0),
        (0.35, 1.0, 0.55),
    ]
    if rng.random() < 0.65:
        return 1.0, 1.0, 1.0
    if rng.random() < 0.75:
        return tuple(float(v) for v in rng.choice(palette))
    return tuple(float(rng.uniform(0.45, 1.0)) for _ in range(3))


class RelightingComponentAdapterDataset(Dataset):
    """Adapter for `repos/relighting_dataset` TokenLight component samples.

    External samples are:

      {"input": CHW [-1,1], "target": CHW [-1,1], "condition": {...}}

    Wan export needs:

      {"source": CHW, "target": CHW, "attrs": {...}}
    """

    def __init__(
        self,
        component_root: str | Path,
        repo_path: str | Path = "repos/relighting_dataset",
        length: int = 100_000,
        modes: tuple[str, ...] = ("spatial", "ambient", "diffuse", "fixture"),
        seed: int = 1234,
        max_lights: int = 1,
        image_range: str = "minus_one_one",
        include_masks: bool = False,
        include_object_masks: bool = False,
    ) -> None:
        self.root = resolve_absolute_path(component_root)
        self.repo_path = resolve_absolute_path(repo_path)
        self.image_range = image_range
        self.include_masks = bool(include_masks)
        self.include_object_masks = bool(include_object_masks)
        if self.image_range not in {"minus_one_one", "zero_one"}:
            raise ValueError("image_range must be 'minus_one_one' or 'zero_one'")
        if int(max_lights) != 1:
            raise ValueError("Wan TokenLight export currently packs one selected light per sample.")
        if not self.repo_path.exists():
            raise FileNotFoundError(f"relighting_dataset repo not found: {self.repo_path}")

        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            from tokenlight_dataset.component_dataset import TokenLightComponentDataset
        except Exception as exc:  # pragma: no cover - requires external repo
            raise RuntimeError(
                "Could not import tokenlight_dataset from repos/relighting_dataset. "
                "Check --component-repo and container PYTHONPATH."
            ) from exc

        self.dataset = TokenLightComponentDataset(
            root=str(self.root),
            length=int(length),
            modes=tuple(modes),
            seed=int(seed),
            max_lights=1,
            return_torch=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        source = self._convert_image(sample["input"].float())
        target = self._convert_image(sample["target"].float())
        condition = dict(sample["condition"])
        item = {
            "source": source,
            "target": target,
            "attrs": self.attrs_from_condition(condition),
            "condition": condition,
        }
        mask = self._load_mask(condition)
        if mask is not None:
            item["mask"] = mask
        return item

    def _convert_image(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.image_range == "zero_one":
            return ((tensor + 1.0) * 0.5).clamp(0.0, 1.0).contiguous()
        return tensor.contiguous()

    def _load_mask(self, condition: dict[str, Any]) -> torch.Tensor | None:
        if not (self.include_masks or self.include_object_masks):
            return None
        scene_id = condition.get("scene_id")
        if not scene_id:
            return None
        scene_dir = self.root / "scenes" / str(scene_id)
        candidates: list[Path] = []
        if self.include_masks and condition.get("mask"):
            candidates.append(scene_dir / str(condition["mask"]))
        if self.include_object_masks:
            candidates.append(scene_dir / "masks" / "object_mask.png")
        for path in candidates:
            if path.exists():
                return load_tensor_image(path).float().clamp(0.0, 1.0).contiguous()
        return None

    @staticmethod
    def attrs_from_condition(condition: dict[str, Any]) -> dict[str, float]:
        task = condition.get("task")
        attrs: dict[str, float] = {}

        if task == "spatial":
            attrs["a"] = _float_or_nan(condition.get("ambient_scale"))
            lights = list(condition.get("lights") or [])
            if lights:
                light = lights[0]
                x, y, z = _triple(light.get("position"))
                r, g, b = _triple(light.get("color"))
                attrs.update(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "r": r,
                        "g": g,
                        "b": b,
                        "lambda": _float_or_nan(light.get("intensity")),
                        "d": _float_or_nan(light.get("radius")),
                    }
                )
            return attrs

        if task == "ambient":
            attrs["a"] = _float_or_nan(condition.get("ambient_scale_out", condition.get("ambient_scale_delta")))
            return attrs

        if task == "diffuse":
            r, g, b = _triple(condition.get("color"))
            attrs.update(
                {
                    "a": _float_or_nan(condition.get("ambient_scale")),
                    "dg": _float_or_nan(condition.get("spread_delta")),
                    "d": _float_or_nan(condition.get("spread_out")),
                    "r": r,
                    "g": g,
                    "b": b,
                    "lambda": _float_or_nan(condition.get("intensity")),
                }
            )
            return attrs

        if task == "fixture":
            r, g, b = _triple(condition.get("color"))
            attrs.update(
                {
                    "a": _float_or_nan(condition.get("ambient_scale")),
                    "r": r,
                    "g": g,
                    "b": b,
                    "lambda": _float_or_nan(condition.get("intensity")),
                    "t": _float_or_nan(condition.get("transition_on")),
                }
            )
            return attrs

        return attrs


class DirectPointLightPngDataset(Dataset):
    """Source/target pairs from the direct `data/sample` layout.

    `data/sample` is laid out as:

      scene_xxxxxx/spatial/point_lights/light_000.png ... light_063.png
      scene_xxxxxx/spatial/ambient.exr
      scene_xxxxxx/diffuse/spread_000.exr ...
      scene_xxxxxx/masks/object_mask.png
      scene_xxxxxx/meta.json

    Spatial point lights use the final PNGs directly. Ambient and diffuse modes
    synthesize source/target PNGs from EXR components at export time.
    """

    AMBIENT_PAIRS = ((0.006, 0.014), (0.014, 0.028), (0.028, 0.010), (0.020, 0.040))
    RANDOM_AMBIENT_RANGE = (0.006, 0.040)
    SPATIAL_SOURCE_AMBIENT_SCALE = 0.014
    DIFFUSE_AMBIENT_SCALE = 0.035
    RANDOM_DIFFUSE_AMBIENT_RANGE = (0.012, 0.060)
    DIFFUSE_INTENSITY = 0.120
    RANDOM_DIFFUSE_INTENSITY_RANGE = (0.040, 0.180)

    def __init__(
        self,
        root: str | Path,
        length: int | None = None,
        modes: tuple[str, ...] = ("spatial",),
        seed: int = 1234,
        valid_only: bool = True,
        pairing: str = "random",
        source_light_id: int = 0,
        target_light_ids: tuple[int, ...] | None = None,
        allow_self_pairs: bool = False,
        include_object_masks: bool = True,
        light_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        light_intensity: float = 1.0,
    ) -> None:
        self.root = resolve_absolute_path(root)
        self.length = None if length is None or int(length) <= 0 else int(length)
        self.modes = tuple(modes)
        self.seed = int(seed)
        self.valid_only = bool(valid_only)
        self.pairing = str(pairing)
        self.source_light_id = int(source_light_id)
        self.target_light_ids = None if target_light_ids is None else {int(value) for value in target_light_ids}
        self.allow_self_pairs = bool(allow_self_pairs)
        self.include_object_masks = bool(include_object_masks)
        self.light_color = tuple(float(v) for v in light_color)
        self.light_intensity = float(light_intensity)
        if self.pairing not in {"random", "all-targets"}:
            raise ValueError("pairing must be 'random' or 'all-targets'")
        unknown_modes = set(self.modes) - {"spatial", "ambient", "diffuse"}
        if unknown_modes:
            raise ValueError(f"Unsupported direct sample modes: {sorted(unknown_modes)}")

        self.scenes = self._load_scenes()
        if not self.scenes:
            raise FileNotFoundError(f"No point-light PNG scenes found under {self.root}")
        self.all_targets = self._build_all_targets()

    def _load_scenes(self) -> list[dict[str, Any]]:
        scenes: list[dict[str, Any]] = []
        for scene_dir in _scene_dirs(self.root):
            with (scene_dir / "meta.json").open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            spatial = meta.get("spatial", {})
            diffuse = meta.get("diffuse", {})
            lights = []
            for light in spatial.get("point_lights", []):
                if self.target_light_ids is not None and int(light.get("id", -1)) not in self.target_light_ids:
                    continue
                if self.valid_only and not light.get("valid", True):
                    continue
                image_path = scene_dir / str(light.get("render", ""))
                if not image_path.exists():
                    continue
                lights.append(dict(light))
            if len(lights) < 1:
                continue
            scenes.append(
                {
                    "dir": scene_dir,
                    "meta": meta,
                    "scene_id": meta.get("scene_id", scene_dir.name),
                    "lights": lights,
                    "lights_by_id": {int(light["id"]): light for light in lights if "id" in light},
                    "diffuse_spreads": [dict(item) for item in diffuse.get("spreads", []) if (scene_dir / item.get("render", "")).exists()],
                }
            )
        return scenes

    def _build_all_targets(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for scene_idx, scene in enumerate(self.scenes):
            if "spatial" in self.modes:
                rows.extend(("spatial", scene_idx, light_idx) for light_idx in range(len(scene["lights"])))
            if "ambient" in self.modes:
                rows.extend(("ambient", scene_idx, pair_idx) for pair_idx in range(len(self.AMBIENT_PAIRS)))
            if "diffuse" in self.modes:
                spreads = scene["diffuse_spreads"]
                rows.extend(
                    ("diffuse", scene_idx, source_idx, target_idx)
                    for source_idx in range(len(spreads))
                    for target_idx in range(len(spreads))
                    if self.allow_self_pairs or source_idx != target_idx
                )
        return rows

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if self.pairing == "all-targets":
            return len(self.all_targets)
        return len(self.scenes) * 64

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.pairing == "all-targets":
            target = self.all_targets[index % len(self.all_targets)]
            mode = target[0]
            scene = self.scenes[target[1]]
            if mode == "spatial":
                target_light = scene["lights"][target[2]]
                return self._spatial_item(scene, target_light)
            if mode == "ambient":
                a_in, a_out = self.AMBIENT_PAIRS[target[2]]
                return self._ambient_item(scene, a_in, a_out)
            if mode == "diffuse":
                return self._diffuse_item(scene, target[2], target[3], rng=None)
            raise ValueError(f"Unknown target mode: {mode}")

        rng = random.Random(self.seed + int(index))
        mode = rng.choice(self.modes)
        scene = rng.choice(self.scenes)
        if mode == "spatial":
            target_light = rng.choice(scene["lights"])
            return self._spatial_item(scene, target_light)
        if mode == "ambient":
            low, high = self.RANDOM_AMBIENT_RANGE
            return self._ambient_item(scene, rng.uniform(low, high), rng.uniform(low, high))
        if mode == "diffuse":
            spreads = scene["diffuse_spreads"]
            if len(spreads) < 2:
                return self.__getitem__(index + 1)
            source_idx, target_idx = rng.sample(range(len(spreads)), 2)
            return self._diffuse_item(scene, source_idx, target_idx, rng=rng)
        raise ValueError(f"Unknown mode: {mode}")

    def _spatial_item(self, scene: dict[str, Any], target_light: dict[str, Any]) -> dict[str, Any]:
        source = self._scene_source_image(scene)
        target = load_tensor_image(scene["dir"] / target_light["render"]).float().clamp(0.0, 1.0)
        condition = self._condition(scene, target_light)
        item = {
            "source": source,
            "target": target,
            "attrs": self.attrs_from_light(target_light),
            "condition": condition,
        }
        mask = self._load_mask(scene)
        if mask is not None:
            item["mask"] = mask
        return item

    def _ambient_item(self, scene: dict[str, Any], a_in: float, a_out: float) -> dict[str, Any]:
        ambient = load_tensor_image(scene["dir"] / scene["meta"]["spatial"]["ambient_render"]).float()
        condition = {
            "task": "ambient",
            "scene_id": scene["scene_id"],
            "ambient_scale_in": float(a_in),
            "ambient_scale_out": float(a_out),
            "ambient_scale_delta": float(a_out) / max(float(a_in), 1e-6),
            "source_relpath": f"{scene['scene_id']}/source/ambient_a_{a_in:.6f}.png",
            "target_relpath": f"{scene['scene_id']}/target/ambient_a_{a_out:.6f}.png",
            "mask_relpath": self._mask_relpath(scene),
        }
        item = {
            "source": _reinhard_tensor(a_in * ambient),
            "target": _reinhard_tensor(a_out * ambient),
            "attrs": RelightingComponentAdapterDataset.attrs_from_condition(condition),
            "condition": condition,
        }
        mask = self._load_mask(scene)
        if mask is not None:
            item["mask"] = mask
        return item

    def _diffuse_item(self, scene: dict[str, Any], source_idx: int, target_idx: int, rng: random.Random | None) -> dict[str, Any]:
        spreads = scene["diffuse_spreads"]
        source_spread = spreads[source_idx]
        target_spread = spreads[target_idx]
        color = _sample_color(rng) if rng is not None else self.light_color
        intensity = rng.uniform(*self.RANDOM_DIFFUSE_INTENSITY_RANGE) if rng is not None else self.DIFFUSE_INTENSITY
        ambient_scale = rng.uniform(*self.RANDOM_DIFFUSE_AMBIENT_RANGE) if rng is not None else self.DIFFUSE_AMBIENT_SCALE
        ambient = load_tensor_image(scene["dir"] / scene["meta"]["diffuse"]["ambient_render"]).float()
        source = load_tensor_image(scene["dir"] / source_spread["render"]).float()
        target = load_tensor_image(scene["dir"] / target_spread["render"]).float()
        color_tensor = torch.tensor(color, dtype=ambient.dtype).view(3, 1, 1)
        condition = {
            "task": "diffuse",
            "scene_id": scene["scene_id"],
            "spread_in": source_spread["normalized_spread"],
            "spread_out": target_spread["normalized_spread"],
            "spread_delta": target_spread["normalized_spread"] - source_spread["normalized_spread"],
            "color": list(color),
            "intensity": float(intensity),
            "ambient_scale": float(ambient_scale),
            "source_spread_id": int(source_spread["id"]),
            "target_spread_id": int(target_spread["id"]),
            "source_relpath": (
                f"{scene['scene_id']}/source/diffuse_spread_{int(source_spread['id']):03d}"
                f"_a_{ambient_scale:.6f}_i_{intensity:.6f}"
                f"_c_{color[0]:.4f}_{color[1]:.4f}_{color[2]:.4f}.png"
            ),
            "target_relpath": (
                f"{scene['scene_id']}/target/diffuse_spread_{int(target_spread['id']):03d}"
                f"_a_{ambient_scale:.6f}_i_{intensity:.6f}"
                f"_c_{color[0]:.4f}_{color[1]:.4f}_{color[2]:.4f}.png"
            ),
            "mask_relpath": self._mask_relpath(scene),
        }
        item = {
            "source": _reinhard_tensor(ambient_scale * ambient + intensity * source * color_tensor),
            "target": _reinhard_tensor(ambient_scale * ambient + intensity * target * color_tensor),
            "attrs": RelightingComponentAdapterDataset.attrs_from_condition(condition),
            "condition": condition,
        }
        mask = self._load_mask(scene)
        if mask is not None:
            item["mask"] = mask
        return item

    def _source_for_target(self, scene: dict[str, Any], target_light: dict[str, Any]) -> dict[str, Any]:
        if self.source_light_id in scene["lights_by_id"]:
            source = scene["lights_by_id"][self.source_light_id]
            if self.allow_self_pairs or int(source["id"]) != int(target_light["id"]):
                return source
        for light in scene["lights"]:
            if self.allow_self_pairs or int(light["id"]) != int(target_light["id"]):
                return light
        return target_light

    def _scene_source_light(self, scene: dict[str, Any]) -> dict[str, Any]:
        if self.source_light_id in scene["lights_by_id"]:
            return scene["lights_by_id"][self.source_light_id]
        return scene["lights"][0]

    def _scene_source_image(self, scene: dict[str, Any]) -> torch.Tensor:
        ambient = load_tensor_image(scene["dir"] / scene["meta"]["spatial"]["ambient_render"]).float()
        return _reinhard_tensor(self.SPATIAL_SOURCE_AMBIENT_SCALE * ambient)

    @staticmethod
    def _scene_number(scene: dict[str, Any]) -> str:
        scene_id = str(scene["scene_id"])
        return scene_id.removeprefix("scene_")

    def _source_relpath(self, scene: dict[str, Any]) -> str:
        return f"{scene['scene_id']}/source_{self._scene_number(scene)}.png"

    def _mask_relpath(self, scene: dict[str, Any]) -> str:
        return f"{scene['scene_id']}/mask_{self._scene_number(scene)}.png"

    def _load_mask(self, scene: dict[str, Any]) -> torch.Tensor | None:
        if not self.include_object_masks:
            return None
        rel = scene["meta"].get("masks", {}).get("object", "masks/object_mask.png")
        path = scene["dir"] / str(rel)
        if not path.exists():
            return None
        return load_tensor_image(path).float().clamp(0.0, 1.0).contiguous()

    def _condition(self, scene: dict[str, Any], target_light: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": "spatial",
            "scene_id": scene["scene_id"],
            "source_kind": "ambient_only",
            "source_light_id": -1,
            "target_light_id": int(target_light["id"]),
            "target_valid": bool(target_light.get("valid", True)),
            "target_position": target_light.get("canonical_position"),
            "target_render": target_light.get("render"),
            "source_relpath": self._source_relpath(scene),
            "target_relpath": f"{scene['scene_id']}/target/spatial_light_{int(target_light['id']):03d}.png",
            "mask_relpath": self._mask_relpath(scene),
        }

    def attrs_from_light(self, light: dict[str, Any]) -> dict[str, float]:
        x, y, z = _triple(light.get("canonical_position"))
        r, g, b = _triple(self.light_color)
        return {
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "g": g,
            "b": b,
            "lambda": self.light_intensity,
            "d": _float_or_nan(light.get("canonical_radius")),
        }
