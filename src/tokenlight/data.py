from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .color import compose_diffuse_pair, compose_relight, reinhard_tonemap


def load_tensor_image(path: str | Path) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        tensor = torch.load(path, map_location="cpu")
        return tensor.float()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".exr", ".hdr"}:
        try:
            import imageio.v3 as iio
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install with `pip install -e .[exr]` to read EXR/HDR images.") from exc
        array = iio.imread(path)
    else:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(np.asarray(array)).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    return tensor.contiguous()


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def resolve_absolute_path(value: str | Path, base: str | Path = ".") -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


class TokenLightManifestDataset(Dataset):
    """JSONL dataset of aligned (I, DeltaL, Ir) samples."""

    def __init__(self, manifest: str | Path, root: str | Path = "."):
        self.manifest = Path(manifest)
        self.root = Path(root)
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source = load_tensor_image(resolve_path(self.root, record["source"]))
        target = load_tensor_image(resolve_path(self.root, record["target"]))
        mask_path = resolve_path(self.root, record.get("mask"))
        item = {
            "source": source,
            "target": target,
            "attrs": dict(record.get("attrs", {})),
        }
        if mask_path is not None:
            item["mask"] = load_tensor_image(mask_path)
        return item


class ComponentRelightDataset(Dataset):
    """On-the-fly component composition for spatial and fixture data.

    Each record must contain `ambient` and `contribution` component paths. The
    dataset samples paper-style `a`, `lambda`, and `c` values during loading and
    returns the tone-mapped ambient source plus target relit image.
    """

    def __init__(self, manifest: str | Path, root: str | Path = "."):
        self.manifest = Path(manifest)
        self.root = Path(root)
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        ambient = load_tensor_image(resolve_path(self.root, record["ambient"]))
        contribution = load_tensor_image(resolve_path(self.root, record["contribution"]))
        attrs = dict(record.get("attrs", {}))
        a = float(attrs.get("a", torch.rand(()).item()))
        lam = float(attrs.get("lambda", torch.rand(()).item()))
        color = torch.tensor(
            [
                float(attrs.get("r", torch.rand(()).item())),
                float(attrs.get("g", torch.rand(()).item())),
                float(attrs.get("b", torch.rand(()).item())),
            ],
            dtype=ambient.dtype,
        )
        source = reinhard_tonemap(a * ambient).clamp(0.0, 1.0)
        target = compose_relight(ambient, contribution, a, lam, color)
        attrs.update({"a": a, "lambda": lam, "r": float(color[0]), "g": float(color[1]), "b": float(color[2])})
        item = {"source": source, "target": target, "attrs": attrs}
        mask_path = resolve_path(self.root, record.get("mask"))
        if mask_path is not None:
            item["mask"] = load_tensor_image(mask_path)
        return item


class DiffuseSpreadDataset(Dataset):
    """On-the-fly diffuse-level pair composition.

    Each record must contain:

      ambient: path
      spreads: [{"value": 0.0, "path": "..."}, ...]

    The dataset samples two spread levels and returns `dg = target - source`,
    matching the paper's diffuse-control supervision.
    """

    def __init__(self, manifest: str | Path, root: str | Path = "."):
        self.manifest = Path(manifest)
        self.root = Path(root)
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        spreads = record["spreads"]
        if len(spreads) < 2:
            raise ValueError("DiffuseSpreadDataset requires at least two spread renders per record")
        source_index = int(torch.randint(0, len(spreads), ()).item())
        target_index = int(torch.randint(0, len(spreads) - 1, ()).item())
        if target_index >= source_index:
            target_index += 1

        ambient = load_tensor_image(resolve_path(self.root, record["ambient"]))
        source_spread = load_tensor_image(resolve_path(self.root, spreads[source_index]["path"]))
        target_spread = load_tensor_image(resolve_path(self.root, spreads[target_index]["path"]))

        source_value = float(spreads[source_index]["value"])
        target_value = float(spreads[target_index]["value"])
        attrs = dict(record.get("attrs", {}))
        a = float(attrs.get("a", torch.rand(()).item()))
        lam = float(attrs.get("lambda", torch.rand(()).item()))
        ambient_color = torch.tensor(
            [
                float(attrs.get("ambient_r", 1.0)),
                float(attrs.get("ambient_g", 1.0)),
                float(attrs.get("ambient_b", 1.0)),
            ],
            dtype=ambient.dtype,
        )
        light_color = torch.tensor(
            [
                float(attrs.get("r", torch.rand(()).item())),
                float(attrs.get("g", torch.rand(()).item())),
                float(attrs.get("b", torch.rand(()).item())),
            ],
            dtype=ambient.dtype,
        )
        source, target = compose_diffuse_pair(
            ambient=ambient,
            source_spread=source_spread,
            target_spread=target_spread,
            ambient_scale=a,
            ambient_color=ambient_color,
            intensity=lam,
            light_color=light_color,
        )
        attrs.update(
            {
                "a": a,
                "lambda": lam,
                "r": float(light_color[0]),
                "g": float(light_color[1]),
                "b": float(light_color[2]),
                "dg": target_value - source_value,
                "d": target_value,
            }
        )
        return {"source": source, "target": target, "attrs": attrs}


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


class RelightingComponentAdapterDataset(Dataset):
    """Adapter for `repos/relighting_dataset` component samples.

    The external repo synthesizes pairs as:

      {"input": CHW [-1,1], "target": CHW [-1,1], "condition": {...}}

    TokenLight training expects:

      {"source": CHW, "target": CHW, "attrs": {...}, "mask": optional CHW}

    This adapter keeps the external repo untouched and performs only the schema
    translation needed by this training code.
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
        include_masks: bool = True,
        include_object_masks: bool = True,
    ) -> None:
        self.root = resolve_absolute_path(component_root)
        self.repo_path = resolve_absolute_path(repo_path)
        self.image_range = image_range
        self.include_masks = bool(include_masks)
        self.include_object_masks = bool(include_object_masks)
        if self.image_range not in {"minus_one_one", "zero_one"}:
            raise ValueError("image_range must be 'minus_one_one' or 'zero_one'")
        if int(max_lights) != 1:
            raise ValueError(
                "The current TokenLight tokenizer has one x/y/z/r/g/b/lambda/d slot. "
                "Use max_lights=1 until multi-light token packing is implemented."
            )
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
                "Check --component-repo and the container PYTHONPATH."
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
        item: dict[str, Any] = {
            "source": source,
            "target": target,
            "attrs": self.attrs_from_condition(condition),
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
        if not self.include_masks:
            return None
        scene_id = condition.get("scene_id")
        if not scene_id:
            return None
        scene_dir = self.root / "scenes" / str(scene_id)
        candidates: list[Path] = []
        if condition.get("mask"):
            candidates.append(scene_dir / str(condition["mask"]))
        if self.include_object_masks:
            candidates.append(scene_dir / "masks" / "object_mask.png")
        for path in candidates:
            if path.exists():
                mask = load_tensor_image(path).float()
                if self.image_range == "minus_one_one":
                    mask = mask * 2.0 - 1.0
                return mask.contiguous()
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


def collate_tokenlight(batch: list[dict[str, Any]]) -> dict[str, Any]:
    source = torch.stack([item["source"] for item in batch])
    target = torch.stack([item["target"] for item in batch])
    out: dict[str, Any] = {"source": source, "target": target}
    if all("mask" in item for item in batch):
        out["mask"] = torch.stack([item["mask"] for item in batch])
    keys = sorted({key for item in batch for key in item.get("attrs", {}).keys()})
    attrs: dict[str, torch.Tensor] = {}
    for key in keys:
        values = []
        for item in batch:
            value = item.get("attrs", {}).get(key, float("nan"))
            values.append(float(value) if value is not None else float("nan"))
        attrs[key] = torch.tensor(values, dtype=torch.float32)
    out["attrs"] = attrs
    return out


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            out[key] = {
                attr_key: attr_value.to(device, non_blocking=True) if isinstance(attr_value, torch.Tensor) else attr_value
                for attr_key, attr_value in value.items()
            }
        else:
            out[key] = value
    return out
