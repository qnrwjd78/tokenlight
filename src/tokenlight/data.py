from __future__ import annotations

import json
from pathlib import Path
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
