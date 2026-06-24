#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenlight_dataset.exr_io import read_exr
from tokenlight_dataset.tonemap import reinhard, to_uint8


DEFAULT_PROMPT = "photorealistic object relighting, preserve geometry and materials"


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    scene_dir: Path
    valid_ids: list[int]
    meta: dict[str, Any]
    valid_lights: dict[str, Any]


def parse_csv_floats(value: str) -> list[float]:
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            result.append(float(item))
    return result


def scale_slug(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relpath(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def triple(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)):
        return default
    items = list(value)[:3]
    items += [None] * (3 - len(items))
    return tuple(finite_float(item, default[index]) for index, item in enumerate(items))  # type: ignore[return-value]


def point_lights_by_id(meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in meta.get("spatial", {}).get("point_lights", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            result[int(item["id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return result


def light_energy(light: dict[str, Any]) -> float | None:
    for key in ("canonical_energy", "energy", "world_energy", "component_energy"):
        number = finite_float(light.get(key))
        if number is not None:
            return number
    return None


def lambda_value(light: dict[str, Any], args: argparse.Namespace, energy_range: tuple[float, float] | None) -> float:
    if args.lambda_mode == "constant" or energy_range is None:
        return float(args.lambda_constant)
    energy = light_energy(light)
    if energy is None:
        return float(args.lambda_constant)
    min_energy, max_energy = energy_range
    if max_energy <= min_energy:
        return float(args.lambda_constant)
    alpha = (energy - min_energy) / (max_energy - min_energy)
    alpha = min(1.0, max(0.0, alpha))
    return float(args.lambda_min + alpha * (args.lambda_max - args.lambda_min))


def light_attrs(light: dict[str, Any], args: argparse.Namespace, energy_range: tuple[float, float] | None) -> dict[str, float]:
    x, y, z = triple(light.get("canonical_position"), (0.0, 0.0, 0.0))
    r, g, b = triple(light.get("component_color"), (1.0, 1.0, 1.0))
    radius = finite_float(light.get("canonical_radius"), finite_float(light.get("world_radius"), 0.15))
    return {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "g": g,
        "b": b,
        "lambda": lambda_value(light, args, energy_range),
        "d": float(radius if radius is not None else 0.15),
    }


def attrs_json(*, ambient_scale: float, task_code: float, lights: list[dict[str, float]]) -> str:
    payload = {
        "a": float(ambient_scale),
        "dg": 0.0,
        "t": float(task_code),
        "lights": lights,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def save_rgb(path: Path, image: np.ndarray, *, overwrite: bool, gamma: float) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(reinhard(image), gamma=gamma), mode="RGB").save(path)


def save_uint8_rgb(path: Path, image: np.ndarray, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr[..., :3], mode="RGB").save(path)


def save_depth_png(scene_dir: Path, out_path: Path, *, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return
    source_png = scene_dir / "pbr" / "depth.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if source_png.exists():
        Image.open(source_png).convert("RGB").save(out_path)
        return

    depth_exr = scene_dir / "pbr" / "depth.exr"
    depth = read_exr(depth_exr)[..., 0]
    finite = np.isfinite(depth)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [1.0, 99.0])
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    save_uint8_rgb(out_path, (norm * 255.0 + 0.5).astype(np.uint8), overwrite=True)


def read_scene_records(raw_root: Path, *, max_scenes: int | None) -> list[SceneRecord]:
    scene_root = raw_root / "scenes" if (raw_root / "scenes").is_dir() else raw_root
    records = []
    for scene_dir in sorted(scene_root.glob("scene_*")):
        if not scene_dir.is_dir():
            continue
        valid_path = scene_dir / "valid_lights.json"
        meta_path = scene_dir / "meta.json"
        if not valid_path.exists() or not meta_path.exists():
            continue
        valid_lights = load_json(valid_path)
        meta = load_json(meta_path)
        valid_ids = [int(item) for item in valid_lights.get("valid_light_ids", [])]
        records.append(
            SceneRecord(
                scene_id=scene_dir.name,
                scene_dir=scene_dir,
                valid_ids=valid_ids,
                meta=meta,
                valid_lights=valid_lights,
            )
        )
        if max_scenes is not None and len(records) >= max_scenes:
            break
    return records


def collect_energy_range(records: list[SceneRecord]) -> tuple[float, float] | None:
    energies = []
    for record in records:
        lights = point_lights_by_id(record.meta)
        for light_id in record.valid_ids:
            energy = light_energy(lights.get(light_id, {}))
            if energy is not None:
                energies.append(energy)
    if not energies:
        return None
    return min(energies), max(energies)


def select_single_ids(valid_ids: list[int], args: argparse.Namespace, rng: random.Random) -> list[int]:
    if args.single_lights_per_scene <= 0 or args.single_lights_per_scene >= len(valid_ids):
        return list(valid_ids)
    return sorted(rng.sample(valid_ids, args.single_lights_per_scene))


def select_double_pairs(valid_ids: list[int], args: argparse.Namespace, rng: random.Random) -> list[tuple[int, int]]:
    if len(valid_ids) < 2:
        return []
    all_pairs = [(a, b) for index, a in enumerate(valid_ids) for b in valid_ids[index + 1 :]]
    if args.double_pairs_per_scene < 0:
        return all_pairs
    count = len(valid_ids) if args.double_pairs_per_scene == 0 else args.double_pairs_per_scene
    if count >= len(all_pairs):
        return all_pairs
    return sorted(rng.sample(all_pairs, count))


def task_pbr_mode(aux_type: str, args: argparse.Namespace, rng: random.Random) -> str:
    if aux_type == "shading":
        return "target"
    return "target" if rng.random() < float(args.depth_target_prob) else "condition"


def copy_scene_metadata(record: SceneRecord, output_root: Path) -> None:
    out_scene = output_root / "scenes" / record.scene_id
    out_scene.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record.scene_dir / "meta.json", out_scene / "meta.json")
    shutil.copy2(record.scene_dir / "valid_lights.json", out_scene / "valid_lights.json")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = repo_path(args.raw_root)
    output_name = args.output_root or f"data/objaverse_pbr_{args.aux_type}"
    output_root = repo_path(output_name)
    metadata_path = output_root / "metadata.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)

    records = read_scene_records(raw_root, max_scenes=args.max_scenes)
    energy_range = collect_energy_range(records) if args.lambda_mode == "energy_minmax" else None
    selection_rng = random.Random(args.seed)
    mode_rng = random.Random(args.seed + 1000003)

    rows: list[dict[str, Any]] = []
    counts = {"single_light": 0, "double_light": 0, "ambient_only": 0}
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record.scene_id}: valid_lights={len(record.valid_ids)}")
        copy_scene_metadata(record, output_root)
        lights = point_lights_by_id(record.meta)
        ambient = read_exr(record.scene_dir / "spatial" / "ambient.exr")
        source_path = output_root / "images" / record.scene_id / "source" / "ambient.png"
        save_rgb(source_path, ambient, overwrite=args.overwrite_images, gamma=args.gamma)

        point_cache: dict[int, np.ndarray] = {}

        def point(light_id: int) -> np.ndarray:
            if light_id not in point_cache:
                point_cache[light_id] = read_exr(record.scene_dir / "spatial" / "point_lights" / f"light_{light_id:03d}.exr")
            return point_cache[light_id]

        pbr_depth_path = output_root / "images" / record.scene_id / "depth" / "depth.png"
        if args.aux_type == "depth":
            save_depth_png(record.scene_dir, pbr_depth_path, overwrite=args.overwrite_images)

        for light_id in select_single_ids(record.valid_ids, args, selection_rng):
            if light_id not in lights:
                continue
            light_image = point(light_id)
            target_path = output_root / "images" / record.scene_id / "rgb" / f"single_light_{light_id:03d}.png"
            save_rgb(target_path, ambient + light_image, overwrite=args.overwrite_images, gamma=args.gamma)
            if args.aux_type == "shading":
                pbr_path = output_root / "images" / record.scene_id / "shading" / f"single_light_{light_id:03d}.png"
                save_rgb(pbr_path, light_image, overwrite=args.overwrite_images, gamma=args.gamma)
            else:
                pbr_path = pbr_depth_path
            rows.append(
                {
                    "scene_id": record.scene_id,
                    "task": "single_light",
                    "light_id": light_id,
                    "light_ids": [light_id],
                    "input_image": relpath(source_path, output_root),
                    "video": relpath(target_path, output_root),
                    "pbr_image": relpath(pbr_path, output_root),
                    "pbr_aux_type": args.aux_type,
                    "pbr_mode": task_pbr_mode(args.aux_type, args, mode_rng),
                    "prompt": DEFAULT_PROMPT,
                    "attrs_json": attrs_json(
                        ambient_scale=1.0,
                        task_code=1.0,
                        lights=[light_attrs(lights[light_id], args, energy_range)],
                    ),
                    "valid": True,
                }
            )
            counts["single_light"] += 1

        for first_id, second_id in select_double_pairs(record.valid_ids, args, selection_rng):
            if first_id not in lights or second_id not in lights:
                continue
            first = point(first_id)
            second = point(second_id)
            target_path = output_root / "images" / record.scene_id / "rgb" / f"double_light_{first_id:03d}_{second_id:03d}.png"
            save_rgb(target_path, ambient + first + second, overwrite=args.overwrite_images, gamma=args.gamma)
            if args.aux_type == "shading":
                pbr_path = output_root / "images" / record.scene_id / "shading" / f"double_light_{first_id:03d}_{second_id:03d}.png"
                save_rgb(pbr_path, first + second, overwrite=args.overwrite_images, gamma=args.gamma)
            else:
                pbr_path = pbr_depth_path
            rows.append(
                {
                    "scene_id": record.scene_id,
                    "task": "double_light",
                    "light_id": None,
                    "light_ids": [first_id, second_id],
                    "input_image": relpath(source_path, output_root),
                    "video": relpath(target_path, output_root),
                    "pbr_image": relpath(pbr_path, output_root),
                    "pbr_aux_type": args.aux_type,
                    "pbr_mode": task_pbr_mode(args.aux_type, args, mode_rng),
                    "prompt": DEFAULT_PROMPT,
                    "attrs_json": attrs_json(
                        ambient_scale=1.0,
                        task_code=2.0,
                        lights=[
                            light_attrs(lights[first_id], args, energy_range),
                            light_attrs(lights[second_id], args, energy_range),
                        ],
                    ),
                    "valid": True,
                }
            )
            counts["double_light"] += 1

        for ambient_scale in parse_csv_floats(args.ambient_scales):
            target_path = output_root / "images" / record.scene_id / "rgb" / f"ambient_only_a{scale_slug(ambient_scale)}.png"
            save_rgb(target_path, ambient * float(ambient_scale), overwrite=args.overwrite_images, gamma=args.gamma)
            if args.aux_type == "shading":
                pbr_path = output_root / "images" / record.scene_id / "shading" / f"ambient_only_a{scale_slug(ambient_scale)}.png"
                save_uint8_rgb(pbr_path, np.zeros_like(to_uint8(reinhard(ambient), gamma=args.gamma)), overwrite=args.overwrite_images)
            else:
                pbr_path = pbr_depth_path
            rows.append(
                {
                    "scene_id": record.scene_id,
                    "task": "ambient_only",
                    "light_id": None,
                    "light_ids": [],
                    "input_image": relpath(source_path, output_root),
                    "video": relpath(target_path, output_root),
                    "pbr_image": relpath(pbr_path, output_root),
                    "pbr_aux_type": args.aux_type,
                    "pbr_mode": task_pbr_mode(args.aux_type, args, mode_rng),
                    "prompt": DEFAULT_PROMPT,
                    "attrs_json": attrs_json(ambient_scale=float(ambient_scale), task_code=0.0, lights=[]),
                    "valid": True,
                }
            )
            counts["ambient_only"] += 1

    with metadata_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")

    summary = {
        "raw_root": raw_root.as_posix(),
        "output_root": output_root.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "aux_type": args.aux_type,
        "scene_count": len(records),
        "row_count": len(rows),
        "task_counts": counts,
        "lambda_mode": args.lambda_mode,
        "lambda_energy_range": list(energy_range) if energy_range is not None else None,
        "depth_target_prob": args.depth_target_prob if args.aux_type == "depth" else None,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build TokenLight Objaverse PBR PNG training metadata.")
    p.add_argument("--raw-root", default="data/objaverse_sample_completed_20260624_144354")
    p.add_argument("--output-root", default=None)
    p.add_argument("--aux-type", choices=("shading", "depth"), default="shading")
    p.add_argument("--single-lights-per-scene", type=int, default=0, help="0 means all valid lights.")
    p.add_argument("--double-pairs-per-scene", type=int, default=0, help="0 means one pair per valid light; -1 means all pairs.")
    p.add_argument("--ambient-scales", default="0.5,0.75,1.0,1.25")
    p.add_argument("--depth-target-prob", type=float, default=0.5)
    p.add_argument("--lambda-mode", choices=("constant", "energy_minmax"), default="energy_minmax")
    p.add_argument("--lambda-constant", type=float, default=1.0)
    p.add_argument("--lambda-min", type=float, default=0.25)
    p.add_argument("--lambda-max", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--overwrite-images", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
