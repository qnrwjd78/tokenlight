#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKENLIGHT_PROMPT = "photorealistic object relighting, preserve geometry and materials"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TokenLight evaluation manifest.")
    parser.add_argument("--raw-root", default="data/sample_exr_hf/sample_exr")
    parser.add_argument("--sample-png-root", default="outputs/sample_exr_png")
    parser.add_argument("--metadata", default="outputs/sample_exr_png/metadata.jsonl")
    parser.add_argument("--output-root", default="outputs/eval500")
    parser.add_argument("--manifest-name", default="eval500_manifest.jsonl")
    parser.add_argument("--summary-name", default="eval500_summary.json")
    parser.add_argument("--scene-count", type=int, default=50)
    parser.add_argument("--lights-per-scene", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt", default=TOKENLIGHT_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_metadata(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "single_light":
                continue
            scene_id = str(row.get("scene_id") or "")
            if not scene_id:
                raise ValueError(f"Missing scene_id in {path}:{line_number}")
            light_id = int(row["light_id"])
            rows.setdefault(scene_id, {})[light_id] = row
    return rows


def read_exr_rgb(path: Path) -> np.ndarray:
    from tokenlight_dataset.exr_io import read_exr

    return read_exr(path)


def save_tonemapped_png(linear: np.ndarray, path: Path) -> None:
    from tokenlight_dataset.tonemap import reinhard, to_uint8

    image = np.asarray(linear)
    while image.ndim > 3 and 1 in image.shape[:-1]:
        axis = next(index for index, size in enumerate(image.shape[:-1]) if size == 1)
        image = np.squeeze(image, axis=axis)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 image for {path}, got shape {image.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(reinhard(image[..., :3])), mode="RGB").save(path, compress_level=1)


def convert_common_source(raw_root: Path, output_root: Path, scene_id: str, overwrite: bool) -> tuple[Path, Path]:
    raw_path = raw_root / scene_id / "spatial" / "ambient.exr"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw source EXR: {raw_path}")
    png_path = output_root / "common_sources" / scene_id / "source.png"
    if overwrite or not png_path.exists():
        save_tonemapped_png(read_exr_rgb(raw_path), png_path)
    return raw_path, png_path


def selected_light_ids(scene_id: str, available_ids: list[int], count: int, seed: int) -> list[int]:
    if len(available_ids) < count:
        raise ValueError(f"{scene_id}: requested {count} lights but only {len(available_ids)} are available")
    rng = random.Random(f"{seed}:{scene_id}")
    return sorted(rng.sample(sorted(available_ids), count))


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_root = resolve(args.raw_root)
    sample_png_root = resolve(args.sample_png_root)
    metadata_path = resolve(args.metadata)
    output_root = resolve(args.output_root)

    metadata = load_metadata(metadata_path)
    scene_ids = sorted(metadata)[: int(args.scene_count)]
    if len(scene_ids) < int(args.scene_count):
        raise ValueError(f"Requested {args.scene_count} scenes but only found {len(scene_ids)}")

    rows: list[dict[str, Any]] = []
    selected_by_scene: dict[str, list[int]] = {}
    for scene_id in scene_ids:
        raw_source, source_png = convert_common_source(raw_root, output_root, scene_id, bool(args.overwrite))
        light_ids = selected_light_ids(
            scene_id,
            sorted(metadata[scene_id]),
            int(args.lights_per_scene),
            int(args.seed),
        )
        selected_by_scene[scene_id] = light_ids
        for light_id in light_ids:
            source_row = metadata[scene_id][light_id]
            target_path = sample_png_root / str(source_row["video"])
            mask_value = source_row.get("mask")
            mask_path = sample_png_root / str(mask_value) if mask_value else None
            original_input = sample_png_root / str(source_row["input_image"])
            if not target_path.exists():
                raise FileNotFoundError(f"Missing target PNG: {target_path}")
            if mask_path is not None and not mask_path.exists():
                raise FileNotFoundError(f"Missing mask PNG: {mask_path}")

            rows.append(
                {
                    "index": len(rows),
                    "scene_id": scene_id,
                    "light_id": light_id,
                    "task": "single_light",
                    "prompt": args.prompt,
                    "attrs_json": source_row["attrs_json"],
                    "input_image": project_path(source_png),
                    "video": project_path(target_path),
                    "target_image": project_path(target_path),
                    "mask": project_path(mask_path) if mask_path is not None else "",
                    "source_raw": project_path(raw_source),
                    "original_metadata_input_image": project_path(original_input),
                    "selection_seed": int(args.seed),
                    "valid": True,
                }
            )

    summary = {
        "schema": "tokenlight_eval500_manifest_v1",
        "scene_count": len(scene_ids),
        "lights_per_scene": int(args.lights_per_scene),
        "sample_count": len(rows),
        "seed": int(args.seed),
        "raw_root": project_path(raw_root),
        "sample_png_root": project_path(sample_png_root),
        "metadata": project_path(metadata_path),
        "output_root": project_path(output_root),
        "manifest": project_path(output_root / args.manifest_name),
        "common_sources": project_path(output_root / "common_sources"),
        "selected_light_ids": selected_by_scene,
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    output_root = resolve(args.output_root)
    rows, summary = build_manifest(args)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name
    summary_path = output_root / args.summary_name
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[manifest] wrote {len(rows)} rows: {manifest_path}")
    print(f"[manifest] wrote summary: {summary_path}")
    print(f"[manifest] common source PNGs: {output_root / 'common_sources'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
