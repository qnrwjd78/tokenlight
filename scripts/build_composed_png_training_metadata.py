#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "photorealistic object relighting, preserve geometry and materials"
TASK_MAP = {
    "global_ambient": "ambient_only",
    "single_light": "single_light",
    "two_light": "double_light",
}
TASK_CODE = {
    "ambient_only": 0.0,
    "single_light": 1.0,
    "double_light": 2.0,
}
PBR_AUX_PATH = {
    "albedo": "pbr/albedo.png",
    "normal": "pbr/normal.png",
    "roughness": "pbr/roughness.png",
    "depth": "pbr/depth.png",
}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: Any, default: float) -> float:
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


def light_attrs(light: dict[str, Any]) -> dict[str, float]:
    x, y, z = triple(light.get("position"), (0.0, 0.0, 0.0))
    r, g, b = triple(light.get("color"), (1.0, 1.0, 1.0))
    return {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "g": g,
        "b": b,
        "lambda": finite_float(light.get("intensity"), 1.0),
        "d": finite_float(light.get("radius"), 0.15),
    }


def attrs_json(task: str, sample: dict[str, Any], lights: list[dict[str, Any]]) -> str:
    if task == "ambient_only":
        ambient_scale = finite_float(sample.get("ambient_scale"), 1.0)
    else:
        ambient_scale = finite_float(sample.get("global_control", {}).get("ambient_scale"), 1.0)
    payload = {
        "a": ambient_scale,
        "dg": 0.0,
        "t": TASK_CODE[task],
        "lights": [light_attrs(light) for light in lights],
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def row_from_sample(
    *,
    data_root: Path,
    scene_dir: Path,
    sample: dict[str, Any],
    pbr_aux: str,
    pbr_mode: str,
    prompt: str,
    check_files: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    raw_task = str(sample.get("task", ""))
    task = TASK_MAP.get(raw_task)
    if task is None:
        return None, "unsupported_task"

    scene_id = scene_dir.name
    scene_rel = Path("scenes") / scene_id
    sample_image = sample.get("image")
    if not isinstance(sample_image, str) or not sample_image:
        return None, "missing_image"

    source_image = sample.get("source_image") if isinstance(sample.get("source_image"), str) else "source.png"
    video = scene_rel / sample_image
    input_image = scene_rel / source_image
    mask = scene_rel / "masks" / "object_mask.png"
    lights = [item for item in sample.get("lights", []) if isinstance(item, dict)]
    light_ids = [int(item["id"]) for item in lights if "id" in item]

    pbr_paths = {aux: scene_rel / rel for aux, rel in PBR_AUX_PATH.items()}
    required = [video, input_image]
    if pbr_aux != "none":
        required.append(pbr_paths[pbr_aux])
    if check_files:
        for rel in required:
            if not (data_root / rel).exists():
                return None, f"missing_file:{rel.as_posix()}"

    row: dict[str, Any] = {
        "attrs_json": attrs_json(task, sample, lights),
        "input_image": input_image.as_posix(),
        "light_id": light_ids[0] if len(light_ids) == 1 else None,
        "light_ids": light_ids,
        "prompt": prompt,
        "scene_folder": scene_id,
        "scene_id": scene_id,
        "source": "objaverse_ratio3p5_cube1p6_direct",
        "task": task,
        "valid": True,
        "video": video.as_posix(),
    }
    if (not check_files) or (data_root / mask).exists():
        row["mask"] = mask.as_posix()
    for aux, rel in pbr_paths.items():
        if (not check_files) or (data_root / rel).exists():
            row[f"pbr_{aux}_image"] = rel.as_posix()
    if pbr_aux != "none":
        row["pbr_aux_type"] = pbr_aux
        row["pbr_image"] = pbr_paths[pbr_aux].as_posix()
        row["pbr_mode"] = pbr_mode
    return row, None


def scene_dirs(data_root: Path, max_scenes: int | None) -> list[Path]:
    scenes_root = data_root / "scenes"
    scenes = [path for path in sorted(scenes_root.glob("scene_*")) if path.is_dir()]
    return scenes[:max_scenes] if max_scenes is not None else scenes


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    data_root = repo_path(args.data_root)
    output_dir = repo_path(args.output_dir or Path("data_train") / data_root.name)
    metadata_path = output_dir / args.metadata_name
    summary_path = output_dir / args.summary_name

    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    rows_written = 0
    with metadata_path.open("w", encoding="utf-8") as f:
        for scene_dir in scene_dirs(data_root, args.max_scenes):
            manifest_path = scene_dir / "samples_manifest.json"
            if not manifest_path.exists():
                skipped["missing_samples_manifest"] += 1
                continue
            manifest = load_json(manifest_path)
            for sample in manifest.get("samples", []):
                if not isinstance(sample, dict):
                    skipped["bad_sample"] += 1
                    continue
                row, reason = row_from_sample(
                    data_root=data_root,
                    scene_dir=scene_dir,
                    sample=sample,
                    pbr_aux=args.pbr_aux,
                    pbr_mode=args.pbr_mode,
                    prompt=args.prompt,
                    check_files=not args.no_check_files,
                )
                if row is None:
                    skipped[str(reason or "skipped")] += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                counts[str(row["task"])] += 1
                rows_written += 1

    summary = {
        "data_root": data_root.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "pbr_aux": args.pbr_aux,
        "pbr_image_keys": [f"pbr_{aux}_image" for aux in PBR_AUX_PATH],
        "pbr_mode": None if args.pbr_aux == "none" else args.pbr_mode,
        "row_count": rows_written,
        "scene_count": len(scene_dirs(data_root, args.max_scenes)),
        "task_counts": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build TokenLight metadata from composed PNG samples_manifest.json files.")
    p.add_argument("--data-root", default="data/objaverse_ratio3p5_cube1p6_direct_scene0000_1999_640_png")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--metadata-name", default="metadata.jsonl")
    p.add_argument("--summary-name", default="metadata_summary.json")
    p.add_argument("--pbr-aux", choices=("none", "albedo", "normal", "roughness", "depth"), default="depth")
    p.add_argument("--pbr-mode", choices=("target", "condition"), default="target")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--no-check-files", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    summary = build_metadata(parser().parse_args())
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
