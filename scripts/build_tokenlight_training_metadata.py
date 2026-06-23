#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.convert_completed_exr_dataset_to_png import TOKENLIGHT_PROMPT, attrs_from_light_sample


DATASET_SPECS = {
    "objaverse": {
        "png_globs": ("data/cuda*_scenes_*_batch_*",),
        "input_kind": "objaverse_preview",
    },
    "portrait": {
        "png_globs": ("data/all_batch_*_202*",),
        "input_kind": "portrait_input",
    },
    "blenderkit": {
        "png_globs": ("data/object_batch_*_202*",),
        "input_kind": "blenderkit_input",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TokenLight A-plan metadata using original input images.")
    parser.add_argument("--workspace", default="/workspace", help="Path prefix stored in Docker; only used for help text.")
    parser.add_argument("--root", default=".", help="Local repo/workspace root to scan.")
    parser.add_argument("--out-dir", default="data_train/tokenlight_all")
    parser.add_argument("--prompt", default=TOKENLIGHT_PROMPT)
    parser.add_argument("--include-sources", default="objaverse,portrait,blenderkit")
    parser.add_argument("--include-tasks", default="global_ambient,single_light,two_light")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def csv_set(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def scene_folder_from_manifest(manifest_path: Path) -> str:
    return manifest_path.parent.name


def index_inputs(root: Path) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], Path], dict[tuple[str, str], Path]]:
    portrait: dict[tuple[str, str], Path] = {}
    for path in (root / "downloaded_inputs" / "portrait_exr_objaverse").glob("*/scenes/scene_*/input.png"):
        portrait[(path.parents[2].name, path.parent.name)] = path

    blenderkit: dict[tuple[str, str], Path] = {}
    for path in (root / "downloaded_inputs" / "blenderkit_manual_exr").glob("*/scenes/scene_*/input.png"):
        blenderkit[(path.parents[2].name, path.parent.name)] = path

    objaverse: dict[tuple[str, str], Path] = {}
    preview_root = root / "downloaded_previews" / "reinhard_only"
    for path in preview_root.glob("objaverse_ambient_reinhard_stage/cuda*_scenes_*/scenes/preview/scene_*_ambient.png"):
        scene_name = path.stem.removesuffix("_ambient")
        batch_name = path.parents[2].name
        objaverse[(batch_name, scene_name)] = path

    return objaverse, portrait, blenderkit


def source_batches(root: Path, source: str) -> list[Path]:
    specs = DATASET_SPECS[source]
    batches: list[Path] = []
    for pattern in specs["png_globs"]:
        batches.extend(path for path in root.glob(pattern) if (path / "scenes").is_dir())
    return sorted(set(batches), key=lambda path: path.as_posix())


def input_for_scene(
    *,
    source: str,
    batch_path: Path,
    scene_folder: str,
    manifest: dict[str, Any],
    objaverse_inputs: dict[tuple[str, str], Path],
    portrait_inputs: dict[tuple[str, str], Path],
    blenderkit_inputs: dict[tuple[str, str], Path],
) -> Path | None:
    batch_name = batch_path.name
    if source == "portrait":
        input_batch = batch_name.split("_202", 1)[0] + "_inputs"
        return portrait_inputs.get((input_batch, scene_folder))
    if source == "blenderkit":
        input_batch = batch_name.split("_202", 1)[0] + "_inputs"
        return blenderkit_inputs.get((input_batch, scene_folder))
    if source == "objaverse":
        source_scene = str(manifest.get("source_scene") or "")
        parts = Path(source_scene).parts
        source_batch = ""
        for part in parts:
            if part.startswith("cuda") and "_scenes_" in part:
                source_batch = part
                break
        if not source_batch:
            source_batch = batch_name.split("_batch_", 1)[0]
        return objaverse_inputs.get((source_batch, scene_folder))
    raise ValueError(f"Unknown source: {source}")


def light_ids(sample: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for light in sample.get("lights", []) or []:
        if not isinstance(light, dict) or light.get("id") is None:
            continue
        try:
            ids.append(int(light["id"]))
        except (TypeError, ValueError):
            continue
    return ids


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(args.root).resolve()
    include_sources = csv_set(args.include_sources)
    include_tasks = csv_set(args.include_tasks)
    objaverse_inputs, portrait_inputs, blenderkit_inputs = index_inputs(root)

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    per_source_task: Counter[tuple[str, str]] = Counter()

    for source in sorted(include_sources):
        if source not in DATASET_SPECS:
            raise SystemExit(f"Unknown source {source!r}. Choices: {', '.join(DATASET_SPECS)}")
        for batch_path in source_batches(root, source):
            for manifest_path in sorted(batch_path.glob("scenes/scene_*/samples_manifest.json")):
                scene_folder = scene_folder_from_manifest(manifest_path)
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    stats[f"{source}:bad_manifest"] += 1
                    continue
                input_path = input_for_scene(
                    source=source,
                    batch_path=batch_path,
                    scene_folder=scene_folder,
                    manifest=manifest,
                    objaverse_inputs=objaverse_inputs,
                    portrait_inputs=portrait_inputs,
                    blenderkit_inputs=blenderkit_inputs,
                )
                if input_path is None or not input_path.exists():
                    stats[f"{source}:missing_input"] += 1
                    continue

                scene_rel = manifest_path.parent.relative_to(root).as_posix()
                mask_path = manifest_path.parent / "masks" / "object_mask.png"
                mask_rel = rel(mask_path, root) if mask_path.exists() else None

                for sample in manifest.get("samples", []):
                    task = sample.get("task")
                    if task not in include_tasks:
                        continue
                    image = sample.get("image")
                    if not image:
                        stats[f"{source}:missing_sample_image"] += 1
                        continue
                    video_path = manifest_path.parent / image
                    if not video_path.exists():
                        stats[f"{source}:missing_target"] += 1
                        continue
                    attrs = attrs_from_light_sample(sample)
                    if not attrs:
                        stats[f"{source}:missing_attrs"] += 1
                        continue
                    ids = light_ids(sample)
                    row: dict[str, Any] = {
                        "video": rel(video_path, root),
                        "input_image": rel(input_path, root),
                        "prompt": args.prompt,
                        "attrs_json": json.dumps(attrs, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                        "source": source,
                        "batch": batch_path.name,
                        "scene_folder": scene_folder,
                        "scene_id": sample.get("scene_id") or manifest.get("scene_id") or scene_folder,
                        "task": task,
                        "valid": True,
                    }
                    if len(ids) == 1:
                        row["light_id"] = ids[0]
                    elif len(ids) > 1:
                        row["light_ids"] = ids
                    if mask_rel:
                        row["mask"] = mask_rel
                    rows.append(row)
                    stats["rows"] += 1
                    per_source_task[(source, task)] += 1

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    summary = {
        "schema": "tokenlight_a_plan_metadata_v1",
        "row_count": len(rows),
        "sources": sorted(include_sources),
        "tasks": sorted(include_tasks),
        "stats": dict(stats),
        "per_source_task": {
            f"{source}/{task}": count
            for (source, task), count in sorted(per_source_task.items())
        },
        "input_counts": {
            "objaverse_preview": len(objaverse_inputs),
            "portrait_input": len(portrait_inputs),
            "blenderkit_input": len(blenderkit_inputs),
        },
    }
    return rows, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    rows, summary = build_rows(args)
    out_dir = Path(args.root).resolve() / args.out_dir
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "metadata_all.jsonl", rows)
    for source in DATASET_SPECS:
        write_jsonl(out_dir / f"metadata_{source}.jsonl", [row for row in rows if row["source"] == source])
    for task in ("global_ambient", "single_light", "two_light"):
        write_jsonl(out_dir / f"metadata_{task}.jsonl", [row for row in rows if row["task"] == task])
    rows_nomask = []
    for row in rows:
        item = dict(row)
        item.pop("mask", None)
        rows_nomask.append(item)
    write_jsonl(out_dir / "metadata_all_nomask.jsonl", rows_nomask)
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {len(rows)} rows to {out_dir / 'metadata_all.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
