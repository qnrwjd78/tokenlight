#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MASK_ROOT = ROOT / "outputs/sample_exr_png_subset_masks_640_all"
DEFAULT_INFER_ROOT = ROOT / "model/infer/sample_exr_png_subset_tokenlight_full_a_zero3_720_20260622_222426"
DEFAULT_DATA_ROOT = ROOT / "data/sample_exr_png_subset"

SAMPLE_RE = re.compile(r"sample_(?P<sample>\d+)_light_(?P<light>\d+)_c(?P<color>\d+)_a(?P<ambient>\d+)$")
EPOCH_RE = re.compile(r"epoch-(?P<epoch>\d+)$")

MASK_NAMES = (
    "direct_lit_clear",
    "receiver_shadow",
    "object_lit_clear",
    "object_self_shadow",
    "outside_light",
    "preserve",
    "color_valid",
)

METRIC_NAMES = (
    "global_rgb_l1",
    "global_res_l1",
    "global_extra_light",
    "global_missing_light",
    "direct_lit_l1",
    "direct_missing",
    "direct_extra",
    "shadow_l1",
    "shadow_leak",
    "shadow_overdark",
    "object_lit_l1",
    "object_missing_light",
    "object_extra_light",
    "object_self_shadow_l1",
    "object_shadow_leak",
    "object_overdark",
    "outside_l1",
    "outside_spill",
    "outside_missing",
    "preserve_l1",
    "preserve_res_l1",
    "color_error",
)


def resolve_repo(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate relighting outputs inside GT mask regions.")
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT.as_posix())
    parser.add_argument("--infer-root", default=DEFAULT_INFER_ROOT.as_posix())
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT.as_posix())
    parser.add_argument("--output-dir", default="", help="Default: <infer-root>/mask_metrics")
    parser.add_argument("--epoch", action="append", default=[], help="Epoch name or number. Can be passed multiple times.")
    parser.add_argument("--color-valid-threshold", type=float, default=0.03)
    return parser.parse_args()


def epoch_number(path: Path) -> int:
    match = EPOCH_RE.match(path.name)
    return int(match.group("epoch")) if match else 10**9


def selected_epoch_dirs(infer_root: Path, values: list[str]) -> list[Path]:
    epoch_dirs = sorted((p for p in infer_root.iterdir() if p.is_dir() and EPOCH_RE.match(p.name)), key=epoch_number)
    if not values:
        return epoch_dirs

    wanted = set()
    for value in values:
        value = value.strip()
        wanted.add(value if value.startswith("epoch-") else f"epoch-{int(value)}")
    return [path for path in epoch_dirs if path.name in wanted]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_sample_asset(ref: str | None, *, sample_dir: Path, scene_dir: Path, data_root: Path, scene_id: str) -> Path | None:
    if not ref:
        return None
    value = Path(ref)
    if value.is_absolute():
        return value if value.exists() else None
    return first_existing(
        [
            sample_dir / value,
            scene_dir / value,
            data_root / value,
            data_root / "scenes" / scene_id / value,
        ]
    )


def collect_samples(mask_root: Path, data_root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    scenes_root = mask_root / "scenes"
    for scene_dir in sorted(scenes_root.glob("scene_*")):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        pairs_dir = scene_dir / "subset_pairs"
        for sample_dir in sorted(p for p in pairs_dir.glob("sample_*") if p.is_dir()):
            match = SAMPLE_RE.match(sample_dir.name)
            if not match:
                continue
            manifest_path = sample_dir / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            source = first_existing([sample_dir / "source.png"]) or resolve_sample_asset(
                manifest.get("source_png") or manifest.get("source_image"),
                sample_dir=sample_dir,
                scene_dir=scene_dir,
                data_root=data_root,
                scene_id=scene_id,
            )
            target = first_existing([sample_dir / "target.png"]) or resolve_sample_asset(
                manifest.get("target_png") or manifest.get("target_image"),
                sample_dir=sample_dir,
                scene_dir=scene_dir,
                data_root=data_root,
                scene_id=scene_id,
            )
            samples.append(
                {
                    "scene_id": scene_id,
                    "sample_id": int(match.group("sample")),
                    "light_id": int(match.group("light")),
                    "color_id": int(match.group("color")),
                    "ambient_id": int(match.group("ambient")),
                    "sample_dir": sample_dir,
                    "mask_dir": sample_dir / "masks",
                    "source": source,
                    "target": target,
                }
            )
    return samples


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


def luminance(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def masked_mean(value_map: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None:
        return float("nan")
    denom = float(mask.sum())
    if denom <= 1e-6:
        return float("nan")
    return float((value_map * mask).sum() / denom)


def color_cosine_error(delta_pred: np.ndarray, delta_gt: np.ndarray, mask: np.ndarray | None, threshold: float) -> tuple[float, float]:
    if mask is None:
        return float("nan"), 0.0
    dot = (delta_pred * delta_gt).sum(axis=2)
    norm_pred = np.sqrt((delta_pred * delta_pred).sum(axis=2))
    norm_gt = np.sqrt((delta_gt * delta_gt).sum(axis=2))
    cos = dot / (norm_pred * norm_gt + 1e-6)
    cos = np.clip(cos, -1.0, 1.0)
    valid = mask * (norm_gt > threshold).astype(np.float32)
    denom = float(valid.sum())
    if denom <= 1e-6:
        return float("nan"), 0.0
    return float(((1.0 - cos) * valid).sum() / denom), denom


def prediction_path(epoch_dir: Path, scene_id: str, light_id: int) -> Path:
    return epoch_dir / f"{scene_id}_light_{light_id:03d}.png"


def evaluate_sample(sample: dict[str, Any], epoch_dir: Path, color_valid_threshold: float) -> dict[str, Any] | None:
    pred_path = prediction_path(epoch_dir, sample["scene_id"], sample["light_id"])
    source_path = sample["source"]
    target_path = sample["target"]
    if not pred_path.exists() or source_path is None or target_path is None or not source_path.exists() or not target_path.exists():
        return None

    pred_image = Image.open(pred_path)
    size = pred_image.size
    pred = np.asarray(pred_image.convert("RGB"), dtype=np.float32) / 255.0
    source = load_rgb(source_path, size=size)
    target = load_rgb(target_path, size=size)
    height, width = pred.shape[:2]

    masks = {
        name: load_mask(sample["mask_dir"] / f"{name}.png", size=size)
        for name in MASK_NAMES
    }

    rgb_l1_map = np.abs(pred - target).mean(axis=2)
    delta_gt = target - source
    delta_pred = pred - source
    res_l1_map = np.abs(delta_pred - delta_gt).mean(axis=2)

    diff = luminance(delta_pred) - luminance(delta_gt)
    extra_map = np.maximum(diff, 0.0)
    missing_map = np.maximum(-diff, 0.0)

    color_error, color_valid_eval_pixels = color_cosine_error(
        delta_pred,
        delta_gt,
        masks["color_valid"],
        color_valid_threshold,
    )

    record: dict[str, Any] = {
        "epoch": epoch_dir.name,
        "scene_id": sample["scene_id"],
        "sample_id": sample["sample_id"],
        "light_id": sample["light_id"],
        "color_id": sample["color_id"],
        "ambient_id": sample["ambient_id"],
        "width": width,
        "height": height,
        "pred": pred_path.as_posix(),
        "source": source_path.as_posix(),
        "target": target_path.as_posix(),
        "mask_dir": sample["mask_dir"].as_posix(),
        "global_rgb_l1": float(rgb_l1_map.mean()),
        "global_res_l1": float(res_l1_map.mean()),
        "global_extra_light": float(extra_map.mean()),
        "global_missing_light": float(missing_map.mean()),
        "direct_lit_l1": masked_mean(res_l1_map, masks["direct_lit_clear"]),
        "direct_missing": masked_mean(missing_map, masks["direct_lit_clear"]),
        "direct_extra": masked_mean(extra_map, masks["direct_lit_clear"]),
        "shadow_l1": masked_mean(res_l1_map, masks["receiver_shadow"]),
        "shadow_leak": masked_mean(extra_map, masks["receiver_shadow"]),
        "shadow_overdark": masked_mean(missing_map, masks["receiver_shadow"]),
        "object_lit_l1": masked_mean(res_l1_map, masks["object_lit_clear"]),
        "object_missing_light": masked_mean(missing_map, masks["object_lit_clear"]),
        "object_extra_light": masked_mean(extra_map, masks["object_lit_clear"]),
        "object_self_shadow_l1": masked_mean(res_l1_map, masks["object_self_shadow"]),
        "object_shadow_leak": masked_mean(extra_map, masks["object_self_shadow"]),
        "object_overdark": masked_mean(missing_map, masks["object_self_shadow"]),
        "outside_l1": masked_mean(res_l1_map, masks["outside_light"]),
        "outside_spill": masked_mean(extra_map, masks["outside_light"]),
        "outside_missing": masked_mean(missing_map, masks["outside_light"]),
        "preserve_l1": masked_mean(rgb_l1_map, masks["preserve"]),
        "preserve_res_l1": masked_mean(res_l1_map, masks["preserve"]),
        "color_error": color_error,
        "color_valid_eval_pixels": color_valid_eval_pixels,
        "color_valid_eval_ratio": color_valid_eval_pixels / float(width * height),
    }
    for name, mask in masks.items():
        pixels = 0.0 if mask is None else float(mask.sum())
        record[f"{name}_pixels"] = pixels
        record[f"{name}_ratio"] = pixels / float(width * height)
    return record


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def csv_value(value: Any) -> Any:
    number = finite_float(value)
    if number is None:
        return "" if isinstance(value, float) else value
    return f"{number:.9g}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def summarize(records: list[dict[str, Any]], samples: list[dict[str, Any]], epoch_dirs: list[Path]) -> list[dict[str, Any]]:
    by_epoch = {epoch.name: [] for epoch in epoch_dirs}
    for record in records:
        by_epoch.setdefault(str(record["epoch"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for epoch in epoch_dirs:
        epoch_records = by_epoch.get(epoch.name, [])
        row: dict[str, Any] = {
            "epoch": epoch.name,
            "expected_count": len(samples),
            "evaluated_count": len(epoch_records),
            "missing_count": len(samples) - len(epoch_records),
        }
        for metric in METRIC_NAMES:
            values = np.array(
                [value for value in (finite_float(record.get(metric)) for record in epoch_records) if value is not None],
                dtype=np.float64,
            )
            row[f"{metric}_valid_count"] = int(values.size)
            row[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        rows.append(row)
    return rows


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    mask_root = resolve_repo(args.mask_root)
    infer_root = resolve_repo(args.infer_root)
    data_root = resolve_repo(args.data_root)
    output_dir = resolve_repo(args.output_dir) if args.output_dir else infer_root / "mask_metrics"

    samples = collect_samples(mask_root, data_root)
    epoch_dirs = selected_epoch_dirs(infer_root, args.epoch)
    if not samples:
        raise SystemExit(f"No mask samples found under {mask_root}")
    if not epoch_dirs:
        raise SystemExit(f"No epoch dirs found under {infer_root}")

    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for epoch_dir in epoch_dirs:
        for sample in samples:
            record = evaluate_sample(sample, epoch_dir, float(args.color_valid_threshold))
            if record is None:
                missing.append(
                    {
                        "epoch": epoch_dir.name,
                        "scene_id": sample["scene_id"],
                        "light_id": sample["light_id"],
                        "pred": prediction_path(epoch_dir, sample["scene_id"], sample["light_id"]).as_posix(),
                        "source": "" if sample["source"] is None else sample["source"].as_posix(),
                        "target": "" if sample["target"] is None else sample["target"].as_posix(),
                    }
                )
                continue
            records.append(record)

    per_sample_fields = [
        "epoch",
        "scene_id",
        "sample_id",
        "light_id",
        "color_id",
        "ambient_id",
        "width",
        "height",
        *METRIC_NAMES,
        "color_valid_eval_pixels",
        "color_valid_eval_ratio",
        *[field for name in MASK_NAMES for field in (f"{name}_pixels", f"{name}_ratio")],
        "pred",
        "source",
        "target",
        "mask_dir",
    ]
    summary_rows = summarize(records, samples, epoch_dirs)
    summary_fields = ["epoch", "expected_count", "evaluated_count", "missing_count"]
    for metric in METRIC_NAMES:
        summary_fields.extend(
            [
                f"{metric}_valid_count",
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_std",
            ]
        )

    write_csv(output_dir / "per_sample.csv", records, per_sample_fields)
    write_csv(output_dir / "per_epoch_summary.csv", summary_rows, summary_fields)
    (output_dir / "run_info.json").write_text(
        json.dumps(
            json_ready(
                {
                    "mask_root": mask_root.as_posix(),
                    "infer_root": infer_root.as_posix(),
                    "data_root": data_root.as_posix(),
                    "output_dir": output_dir.as_posix(),
                    "sample_count": len(samples),
                    "epoch_count": len(epoch_dirs),
                    "evaluated_count": len(records),
                    "missing_count": len(missing),
                    "color_valid_threshold": float(args.color_valid_threshold),
                    "missing": missing,
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'per_sample.csv'}")
    print(f"wrote {output_dir / 'per_epoch_summary.csv'}")
    print(f"evaluated={len(records)} missing={len(missing)} samples={len(samples)} epochs={len(epoch_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
