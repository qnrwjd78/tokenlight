#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create source/output/GT triplets for predictions.")
    parser.add_argument("--manifest", default="outputs/eval500/eval500_manifest.jsonl")
    parser.add_argument("--base-path", default="", help="Base path for relative image paths inside the manifest.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--source-key", default="input_image")
    parser.add_argument("--target-key", default="target_image")
    parser.add_argument("--target-fallback-key", default="video")
    parser.add_argument("--mask-key", default="inf_mask")
    parser.add_argument("--mask-fallback-key", default="mask")
    parser.add_argument("--gt", "--with-gt", dest="with_gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-mask", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def resolve_data(path: str | Path, base_path: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return base_path / value if str(base_path) else ROOT / value


def row_value(row: dict, key: str, fallback_key: str = ""):
    value = row.get(key)
    if value not in (None, ""):
        return value
    return row.get(fallback_key) if fallback_key else None


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def pred_path(pred_dir: Path, row: dict) -> Path:
    return pred_dir / f"{row['scene_id']}_light_{int(row['light_id']):03d}.png"


def preview_path(output_dir: Path, row: dict, with_gt: bool) -> Path:
    suffix = "triplet" if with_gt else "pair"
    return output_dir / f"{row['scene_id']}_light_{int(row['light_id']):03d}_{suffix}.png"


def load_panel(path: Path, size: tuple[int, int], *, mask: bool = False) -> Image.Image:
    image = Image.open(path).convert("L" if mask else "RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC)
    if mask:
        image = image.convert("RGB")
    return image


def label_font(panel_height: int) -> ImageFont.ImageFont:
    font_size = max(14, min(18, panel_height // 30))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_labeled_panels(panels: list[tuple[str, Image.Image]], out_file: Path) -> None:
    if not panels:
        raise ValueError("No panels to render")
    width, height = panels[0][1].size
    font = label_font(height)
    label_height = max(34, int(getattr(font, "size", 14) * 2.2))
    canvas = Image.new("RGB", (width * len(panels), height + label_height), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * width
        canvas.paste(image, (x, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (
                x + width // 2 - text_w // 2 - bbox[0],
                height + label_height // 2 - text_h // 2 - bbox[1],
            ),
            label,
            fill=(245, 245, 245),
            font=font,
        )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)


def make_triplet(
    row: dict,
    pred_file: Path,
    out_file: Path,
    source_key: str,
    target_key: str,
    target_fallback_key: str,
    mask_key: str,
    mask_fallback_key: str,
    base_path: Path,
    include_mask: bool,
    with_gt: bool,
) -> None:
    pred = Image.open(pred_file).convert("RGB")
    size = pred.size
    panels = [
        ("source", load_panel(resolve_data(row[source_key], base_path), size)),
        ("output", pred),
    ]
    if with_gt:
        target = row_value(row, target_key, target_fallback_key)
        if not target:
            raise KeyError(f"target key {target_key!r}/{target_fallback_key!r} not found in manifest row")
        panels.append(("gt", load_panel(resolve_data(target, base_path), size)))
    mask = row_value(row, mask_key, mask_fallback_key)
    if include_mask and mask:
        panels.append(("mask", load_panel(resolve_data(mask, base_path), size, mask=True)))
    render_labeled_panels(panels, out_file)


def main() -> int:
    args = parse_args()
    pred_dir = resolve(args.pred_dir)
    base_path = resolve(args.base_path) if args.base_path else Path("")
    output_dir = resolve(args.output_dir) if args.output_dir else pred_dir / ("triplets" if args.with_gt else "pairs")
    rows = load_rows(resolve(args.manifest), int(args.limit))
    if rows and args.source_key not in rows[0]:
        raise KeyError(f"source key {args.source_key!r} not found in manifest rows")

    made = 0
    missing = 0
    for row in rows:
        pred_file = pred_path(pred_dir, row)
        if not pred_file.exists():
            missing += 1
            continue
        out_file = preview_path(output_dir, row, args.with_gt)
        if args.skip_existing and out_file.exists():
            continue
        make_triplet(
            row,
            pred_file,
            out_file,
            args.source_key,
            args.target_key,
            args.target_fallback_key,
            args.mask_key,
            args.mask_fallback_key,
            base_path,
            args.include_mask,
            args.with_gt,
        )
        made += 1
    print(f"[triplets] made={made} missing_pred={missing} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
