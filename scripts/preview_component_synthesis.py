from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


MODE_ORDER = ("spatial", "ambient", "diffuse", "fixture")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview TokenLight component-pair synthesis as PNGs and contact sheets."
    )
    parser.add_argument("--component-root", default="/workspace/data/sample")
    parser.add_argument("--component-repo", default="/workspace/repos/relighting_dataset")
    parser.add_argument("--output", default="/workspace/runs/component_preview")
    parser.add_argument("--modes", nargs="+", default=["auto"], help="Use 'auto' or any of: spatial ambient diffuse fixture")
    parser.add_argument("--count-per-mode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-lights", type=int, default=1)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--diff-scale", type=float, default=4.0)
    parser.add_argument("--sweep-scenes", action="store_true", help="Generate deterministic per-scene sweeps.")
    parser.add_argument("--max-scenes", type=int, default=0, help="Limit scenes in --sweep-scenes mode; 0 means all.")
    parser.add_argument("--max-spatial-lights", type=int, default=0, help="Limit point lights per scene; 0 means all.")
    parser.add_argument("--max-diffuse-pairs", type=int, default=0, help="Limit ordered diffuse pairs per scene; 0 means all.")
    parser.add_argument("--ambient-pairs", default="0.35:1.10,1.10:0.35,0.70:1.25")
    parser.add_argument("--sweep-color", default="1,1,1")
    parser.add_argument("--sweep-intensity", type=float, default=1.0)
    parser.add_argument("--sweep-ambient-scale", type=float, default=0.8)
    parser.add_argument("--rows-per-sheet", type=int, default=24)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_scene_meta(component_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    scenes_dir = component_root / "scenes"
    metas = []
    for meta_path in sorted(scenes_dir.glob("scene_*/meta.json")):
        metas.append((meta_path.parent, load_json(meta_path)))
    if not metas:
        raise FileNotFoundError(f"No scene meta files found under {scenes_dir}")
    return metas


def detect_modes(component_root: Path) -> list[str]:
    modes: set[str] = set()
    for _scene_dir, meta in list_scene_meta(component_root):
        spatial = meta.get("spatial") or {}
        if spatial.get("ambient_render") and spatial.get("point_lights"):
            modes.add("spatial")
            modes.add("ambient")
        diffuse = meta.get("diffuse") or {}
        if diffuse.get("ambient_render") and diffuse.get("spreads"):
            modes.add("diffuse")
        fixtures = meta.get("fixtures") or {}
        if fixtures.get("environment_render") and fixtures.get("fixtures"):
            modes.add("fixture")
    return [mode for mode in MODE_ORDER if mode in modes]


def choose_modes(requested: list[str], available: list[str]) -> list[str]:
    if requested == ["auto"] or "auto" in requested:
        return available
    unknown = sorted(set(requested) - set(MODE_ORDER))
    if unknown:
        raise ValueError(f"Unknown mode(s): {', '.join(unknown)}")
    selected = [mode for mode in MODE_ORDER if mode in requested and mode in available]
    missing = [mode for mode in requested if mode not in available]
    if missing:
        print(f"Skipping unavailable mode(s): {', '.join(missing)}", file=sys.stderr)
    if not selected:
        raise ValueError(f"No requested modes are available. Available modes: {', '.join(available)}")
    return selected


def chw_to_uint8(chw: np.ndarray) -> np.ndarray:
    image = np.asarray(chw, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected CHW image, got shape {image.shape}")
    image = image[:3]
    image = np.transpose(image, (1, 2, 0))
    image = ((image + 1.0) * 0.5).clip(0.0, 1.0)
    return (image * 255.0 + 0.5).astype(np.uint8)


def diff_to_uint8(input_chw: np.ndarray, target_chw: np.ndarray, scale: float) -> np.ndarray:
    input_img = ((np.asarray(input_chw, dtype=np.float32)[:3] + 1.0) * 0.5).clip(0.0, 1.0)
    target_img = ((np.asarray(target_chw, dtype=np.float32)[:3] + 1.0) * 0.5).clip(0.0, 1.0)
    diff = np.abs(target_img - input_img) * float(scale)
    diff = np.transpose(diff.clip(0.0, 1.0), (1, 2, 0))
    return (diff * 255.0 + 0.5).astype(np.uint8)


def make_thumb(image: Image.Image, size: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - thumb.width) // 2
    y = (size - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(20, 20, 20))


def parse_rgb(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--sweep-color must be formatted as r,g,b")
    return np.array(parts, dtype=np.float32)


def parse_ambient_pairs(value: str) -> list[tuple[float, float]]:
    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        left, right = item.split(":", 1)
        pairs.append((float(left), float(right)))
    if not pairs:
        raise ValueError("--ambient-pairs must contain at least one a_in:a_out pair")
    return pairs


def condition_summary(condition: dict[str, Any]) -> str:
    task = condition.get("task", "?")
    scene_id = condition.get("scene_id", "?")
    if task == "spatial":
        lights = condition.get("lights") or []
        if lights:
            light = lights[0]
            pos = light.get("position") or ["?", "?", "?"]
            return f"{scene_id} spatial p=({float(pos[0]):.2f},{float(pos[1]):.2f},{float(pos[2]):.2f})"
        return f"{scene_id} spatial"
    if task == "ambient":
        return f"{scene_id} ambient a={float(condition.get('ambient_scale_out', 0.0)):.2f}"
    if task == "diffuse":
        return f"{scene_id} diffuse d={float(condition.get('spread_out', 0.0)):.2f}"
    if task == "fixture":
        return f"{scene_id} fixture id={condition.get('fixture_id', '?')}"
    return f"{scene_id} {task}"


def make_contact_sheet(records: list[dict[str, Any]], output: Path, thumb_size: int) -> None:
    if not records:
        return
    label_w = max(300, thumb_size)
    header_h = 26
    row_h = thumb_size + 34
    col_gap = 12
    width = label_w + 3 * thumb_size + 4 * col_gap
    height = header_h + len(records) * row_h + 12
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw_label(draw, (label_w + col_gap, 6), "input")
    draw_label(draw, (label_w + 2 * col_gap + thumb_size, 6), "target")
    draw_label(draw, (label_w + 3 * col_gap + 2 * thumb_size, 6), "abs diff")

    y = header_h
    for record in records:
        draw_label(draw, (12, y + 6), record["summary"])
        for col, key in enumerate(("input_path", "target_path", "diff_path")):
            image = Image.open(record[key]).convert("RGB")
            thumb = make_thumb(image, thumb_size)
            x = label_w + (col + 1) * col_gap + col * thumb_size
            sheet.paste(thumb, (x, y))
        y += row_h
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def make_contact_sheets(records: list[dict[str, Any]], output: Path, thumb_size: int, rows_per_sheet: int) -> None:
    rows_per_sheet = max(1, int(rows_per_sheet))
    if len(records) <= rows_per_sheet:
        make_contact_sheet(records, output, thumb_size)
        return
    stem = output.stem
    suffix = output.suffix or ".png"
    for start in range(0, len(records), rows_per_sheet):
        chunk = records[start : start + rows_per_sheet]
        chunk_index = start // rows_per_sheet
        make_contact_sheet(chunk, output.with_name(f"{stem}_{chunk_index:03d}{suffix}"), thumb_size)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_record_images(
    mode_dir: Path,
    stem: str,
    input_chw: np.ndarray,
    target_chw: np.ndarray,
    condition: dict[str, Any],
    diff_scale: float,
    mode: str,
    index: int,
) -> dict[str, Any]:
    mode_dir.mkdir(parents=True, exist_ok=True)
    input_img = Image.fromarray(chw_to_uint8(input_chw))
    target_img = Image.fromarray(chw_to_uint8(target_chw))
    diff_img = Image.fromarray(diff_to_uint8(input_chw, target_chw, diff_scale))
    input_path = mode_dir / f"{stem}_input.png"
    target_path = mode_dir / f"{stem}_target.png"
    diff_path = mode_dir / f"{stem}_diff_x{diff_scale:g}.png"
    input_img.save(input_path)
    target_img.save(target_path)
    diff_img.save(diff_path)
    condition = json_safe(condition)
    return {
        "mode": mode,
        "index": index,
        "input_path": str(input_path),
        "target_path": str(target_path),
        "diff_path": str(diff_path),
        "summary": condition_summary(condition),
        "condition": condition,
    }


def hwc_to_chw_minus_one_one(image: np.ndarray) -> np.ndarray:
    return np.transpose(image * 2.0 - 1.0, (2, 0, 1)).astype(np.float32)


def synthesize_spatial_sweep(scene_dir: Path, meta: dict[str, Any], color: np.ndarray, intensity: float, ambient_scale: float):
    from tokenlight_dataset.exr_io import read_exr
    from tokenlight_dataset.tonemap import reinhard

    spatial = meta["spatial"]
    ambient = read_exr(scene_dir / spatial["ambient_render"])
    source = reinhard(ambient)
    for light_index, light in enumerate(spatial["point_lights"]):
        contribution = read_exr(scene_dir / light["render"])
        target = reinhard(ambient_scale * ambient + intensity * contribution * color.reshape(1, 1, 3))
        condition = {
            "task": "spatial",
            "scene_id": meta["scene_id"],
            "ambient_scale": ambient_scale,
            "lights": [
                {
                    "position": light["canonical_position"],
                    "color": color.tolist(),
                    "intensity": intensity,
                    "radius": light.get("canonical_radius"),
                    "light_id": light.get("id", light_index),
                }
            ],
        }
        yield f"light_{light_index:03d}", hwc_to_chw_minus_one_one(source), hwc_to_chw_minus_one_one(target), condition


def synthesize_ambient_sweep(scene_dir: Path, meta: dict[str, Any], pairs: list[tuple[float, float]]):
    from tokenlight_dataset.exr_io import read_exr
    from tokenlight_dataset.tonemap import reinhard

    ambient = read_exr(scene_dir / meta["spatial"]["ambient_render"])
    for pair_index, (a_in, a_out) in enumerate(pairs):
        source = reinhard(a_in * ambient)
        target = reinhard(a_out * ambient)
        condition = {
            "task": "ambient",
            "scene_id": meta["scene_id"],
            "ambient_scale_in": a_in,
            "ambient_scale_out": a_out,
            "ambient_scale_delta": a_out / max(a_in, 1e-6),
        }
        yield f"ambient_{pair_index:03d}", hwc_to_chw_minus_one_one(source), hwc_to_chw_minus_one_one(target), condition


def synthesize_diffuse_sweep(scene_dir: Path, meta: dict[str, Any], color: np.ndarray, intensity: float, ambient_scale: float):
    from tokenlight_dataset.exr_io import read_exr
    from tokenlight_dataset.tonemap import reinhard

    diffuse = meta["diffuse"]
    ambient = read_exr(scene_dir / diffuse["ambient_render"])
    spreads = diffuse["spreads"]
    rendered_spreads = {idx: read_exr(scene_dir / spread["render"]) for idx, spread in enumerate(spreads)}
    pair_index = 0
    for src_index, src in enumerate(spreads):
        for dst_index, dst in enumerate(spreads):
            if src_index == dst_index:
                continue
            source_linear = ambient_scale * ambient + intensity * rendered_spreads[src_index] * color.reshape(1, 1, 3)
            target_linear = ambient_scale * ambient + intensity * rendered_spreads[dst_index] * color.reshape(1, 1, 3)
            condition = {
                "task": "diffuse",
                "scene_id": meta["scene_id"],
                "spread_in": src["normalized_spread"],
                "spread_out": dst["normalized_spread"],
                "spread_delta": dst["normalized_spread"] - src["normalized_spread"],
                "color": color.tolist(),
                "intensity": intensity,
                "ambient_scale": ambient_scale,
                "source_spread_id": src_index,
                "target_spread_id": dst_index,
            }
            yield (
                f"spread_{src_index:03d}_to_{dst_index:03d}",
                hwc_to_chw_minus_one_one(reinhard(source_linear)),
                hwc_to_chw_minus_one_one(reinhard(target_linear)),
                condition,
            )
            pair_index += 1


def run_scene_sweep(args: argparse.Namespace, component_root: Path, output: Path, modes: list[str]) -> list[dict[str, Any]]:
    color = parse_rgb(args.sweep_color)
    ambient_pairs = parse_ambient_pairs(args.ambient_pairs)
    scene_metas = list_scene_meta(component_root)
    if args.max_scenes > 0:
        scene_metas = scene_metas[: args.max_scenes]

    records: list[dict[str, Any]] = []
    failures = []
    for scene_offset, (scene_dir, meta) in enumerate(scene_metas):
        scene_id = meta["scene_id"]
        scene_records: list[dict[str, Any]] = []
        generators = []
        if "spatial" in modes:
            generators.append(("spatial", synthesize_spatial_sweep(scene_dir, meta, color, args.sweep_intensity, args.sweep_ambient_scale)))
        if "ambient" in modes:
            generators.append(("ambient", synthesize_ambient_sweep(scene_dir, meta, ambient_pairs)))
        if "diffuse" in modes:
            generators.append(("diffuse", synthesize_diffuse_sweep(scene_dir, meta, color, args.sweep_intensity, args.sweep_ambient_scale)))

        for mode, generator in generators:
            mode_records: list[dict[str, Any]] = []
            limit = 0
            if mode == "spatial":
                limit = args.max_spatial_lights
            elif mode == "diffuse":
                limit = args.max_diffuse_pairs
            try:
                for item_index, (stem, source, target, condition) in enumerate(generator):
                    if limit > 0 and item_index >= limit:
                        break
                    record = save_record_images(
                        output / scene_id / mode,
                        stem,
                        source,
                        target,
                        condition,
                        args.diff_scale,
                        mode,
                        item_index,
                    )
                    records.append(record)
                    mode_records.append(record)
                    scene_records.append(record)
            except Exception as exc:
                failures.append({"scene_id": scene_id, "mode": mode, "error": str(exc)})
            make_contact_sheets(
                mode_records,
                output / scene_id / mode / f"{mode}_contact_sheet.png",
                args.thumb_size,
                args.rows_per_sheet,
            )
        make_contact_sheets(
            scene_records,
            output / scene_id / "scene_contact_sheet.png",
            args.thumb_size,
            args.rows_per_sheet,
        )
        print(f"[{scene_offset + 1}/{len(scene_metas)}] {scene_id}: wrote {len(scene_records)} previews")
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def main() -> int:
    args = parse_args()
    component_root = resolve_path(args.component_root)
    component_repo = resolve_path(args.component_repo)
    output = resolve_path(args.output)

    if not component_repo.exists():
        raise FileNotFoundError(f"component repo not found: {component_repo}")
    if str(component_repo) not in sys.path:
        sys.path.insert(0, str(component_repo))

    available_modes = detect_modes(component_root)
    modes = choose_modes(args.modes, available_modes)
    output.mkdir(parents=True, exist_ok=True)

    if args.sweep_scenes:
        records = run_scene_sweep(args, component_root, output, modes)
        make_contact_sheets(records, output / "contact_sheet.png", args.thumb_size, args.rows_per_sheet)
        with (output / "preview_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                serializable = dict(record)
                serializable["input_path"] = str(Path(record["input_path"]).relative_to(output))
                serializable["target_path"] = str(Path(record["target_path"]).relative_to(output))
                serializable["diff_path"] = str(Path(record["diff_path"]).relative_to(output))
                handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        summary = {
            "component_root": str(component_root),
            "component_repo": str(component_repo),
            "available_modes": available_modes,
            "selected_modes": modes,
            "sweep_scenes": True,
            "records": len(records),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(records)} previews to {output}")
        return 0

    from tokenlight_dataset.component_dataset import TokenLightComponentDataset

    records: list[dict[str, Any]] = []
    for mode in modes:
        dataset = TokenLightComponentDataset(
            root=component_root,
            length=max(args.start_index + args.count_per_mode + 1, 1),
            modes=(mode,),
            seed=args.seed,
            max_lights=args.max_lights,
            return_torch=False,
        )
        mode_dir = output / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_records = []
        for offset in range(args.count_per_mode):
            index = args.start_index + offset
            sample = dataset[index]
            stem = f"{mode}_{offset:03d}"
            record = save_record_images(
                mode_dir,
                stem,
                sample["input"],
                sample["target"],
                sample["condition"],
                args.diff_scale,
                mode,
                index,
            )
            records.append(record)
            mode_records.append(record)

        make_contact_sheets(mode_records, mode_dir / f"{mode}_contact_sheet.png", args.thumb_size, args.rows_per_sheet)

    make_contact_sheets(records, output / "contact_sheet.png", args.thumb_size, args.rows_per_sheet)
    with (output / "preview_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            serializable = dict(record)
            serializable["input_path"] = str(Path(record["input_path"]).relative_to(output))
            serializable["target_path"] = str(Path(record["target_path"]).relative_to(output))
            serializable["diff_path"] = str(Path(record["diff_path"]).relative_to(output))
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    summary = {
        "component_root": str(component_root),
        "component_repo": str(component_repo),
        "available_modes": available_modes,
        "selected_modes": modes,
        "count_per_mode": args.count_per_mode,
        "records": len(records),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} previews to {output}")
    print(f"Open {output / 'contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
