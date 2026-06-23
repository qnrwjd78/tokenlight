#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKENLIGHT_PROMPT = "photorealistic object relighting, preserve geometry and materials"
PRESET10_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.00, 1.00, 1.00),
    (1.00, 0.82, 0.62),
    (0.62, 0.78, 1.00),
    (1.00, 0.45, 0.45),
    (0.45, 1.00, 0.45),
    (0.45, 0.45, 1.00),
    (1.00, 1.00, 0.45),
    (0.45, 1.00, 1.00),
    (1.00, 0.45, 1.00),
    (0.75, 0.75, 0.75),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final composed PNG samples from completed TokenLight EXR scenes."
    )
    parser.add_argument("--source", default="outputs/objaverse_dataset_exr")
    parser.add_argument("--dest", default="outputs/objaverse_dataset_png")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--two-light-samples", type=int, default=32)
    parser.add_argument("--global-ambient-samples", type=int, default=7)
    parser.add_argument("--global-diffuse-samples", type=int, default=0)
    parser.add_argument("--include-global-diffuse", action="store_true")
    parser.add_argument("--single-lights", choices=["all", "none"], default="all")
    parser.add_argument("--max-single-lights", type=int, default=0, help="0 means no cap.")
    parser.add_argument(
        "--fixed-colors",
        default="",
        help="Use fixed RGB colors for single-light samples. Use 'preset10' or ';'-separated triples.",
    )
    parser.add_argument(
        "--color-policy",
        choices=["sample", "cycle", "grid"],
        default="sample",
        help="For fixed colors: choose one color per light, cycle colors, or render all color combinations.",
    )
    parser.add_argument(
        "--ambient-scales",
        default="",
        help="Comma-separated ambient scales. When set, writes matching global_ambient source PNGs.",
    )
    parser.add_argument(
        "--ambient-policy",
        choices=["sample", "cycle", "grid"],
        default="sample",
        help="For fixed colors: choose one ambient scale per light/color, cycle scales, or render all scale combinations.",
    )
    parser.add_argument(
        "--fixed-intensity",
        type=float,
        default=1.0,
        help="Point-light intensity multiplier for fixed-color samples.",
    )
    parser.add_argument("--write-train-metadata", action="store_true")
    parser.add_argument("--metadata-name", default="metadata.jsonl")
    parser.add_argument("--metadata-nomask-name", default="metadata_nomask.jsonl")
    parser.add_argument("--metadata-prompt", default=TOKENLIGHT_PROMPT)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-existing", action="store_true", help="Copy metadata/masks/previews instead of hardlinking.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    if not text:
        return []
    values = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def parse_color_palette(text: str) -> list[tuple[float, float, float]]:
    if not text:
        return []
    if text.strip().lower() in {"preset10", "tokenlight10"}:
        return list(PRESET10_COLORS)
    colors: list[tuple[float, float, float]] = []
    for item in text.split(";"):
        values = [float(value) for value in item.replace(",", " ").split()]
        if len(values) != 3:
            raise SystemExit(f"Each fixed color must have exactly 3 values, got {item!r}")
        colors.append((values[0], values[1], values[2]))
    return colors


def discover_source_roots(source: Path) -> list[Path]:
    if (source / "scenes").is_dir():
        return [source]
    roots = sorted(path.parent for path in source.glob("*/scenes") if path.is_dir())
    if not roots:
        raise SystemExit(f"SOURCE does not contain scenes/ directly or one level below it: {source}")
    return roots


def completed_scene_dirs(source_roots: list[Path], scene_offset: int, scene_limit: int) -> list[Path]:
    scene_dirs: list[Path] = []
    for source_root in source_roots:
        scene_dirs.extend(meta.parent for meta in sorted((source_root / "scenes").glob("scene_*/meta.json")))
    if scene_offset > 0:
        scene_dirs = scene_dirs[scene_offset:]
    if scene_limit > 0:
        scene_dirs = scene_dirs[:scene_limit]
    if not scene_dirs:
        raise SystemExit("No completed scenes found. A completed scene must contain scenes/<scene_id>/meta.json.")
    return scene_dirs


def dest_root_for(source: Path, dest: Path, source_root: Path) -> Path:
    if source_root == source:
        return dest
    return dest / source_root.relative_to(source)


def link_or_copy(src: Path, dst: Path, copy_existing: bool, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if copy_existing:
        shutil.copy2(src, dst)
    else:
        os.link(src, dst)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entry_path(scene_dir: Path, entry: dict[str, Any] | str) -> Path:
    if isinstance(entry, dict):
        value = entry.get("render_exr") or entry.get("exr") or entry.get("render")
    else:
        value = entry
    path = scene_dir / str(value)
    if path.suffix.lower() != ".exr":
        path = path.with_suffix(".exr")
    return path


def read_component(scene_dir: Path, entry: dict[str, Any] | str):
    from tokenlight_dataset.exr_io import read_exr

    return read_exr(entry_path(scene_dir, entry))


def valid_lights(scene_dir: Path, spatial: dict[str, Any]) -> list[dict[str, Any]]:
    lights = []
    for light in spatial.get("point_lights", []):
        if not light.get("valid", True) or not light.get("render"):
            continue
        if entry_path(scene_dir, light).exists():
            lights.append(light)
    if not lights:
        raise RuntimeError(f"No valid spatial point lights in {scene_dir}")
    return lights


def sample_color(rng: random.Random):
    import numpy as np

    return np.array([rng.uniform(0.45, 1.0) for _ in range(3)], dtype=np.float32)


def spatial_light_component(scene_dir: Path, spatial: dict[str, Any], light: dict[str, Any], ambient):
    import numpy as np

    component = read_component(scene_dir, light)
    if spatial.get("point_light_output_semantics") == "ambient_plus_point_light_target":
        component = component - ambient
    source_color = np.asarray(light.get("component_color", [1.0, 1.0, 1.0]), dtype=np.float32)
    component = component / np.maximum(source_color.reshape(1, 1, 3), 1e-4)
    return np.maximum(component, 0.0)


def global_diffuse_meta(meta: dict[str, Any]) -> dict[str, Any] | None:
    return meta.get("global_diffuse") or meta.get("spatial", {}).get("global_diffuse")


def sample_global_background(scene_dir: Path, meta: dict[str, Any], rng: random.Random, include_global_diffuse: bool):
    import numpy as np

    spatial = meta["spatial"]
    if not include_global_diffuse:
        ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        scale = rng.uniform(0.25, 1.15)
        return scale * ambient, {
            "ambient_render": spatial.get("ambient_render"),
            "ambient_scale": scale,
            "global_diffuse": None,
        }

    diffuse = global_diffuse_meta(meta)
    if not diffuse or not diffuse.get("variants"):
        ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        scale = rng.uniform(0.25, 1.15)
        return scale * ambient, {"ambient_scale": scale, "global_diffuse": None}

    variants = [row for row in diffuse.get("variants", []) if row.get("render")]
    if not variants:
        raise RuntimeError(f"No global_diffuse variants in {scene_dir}")
    variant = rng.choice(variants)
    dg = float(variant.get("dg", variant.get("normalized_diffuse", 0.0)))
    complete_targets = bool(diffuse.get("complete_target_variants", True))
    if complete_targets:
        return read_component(scene_dir, variant), {
            "complete_target_variants": True,
            "variant_id": variant.get("id"),
            "dg": dg,
        }

    ambient_entry = diffuse.get("ambient_output", diffuse.get("ambient_render"))
    if ambient_entry is None:
        raise RuntimeError(f"Component global_diffuse metadata needs ambient_output or ambient_render in {scene_dir}")
    ambient = read_component(scene_dir, ambient_entry)
    component = read_component(scene_dir, variant)
    ambient_range = diffuse.get("ambient_scale_range", [0.85, 1.15])
    intensity_range = diffuse.get("intensity_range", [0.85, 1.15])
    ambient_scale = rng.uniform(float(ambient_range[0]), float(ambient_range[1]))
    intensity = rng.uniform(float(intensity_range[0]), float(intensity_range[1]))
    color = np.asarray(diffuse.get("light", {}).get("color", [1.0, 1.0, 1.0]), dtype=np.float32)
    linear = ambient_scale * ambient + intensity * component * color.reshape(1, 1, 3)
    return linear, {
        "complete_target_variants": False,
        "variant_id": variant.get("id"),
        "dg": dg,
        "ambient_scale": ambient_scale,
        "intensity": intensity,
        "color": color.tolist(),
    }


def save_png(linear, path: Path) -> None:
    import numpy as np
    from PIL import Image

    from tokenlight_dataset.tonemap import reinhard, to_uint8

    linear = np.asarray(linear)
    while linear.ndim > 3 and 1 in linear.shape[:-1]:
        axis = next(index for index, size in enumerate(linear.shape[:-1]) if size == 1)
        linear = np.squeeze(linear, axis=axis)
    if linear.ndim != 3 or linear.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 image for {path}, got shape {linear.shape}")
    linear = linear[..., :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(reinhard(linear)), mode="RGB").save(path, compress_level=1)


def render_global_ambient_samples(
    scene_dest: Path,
    meta: dict[str, Any],
    ambient,
    count: int,
    rng: random.Random,
    overwrite: bool,
    ambient_scales: list[float] | None = None,
) -> list[dict[str, Any]]:
    samples = []
    scale_values = list(ambient_scales or [])
    if not scale_values:
        scale_values = [rng.uniform(0.1, 1.3) for _ in range(max(0, count))]
    for idx, scale in enumerate(scale_values):
        rel_path = Path("samples") / f"global_ambient_{idx:03d}.png"
        out_path = scene_dest / rel_path
        if overwrite or not out_path.exists():
            save_png(float(scale) * ambient, out_path)
        samples.append(
            {
                "image": rel_path.as_posix(),
                "input_image": rel_path.as_posix(),
                "scene_id": meta.get("scene_id"),
                "task": "global_ambient",
                "ambient_scale": scale,
                "global_control": {
                    "ambient_scale": scale,
                },
                "lights": [],
            }
        )
    return samples


def choose_ambient_scales(
    ambient_scales: list[float],
    ambient_policy: str,
    sample_rng: random.Random,
    sample_index: int,
) -> list[tuple[int | None, float]]:
    if not ambient_scales:
        return [(None, sample_rng.uniform(0.25, 1.15))]
    if ambient_policy == "grid":
        return list(enumerate(ambient_scales))
    if ambient_policy == "cycle":
        idx = sample_index % len(ambient_scales)
    else:
        idx = sample_rng.randrange(len(ambient_scales))
    return [(idx, float(ambient_scales[idx]))]


def choose_color_indices(
    fixed_colors: list[tuple[float, float, float]],
    color_policy: str,
    sample_rng: random.Random,
    sample_index: int,
) -> list[int]:
    if not fixed_colors:
        return []
    if color_policy == "grid":
        return list(range(len(fixed_colors)))
    if color_policy == "cycle":
        return [sample_index % len(fixed_colors)]
    return [sample_rng.randrange(len(fixed_colors))]


def render_fixed_single_light_samples(
    scene_dest: Path,
    meta: dict[str, Any],
    ambient,
    light_components: dict[int, Any],
    selected_lights: list[dict[str, Any]],
    fixed_colors: list[tuple[float, float, float]],
    color_policy: str,
    ambient_scales: list[float],
    ambient_policy: str,
    fixed_intensity: float,
    seed: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    import numpy as np

    samples = []
    sample_index = 0
    for light in selected_lights:
        light_id = int(light["id"])
        sample_rng = random.Random(seed * 1_000_000 + light_id)
        for color_idx in choose_color_indices(fixed_colors, color_policy, sample_rng, sample_index):
            color_values = fixed_colors[color_idx]
            for ambient_idx, ambient_scale in choose_ambient_scales(
                ambient_scales,
                ambient_policy,
                sample_rng,
                sample_index,
            ):
                color = np.asarray(color_values, dtype=np.float32)
                intensity = float(fixed_intensity)
                linear = float(ambient_scale) * ambient
                linear = linear + intensity * light_components[light_id] * color.reshape(1, 1, 3)
                suffix = f"c{color_idx:02d}"
                if ambient_idx is not None:
                    suffix = f"{suffix}_a{ambient_idx:02d}"
                rel_path = Path("samples") / f"light_{light_id:03d}_{suffix}.png"
                out_path = scene_dest / rel_path
                if overwrite or not out_path.exists():
                    save_png(linear, out_path)
                sample = {
                    "image": rel_path.as_posix(),
                    "scene_id": meta.get("scene_id"),
                    "task": "single_light",
                    "color_id": color_idx,
                    "ambient_id": ambient_idx,
                    "global_control": {
                        "ambient_scale": float(ambient_scale),
                    },
                    "lights": [
                        {
                            "id": light_id,
                            "position": light.get("canonical_position"),
                            "color": color.tolist(),
                            "intensity": intensity,
                            "radius": light.get("canonical_radius"),
                            "base_energy": light.get("canonical_energy"),
                        }
                    ],
                }
                if ambient_idx is not None:
                    sample["input_image"] = (Path("samples") / f"global_ambient_{ambient_idx:03d}.png").as_posix()
                samples.append(sample)
                sample_index += 1
    return samples


def render_global_diffuse_samples(
    scene_dir: Path,
    scene_dest: Path,
    meta: dict[str, Any],
    count: int,
    rng: random.Random,
    overwrite: bool,
) -> list[dict[str, Any]]:
    import numpy as np

    diffuse = global_diffuse_meta(meta)
    if not diffuse:
        return []
    variants = sorted(
        [row for row in diffuse.get("variants", []) if row.get("render")],
        key=lambda row: float(row.get("dg", row.get("normalized_diffuse", 0.0))),
    )
    selected = [rng.choice(variants) for _ in range(max(0, count))] if variants else []
    if not selected:
        return []

    complete_targets = bool(diffuse.get("complete_target_variants", True))
    ambient = None
    color = None
    if not complete_targets:
        ambient_entry = diffuse.get("ambient_output", diffuse.get("ambient_render"))
        if ambient_entry is None:
            raise RuntimeError(f"Component global_diffuse metadata needs ambient_output or ambient_render in {scene_dir}")
        ambient = read_component(scene_dir, ambient_entry)
        color = np.asarray(diffuse.get("light", {}).get("color", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)

    samples = []
    for idx, variant in enumerate(selected):
        rel_path = Path("samples") / f"global_diffuse_{idx:03d}.png"
        out_path = scene_dest / rel_path
        dg = float(variant.get("dg", variant.get("normalized_diffuse", 0.0)))
        if complete_targets:
            linear = read_component(scene_dir, variant)
            condition = {"complete_target_variants": True, "variant_id": variant.get("id"), "dg": dg}
        else:
            component = read_component(scene_dir, variant)
            linear = ambient + component * color
            condition = {
                "complete_target_variants": False,
                "variant_id": variant.get("id"),
                "dg": dg,
                "ambient_scale": 1.0,
                "intensity": 1.0,
                "color": color.reshape(3).tolist(),
            }
        if overwrite or not out_path.exists():
            save_png(linear, out_path)
        samples.append(
            {
                "image": rel_path.as_posix(),
                "scene_id": meta.get("scene_id"),
                "task": "global_diffuse",
                "global_control": condition,
            }
        )
    return samples


def render_sample(
    scene_dir: Path,
    scene_dest: Path,
    meta: dict[str, Any],
    ambient,
    light_components: dict[int, Any],
    lights: list[dict[str, Any]],
    selected_lights: list[dict[str, Any]],
    out_name: str,
    rng: random.Random,
    include_global_diffuse: bool,
    overwrite: bool,
    ambient_override: tuple[int | None, float] | None = None,
) -> dict[str, Any]:
    if ambient_override is not None and not include_global_diffuse:
        ambient_idx, ambient_scale = ambient_override
        linear = float(ambient_scale) * ambient
        global_condition = {
            "ambient_render": meta["spatial"].get("ambient_output", meta["spatial"].get("ambient_render")),
            "ambient_scale": float(ambient_scale),
            "global_diffuse": None,
        }
        if ambient_idx is not None:
            input_rel_path = Path("samples") / f"global_ambient_{ambient_idx:03d}.png"
        else:
            input_rel_path = Path("samples") / f"{Path(out_name).stem}_input.png"
            input_out_path = scene_dest / input_rel_path
            if overwrite or not input_out_path.exists():
                save_png(linear.copy() if hasattr(linear, "copy") else linear, input_out_path)
    else:
        linear, global_condition = sample_global_background(scene_dir, meta, rng, include_global_diffuse)
        input_rel_path = Path("samples") / f"{Path(out_name).stem}_input.png"
        input_out_path = scene_dest / input_rel_path
        if overwrite or not input_out_path.exists():
            save_png(linear.copy() if hasattr(linear, "copy") else linear, input_out_path)
    spatial = meta["spatial"]
    intensity_range = spatial.get("intensity_range", [0.15, 1.25])
    intensity_lo, intensity_hi = float(intensity_range[0]), float(intensity_range[1])
    light_conditions = []
    for light in selected_lights:
        color = sample_color(rng)
        intensity = rng.uniform(intensity_lo, intensity_hi)
        linear = linear + intensity * light_components[int(light["id"])] * color.reshape(1, 1, 3)
        light_conditions.append(
            {
                "id": int(light["id"]),
                "position": light.get("canonical_position"),
                "color": color.tolist(),
                "intensity": intensity,
                "radius": light.get("canonical_radius"),
                "base_energy": light.get("canonical_energy"),
            }
        )

    rel_path = Path("samples") / out_name
    out_path = scene_dest / rel_path
    if overwrite or not out_path.exists():
        save_png(linear, out_path)
    return {
        "image": rel_path.as_posix(),
        "scene_id": meta.get("scene_id", scene_dir.name),
        "task": "single_light" if len(selected_lights) == 1 else "two_light",
        "input_image": input_rel_path.as_posix(),
        "global_control": global_condition,
        "lights": light_conditions,
    }


def stage_reference_files(scene_dir: Path, scene_dest: Path, copy_existing: bool, overwrite: bool) -> int:
    count = 0
    for src in scene_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() == ".exr":
            continue
        rel = src.relative_to(scene_dir)
        if rel.parts and rel.parts[0] == "samples":
            continue
        link_or_copy(src, scene_dest / rel, copy_existing, overwrite)
        count += 1
    return count


def stage_scene(
    scene_dir: Path,
    source: Path,
    dest: Path,
    seed: int,
    single_lights: str,
    max_single_lights: int,
    two_light_samples: int,
    global_ambient_samples: int,
    global_diffuse_samples: int,
    include_global_diffuse: bool,
    fixed_colors: list[tuple[float, float, float]],
    color_policy: str,
    ambient_scales: list[float],
    ambient_policy: str,
    fixed_intensity: float,
    copy_existing: bool,
    overwrite: bool,
) -> tuple[str, int, int]:
    meta = load_json(scene_dir / "meta.json")
    source_root = scene_dir.parents[1]
    scene_dest = dest_root_for(source, dest, source_root) / "scenes" / scene_dir.name
    linked = stage_reference_files(scene_dir, scene_dest, copy_existing, overwrite)

    spatial = meta["spatial"]
    ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
    lights = valid_lights(scene_dir, spatial)
    if max_single_lights > 0:
        single_candidates = lights[:max_single_lights]
    else:
        single_candidates = lights

    light_components = {
        int(light["id"]): spatial_light_component(scene_dir, spatial, light, ambient)
        for light in lights
    }

    scene_seed = seed + sum(ord(ch) for ch in scene_dir.name)
    rng = random.Random(scene_seed)
    samples: list[dict[str, Any]] = []

    samples.extend(
        render_global_ambient_samples(
            scene_dest,
            meta,
            ambient,
            global_ambient_samples,
            random.Random(scene_seed * 100_000 + 20_000),
            overwrite,
            ambient_scales,
        )
    )
    if include_global_diffuse:
        samples.extend(
            render_global_diffuse_samples(
                scene_dir,
                scene_dest,
                meta,
                global_diffuse_samples,
                random.Random(scene_seed * 100_000 + 30_000),
                overwrite,
            )
        )

    if single_lights == "all" and fixed_colors:
        samples.extend(
            render_fixed_single_light_samples(
                scene_dest,
                meta,
                ambient,
                light_components,
                single_candidates,
                fixed_colors,
                color_policy,
                ambient_scales,
                ambient_policy,
                fixed_intensity,
                scene_seed,
                overwrite,
            )
        )
    elif single_lights == "all":
        for light in single_candidates:
            light_id = int(light["id"])
            sample_rng = random.Random(scene_seed * 100_000 + light_id)
            samples.append(
                render_sample(
                    scene_dir,
                    scene_dest,
                    meta,
                    ambient,
                    light_components,
                    lights,
                    [light],
                    f"light_{light_id:03d}.png",
                    sample_rng,
                    include_global_diffuse,
                    overwrite,
                )
            )

    if len(lights) >= 2 and two_light_samples > 0:
        for idx in range(two_light_samples):
            selected = rng.sample(lights, 2)
            sample_rng = random.Random(scene_seed * 100_000 + 10_000 + idx)
            for ambient_idx, ambient_scale in choose_ambient_scales(
                ambient_scales,
                ambient_policy,
                sample_rng,
                idx,
            ):
                suffix = f"_a{ambient_idx:02d}" if ambient_idx is not None else ""
                samples.append(
                    render_sample(
                        scene_dir,
                        scene_dest,
                        meta,
                        ambient,
                        light_components,
                        lights,
                        selected,
                        f"two_lights_{idx:03d}{suffix}.png",
                        sample_rng,
                        include_global_diffuse,
                        overwrite,
                        ambient_override=(ambient_idx, ambient_scale),
                    )
                )

    manifest = {
        "schema": "tokenlight_composed_png_samples_v1",
        "source_scene": str(scene_dir),
        "scene_id": meta.get("scene_id", scene_dir.name),
        "single_light_count": sum(1 for row in samples if row["task"] == "single_light"),
        "two_light_count": sum(1 for row in samples if row["task"] == "two_light"),
        "global_ambient_count": sum(1 for row in samples if row["task"] == "global_ambient"),
        "global_diffuse_count": sum(1 for row in samples if row["task"] == "global_diffuse"),
        "samples": samples,
    }
    manifest_path = scene_dest / "samples_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return scene_dir.name, len(samples), linked


def attrs_from_light_condition(light: dict[str, Any]) -> dict[str, float]:
    position = list(light.get("position") or [None, None, None])
    color = list(light.get("color") or [None, None, None])
    attrs = {
        "x": position[0] if len(position) > 0 else None,
        "y": position[1] if len(position) > 1 else None,
        "z": position[2] if len(position) > 2 else None,
        "r": color[0] if len(color) > 0 else None,
        "g": color[1] if len(color) > 1 else None,
        "b": color[2] if len(color) > 2 else None,
        "lambda": light.get("intensity"),
        "d": light.get("radius"),
    }
    return {
        key: float(value)
        for key, value in attrs.items()
        if value is not None
    }


def attrs_from_light_sample(sample: dict[str, Any]) -> dict[str, Any]:
    global_control = sample.get("global_control", {}) or {}
    attrs: dict[str, Any] = {}
    ambient_scale = global_control.get("ambient_scale", sample.get("ambient_scale"))
    if ambient_scale is not None:
        attrs["a"] = float(ambient_scale)
    diffuse_gain = global_control.get("dg")
    if diffuse_gain is not None:
        attrs["dg"] = float(diffuse_gain)
    attrs["lights"] = [
        attrs_from_light_condition(light)
        for light in sample.get("lights", []) or []
        if isinstance(light, dict)
    ]
    return attrs


def write_train_metadata(
    dest: Path,
    metadata_name: str,
    metadata_nomask_name: str,
    prompt: str,
) -> tuple[int, int]:
    rows = []
    rows_nomask = []
    for manifest_path in sorted(dest.glob("**/samples_manifest.json")):
        scene_dest = manifest_path.parent
        scene_rel = scene_dest.relative_to(dest).as_posix()
        manifest = load_json(manifest_path)
        mask_path = scene_dest / "masks" / "object_mask.png"
        mask_rel = f"{scene_rel}/masks/object_mask.png" if mask_path.exists() else None
        for sample in manifest.get("samples", []):
            if sample.get("task") not in {"global_ambient", "single_light", "two_light"}:
                continue
            attrs = attrs_from_light_sample(sample)
            if not attrs:
                continue
            input_image = sample.get("input_image")
            if not input_image and sample.get("task") == "global_ambient":
                input_image = sample.get("image")
            if not input_image:
                continue
            light_ids = [
                int(light["id"])
                for light in sample.get("lights", []) or []
                if isinstance(light, dict) and light.get("id") is not None
            ]
            row = {
                "video": f"{scene_rel}/{sample['image']}",
                "input_image": f"{scene_rel}/{input_image}",
                "prompt": prompt,
                "attrs_json": json.dumps(
                    attrs,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "scene_id": sample.get("scene_id") or manifest.get("scene_id"),
                "task": sample.get("task"),
                "valid": True,
            }
            if len(light_ids) == 1:
                row["light_id"] = light_ids[0]
            elif len(light_ids) > 1:
                row["light_ids"] = light_ids
            if mask_rel:
                row["mask"] = mask_rel
            rows.append(row)
            row_nomask = dict(row)
            row_nomask.pop("mask", None)
            rows_nomask.append(row_nomask)

    for path, items in ((dest / metadata_name, rows), (dest / metadata_nomask_name, rows_nomask)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return len(rows), len(rows_nomask)


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()
    fixed_colors = parse_color_palette(args.fixed_colors)
    ambient_scales = parse_float_list(args.ambient_scales)
    if not source.is_dir():
        raise SystemExit(f"SOURCE is not a directory: {source}")
    if dest == source or source in dest.parents:
        raise SystemExit("DEST must be outside SOURCE.")

    source_roots = discover_source_roots(source)
    scenes = completed_scene_dirs(source_roots, args.scene_offset, args.scene_limit)

    print(f"[INFO] source={source}")
    print(f"[INFO] dest={dest}")
    print(f"[INFO] source_roots={len(source_roots)} scene_offset={args.scene_offset} completed_scenes={len(scenes)}")
    print(
        f"[INFO] workers={args.workers} single_lights={args.single_lights} "
        f"two_light_samples={args.two_light_samples} "
        f"global_ambient_samples={args.global_ambient_samples} "
        f"global_diffuse_samples={args.global_diffuse_samples} "
        f"include_global_diffuse={args.include_global_diffuse} seed={args.seed}"
    )
    if fixed_colors:
        print(
            f"[INFO] fixed_colors={len(fixed_colors)} ambient_scales={ambient_scales or 'random'} "
            f"color_policy={args.color_policy} ambient_policy={args.ambient_policy} "
            f"fixed_intensity={args.fixed_intensity}"
        )

    if args.dry_run:
        for root in source_roots:
            print(f"[DRY-RUN] root {root} -> {dest_root_for(source, dest, root)}")
        print("[DRY-RUN] no files written")
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_linked = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                stage_scene,
                scene,
                source,
                dest,
                args.seed,
                args.single_lights,
                args.max_single_lights,
                args.two_light_samples,
                args.global_ambient_samples,
                args.global_diffuse_samples,
                args.include_global_diffuse,
                fixed_colors,
                args.color_policy,
                ambient_scales,
                args.ambient_policy,
                args.fixed_intensity,
                args.copy_existing,
                args.overwrite,
            )
            for scene in scenes
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            scene_id, sample_count, linked = future.result()
            total_samples += sample_count
            total_linked += linked
            if idx == 1 or idx % 25 == 0 or idx == len(futures):
                print(
                    f"[PROGRESS] scenes={idx}/{len(futures)} last={scene_id} "
                    f"samples={total_samples} linked={total_linked}",
                    flush=True,
                )

    dataset_manifest = {
        "schema": "tokenlight_composed_png_dataset_v1",
        "source": str(source),
        "scene_count": len(scenes),
        "sample_count": total_samples,
        "single_lights": args.single_lights,
        "two_light_samples_per_scene": args.two_light_samples,
        "global_ambient_samples_per_scene": args.global_ambient_samples,
        "global_diffuse_samples_per_scene": args.global_diffuse_samples,
        "include_global_diffuse": args.include_global_diffuse,
        "fixed_colors": fixed_colors,
        "color_policy": args.color_policy if fixed_colors else None,
        "ambient_scales": ambient_scales,
        "ambient_policy": args.ambient_policy,
        "fixed_intensity": args.fixed_intensity if fixed_colors else None,
        "seed": args.seed,
    }
    (dest / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] wrote composed PNG stage: {dest}")
    print(f"[DONE] samples={total_samples} linked={total_linked}")
    if args.write_train_metadata:
        with_mask, without_mask = write_train_metadata(
            dest,
            args.metadata_name,
            args.metadata_nomask_name,
            args.metadata_prompt,
        )
        print(f"[DONE] wrote train metadata: {dest / args.metadata_name} rows={with_mask}")
        print(f"[DONE] wrote nomask metadata: {dest / args.metadata_nomask_name} rows={without_mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
