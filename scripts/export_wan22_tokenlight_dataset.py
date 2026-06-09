from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

WAN_HELPER_PATH = SRC_ROOT / "tokenlight" / "wan.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TokenLight PNG/component pairs to DiffSynth Wan2.2-TI2V metadata."
    )
    parser.add_argument("--dataset-kind", choices=["point-light-png", "component"], default="point-light-png")
    parser.add_argument("--data-root", "--component-root", dest="component_root", default="/workspace/data/sample")
    parser.add_argument("--component-repo", default="/workspace/repos/relighting_dataset")
    parser.add_argument("--output", default="/workspace/data/tokenlight_wan22_train")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--component-length", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=["spatial", "ambient", "diffuse"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-lights", type=int, default=1)
    parser.add_argument("--image-range", choices=["minus_one_one", "zero_one"], default="minus_one_one")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--target-format", choices=["png", "gif"], default="png")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--prompt-prefix", default=None)
    parser.add_argument("--prompt-mode", choices=["generic", "attrs"], default="generic")
    parser.add_argument("--include-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-object-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pairing", choices=["random", "all-targets"], default="random")
    parser.add_argument("--source-light-id", type=int, default=0)
    parser.add_argument("--target-light-id", type=int, default=None)
    parser.add_argument("--target-light-ids", nargs="+", type=int, default=None)
    parser.add_argument("--allow-self-pairs", action="store_true")
    parser.add_argument("--include-invalid-lights", action="store_true")
    parser.add_argument("--light-intensity", type=float, default=1.0)
    parser.add_argument("--light-color", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--dedupe-source-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dedupe-mask-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_wan_helpers():
    spec = importlib.util.spec_from_file_location("_tokenlight_wan_helpers", WAN_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Wan helper module: {WAN_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.attrs_json, module.light_attrs_to_prompt


def tensor_to_uint8_image(tensor) -> Image.Image:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(tensor.shape)}")
    tensor = tensor.detach().float().cpu()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] > 3:
        tensor = tensor[:3]
    if float(tensor.min()) < -0.05:
        tensor = (tensor + 1.0) * 0.5
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def resize_if_needed(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.size == (width, height):
        return image
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((width, height), resampling)


def save_target(image: Image.Image, path: Path, num_frames: int, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        frames = [image.copy() for _ in range(max(1, int(num_frames)))]
        duration_ms = max(1, int(1000 / max(1, int(fps))))
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
        return
    image.save(path)


def safe_relative_png(root_name: str, key: Any, fallback_index: int) -> Path:
    if key is None or str(key).strip() == "":
        return Path(root_name) / f"{fallback_index:08d}.png"
    parts = []
    for part in Path(str(key)).parts:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in part)
        if safe not in {"", ".", ".."}:
            parts.append(safe)
    if not parts:
        return Path(root_name) / f"{fallback_index:08d}.png"
    return Path(root_name, *parts).with_suffix(".png")


def safe_relative_path(value: Any, fallback: Path) -> Path:
    if value is None or str(value).strip() == "":
        return fallback
    parts = []
    for part in Path(str(value)).parts:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in part)
        if safe not in {"", ".", ".."}:
            parts.append(safe)
    if not parts:
        return fallback
    return Path(*parts)


def save_once(image: Image.Image, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def infer_task(sample: dict[str, Any]) -> str:
    attrs = sample.get("attrs", {})
    if "x" in attrs or "y" in attrs or "z" in attrs:
        return "spatial"
    if "dg" in attrs:
        return "diffuse"
    if "t" in attrs:
        return "fixture"
    return "ambient"


def main() -> int:
    args = parse_args()
    from tokenlight.data import DirectPointLightPngDataset, RelightingComponentAdapterDataset

    attrs_json, light_attrs_to_prompt = load_wan_helpers()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        existing = output / "metadata.csv"
        if existing.exists():
            raise FileExistsError(f"{existing} already exists. Pass --overwrite to replace the export.")

    output.mkdir(parents=True, exist_ok=True)
    target_light_ids = args.target_light_ids
    if args.target_light_id is not None:
        target_light_ids = [args.target_light_id] if target_light_ids is None else [args.target_light_id, *target_light_ids]

    if args.dataset_kind == "point-light-png":
        dataset = DirectPointLightPngDataset(
            root=args.component_root,
            length=args.count if args.count > 0 else None,
            modes=tuple(args.modes),
            seed=args.seed,
            valid_only=not args.include_invalid_lights,
            pairing=args.pairing,
            source_light_id=args.source_light_id,
            target_light_ids=None if target_light_ids is None else tuple(target_light_ids),
            allow_self_pairs=args.allow_self_pairs,
            include_object_masks=args.include_object_masks,
            light_color=tuple(args.light_color),
            light_intensity=args.light_intensity,
        )
    else:
        dataset = RelightingComponentAdapterDataset(
            component_root=args.component_root,
            repo_path=args.component_repo,
            length=args.component_length or args.count,
            modes=tuple(args.modes),
            seed=args.seed,
            max_lights=args.max_lights,
            image_range=args.image_range,
            include_masks=args.include_masks,
            include_object_masks=args.include_object_masks,
        )

    metadata_csv = output / "metadata.csv"
    metadata_jsonl = output / "metadata.jsonl"
    target_suffix = ".gif" if args.target_format == "gif" else ".png"
    rows: list[dict[str, str]] = []
    export_count = len(dataset) if args.count <= 0 else args.count

    with metadata_jsonl.open("w", encoding="utf-8") as jsonl:
        for index in range(export_count):
            sample = dataset[index]
            source = resize_if_needed(tensor_to_uint8_image(sample["source"]), args.width, args.height)
            target = resize_if_needed(tensor_to_uint8_image(sample["target"]), args.width, args.height)
            mask = None
            if "mask" in sample:
                mask = resize_if_needed(tensor_to_uint8_image(sample["mask"]), args.width, args.height)

            condition = dict(sample.get("condition", {}))
            rel_source = (
                safe_relative_path(
                    condition.get("source_relpath"),
                    safe_relative_png("source", condition.get("source_key"), index),
                )
                if args.dedupe_source_files
                else Path("source") / f"{index:08d}.png"
            )
            rel_target = safe_relative_path(
                condition.get("target_relpath"),
                Path("target") / f"{index:08d}{target_suffix}",
            ).with_suffix(target_suffix)
            save_once(source, output / rel_source) if args.dedupe_source_files else source.save(output / rel_source)
            save_target(target, output / rel_target, args.num_frames, args.fps)
            rel_mask = ""
            if mask is not None:
                rel_mask_path = (
                    safe_relative_path(
                        condition.get("mask_relpath"),
                        safe_relative_png("mask", condition.get("mask_key"), index),
                    )
                    if args.dedupe_mask_files
                    else Path("mask") / f"{index:08d}.png"
                )
                save_once(mask, output / rel_mask_path) if args.dedupe_mask_files else mask.save(output / rel_mask_path)
                rel_mask = rel_mask_path.as_posix()

            attrs = dict(sample.get("attrs", {}))
            task = infer_task(sample)
            prompt = light_attrs_to_prompt(
                attrs,
                task=task,
                prefix=args.prompt_prefix,
                include_values=args.prompt_mode == "attrs",
            )
            record = {
                "video": rel_target.as_posix(),
                "input_image": rel_source.as_posix(),
                "mask": rel_mask,
                "prompt": prompt,
                "task": task,
                "attrs_json": attrs_json(attrs),
            }
            rows.append(record)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (index + 1) % 100 == 0:
                print(f"[TokenLight/Wan] Exported {index + 1}/{export_count}", flush=True)

    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "input_image", "mask", "prompt", "task", "attrs_json"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[TokenLight/Wan] Wrote {len(rows)} rows to {metadata_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
