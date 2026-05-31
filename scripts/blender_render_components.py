"""Blender component-render helper.

Run inside Blender:

  blender -b scene.blend --python scripts/blender_render_components.py -- spec.json

The spec file intentionally mirrors the paper components: ambient environment
renders, per-position point-light contribution renders, spread-control renders,
and optional fixture masks. This script is a deterministic helper, not a full
asset-list replacement for the unpublished TokenLight data pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _after_double_dash(argv):
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def configure_cycles(bpy, resolution: int, samples: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"


def clear_lights(bpy):
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def render_to(bpy, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def add_point_light(bpy, name: str, position, energy: float, radius: float):
    light_data = bpy.data.lights.new(name=name, type="POINT")
    light_data.energy = energy
    light_data.shadow_soft_size = radius
    obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    return obj


def main():
    import bpy

    args = _after_double_dash(sys.argv)
    if len(args) != 1:
        raise SystemExit("Usage: blender -b scene.blend --python scripts/blender_render_components.py -- spec.json")
    spec_path = Path(args[0])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = Path(spec["output"])
    configure_cycles(bpy, resolution=int(spec.get("resolution", 960)), samples=int(spec.get("samples", 128)))

    if "ambient_output" in spec:
        clear_lights(bpy)
        render_to(bpy, output / spec["ambient_output"])

    for light in spec.get("point_lights", []):
        clear_lights(bpy)
        add_point_light(
            bpy,
            light.get("name", "point_light"),
            light["position"],
            float(light.get("energy", 1.0)),
            float(light.get("radius", 0.05)),
        )
        render_to(bpy, output / light["output"])

    for spread in spec.get("spread_lights", []):
        clear_lights(bpy)
        add_point_light(
            bpy,
            spread.get("name", "spread_light"),
            spread["position"],
            float(spread.get("energy", 1.0)),
            float(spread["radius"]),
        )
        render_to(bpy, output / spread["output"])


if __name__ == "__main__":
    main()
