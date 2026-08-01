#!/usr/bin/env python3
"""Convert the user-provided hospital archives into MuJoCo-ready local meshes.

Run through Blender:
  blender --background --python scripts/prepare_hospital_assets.py -- \
    --source-dir ~/Downloads/hospital_assets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "baymax_nurse" / "assets" / "local"
ARCHIVES = {
    "room": "lowpoly-medical-room.zip",
    "bed": "medical-examination-bed-game-ready-asset.zip",
    "monitor": "aero-monitor.zip",
    "patient": "grandma-on-bench-free.zip",
}
SOURCES = {
    "room": "medical_room.fbx",
    "bed": "Hospital Bed.glb",
    "monitor": "monitor.fbx",
    "patient": "GrandmaOnBench.fbx",
}
TEXTURES = {
    "room": "palette.png",
    "bed": "Image_0_0.png",
    "monitor": "monitor_BaseColor.jpg",
    "grandma_sitting": "9_meshes_Merge_Diffuse.png",
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.home() / "Downloads" / "hospital_assets",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def extract_archives(source_dir: Path, work_dir: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for name, filename in ARCHIVES.items():
        archive = source_dir / filename
        if not archive.is_file():
            raise FileNotFoundError(f"Missing hospital asset archive: {archive}")
        destination = work_dir / name
        destination.mkdir(parents=True)
        with zipfile.ZipFile(archive) as package:
            package.extractall(destination)
        matches = list(destination.rglob(SOURCES[name]))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {SOURCES[name]} in {archive.name}, found {len(matches)}"
            )
        roots[name] = matches[0]
    boy = source_dir / "boy.glb"
    if not boy.is_file():
        raise FileNotFoundError(f"Missing hospital patient asset: {boy}")
    roots["boy"] = boy
    return roots


def import_asset(path: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if path.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))


def is_skinned(obj: bpy.types.Object) -> bool:
    return bool(
        obj.parent is not None and obj.parent.type == "ARMATURE"
        or any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    )


def select_meshes(kind: str) -> list[bpy.types.Object]:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if kind == "room":
        # The source room includes its own decorative bed. Remove those pieces
        # so the standalone detailed bed and its collision geometry stay aligned.
        source_bed_parts = {f"cube.{index:03d}" for index in range(23, 31)}
        meshes = [
            obj
            for obj in meshes
            if not is_skinned(obj)
            and obj.name.lower() not in {"walls", "lowerwall", "floor", "floortile"}
            and obj.name.lower() not in source_bed_parts
            and not (
                min(obj.dimensions) <= 0.08
                and max(obj.dimensions.x, obj.dimensions.y) >= 2.0
            )
        ]
    elif kind == "grandma":
        meshes = [obj for obj in meshes if obj.name != "Cube_002"]
    elif kind == "boy":
        meshes = [obj for obj in meshes if is_skinned(obj)]
    if not meshes:
        raise RuntimeError(f"No usable meshes found for {kind}")
    return meshes


def join_meshes(meshes: list[bpy.types.Object]) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    converted: list[bpy.types.Object] = []
    for obj in meshes:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target="MESH")
        converted.append(bpy.context.active_object)
        bpy.ops.object.select_all(action="DESELECT")
    for obj in converted:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = converted[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = "hospital_asset"
    return joined


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector(tuple(min(point[index] for point in points) for index in range(3))),
        Vector(tuple(max(point[index] for point in points) for index in range(3))),
    )


def normalize(
    obj: bpy.types.Object,
    kind: str,
    *,
    patient_rotation: tuple[float, float, float] | None = None,
) -> None:
    low, high = bounds(obj)
    dimensions = high - low
    if kind == "room":
        scale = 4.0 / max(dimensions.x, dimensions.y)
    elif kind == "bed":
        scale = 2.08 / max(dimensions.x, dimensions.y)
    elif kind == "monitor":
        scale = 0.46 / max(dimensions)
    else:
        target_height = 1.55 if kind == "grandma" else 1.45
        scale = target_height / max(dimensions.z, 1e-6)
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if patient_rotation:
        obj.rotation_euler = patient_rotation
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    low, high = bounds(obj)
    center = (low + high) * 0.5
    obj.location += Vector((-center.x, -center.y, -low.z))
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


def export_obj(obj: bpy.types.Object, destination: Path) -> int:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=str(destination.resolve()),
        export_selected_objects=True,
        export_materials=False,
        export_triangulated_mesh=True,
        forward_axis="Y",
        up_axis="Z",
    )
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def normalize_exported_obj(path: Path, kind: str) -> None:
    """Enforce metric bounds after FBX armature/unit transforms are baked."""
    lines = path.read_text(encoding="utf-8").splitlines()
    vertices: list[tuple[float, float, float]] = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "v":
            vertices.append(tuple(float(value) for value in fields[1:4]))
    if not vertices:
        raise RuntimeError(f"No vertices exported to {path}")
    minimum = [min(point[index] for point in vertices) for index in range(3)]
    maximum = [max(point[index] for point in vertices) for index in range(3)]
    dimensions = [maximum[index] - minimum[index] for index in range(3)]
    target = {
        "room": 4.0,
        "bed": 2.08,
        "monitor": 0.46,
        "grandma": 1.55,
        "boy": 1.45,
    }[kind]
    denominator = (
        max(dimensions[0], dimensions[1])
        if kind in {"room", "bed"}
        else max(dimensions)
    )
    scale = target / max(denominator, 1e-9)
    center_x = (minimum[0] + maximum[0]) * 0.5
    center_y = (minimum[1] + maximum[1]) * 0.5
    vertex_index = 0
    output: list[str] = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "v":
            x, y, z = vertices[vertex_index]
            vertex_index += 1
            output.append(
                "v "
                + " ".join(
                    f"{value:.9g}"
                    for value in (
                        (x - center_x) * scale,
                        (y - center_y) * scale,
                        (z - minimum[2]) * scale,
                    )
                )
            )
        else:
            output.append(line)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def convert(
    source: Path,
    output: Path,
    kind: str,
    *,
    patient_rotation: tuple[float, float, float] | None = None,
) -> int:
    import_asset(source)
    obj = join_meshes(select_meshes(kind))
    normalize(obj, kind, patient_rotation=patient_rotation)
    triangles = export_obj(obj, output)
    normalize_exported_obj(output, kind)
    return triangles


def find_texture(work_dir: Path, filename: str) -> Path:
    matches = list(work_dir.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename}, found {len(matches)}")
    return matches[0]


def export_boy_texture(source: Path, destination: Path) -> None:
    import_asset(source)
    image = bpy.data.images.get("Image_0")
    if image is None:
        raise RuntimeError("boy.glb is missing its embedded body texture")
    bpy.context.scene.render.image_settings.file_format = "PNG"
    image.save_render(str(destination), scene=bpy.context.scene)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hospital-assets-") as temporary:
        work_dir = Path(temporary)
        sources = extract_archives(source_dir, work_dir)
        triangles = {
            "room": convert(sources["room"], output_dir / "room.obj", "room"),
            "bed": convert(sources["bed"], output_dir / "bed.obj", "bed"),
            "monitor": convert(
                sources["monitor"], output_dir / "monitor.obj", "monitor"
            ),
            "grandma_sitting": convert(
                sources["patient"],
                output_dir / "grandma_sitting.obj",
                "grandma",
            ),
            "boy_fallen": convert(
                sources["boy"],
                output_dir / "boy_fallen.obj",
                "boy",
                patient_rotation=(math.pi / 2, 0.0, math.radians(12)),
            ),
        }
        texture_files: dict[str, str] = {}
        for asset_name, filename in TEXTURES.items():
            source_texture = find_texture(work_dir, filename)
            destination = output_dir / f"{asset_name}_albedo.png"
            if source_texture.suffix.lower() == ".png":
                shutil.copy2(source_texture, destination)
            else:
                image = bpy.data.images.load(str(source_texture), check_existing=False)
                bpy.context.scene.render.image_settings.file_format = "PNG"
                image.save_render(str(destination), scene=bpy.context.scene)
                bpy.data.images.remove(image)
            texture_files[asset_name] = destination.name
        boy_texture = output_dir / "boy_fallen_albedo.png"
        export_boy_texture(sources["boy"], boy_texture)
        texture_files["boy_fallen"] = boy_texture.name

    assets: dict[str, dict[str, object]] = {}
    for name in ("room", "bed", "monitor", "grandma_sitting", "boy_fallen"):
        assets[name] = {
            "mesh": f"{name}.obj",
            "texture": texture_files[name],
            "rgba": [1.0, 1.0, 1.0, 1.0],
            "triangles": triangles[name],
        }
    # The source character was authored oversized relative to the hospital bed.
    # Keep her original seated pose but bring her to a believable seated scale.
    assets["grandma_sitting"]["scale"] = [0.72, 0.72, 0.72]
    manifest = {
        "version": 1,
        "localOnly": True,
        "sourceDirectory": str(source_dir),
        "notice": "Generated from user-provided archives; review source licenses before redistribution.",
        "archives": {
            filename: sha256(source_dir / filename) for filename in ARCHIVES.values()
        },
        "standaloneSources": {"boy.glb": sha256(source_dir / "boy.glb")},
        "assets": assets,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared hospital assets: {output_dir}")
    print(json.dumps({"triangles": triangles}, indent=2))


if __name__ == "__main__":
    main()
