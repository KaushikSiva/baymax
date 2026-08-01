from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from baymax_nurse.runtime.robot_model import (
    LEG_JOINT_NAMES,
    TORQUE_LIMITS,
    add_robot_sites as _add_robot_sites,
    add_visual as _add_visual,
    default_g1_path,
    numbers as _numbers,
    prefix_tree as _prefix_tree,
)


ROBOT_SPAWN = (0.0, -4.05, 0.78)
HOME_POSITION = (0.0, -4.05)
ROOM_INSPECTION_POINTS = {
    "room_1": (-0.35, -1.15),
    "room_2": (0.35, 2.15),
}
BED_POSITIONS = {
    "room_1": (-1.75, -1.05, 0.46),
    "room_2": (-1.70, 2.05, 0.46),
}
MONITOR_POSITIONS = {
    "room_1": (-2.62, -0.78, 1.22),
    "room_2": (-2.62, 2.32, 1.22),
}
PATIENT_POSITIONS = {
    "patient_101": (-1.75, -1.05, 0.50),
    "patient_202": (1.58, 2.15, 0.205),
}


def default_asset_manifest() -> Path:
    return Path(__file__).with_name("assets") / "local" / "manifest.json"


def build_hospital_xml(
    g1_path: Path | str | None = None,
    asset_manifest: Path | str | None = None,
) -> str:
    source_path = Path(g1_path) if g1_path else default_g1_path()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"G1-with-hands MJCF not found at {source_path}. "
            "Run scripts/download_g1_mjcf.sh."
        )
    source = ET.parse(source_path).getroot()
    pelvis = source.find("./worldbody/body[@name='pelvis']")
    source_actuators = source.find("actuator")
    if pelvis is None or source_actuators is None:
        raise ValueError("G1-with-hands model is missing pelvis or actuators")

    root = ET.Element("mujoco", model="robot_gym_hospital_patrol")
    compiler = copy.deepcopy(source.find("compiler"))
    if compiler is None:
        compiler = ET.Element("compiler")
    compiler.set("meshdir", str((source_path.parent / "assets").resolve()))
    root.append(compiler)
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        gravity="0 0 -9.81",
        cone="elliptic",
    )
    for tag in ("default", "asset"):
        element = source.find(tag)
        if element is not None:
            root.append(copy.deepcopy(element))
    _add_visual(root)
    visual_global = root.find("./visual/global")
    if visual_global is not None:
        visual_global.set("offwidth", "1920")
        visual_global.set("offheight", "1080")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    _add_materials(asset)
    prepared = _add_prepared_assets(asset, asset_manifest)

    worldbody = ET.SubElement(root, "worldbody")
    _add_hospital_shell(worldbody)
    _add_room_contents(worldbody, prepared)

    robot = copy.deepcopy(pelvis)
    _add_robot_sites(robot)
    robot.set("pos", _numbers(ROBOT_SPAWN))
    # The pinned locomotion policy's forward axis is +X; face it toward +Y.
    robot.set("quat", "0.7071068 0 0 0.7071068")
    torso = robot.find(".//body[@name='torso_link']")
    if torso is not None:
        ET.SubElement(
            torso,
            "geom",
            name="clinical_panel",
            type="box",
            pos="0.02 0 0.19",
            size="0.055 0.105 0.075",
            rgba="0.14 0.86 0.69 1",
            contype="0",
            conaffinity="0",
        )
    _prefix_tree(robot, "p1_")
    worldbody.append(robot)

    actuator = ET.SubElement(root, "actuator")
    for source_item in source_actuators:
        joint_name = source_item.get("joint")
        if not joint_name:
            continue
        if joint_name in LEG_JOINT_NAMES:
            index = LEG_JOINT_NAMES.index(joint_name)
            item = ET.Element(
                "motor",
                name=f"p1_{joint_name}",
                joint=f"p1_{joint_name}",
                gear="1",
                ctrllimited="true",
                ctrlrange=f"{-TORQUE_LIMITS[index]} {TORQUE_LIMITS[index]}",
            )
        else:
            item = copy.deepcopy(source_item)
            _prefix_tree(item, "p1_")
        actuator.append(item)

    sensor = ET.SubElement(root, "sensor")
    source_sensors = source.find("sensor")
    if source_sensors is not None:
        for source_item in source_sensors:
            item = copy.deepcopy(source_item)
            _prefix_tree(item, "p1_")
            sensor.append(item)
    return ET.tostring(root, encoding="unicode")


def _add_materials(asset: ET.Element) -> None:
    for name, rgba in (
        ("floor", "0.13 0.17 0.16 1"),
        ("wall", "0.86 0.89 0.87 1"),
        ("rail", "0.15 0.22 0.25 1"),
        ("bed", "0.78 0.83 0.84 1"),
        ("linen", "0.84 0.92 0.91 1"),
        ("monitor", "0.045 0.065 0.075 1"),
        ("critical", "0.94 0.11 0.10 1"),
        ("normal", "0.18 0.90 0.61 1"),
        ("skin", "0.62 0.38 0.28 1"),
        ("gown", "0.28 0.55 0.72 1"),
    ):
        ET.SubElement(asset, "material", name=f"hospital_{name}", rgba=rgba)


def _add_prepared_assets(
    asset: ET.Element, manifest_path: Path | str | None
) -> dict[str, dict[str, str]]:
    path = Path(manifest_path) if manifest_path else default_asset_manifest()
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prepared: dict[str, dict[str, str]] = {}
    for name, definition in manifest.get("assets", {}).items():
        mesh_path = (path.parent / definition["mesh"]).resolve()
        if not mesh_path.is_file():
            continue
        mesh_name = f"hospital_local_{name}_mesh"
        material_name = f"hospital_local_{name}_material"
        mesh_attributes = {"name": mesh_name, "file": str(mesh_path)}
        if scale := definition.get("scale"):
            mesh_attributes["scale"] = _numbers(scale)
        ET.SubElement(asset, "mesh", **mesh_attributes)
        attributes = {
            "name": material_name,
            "rgba": _numbers(definition.get("rgba", [1, 1, 1, 1])),
            "specular": "0.12",
            "shininess": "0.08",
        }
        texture = definition.get("texture")
        if texture:
            texture_path = (path.parent / texture).resolve()
            if texture_path.is_file():
                texture_name = f"hospital_local_{name}_texture"
                ET.SubElement(
                    asset,
                    "texture",
                    name=texture_name,
                    type="2d",
                    file=str(texture_path),
                )
                attributes["texture"] = texture_name
        ET.SubElement(asset, "material", **attributes)
        prepared[name] = {"mesh": mesh_name, "material": material_name}
    return prepared


def _add_hospital_shell(worldbody: ET.Element) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        name="hospital_floor",
        type="plane",
        size="3.7 5.0 0.05",
        material="hospital_floor",
        friction="0.96 0.02 0.002",
        contype="4",
        conaffinity="3",
    )
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 7.4",
        dir="0 0 -1",
        directional="true",
        diffuse="0.78 0.82 0.84",
    )
    for x, y in ((-2.1, -1.2), (2.0, 2.2), (0.0, 0.5)):
        ET.SubElement(worldbody, "light", pos=f"{x} {y} 3.0", diffuse="0.68 0.75 0.73")
    walls = (
        ("west", (-3.55, 0.0, 0.72), (0.07, 4.85, 0.72)),
        ("east", (3.55, 0.0, 0.72), (0.07, 4.85, 0.72)),
        ("south", (0.0, -4.85, 0.72), (3.55, 0.07, 0.72)),
        ("north", (0.0, 4.35, 0.72), (3.55, 0.07, 0.72)),
        ("divider_left", (-2.18, 0.45, 0.86), (1.37, 0.07, 0.86)),
        ("divider_right", (2.18, 0.45, 0.86), (1.37, 0.07, 0.86)),
        ("divider_header", (0.0, 0.45, 2.62), (0.82, 0.07, 0.28)),
    )
    for name, pos, size in walls:
        ET.SubElement(
            worldbody,
            "geom",
            name=f"hospital_wall_{name}",
            type="box",
            pos=_numbers(pos),
            size=_numbers(size),
            material="hospital_wall",
            friction="0.86 0.02 0.002",
            contype="4",
            conaffinity="3",
        )
    for room_id, point in ROOM_INSPECTION_POINTS.items():
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{room_id}_inspection_marker",
            type="cylinder",
            pos=f"{point[0]} {point[1]} 0.008",
            size="0.28 0.008",
            rgba="0.10 0.78 0.62 0.28",
            contype="0",
            conaffinity="0",
        )
    ET.SubElement(
        worldbody,
        "camera",
        name="broadcast_camera",
        pos="0 -10.2 7.2",
        xyaxes="1 0 0 0 0.58 0.815",
        fovy="54",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="overhead_camera",
        pos="0 0 11.0",
        quat="0 1 0 0",
        fovy="48",
    )


def _visual_geom(
    worldbody: ET.Element,
    prepared: dict[str, dict[str, str]],
    asset_name: str,
    instance_name: str,
    pos: tuple[float, float, float],
    quat: str = "1 0 0 0",
) -> None:
    definition = prepared.get(asset_name)
    if not definition:
        return
    ET.SubElement(
        worldbody,
        "geom",
        name=instance_name,
        type="mesh",
        mesh=definition["mesh"],
        material=definition["material"],
        pos=_numbers(pos),
        quat=quat,
        contype="0",
        conaffinity="0",
        group="1",
    )


def _add_room_contents(
    worldbody: ET.Element, prepared: dict[str, dict[str, str]]
) -> None:
    _visual_geom(worldbody, prepared, "room", "room_1_detailed_props", (0.0, -1.65, 0.0))
    _visual_geom(worldbody, prepared, "room", "room_2_detailed_props", (0.0, 2.05, 0.0), "0 0 0 1")
    for room_id, bed_pos in BED_POSITIONS.items():
        _add_physical_bed(worldbody, room_id, bed_pos)
        _visual_geom(
            worldbody,
            prepared,
            "bed",
            f"{room_id}_detailed_bed",
            (bed_pos[0], bed_pos[1], 0.0),
            "0.7071068 0 0 0.7071068",
        )
    for room_id, monitor_pos in MONITOR_POSITIONS.items():
        critical = room_id == "room_1"
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{room_id}_monitor_proxy",
            type="box",
            pos=_numbers(monitor_pos),
            size="0.22 0.10 0.17",
            material="hospital_monitor",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{room_id}_monitor_screen",
            type="box",
            pos=_numbers((monitor_pos[0] + 0.23, monitor_pos[1], monitor_pos[2])),
            size="0.008 0.082 0.125",
            material="hospital_critical" if critical else "hospital_normal",
            contype="0",
            conaffinity="0",
        )
        _visual_geom(
            worldbody,
            prepared,
            "monitor",
            f"{room_id}_detailed_monitor",
            (monitor_pos[0], monitor_pos[1], monitor_pos[2] - 0.17),
            "0.7071068 0 0 0.7071068",
        )
    # Grandma retains the source asset's natural seated pose on the mattress.
    _visual_geom(
        worldbody,
        prepared,
        "grandma_sitting",
        "patient_101_detailed",
        PATIENT_POSITIONS["patient_101"],
        "0.7071068 0 0 0.7071068",
    )
    # Rotate the separately supplied upright mesh onto its side at scene level.
    _visual_geom(
        worldbody,
        prepared,
        "boy_fallen",
        "patient_202_detailed",
        PATIENT_POSITIONS["patient_202"],
        "0.7071068 0.7071068 0 0",
    )
    if "grandma_sitting" not in prepared:
        ET.SubElement(
            worldbody,
            "geom",
            name="patient_101_torso_proxy",
            type="capsule",
            pos="-1.75 -1.05 1.28",
            size="0.18 0.34",
            material="hospital_gown",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            worldbody,
            "geom",
            name="patient_101_head_proxy",
            type="sphere",
            pos="-1.75 -1.05 1.78",
            size="0.14",
            material="hospital_skin",
            contype="0",
            conaffinity="0",
        )
    if "boy_fallen" not in prepared:
        ET.SubElement(
            worldbody,
            "geom",
            name="patient_202_torso_proxy",
            type="capsule",
            pos="1.58 2.15 0.20",
            quat="0.7071068 0.7071068 0 0",
            size="0.17 0.42",
            material="hospital_gown",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            worldbody,
            "geom",
            name="patient_202_head_proxy",
            type="sphere",
            pos="1.58 2.64 0.16",
            size="0.13",
            material="hospital_skin",
            contype="0",
            conaffinity="0",
        )
    # Invisible body proxy gives the fallen patient physical occupancy without
    # making the high-detail visual mesh authoritative for contact.
    ET.SubElement(
        worldbody,
        "geom",
        name="patient_202_collision",
        type="capsule",
        pos="1.58 2.15 0.205",
        quat="0.7071068 0.7071068 0 0",
        size="0.20 0.46",
        rgba="0 0 0 0",
        friction="0.95 0.02 0.002",
        contype="4",
        conaffinity="3",
    )


def _add_physical_bed(
    worldbody: ET.Element,
    room_id: str,
    bed_pos: tuple[float, float, float],
) -> None:
    x, y, _ = bed_pos
    parts = (
        ("bed_collision", (x, y, 0.55), (0.53, 1.05, 0.09), "hospital_bed"),
        ("mattress", (x, y, 0.76), (0.49, 0.96, 0.13), "hospital_linen"),
        ("headboard", (x, y + 1.02, 0.87), (0.53, 0.045, 0.43), "hospital_rail"),
        ("footboard", (x, y - 1.02, 0.78), (0.53, 0.045, 0.34), "hospital_rail"),
        ("left_rail", (x - 0.52, y, 0.84), (0.035, 0.68, 0.20), "hospital_rail"),
        ("right_rail", (x + 0.52, y, 0.84), (0.035, 0.68, 0.20), "hospital_rail"),
    )
    for suffix, pos, size, material in parts:
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{room_id}_{suffix}",
            type="box",
            pos=_numbers(pos),
            size=_numbers(size),
            material=material,
            friction="0.92 0.02 0.002",
            contype="4",
            conaffinity="3",
        )
    for index, (dx, dy) in enumerate(
        ((-0.43, -0.86), (-0.43, 0.86), (0.43, -0.86), (0.43, 0.86))
    ):
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{room_id}_bed_leg_{index}",
            type="cylinder",
            pos=_numbers((x + dx, y + dy, 0.27)),
            size="0.055 0.27",
            material="hospital_rail",
            friction="0.95 0.02 0.002",
            contype="4",
            conaffinity="3",
        )
