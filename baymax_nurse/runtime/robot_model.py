from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEG_JOINT_NAMES = BODY_JOINT_NAMES[:12]
RIGHT_HAND_JOINT_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
TORQUE_LIMITS = (88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50)

_REFERENCE_ATTRIBUTES = {
    "body",
    "body1",
    "body2",
    "joint",
    "joint1",
    "joint2",
    "site",
    "site1",
    "site2",
    "target",
    "objname",
    "tendon",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_g1_path() -> Path:
    return project_root() / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"


def add_visual(root: ET.Element) -> None:
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.72 0.75 0.82",
        ambient="0.20 0.22 0.27",
        specular="0.55 0.58 0.65",
    )
    ET.SubElement(
        visual,
        "global",
        azimuth="128",
        elevation="-23",
        offwidth="1920",
        offheight="1080",
    )
    ET.SubElement(visual, "rgba", haze="0.08 0.10 0.14 1")


def add_robot_sites(robot: ET.Element) -> None:
    wrist = robot.find(".//body[@name='right_wrist_yaw_link']")
    torso = robot.find(".//body[@name='torso_link']")
    if wrist is None or torso is None:
        raise ValueError("G1 model is missing right wrist or torso")
    ET.SubElement(
        wrist,
        "site",
        name="right_grasp_site",
        pos="0.145 -0.0046 0",
        size="0.012",
        rgba="0.2 1 0.3 0.8",
    )
    ET.SubElement(
        torso,
        "camera",
        name="ego_camera",
        pos="0.10 0 0.34",
        quat="0.7071068 0 -0.7071068 0",
        fovy="66",
    )


def prefix_tree(element: ET.Element, prefix: str) -> None:
    for node in element.iter():
        if node.get("name"):
            node.set("name", prefix + str(node.get("name")))
        for attribute in _REFERENCE_ATTRIBUTES:
            value = node.get(attribute)
            if value:
                node.set(attribute, prefix + value)


def numbers(values: Any) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
