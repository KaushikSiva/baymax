from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from baymax_nurse.runtime.robot_model import (
    LEG_JOINT_NAMES,
    TORQUE_LIMITS,
    project_root,
)


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.asarray(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


@dataclass
class PolicyState:
    action: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    observation: np.ndarray = field(default_factory=lambda: np.zeros(47, dtype=np.float32))


class UnitreeLocomotion:
    """Runs the pinned official Unitree G1 locomotion policy."""

    def __init__(
        self,
        model: Any,
        data: Any,
        mujoco_module: Any,
        *,
        command_limits: tuple[float, float, float] = (0.65, 0.35, 1.10),
        player_ids: tuple[str, ...] = ("p1",),
    ) -> None:
        self.model = model
        self.data = data
        self.mujoco = mujoco_module
        self.command_limits = np.asarray(command_limits, dtype=np.float32)
        self.player_ids = tuple(player_ids)
        official = project_root() / "vendor" / "unitree_rl_gym"
        config_path = official / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
        policy_path = official / "deploy" / "pre_train" / "g1" / "motion.pt"
        if not config_path.is_file() or not policy_path.is_file():
            raise FileNotFoundError(
                "Pinned Unitree policy is missing. Run scripts/setup_unitree_policy.sh."
            )
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The Unitree locomotion policy requires PyTorch") from exc
        self.torch = torch
        policy = torch.jit.load(str(policy_path))
        policy.eval()
        if hasattr(policy, "reset_memory"):
            policy.reset_memory()
        self.policies = {"p1": policy}
        default = np.asarray(self.config["default_angles"], dtype=np.float32)
        self.states = {"p1": PolicyState(target=default.copy())}
        self.commands = {"p1": np.zeros(3, dtype=np.float32)}
        joints = [model.joint(f"p1_{name}") for name in LEG_JOINT_NAMES]
        self._leg_qpos = {"p1": np.asarray([joint.qposadr[0] for joint in joints])}
        self._leg_dof = {"p1": np.asarray([joint.dofadr[0] for joint in joints])}
        floating = model.joint("p1_floating_base_joint")
        self._base_qpos = {"p1": int(floating.qposadr[0])}
        self._base_dof = {"p1": int(floating.dofadr[0])}
        self._leg_actuator = {
            "p1": np.asarray([model.actuator(f"p1_{name}").id for name in LEG_JOINT_NAMES])
        }

    @property
    def decimation(self) -> int:
        return int(self.config["control_decimation"])

    def set_command(self, player_id: str, vx: float, vy: float, yaw_rate: float) -> None:
        command = np.asarray([vx, vy, yaw_rate], dtype=np.float32)
        if not np.all(np.isfinite(command)):
            raise ValueError("locomotion command must be finite")
        self.commands[player_id] = np.clip(
            command, -self.command_limits, self.command_limits
        )

    def apply_torques(self) -> None:
        kps = np.asarray(self.config["kps"], dtype=np.float32)
        kds = np.asarray(self.config["kds"], dtype=np.float32)
        limits = np.asarray(TORQUE_LIMITS, dtype=np.float32)
        state = self.states["p1"]
        q = self.data.qpos[self._leg_qpos["p1"]]
        dq = self.data.qvel[self._leg_dof["p1"]]
        self.data.ctrl[self._leg_actuator["p1"]] = np.clip(
            (state.target - q) * kps - dq * kds, -limits, limits
        )

    def update(self, simulation_time_s: float) -> None:
        state = self.states["p1"]
        q = self.data.qpos[self._leg_qpos["p1"]]
        dq = self.data.qvel[self._leg_dof["p1"]]
        base_qpos = self._base_qpos["p1"]
        base_dof = self._base_dof["p1"]
        quat = self.data.qpos[base_qpos + 3 : base_qpos + 7]
        phase = simulation_time_s % 0.8 / 0.8
        obs = state.observation
        obs[:3] = self.data.qvel[base_dof + 3 : base_dof + 6] * float(
            self.config["ang_vel_scale"]
        )
        obs[3:6] = gravity_orientation(quat)
        obs[6:9] = self.commands["p1"] * np.asarray(
            self.config["cmd_scale"], dtype=np.float32
        )
        default = np.asarray(self.config["default_angles"], dtype=np.float32)
        obs[9:21] = (q - default) * float(self.config["dof_pos_scale"])
        obs[21:33] = dq * float(self.config["dof_vel_scale"])
        obs[33:45] = state.action
        obs[45:47] = [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]
        with self.torch.no_grad():
            output = self.policies["p1"](self.torch.from_numpy(obs).unsqueeze(0))
        state.action = output.detach().cpu().numpy().squeeze().astype(np.float32)
        state.target = state.action * float(self.config["action_scale"]) + default
