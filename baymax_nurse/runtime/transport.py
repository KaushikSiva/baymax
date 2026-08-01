from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DryRunTransport:
    """Records SDK-shaped commands without contacting physical hardware."""

    backend_name: str = "dry-run"
    commands: list[dict[str, object]] = field(default_factory=list)
    initialized: bool = False

    def initialize(self) -> None:
        self.initialized = True
        self.commands.append({"command": "initialize"})

    def set_velocity(self, vx: float, vy: float, yaw_rate: float, duration_s: float) -> None:
        if not self.initialized:
            raise RuntimeError("Transport has not been initialized")
        self.commands.append(
            {
                "command": "set_velocity",
                "vx": vx,
                "vy": vy,
                "yaw_rate": yaw_rate,
                "duration_s": duration_s,
            }
        )

    def stop(self, duration_s: float) -> None:
        if not self.initialized:
            raise RuntimeError("Transport has not been initialized")
        self.commands.append({"command": "stop", "duration_s": duration_s})
