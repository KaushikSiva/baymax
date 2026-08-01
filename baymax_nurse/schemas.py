from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MatchPhase(str, Enum):
    LOBBY = "lobby"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class MatchEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = "10.0"
    event_type: str = Field(min_length=1, max_length=80)
    match_id: str
    timestamp_s: float
    simulation_time_s: float
    player_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class HospitalSkill(str, Enum):
    WAIT = "wait"
    NAVIGATE_ROOM_1 = "navigate_room_1"
    INSPECT_ROOM_1 = "inspect_room_1"
    NAVIGATE_ROOM_2 = "navigate_room_2"
    INSPECT_ROOM_2 = "inspect_room_2"
    DISPATCH_INCIDENT = "dispatch_incident"
    RETURN_HOME = "return_home"
    RECOVER = "recover"


class IncidentType(str, Enum):
    CRITICAL_MONITOR = "critical_monitor"
    PATIENT_DISTRESS = "patient_distress"
    PATIENT_FALL = "patient_fall"


@dataclass(frozen=True)
class HospitalDecision:
    skill: HospitalSkill
    rationale: str
    expected_outcome: str
    inference_id: str
    model_name: str
    latency_s: float
    incident_type: IncidentType | None = None


def incident_payload(
    *,
    incident_id: str,
    scenario_id: str,
    room_id: str,
    patient_id: str,
    patient_name: str,
    incident_type: IncidentType,
    summary: str,
    evidence: list[str],
    source: str,
    robot_pose: list[float],
    timestamp: float,
) -> dict[str, Any]:
    return {
        "incidentId": incident_id,
        "scenarioId": scenario_id,
        "timestamp": timestamp,
        "roomId": room_id,
        "patientId": patient_id,
        "patientName": patient_name,
        "incidentType": incident_type.value,
        "severity": "critical",
        "summary": summary,
        "evidence": evidence,
        "source": source,
        "robotPose": robot_pose,
        "simulationOnly": True,
    }
