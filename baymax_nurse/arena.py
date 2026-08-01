from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from baymax_nurse.runtime.command_channel import ConstrainedLocomotion
from baymax_nurse.runtime.locomotion import UnitreeLocomotion
from baymax_nurse.runtime.robot_model import (
    BODY_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
    default_g1_path,
)
from baymax_nurse.model import (
    BED_POSITIONS,
    HOME_POSITION,
    ROBOT_SPAWN,
    ROOM_INSPECTION_POINTS,
    build_hospital_xml,
)
from baymax_nurse.schemas import (
    HospitalDecision,
    HospitalSkill,
    IncidentType,
    MatchEvent,
    MatchPhase,
    incident_payload,
)


INSPECTION_RADIUS_M = 0.38
HOME_RADIUS_M = 0.42
INSPECTION_DWELL_S = 0.8
ROOM_1_SILENCE_TIMEOUT_S = 5.0
SCRIPTED_SPEECH_DURATION_S = 4.0
DOORWAY_WAYPOINT_RADIUS_M = 0.30
SCRIPTED_PATIENT_PHRASE = "I have severe chest pain and I'm having trouble breathing."
PATIENT_NAMES = {
    "patient_101": "Eleanor Brooks",
    "patient_202": "Daniel Carter",
}
ROOM_LOOK_TARGETS = {
    "room_1": (-1.85, -0.90),
    "room_2": (1.65, 2.15),
}
ROOM_2_ROUTE = ((0.0, -0.10), (0.0, 0.95), ROOM_INSPECTION_POINTS["room_2"])
HOME_ROUTE = ((0.0, 0.95), (0.0, -0.10), HOME_POSITION)


class HospitalPatrolArena:
    """One G1 patrols two patient rooms using guarded high-level skills."""

    def __init__(
        self,
        *,
        viewer: bool = False,
        realtime: bool = False,
        domain_seed: int = 10,
        asset_manifest: Path | str | None = None,
        speech_mode: str = "scripted",
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Hospital demo requires MuJoCo") from exc
        self.mujoco = mujoco
        self.xml = build_hospital_xml(asset_manifest=asset_manifest)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.scenario_id = f"hospital-{uuid.uuid4().hex[:10]}"
        self.phase = MatchPhase.LOBBY
        self.started_wall_s: float | None = None
        self.realtime = realtime
        self.rng = np.random.default_rng(domain_seed)
        self.viewer = None
        self._renderers: dict[tuple[int, int], Any] = {}
        self._control_counter = 0
        self.current_skill = HospitalSkill.WAIT
        self.rationale = "Waiting for the first grounded patrol decision."
        self.expected_outcome = "Enter Room 1 and begin inspection."
        self.model_name = "Unassigned"
        self.api_calls_remaining: int | None = None
        self.rooms = {
            "room_1": {"visited": False, "dwellSeconds": 0.0},
            "room_2": {"visited": False, "dwellSeconds": 0.0},
        }
        self.speech_mode = speech_mode
        self.patient_transcript = (
            SCRIPTED_PATIENT_PHRASE if speech_mode == "scripted" else ""
        )
        self.transcript_source = "scripted" if self.patient_transcript else None
        self.patient_speaking = False
        self.patient_speech_final = bool(self.patient_transcript)
        self.room_1_listening_started_s: float | None = None
        self.room_1_listening_completed_s: float | None = None
        self.room_1_listening_complete = False
        self.room_1_listening_outcome: str | None = None
        self._route_skill: HospitalSkill | None = None
        self._route_waypoint_index = 0
        self.wall_contact_samples = 0
        self.incidents: list[dict[str, Any]] = []
        self.dispatches: list[dict[str, Any]] = []
        self.events: list[MatchEvent] = []
        self.decisions: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.fallen = False
        self._initialize_pose()
        self._recovery_pose = self._base_qpos().copy()
        self._initialize_actuators()
        self.mujoco.mj_forward(self.model, self.data)
        base = UnitreeLocomotion(
            self.model,
            self.data,
            self.mujoco,
            player_ids=("p1",),
            command_limits=(0.58, 0.28, 0.92),
        )
        self.locomotion = ConstrainedLocomotion(
            base,
            self.model,
            self.data,
            self.rng,
            motor_strength={"p1": 0.98},
            dropout_probability=0.01,
            joint_position_noise_rad=0.002,
            joint_velocity_noise_rps=0.02,
            velocity_limits=(0.52, 0.24, 0.82),
            slew_per_packet=(0.08, 0.05, 0.12),
            player_ids=("p1",),
        )
        if viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.reset_viewer_camera()
            self.viewer.sync()

    @property
    def simulation_time_s(self) -> float:
        return float(self.data.time)

    def start(self) -> None:
        if self.phase != MatchPhase.LOBBY:
            raise RuntimeError("hospital patrol can only start from the lobby")
        self.phase = MatchPhase.RUNNING
        self.started_wall_s = time.monotonic()
        self._event(
            "hospital_patrol_started",
            {"rooms": ["room_1", "room_2"], "simulationOnly": True},
        )

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        for renderer in self._renderers.values():
            if hasattr(renderer, "close"):
                renderer.close()
        self._renderers.clear()

    def reset_viewer_camera(self) -> None:
        if self.viewer is None:
            return
        self.viewer.cam.distance = 10.4
        self.viewer.cam.azimuth = 138
        self.viewer.cam.elevation = -29

    def set_decision(
        self, decision: HospitalDecision, *, api_calls_remaining: int | None = None
    ) -> None:
        previous_skill = self.current_skill
        skill = decision.skill
        if skill == HospitalSkill.RECOVER and not self.fallen:
            skill = HospitalSkill.WAIT
        if skill == HospitalSkill.DISPATCH_INCIDENT and not self.pending_incidents():
            skill = HospitalSkill.WAIT
        self.current_skill = skill
        if skill != previous_skill:
            self._route_skill = None
            self._route_waypoint_index = 0
        self.rationale = decision.rationale
        self.expected_outcome = decision.expected_outcome
        self.model_name = decision.model_name
        self.api_calls_remaining = api_calls_remaining
        record = {
            "simulationTime": round(self.simulation_time_s, 3),
            "skill": skill.value,
            "rationale": decision.rationale,
            "expectedOutcome": decision.expected_outcome,
            "incidentType": (
                decision.incident_type.value if decision.incident_type else None
            ),
            "inferenceId": decision.inference_id,
            "model": decision.model_name,
            "latencySeconds": round(decision.latency_s, 3),
        }
        self.decisions.append(record)
        self._event("hospital_policy_decision", record)

    def patient_speech_started(self, source: str = "browser") -> None:
        self.patient_speaking = True
        self.patient_speech_final = False
        self.transcript_source = source
        self._event("patient_speech_started", {"source": source})

    def patient_speech_finished(self, source: str = "browser") -> None:
        self.patient_speaking = False
        self.patient_speech_final = bool(self.patient_transcript)
        self.transcript_source = source
        self._event(
            "patient_speech_finished",
            {"source": source, "transcript": self.patient_transcript},
        )

    def add_patient_transcript(
        self,
        transcript: str,
        source: str = "browser",
        *,
        final: bool = True,
    ) -> None:
        value = " ".join(str(transcript).split())[:500]
        if not value:
            return
        self.patient_transcript = value
        self.transcript_source = source
        self.patient_speech_final = bool(final)
        if final:
            self.patient_speaking = False
        self._event(
            "patient_transcript",
            {"source": source, "transcript": value, "final": bool(final)},
        )

    def pending_incidents(self) -> list[dict[str, Any]]:
        dispatched = {entry["incidentId"] for entry in self.dispatches if entry["ok"]}
        return [incident for incident in self.incidents if incident["incidentId"] not in dispatched]

    def next_incident(self) -> dict[str, Any] | None:
        pending = self.pending_incidents()
        return pending[0] if pending else None

    def record_dispatch(
        self, incident: dict[str, Any], result: dict[str, Any]
    ) -> None:
        record = {
            "incidentId": incident["incidentId"],
            "incidentType": incident["incidentType"],
            "roomId": incident["roomId"],
            "request": incident,
            **result,
        }
        self.dispatches.append(record)
        self._event("incident_dispatch", record)
        self.current_skill = HospitalSkill.WAIT
        self.rationale = (
            f"Dispatch accepted for {incident['incidentType']}."
            if result.get("ok")
            else f"Dispatch failed for {incident['incidentType']}; retry queued."
        )

    def needs_decision(self) -> bool:
        if self.fallen or self.current_skill == HospitalSkill.WAIT:
            return True
        if self.current_skill == HospitalSkill.DISPATCH_INCIDENT:
            return True
        if self.current_skill == HospitalSkill.NAVIGATE_ROOM_1:
            return self._at(ROOM_INSPECTION_POINTS["room_1"], INSPECTION_RADIUS_M)
        if self.current_skill == HospitalSkill.NAVIGATE_ROOM_2:
            return self._at(ROOM_INSPECTION_POINTS["room_2"], INSPECTION_RADIUS_M)
        if self.current_skill == HospitalSkill.INSPECT_ROOM_1:
            return bool(self.rooms["room_1"]["visited"])
        if self.current_skill == HospitalSkill.INSPECT_ROOM_2:
            return bool(self.rooms["room_2"]["visited"])
        if self.current_skill == HospitalSkill.RETURN_HOME:
            return self._at(HOME_POSITION, HOME_RADIUS_M)
        return False

    def step(self, count: int = 1) -> None:
        if self.phase != MatchPhase.RUNNING:
            return
        for _ in range(count):
            started = time.perf_counter()
            self._apply_skill()
            self.locomotion.apply_torques()
            self.mujoco.mj_step(self.model, self.data)
            self._control_counter += 1
            self._update_inspection()
            if self._control_counter % self.locomotion.decimation == 0:
                self.locomotion.update(self.simulation_time_s)
                self._sample_state()
                self._record_trajectory()
                if self.viewer is not None:
                    if not self.viewer.is_running():
                        self.phase = MatchPhase.ABORTED
                        return
                    self.viewer.sync()
            if self.realtime:
                remaining = float(self.model.opt.timestep) - (
                    time.perf_counter() - started
                )
                if remaining > 0:
                    time.sleep(remaining)
            if self.phase != MatchPhase.RUNNING:
                return

    def render(self, camera: str, width: int = 640, height: int = 360) -> np.ndarray:
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = self.mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = renderer
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def policy_status(self) -> dict[str, Any]:
        return {
            "simulationOnly": True,
            "fallen": self.fallen,
            "atHome": self._at(HOME_POSITION, HOME_RADIUS_M),
            "currentSkill": self.current_skill.value,
            "rooms": {
                room_id: {
                    "visited": bool(state["visited"]),
                    "atInspectionPoint": self._at(point, INSPECTION_RADIUS_M),
                }
                for (room_id, state), point in zip(
                    self.rooms.items(), ROOM_INSPECTION_POINTS.values(), strict=True
                )
            },
            "patientTranscript": self.patient_transcript,
            "room1Listening": {
                "active": self._room_1_listening_active(),
                "complete": self.room_1_listening_complete,
                "speaking": self.patient_speaking,
                "elapsedSeconds": round(self._room_1_listening_elapsed_s(), 2),
                "silenceTimeoutSeconds": ROOM_1_SILENCE_TIMEOUT_S,
            },
            "monitorReadings": {
                "room_1": {"heartRateBpm": 148, "spo2Percent": 82, "alarm": "critical"},
                "room_2": {"heartRateBpm": 76, "spo2Percent": 97, "alarm": "normal"},
            },
            "pendingIncidents": self.pending_incidents(),
            "successfulDispatches": len([item for item in self.dispatches if item["ok"]]),
            "expectedDispatches": 2,
            "bedClearanceM": round(self._bed_clearance_m(), 3),
            "allowedSkills": [skill.value for skill in HospitalSkill],
        }

    def state_payload(self) -> dict[str, Any]:
        elapsed = (
            max(0.0, time.monotonic() - self.started_wall_s)
            if self.started_wall_s is not None
            else 0.0
        )
        position = self._base_position()
        return {
            "protocolVersion": "10.0",
            "scenarioId": self.scenario_id,
            "scenario": "two_room_hospital_patrol",
            "phase": self.phase.value,
            "simulationTime": round(self.simulation_time_s, 3),
            "elapsedTime": round(elapsed, 3),
            "result": "PATROL COMPLETE" if self.phase == MatchPhase.FINISHED else None,
            "expectedDispatches": 2,
            "robot": {
                "displayName": "Clinical G1",
                "currentSkill": self.current_skill.value,
                "rationale": self.rationale,
                "expectedOutcome": self.expected_outcome,
                "model": self.model_name,
                "apiCallsRemaining": self.api_calls_remaining,
                "fallen": self.fallen,
                "pose": [round(float(value), 3) for value in position],
            },
            "rooms": {
                room_id: {
                    "visited": bool(state["visited"]),
                    "atInspectionPoint": self._at(
                        ROOM_INSPECTION_POINTS[room_id], INSPECTION_RADIUS_M
                    ),
                    "patientId": "patient_101" if room_id == "room_1" else "patient_202",
                    "patientName": (
                        PATIENT_NAMES["patient_101"]
                        if room_id == "room_1"
                        else PATIENT_NAMES["patient_202"]
                    ),
                }
                for room_id, state in self.rooms.items()
            },
            "monitors": {
                "room_1": {"heartRateBpm": 148, "spo2Percent": 82, "status": "CRITICAL"},
                "room_2": {"heartRateBpm": 76, "spo2Percent": 97, "status": "NORMAL"},
            },
            "speech": {
                "transcript": self.patient_transcript,
                "source": self.transcript_source,
                "listening": self._room_1_listening_active(),
                "speaking": self.patient_speaking,
                "complete": self.room_1_listening_complete,
                "outcome": self.room_1_listening_outcome,
                "elapsedSeconds": round(self._room_1_listening_elapsed_s(), 2),
                "silenceTimeoutSeconds": ROOM_1_SILENCE_TIMEOUT_S,
            },
            "incidents": self.incidents,
            "pendingIncidents": self.pending_incidents(),
            "dispatches": self.dispatches,
            "safety": {
                "simulationOnly": True,
                "guardrail": "Gemini selects bounded skills; deterministic control enforces routes and dispatch deduplication.",
                "bedClearanceM": round(self._bed_clearance_m(), 3),
                "bedCollisionGeometry": "frame, mattress, rails, boards, and legs",
                "wallContactSamples": self.wall_contact_samples,
            },
            "sdkChannel": self.locomotion.report()["p1"],
        }

    def write_evidence(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        files = {
            "scene.xml": self.xml,
            "result.json": json.dumps(self.state_payload(), indent=2),
            "events.json": json.dumps(
                [event.model_dump(mode="json") for event in self.events], indent=2
            ),
            "trajectory.json": json.dumps(self.trajectory, indent=2),
            "gemini_decisions.json": json.dumps(self.decisions, indent=2),
            "incidents.json": json.dumps(self.incidents, indent=2),
            "dispatches.json": json.dumps(self.dispatches, indent=2),
            "sdk_command_trace.json": json.dumps(
                self.locomotion.channels["p1"].trace, indent=2
            ),
        }
        for filename, content in files.items():
            (directory / filename).write_text(content, encoding="utf-8")

    def _apply_skill(self) -> None:
        if self.current_skill == HospitalSkill.NAVIGATE_ROOM_1:
            self._navigate_route((ROOM_INSPECTION_POINTS["room_1"],))
        elif self.current_skill == HospitalSkill.INSPECT_ROOM_1:
            self._hold_inspection_position("room_1")
        elif self.current_skill == HospitalSkill.NAVIGATE_ROOM_2:
            self._navigate_route(ROOM_2_ROUTE)
        elif self.current_skill == HospitalSkill.INSPECT_ROOM_2:
            self._hold_inspection_position("room_2")
        elif self.current_skill == HospitalSkill.RETURN_HOME:
            self._navigate_route(HOME_ROUTE)
        elif self.current_skill == HospitalSkill.RECOVER:
            self._perform_recovery()
        else:
            self.locomotion.set_command("p1", 0.0, 0.0, 0.0)

    def _hold_inspection_position(self, room_id: str) -> None:
        if not self._at(ROOM_INSPECTION_POINTS[room_id], INSPECTION_RADIUS_M):
            self._navigate_to(np.asarray(ROOM_INSPECTION_POINTS[room_id]))
            return
        self._orient_to(np.asarray(ROOM_LOOK_TARGETS[room_id]))

    def _navigate_route(
        self, waypoints: tuple[tuple[float, float], ...]
    ) -> None:
        if self._route_skill != self.current_skill:
            self._route_skill = self.current_skill
            self._route_waypoint_index = 0
        while self._route_waypoint_index < len(waypoints) - 1 and self._at(
            waypoints[self._route_waypoint_index], DOORWAY_WAYPOINT_RADIUS_M
        ):
            reached = waypoints[self._route_waypoint_index]
            self._event(
                "navigation_waypoint_reached",
                {
                    "skill": self.current_skill.value,
                    "waypoint": [round(float(value), 3) for value in reached],
                },
            )
            self._route_waypoint_index += 1
        self._navigate_to(np.asarray(waypoints[self._route_waypoint_index]))

    def _navigate_to(self, target_xy: np.ndarray) -> None:
        if self._bed_clearance_m() < 0.04:
            nearest = min(
                BED_POSITIONS.values(),
                key=lambda bed: np.linalg.norm(
                    self._base_position()[:2] - np.asarray(bed[:2])
                ),
            )
            target_xy = np.asarray((nearest[0] + 1.05, self._base_position()[1]))
            self.rationale = "Bed exclusion guard is steering the G1 back into the clear lane."
        delta = target_xy - self._base_position()[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 0.20:
            self.locomotion.set_command("p1", 0.0, 0.0, 0.0)
            return
        error = _wrap_angle(
            math.atan2(float(delta[1]), float(delta[0])) - self._base_heading()
        )
        if abs(error) > 0.35:
            self.locomotion.set_command(
                "p1", 0.0, 0.0, float(np.clip(1.65 * error, -0.80, 0.80))
            )
            return
        forward = min(0.44, max(0.0, distance - 0.10)) * max(0.0, math.cos(error))
        lateral = float(np.clip(math.sin(error) * min(distance, 0.6), -0.18, 0.18))
        self.locomotion.set_command(
            "p1", forward, lateral, float(np.clip(1.5 * error, -0.72, 0.72))
        )

    def _orient_to(self, target_xy: np.ndarray) -> None:
        error = self._yaw_error(target_xy)
        self.locomotion.set_command(
            "p1", 0.0, 0.0, float(np.clip(1.5 * error, -0.62, 0.62))
        )

    def _update_inspection(self) -> None:
        for room_id, skill in (
            ("room_1", HospitalSkill.INSPECT_ROOM_1),
            ("room_2", HospitalSkill.INSPECT_ROOM_2),
        ):
            if self.rooms[room_id]["visited"] or self.current_skill != skill:
                continue
            if not self._at(ROOM_INSPECTION_POINTS[room_id], INSPECTION_RADIUS_M):
                self.rationale = f"Inspection blocked: navigate to {room_id}."
                continue
            if abs(self._yaw_error(np.asarray(ROOM_LOOK_TARGETS[room_id]))) > 0.35:
                self.rationale = f"Turning the ego camera toward the patient in {room_id}."
                continue
            if room_id == "room_1" and not self._update_room_1_listening():
                continue
            self.rooms[room_id]["dwellSeconds"] += float(self.model.opt.timestep)
            if self.rooms[room_id]["dwellSeconds"] >= INSPECTION_DWELL_S:
                self.rooms[room_id]["visited"] = True
                self._event("room_inspected", {"roomId": room_id})
                if room_id == "room_1":
                    self._discover_room_1()
                else:
                    self._discover_room_2()

    def _update_room_1_listening(self) -> bool:
        if self.room_1_listening_complete:
            return True
        if self.room_1_listening_started_s is None:
            self.room_1_listening_started_s = self.simulation_time_s
            self._event(
                "room_1_listening_started",
                {"silenceTimeoutSeconds": ROOM_1_SILENCE_TIMEOUT_S},
            )
        elapsed = self._room_1_listening_elapsed_s()
        if self.speech_mode == "scripted" and self.patient_transcript:
            if elapsed < SCRIPTED_SPEECH_DURATION_S:
                self.rationale = (
                    "Eleanor Brooks is speaking; G1 is stopped and listening until she finishes."
                )
                return False
        elif self.patient_speaking:
            self.rationale = (
                "Eleanor Brooks is speaking; G1 is stopped and listening until she finishes."
            )
            return False
        elif not (self.patient_transcript and self.patient_speech_final):
            if elapsed < ROOM_1_SILENCE_TIMEOUT_S:
                remaining = max(0.0, ROOM_1_SILENCE_TIMEOUT_S - elapsed)
                self.rationale = (
                    f"Listening for Eleanor Brooks; silence timeout in {remaining:.1f} seconds."
                )
                return False
            self.room_1_listening_outcome = "silence_timeout"
        if self.room_1_listening_outcome is None:
            self.room_1_listening_outcome = "speech_complete"
        self.room_1_listening_complete = True
        self.room_1_listening_completed_s = self.simulation_time_s
        self._event(
            "room_1_listening_completed",
            {
                "outcome": self.room_1_listening_outcome,
                "elapsedSeconds": round(elapsed, 3),
                "transcript": self.patient_transcript,
            },
        )
        self.rationale = (
            "The listening window for Eleanor Brooks is complete; assessing Room 101."
        )
        return True

    def _discover_room_1(self) -> None:
        monitor = {
            "heartRateBpm": 148,
            "spo2Percent": 82,
            "alarm": "critical",
        }
        speech = {
            "heard": bool(self.patient_transcript),
            "transcript": self.patient_transcript or None,
            "source": self.transcript_source,
            "listeningOutcome": self.room_1_listening_outcome,
            "listeningSeconds": round(self._room_1_listening_elapsed_s(), 3),
        }
        evidence = ["SpO2 82%", "Heart rate 148 BPM", "Red critical alarm visible"]
        if self.patient_transcript:
            evidence.append(f"Patient said: {self.patient_transcript}")
            summary = (
                "Room 101 critical monitor: heart rate 148 BPM and SpO2 82%. "
                f"Eleanor Brooks reported: {self.patient_transcript}"
            )
        else:
            evidence.append("No patient speech during the five-second listening window")
            summary = (
                "Room 101 critical monitor: heart rate 148 BPM and SpO2 82%. "
                "No patient statement was heard during the five-second listening window."
            )
        self._add_incident(
            room_id="room_1",
            patient_id="patient_101",
            incident_type=IncidentType.CRITICAL_MONITOR,
            summary=summary,
            evidence=evidence,
            source=(
                "monitor_vision+patient_speech"
                if self.patient_transcript
                else "monitor_vision"
            ),
            context={"monitorReadings": monitor, "patientSpeech": speech},
        )

    def _discover_room_2(self) -> None:
        self._add_incident(
            room_id="room_2",
            patient_id="patient_202",
            incident_type=IncidentType.PATIENT_FALL,
            summary="Daniel Carter is visibly lying on the floor beside the hospital bed.",
            evidence=["Human form detected on floor", "Patient is outside bed footprint"],
            source="camera_vision",
        )

    def _add_incident(
        self,
        *,
        room_id: str,
        patient_id: str,
        incident_type: IncidentType,
        summary: str,
        evidence: list[str],
        source: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        incident_id = f"{self.scenario_id}-{incident_type.value}"
        if any(item["incidentId"] == incident_id for item in self.incidents):
            return
        payload = incident_payload(
            incident_id=incident_id,
            scenario_id=self.scenario_id,
            room_id=room_id,
            patient_id=patient_id,
            patient_name=PATIENT_NAMES[patient_id],
            incident_type=incident_type,
            summary=summary,
            evidence=evidence,
            source=source,
            robot_pose=[round(float(value), 4) for value in self._base_position()],
            timestamp=time.time(),
        )
        if context:
            payload.update(context)
        self.incidents.append(payload)
        self._event("incident_detected", payload)

    def _perform_recovery(self) -> None:
        joint = self.model.joint("p1_floating_base_joint")
        qpos = joint.qposadr[0]
        dof = joint.dofadr[0]
        self.data.qpos[qpos : qpos + 7] = self._recovery_pose
        self.data.qvel[dof : dof + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.fallen = False
        self.current_skill = HospitalSkill.WAIT
        self.rationale = "Simulation recovery reset completed; reassessing patrol."
        self._event("simulation_recovery", {"privilegedReset": True})

    def _sample_state(self) -> None:
        self.fallen = self._is_fallen()
        wall_contact = self._has_robot_wall_contact()
        if wall_contact:
            self.wall_contact_samples += 1
            self.rationale = "Wall contact detected; stopping and re-centering the route."
        if self.fallen:
            self.rationale = "Robot fall detected; recovery required."
        if (
            all(bool(room["visited"]) for room in self.rooms.values())
            and not self.pending_incidents()
            and self._at(HOME_POSITION, HOME_RADIUS_M)
        ):
            self.phase = MatchPhase.FINISHED
            self.current_skill = HospitalSkill.WAIT
            self.rationale = "Both rooms inspected, alerts dispatched, and G1 returned home."
            self._event(
                "hospital_patrol_completed",
                {"dispatchCount": len([item for item in self.dispatches if item["ok"]])},
            )

    def _record_trajectory(self) -> None:
        self.trajectory.append(
            {
                "simulationTime": round(self.simulation_time_s, 4),
                "robot": self._base_position().tolist(),
                "skill": self.current_skill.value,
                "room1Visited": self.rooms["room_1"]["visited"],
                "room2Visited": self.rooms["room_2"]["visited"],
                "dispatchCount": len([item for item in self.dispatches if item["ok"]]),
                "bedClearanceM": round(self._bed_clearance_m(), 4),
                "wallContact": self._has_robot_wall_contact(),
                "routeWaypointIndex": self._route_waypoint_index,
            }
        )

    def _initialize_pose(self) -> None:
        source_model = self.mujoco.MjModel.from_xml_path(str(default_g1_path()))
        source_data = self.mujoco.MjData(source_model)
        self.mujoco.mj_resetDataKeyframe(source_model, source_data, 0)
        for joint_name in (*BODY_JOINT_NAMES, *RIGHT_HAND_JOINT_NAMES):
            source_joint = source_model.joint(joint_name)
            target_joint = self.model.joint(f"p1_{joint_name}")
            self.data.qpos[target_joint.qposadr[0]] = source_data.qpos[
                source_joint.qposadr[0]
            ]

    def _initialize_actuators(self) -> None:
        for name in BODY_JOINT_NAMES[12:]:
            joint = self.model.joint(f"p1_{name}")
            actuator = self.model.actuator(f"p1_{name}")
            self.data.ctrl[actuator.id] = self.data.qpos[joint.qposadr[0]]
        for name in RIGHT_HAND_JOINT_NAMES:
            actuator = self.model.actuator(f"p1_{name}")
            self.data.ctrl[actuator.id] = 0.0

    def _base_qpos(self) -> np.ndarray:
        joint = self.model.joint("p1_floating_base_joint")
        return self.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 7]

    def _base_position(self) -> np.ndarray:
        return self._base_qpos()[:3].copy()

    def _base_heading(self) -> float:
        q = self._base_qpos()[3:7]
        w, x, y, z = [float(value) for value in q]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return _wrap_angle(yaw)

    def _at(self, point: tuple[float, float], radius: float) -> bool:
        return bool(np.linalg.norm(self._base_position()[:2] - np.asarray(point)) <= radius)

    def _yaw_error(self, target_xy: np.ndarray) -> float:
        delta = target_xy - self._base_position()[:2]
        return _wrap_angle(
            math.atan2(float(delta[1]), float(delta[0])) - self._base_heading()
        )

    def _room_1_listening_elapsed_s(self) -> float:
        if self.room_1_listening_started_s is None:
            return 0.0
        end = self.room_1_listening_completed_s or self.simulation_time_s
        return max(0.0, end - self.room_1_listening_started_s)

    def _room_1_listening_active(self) -> bool:
        return bool(
            self.room_1_listening_started_s is not None
            and not self.room_1_listening_complete
        )

    def _has_robot_wall_contact(self) -> bool:
        for contact in self.data.contact:
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            for robot_geom, other_geom in ((geom_1, geom_2), (geom_2, geom_1)):
                body_name = self.model.body(
                    int(self.model.geom_bodyid[robot_geom])
                ).name
                other_name = self.model.geom(other_geom).name
                if body_name.startswith("p1_") and other_name.startswith(
                    "hospital_wall_"
                ):
                    return True
        return False

    def _bed_clearance_m(self) -> float:
        point = self._base_position()[:2]
        clearances: list[float] = []
        for bed in BED_POSITIONS.values():
            delta = np.abs(point - np.asarray(bed[:2])) - np.asarray((0.82, 1.34))
            outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
            inside = min(max(float(delta[0]), float(delta[1])), 0.0)
            clearances.append(outside + inside)
        return min(clearances)

    def _is_fallen(self) -> bool:
        qpos = self._base_qpos()
        up_z = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
        return bool(qpos[2] < 0.48 or up_z < math.cos(math.radians(60)))

    def _event(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append(
            MatchEvent(
                event_type=event_type,
                match_id=self.scenario_id,
                timestamp_s=time.time(),
                simulation_time_s=self.simulation_time_s,
                player_id="p1",
                payload=payload,
            )
        )


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi
