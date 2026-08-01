from __future__ import annotations

import json
import hashlib
import os
import threading
import time
from typing import Any, Protocol

from baymax_nurse.schemas import HospitalDecision, HospitalSkill, IncidentType


class _RequestLimiter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_started_s: float | None = None

    def acquire(self, minimum_interval_s: float) -> None:
        with self.lock:
            if self.last_started_s is not None:
                remaining = minimum_interval_s - (
                    time.monotonic() - self.last_started_s
                )
                if remaining > 0:
                    time.sleep(remaining)
            self.last_started_s = time.monotonic()


_LIMITERS_LOCK = threading.Lock()
_LIMITERS: dict[str, _RequestLimiter] = {}


def _limiter_for_key(api_key: str) -> _RequestLimiter:
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    with _LIMITERS_LOCK:
        return _LIMITERS.setdefault(fingerprint, _RequestLimiter())


POLICY_SCHEMA = {
    "type": "object",
    "properties": {
        "skill": {"type": "string", "enum": [skill.value for skill in HospitalSkill]},
        "rationale": {"type": "string"},
        "expected_outcome": {"type": "string"},
        "incident_type": {
            "type": ["string", "null"],
            "enum": [incident.value for incident in IncidentType] + [None],
        },
    },
    "required": ["skill", "rationale", "expected_outcome", "incident_type"],
}

SYSTEM_INSTRUCTION = """You control one Unitree G1 during a two-room simulated
hospital patrol. Choose exactly one next high-level skill from the schema. Visit
Room 1 before Room 2, stop at each inspection point, assess camera evidence and
grounded observations, and remain stopped while the Room 1 listening state is
active. Room 1 monitor readings and the completed patient statement are combined
into one queued incident. Route through the doorway waypoints and inspect Room 2
near the fallen patient. Do not choose dispatch_incident until dispatchReady is
true, meaning both rooms have been inspected. Then dispatch every pending
incident exactly once and return home.
If the robot falls, recover. Do not diagnose or recommend treatment. Return only
the requested JSON."""


class HospitalPolicy(Protocol):
    model_name: str

    def decide(
        self, status: dict[str, Any], camera_jpegs: tuple[bytes, bytes]
    ) -> HospitalDecision: ...


class ScriptedHospitalPolicy:
    model_name = "Deterministic hospital validation policy"

    def __init__(self) -> None:
        self.counter = 0

    def decide(
        self, status: dict[str, Any], camera_jpegs: tuple[bytes, bytes]
    ) -> HospitalDecision:
        del camera_jpegs
        self.counter += 1
        pending = status["pendingIncidents"]
        if status["fallen"]:
            skill = HospitalSkill.RECOVER
        elif not status["rooms"]["room_1"]["visited"]:
            skill = (
                HospitalSkill.INSPECT_ROOM_1
                if status["rooms"]["room_1"]["atInspectionPoint"]
                else HospitalSkill.NAVIGATE_ROOM_1
            )
        elif not status["rooms"]["room_2"]["visited"]:
            skill = (
                HospitalSkill.INSPECT_ROOM_2
                if status["rooms"]["room_2"]["atInspectionPoint"]
                else HospitalSkill.NAVIGATE_ROOM_2
            )
        elif pending:
            skill = HospitalSkill.DISPATCH_INCIDENT
        elif not status["atHome"]:
            skill = HospitalSkill.RETURN_HOME
        else:
            skill = HospitalSkill.WAIT
        incident = IncidentType(pending[0]["incidentType"]) if pending else None
        return HospitalDecision(
            skill=skill,
            rationale=f"Grounded patrol state selected {skill.value}.",
            expected_outcome=f"Advance and verify {skill.value}.",
            incident_type=incident,
            inference_id=f"hospital-scripted-{self.counter}",
            model_name=self.model_name,
            latency_s=0.0,
        )


class GeminiHospitalPolicy:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        min_interval_s: float = 12.0,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini Robotics-ER")
        self.model_name = (
            model_name
            or os.getenv("GEMINI_ROBOTICS_MODEL")
            or "gemini-robotics-er-1.6-preview"
        )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai for Gemini Robotics-ER") from exc
        self.client = genai.Client(api_key=resolved_key)
        self.limiter = _limiter_for_key(resolved_key)
        self.min_interval_s = max(12.0, float(min_interval_s))
        self.lock = threading.Lock()
        self.counter = 0

    def decide(
        self, status: dict[str, Any], camera_jpegs: tuple[bytes, bytes]
    ) -> HospitalDecision:
        from google.genai import types

        self.limiter.acquire(self.min_interval_s)
        with self.lock:
            self.counter += 1
            counter = self.counter
        started = time.monotonic()
        contents: list[Any] = [
            types.Part.from_bytes(data=image, mime_type="image/jpeg")
            for image in camera_jpegs
            if image
        ]
        contents.append(
            "Grounded hospital patrol status:\n"
            + json.dumps(status, indent=2)
            + "\nSelect the safest single next skill and cite visible evidence."
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
                response_mime_type="application/json",
                response_json_schema=POLICY_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )
        value = response.parsed if getattr(response, "parsed", None) else json.loads(response.text)
        incident_value = value.get("incident_type")
        return HospitalDecision(
            skill=HospitalSkill(value["skill"]),
            rationale=str(value["rationale"])[:320],
            expected_outcome=str(value["expected_outcome"])[:320],
            incident_type=IncidentType(incident_value) if incident_value else None,
            inference_id=f"hospital-gemini-{counter}",
            model_name=self.model_name,
            latency_s=time.monotonic() - started,
        )


def build_hospital_policy(adapter: str, *, model_name: str | None = None) -> HospitalPolicy:
    if adapter == "gemini-er":
        return GeminiHospitalPolicy(model_name=model_name)
    if adapter == "scripted":
        return ScriptedHospitalPolicy()
    raise ValueError(f"Unknown hospital adapter: {adapter}")
