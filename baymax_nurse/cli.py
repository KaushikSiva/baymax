from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from baymax_nurse.arena import HospitalPatrolArena
from baymax_nurse.dispatch import DummyDispatchReceiver, HospitalDispatchClient
from baymax_nurse.live import HospitalLiveServer
from baymax_nurse.policy import HospitalPolicy, build_hospital_policy
from baymax_nurse.schemas import HospitalDecision, HospitalSkill, MatchPhase


BAYMAX_MONITOR_EVENT_URL = "https://baymax-jet.vercel.app/api/robot/monitor-event"


@dataclass
class PolicySlot:
    adapter: HospitalPolicy
    calls_remaining: int = 12
    pending: Future[HospitalDecision] | None = None
    last_decision_s: float = -1000.0


@dataclass
class DispatchSlot:
    pending: Future[dict[str, Any]] | None = None
    incident: dict[str, Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-room Unitree G1 hospital patrol"
    )
    parser.add_argument("--adapter", choices=("gemini-er", "scripted"), default="gemini-er")
    parser.add_argument("--model")
    parser.add_argument("--speech-mode", choices=("scripted", "browser"), default="scripted")
    parser.add_argument(
        "--dispatch-url",
        default=os.getenv("BAYMAX_DISPATCH_URL")
        or os.getenv("HOSPITAL_DISPATCH_URL"),
    )
    parser.add_argument(
        "--baymax-api",
        action="store_true",
        help=(
            "Dispatch to the deployed Baymax monitor-event API. This may create "
            "records and initiate a doctor call."
        ),
    )
    parser.add_argument("--dispatch-port", type=int, default=8091)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--domain-seed", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no-live-ui", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8090)
    parser.add_argument("--websocket-port", type=int, default=8770)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--result-linger-seconds", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "darwin" and not args.headless:
        raise RuntimeError("Visible hospital demo currently requires macOS and mjpython")
    adapter_name = "scripted" if args.validate_only else args.adapter
    speech_mode = "scripted" if args.validate_only else args.speech_mode
    arena = HospitalPatrolArena(
        viewer=not args.headless,
        realtime=not args.headless or args.realtime,
        domain_seed=args.domain_seed,
        asset_manifest=args.asset_manifest,
        speech_mode=speech_mode,
    )
    policy = build_hospital_policy(adapter_name, model_name=args.model)
    arena.model_name = policy.model_name
    policy_slot = PolicySlot(adapter=policy)
    dispatch_slot = DispatchSlot()
    receiver: DummyDispatchReceiver | None = None
    if args.validate_only:
        receiver = DummyDispatchReceiver(host=args.host, port=args.dispatch_port)
        receiver.start()
        dispatch_url = receiver.url
    elif args.baymax_api:
        dispatch_url = BAYMAX_MONITOR_EVENT_URL
    elif args.dispatch_url:
        dispatch_url = args.dispatch_url
    else:
        receiver = DummyDispatchReceiver(host=args.host, port=args.dispatch_port)
        receiver.start()
        dispatch_url = receiver.url
    dispatch_client = HospitalDispatchClient(dispatch_url)
    live: HospitalLiveServer | None = None
    policy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hospital-policy")
    dispatch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hospital-dispatch")
    output_dir = args.output_dir or _default_output_dir()
    try:
        if not args.no_live_ui and not args.validate_only:
            live = HospitalLiveServer(
                host=args.host,
                http_port=args.http_port,
                websocket_port=args.websocket_port,
            )
            live.start()
            print(f"Baymax clinical operations view: {live.public_url}")
        print(f"Policy: {policy.model_name}")
        print(f"Dispatch endpoint: {dispatch_url}")
        print("Safety: SIMULATION ONLY; alerts are not medical diagnoses")
        arena.start()
        started_wall = time.monotonic()
        last_publish = -1.0
        publish_index = 0
        while arena.phase == MatchPhase.RUNNING:
            _poll_policy(arena, policy_slot)
            _poll_dispatch(arena, dispatch_slot)
            _schedule_dispatch(
                arena, dispatch_slot, dispatch_client, dispatch_executor
            )
            _schedule_policy(arena, policy_slot, policy_executor)
            if live is not None:
                command = live.get_command()
                while command is not None:
                    command_type = command.get("type")
                    if command_type == "patient_speech_started":
                        arena.patient_speech_started(source="browser")
                    elif command_type == "patient_speech_finished":
                        arena.patient_speech_finished(source="browser")
                    elif command_type == "patient_transcript":
                        arena.add_patient_transcript(
                            str(command.get("transcript", "")),
                            source="browser",
                            final=bool(command.get("final", True)),
                        )
                    command = live.get_command()
            arena.step()
            if live is not None and arena.simulation_time_s - last_publish >= 0.2:
                frames = {
                    "broadcast": _jpeg(
                        arena.render("broadcast_camera", width=1280, height=720)
                    )
                }
                if publish_index % 4 == 0:
                    frames["ego"] = _jpeg(
                        arena.render("p1_ego_camera", width=640, height=360)
                    )
                live.publish(arena.state_payload(), frames)
                publish_index += 1
                last_publish = arena.simulation_time_s
            if args.max_seconds > 0 and time.monotonic() - started_wall >= args.max_seconds:
                arena.phase = MatchPhase.ABORTED
                break
            if args.validate_only and arena.simulation_time_s >= 150.0:
                arena.phase = MatchPhase.ABORTED
                break
        _poll_policy(arena, policy_slot)
        _poll_dispatch(arena, dispatch_slot)
        report = {
            "status": (
                "ok"
                if args.validate_only and arena.phase == MatchPhase.FINISHED
                else arena.phase.value
            ),
            "profileVersion": "10.0",
            "state": arena.state_payload(),
            "dispatchEndpoint": dispatch_url,
            "dummyDispatchCount": len(receiver.records) if receiver else None,
        }
        if live is not None:
            live.publish(
                arena.state_payload(),
                {
                    "broadcast": _jpeg(
                        arena.render("broadcast_camera", width=1280, height=720)
                    ),
                    "ego": _jpeg(arena.render("p1_ego_camera", width=640, height=360)),
                },
            )
            if arena.phase == MatchPhase.FINISHED:
                time.sleep(max(0.0, args.result_linger_seconds))
        arena.write_evidence(output_dir)
        (output_dir / "run_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if args.validate_only:
            accepted = [item for item in arena.dispatches if item["ok"]]
            if arena.phase != MatchPhase.FINISHED or len(accepted) != 2:
                raise RuntimeError(
                    f"hospital validation failed: phase={arena.phase.value}, dispatches={len(accepted)}"
                )
        return report
    finally:
        policy_executor.shutdown(wait=False, cancel_futures=True)
        dispatch_executor.shutdown(wait=True, cancel_futures=False)
        if live is not None:
            live.close()
        if receiver is not None:
            receiver.close()
        arena.close()


def _schedule_policy(
    arena: HospitalPatrolArena,
    slot: PolicySlot,
    executor: ThreadPoolExecutor,
) -> None:
    if (
        slot.pending is not None
        or slot.calls_remaining <= 0
        or not arena.needs_decision()
        or arena.current_skill == HospitalSkill.DISPATCH_INCIDENT
        or arena.simulation_time_s - slot.last_decision_s < 0.25
    ):
        return
    images: tuple[bytes, bytes]
    if slot.adapter.model_name.startswith("Deterministic"):
        images = (b"", b"")
    else:
        images = (
            _jpeg(arena.render("p1_ego_camera", width=512, height=288)),
            _jpeg(arena.render("broadcast_camera", width=512, height=288)),
        )
    slot.pending = executor.submit(slot.adapter.decide, arena.policy_status(), images)
    slot.calls_remaining -= 1
    slot.last_decision_s = arena.simulation_time_s
    arena.api_calls_remaining = slot.calls_remaining


def _poll_policy(arena: HospitalPatrolArena, slot: PolicySlot) -> None:
    future = slot.pending
    if future is None or not future.done():
        return
    slot.pending = None
    try:
        decision = future.result()
    except Exception as exc:  # noqa: BLE001
        arena.rationale = f"Gemini policy failed: {exc}"
        arena.current_skill = HospitalSkill.WAIT
        arena._event("policy_error", {"error": str(exc)})
        return
    arena.set_decision(decision, api_calls_remaining=slot.calls_remaining)
    print(
        f"Policy: {decision.skill.value}\n"
        f"Rationale: {decision.rationale}\n"
        f"Expected: {decision.expected_outcome}",
        flush=True,
    )


def _schedule_dispatch(
    arena: HospitalPatrolArena,
    slot: DispatchSlot,
    client: HospitalDispatchClient,
    executor: ThreadPoolExecutor,
) -> None:
    if slot.pending is not None or arena.current_skill != HospitalSkill.DISPATCH_INCIDENT:
        return
    incident = arena.next_incident()
    if incident is None:
        arena.current_skill = HospitalSkill.WAIT
        return
    slot.incident = incident
    slot.pending = executor.submit(client.post, incident)


def _poll_dispatch(arena: HospitalPatrolArena, slot: DispatchSlot) -> None:
    future = slot.pending
    if future is None or not future.done() or slot.incident is None:
        return
    incident = slot.incident
    slot.pending = None
    slot.incident = None
    try:
        result = future.result()
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "status": None, "attempts": 1, "error": str(exc)}
    arena.record_dispatch(incident, result)
    print(
        f"Dispatch: {incident['incidentType']} -> "
        f"{'accepted' if result.get('ok') else 'failed'}",
        flush=True,
    )


def _jpeg(frame: Any) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / f"baymax_patrol_{stamp}"


def main() -> None:
    try:
        report = run(parse_args())
        print(json.dumps(report, indent=2))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001
        print(f"Baymax simulation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
