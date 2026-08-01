from __future__ import annotations

import json

import numpy as np
import pytest

from baymax_nurse.arena import HospitalPatrolArena
from baymax_nurse.cli import parse_args, run
from baymax_nurse.dispatch import DummyDispatchReceiver, HospitalDispatchClient
from baymax_nurse.model import default_asset_manifest, build_hospital_xml
from baymax_nurse.policy import ScriptedHospitalPolicy
from baymax_nurse.schemas import HospitalSkill


def test_hospital_model_contains_one_g1_two_rooms_and_patient_evidence():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(build_hospital_xml())

    assert model.nq == 50
    assert model.nu == 43
    assert model.vis.global_.offwidth == 1920
    assert model.vis.global_.offheight == 1080
    assert model.joint("p1_floating_base_joint").id >= 0
    assert model.geom("room_1_bed_collision").id >= 0
    assert model.geom("room_2_bed_collision").id >= 0
    assert model.geom("room_1_mattress").id >= 0
    assert model.geom("room_1_headboard").id >= 0
    assert model.geom("room_1_left_rail").id >= 0
    assert model.geom("room_1_monitor_screen").id >= 0
    assert model.geom("patient_202_collision").id >= 0
    for name in (
        "room_1_bed_collision",
        "room_1_mattress",
        "room_1_headboard",
        "room_1_footboard",
        "room_1_left_rail",
        "room_1_right_rail",
        "patient_202_collision",
    ):
        geom = model.geom(name)
        assert model.geom_contype[geom.id] == 4
        assert model.geom_conaffinity[geom.id] == 3
    if default_asset_manifest().is_file():
        assert model.geom("patient_101_detailed").id >= 0
        boy = model.geom("patient_202_detailed")
        assert boy.id >= 0
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mesh_id = model.geom_dataid[boy.id]
        vertex_address = model.mesh_vertadr[mesh_id]
        vertex_count = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[vertex_address : vertex_address + vertex_count]
        world_vertices = (
            vertices @ data.geom_xmat[boy.id].reshape(3, 3).T
            + data.geom_xpos[boy.id]
        )
        span = np.ptp(world_vertices, axis=0)
        assert span[1] > 1.4
        assert span[2] < 0.42
        assert world_vertices[:, 2].min() >= -0.001
    else:
        assert model.geom("patient_101_torso_proxy").id >= 0
        assert model.geom("patient_202_torso_proxy").id >= 0
    with pytest.raises(KeyError):
        model.joint("p2_floating_base_joint")


def test_scripted_policy_follows_rooms_incidents_and_return_home():
    policy = ScriptedHospitalPolicy()
    status = {
        "fallen": False,
        "atHome": True,
        "rooms": {
            "room_1": {"visited": False, "atInspectionPoint": False},
            "room_2": {"visited": False, "atInspectionPoint": False},
        },
        "pendingIncidents": [],
    }
    assert policy.decide(status, (b"", b"")).skill == HospitalSkill.NAVIGATE_ROOM_1
    status["rooms"]["room_1"]["atInspectionPoint"] = True
    assert policy.decide(status, (b"", b"")).skill == HospitalSkill.INSPECT_ROOM_1
    status["rooms"]["room_1"]["visited"] = True
    status["pendingIncidents"] = [{"incidentType": "critical_monitor"}]
    assert policy.decide(status, (b"", b"")).skill == HospitalSkill.DISPATCH_INCIDENT
    status["pendingIncidents"] = []
    assert policy.decide(status, (b"", b"")).skill == HospitalSkill.NAVIGATE_ROOM_2
    status["rooms"]["room_2"] = {"visited": True, "atInspectionPoint": False}
    status["atHome"] = False
    assert policy.decide(status, (b"", b"")).skill == HospitalSkill.RETURN_HOME


def test_dummy_dispatch_receiver_accepts_structured_http_post():
    receiver = DummyDispatchReceiver(port=0)
    receiver.start()
    try:
        payload = {
            "incidentId": "incident-test-1",
            "roomId": "room_1",
            "incidentType": "critical_monitor",
        }
        result = HospitalDispatchClient(receiver.url, retries=0).post(payload)
        assert result["ok"] is True
        assert result["status"] == 202
        assert receiver.records == [payload]
    finally:
        receiver.close()


def test_room_1_waits_for_speech_to_finish_even_after_silence_deadline():
    pytest.importorskip("mujoco")
    arena = HospitalPatrolArena(speech_mode="browser")
    try:
        assert arena._update_room_1_listening() is False
        arena.patient_speech_started()
        arena.data.time = 6.0
        arena.add_patient_transcript(
            "I have severe chest pain", source="browser", final=False
        )
        assert arena._update_room_1_listening() is False
        arena.patient_speech_finished()
        assert arena._update_room_1_listening() is True
        assert arena.room_1_listening_outcome == "speech_complete"
    finally:
        arena.close()


def test_room_1_moves_after_five_seconds_without_speech():
    pytest.importorskip("mujoco")
    arena = HospitalPatrolArena(speech_mode="browser")
    try:
        assert arena._update_room_1_listening() is False
        arena.data.time = 4.99
        assert arena._update_room_1_listening() is False
        arena.data.time = 5.01
        assert arena._update_room_1_listening() is True
        assert arena.room_1_listening_outcome == "silence_timeout"
    finally:
        arena.close()


def test_full_scripted_hospital_patrol_routes_through_doorway_and_dispatches_two_incidents(
    tmp_path,
):
    pytest.importorskip("mujoco")
    args = parse_args(
        [
            "--headless",
            "--validate-only",
            "--dispatch-port",
            "0",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report = run(args)

    assert report["status"] == "ok"
    assert report["dummyDispatchCount"] == 2
    assert report["state"]["result"] == "PATROL COMPLETE"
    assert report["state"]["speech"]["outcome"] == "speech_complete"
    assert report["state"]["speech"]["elapsedSeconds"] >= 4.0
    assert report["state"]["safety"]["wallContactSamples"] == 0
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert min(item["bedClearanceM"] for item in trajectory) > 0.0
    assert not any(item["wallContact"] for item in trajectory)
    doorway_samples = [
        item for item in trajectory if 0.15 <= item["robot"][1] <= 0.75
    ]
    assert doorway_samples
    assert max(abs(item["robot"][0]) for item in doorway_samples) < 0.48
    assert max(item["robot"][1] for item in trajectory) > 1.75
    saved = json.loads((tmp_path / "dispatches.json").read_text())
    assert [item["incidentType"] for item in saved] == [
        "critical_monitor",
        "patient_fall",
    ]
    room_1_request = saved[0]["request"]
    assert room_1_request["patientName"] == "Grandma"
    assert room_1_request["monitorReadings"] == {
        "heartRateBpm": 148,
        "spo2Percent": 82,
        "alarm": "critical",
    }
    assert room_1_request["patientSpeech"]["transcript"].startswith(
        "I have severe chest pain"
    )
    assert "Grandma reported" in room_1_request["summary"]
    assert saved[1]["request"]["patientId"] == "patient_202"
    assert saved[1]["request"]["patientName"] == "Boy"
