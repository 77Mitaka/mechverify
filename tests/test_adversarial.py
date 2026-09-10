import os

from mechverify import Mechanism, verify


def _mk(dims, d, driver_start=0.0):
    return Mechanism.from_dict({
        "id": "adv",
        "ground": "ground",
        "links": [
            {"id": "ground", "kind": "ground"},
            {"id": "crank"},
            {"id": "coupler"},
            {"id": "rocker"},
        ],
        "joints": [
            {"id": "A", "type": "revolute", "links": ["ground", "crank"], "position": [0.0, 0.0]},
            {"id": "B", "type": "revolute", "links": ["crank", "coupler"]},
            {"id": "C", "type": "revolute", "links": ["coupler", "rocker"]},
            {"id": "D", "type": "revolute", "links": ["rocker", "ground"], "position": [d, 0.0]},
        ],
        "dimensions": dims,
        "coupler_point": {"link": "coupler", "p": dims["coupler"] / 2.0, "phi": 0.0},
        "driver": {"joint": "A", "link": "crank", "start": driver_start, "span": 360.0, "steps": 720},
    })


def test_non_grashof_crank_cannot_rotate():
    mech = _mk({"crank": 2.0, "coupler": 3.0, "rocker": 4.0}, 6.0)
    res = verify(mech, {"constraints": [{"type": "crank_rotatability", "hard": True}]})
    assert res.status == "fail"


def test_impossible_closure_fails_structural():
    mech = _mk({"crank": 1.0, "coupler": 1.0, "rocker": 1.0}, 10.0)
    res = verify(mech, {})
    assert res.status == "fail"
    assert any(c["name"] == "closure_feasible" and not c["passed"] for c in res.checks)


def test_transmission_angle_requirement_fails():
    mech = _mk({"crank": 1.0, "coupler": 2.5, "rocker": 2.5}, 2.5)
    res = verify(mech, {"constraints": [{"type": "transmission_angle_min", "value": 89.0, "hard": True}]})
    assert res.status == "fail"
    assert res.score == 0.0


def test_straightness_impossible_fails():
    mech = _mk({"crank": 1.0, "coupler": 2.5, "rocker": 2.5}, 2.5)
    res = verify(mech, {"constraints": [{"type": "straightness", "value": 0.0001, "hard": True}]})
    assert res.status == "fail"
