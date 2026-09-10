import json
import os

from mechverify.ir import Mechanism
from mechverify.verifier import verify

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "fourbar_crank_rocker.json")
EX_STRAIGHT = os.path.join(HERE, "..", "examples", "fourbar_straightline.json")
REQ_STRAIGHT = os.path.join(HERE, "..", "examples", "req_straightline.json")


def load():
    return Mechanism.from_json_file(EX)


def test_pass_with_low_threshold():
    req = {
        "constraints": [
            {"type": "crank_rotatability", "hard": True},
            {"type": "transmission_angle_min", "value": 0.0, "hard": True},
        ]
    }
    res = verify(load(), req)
    assert res.status == "pass"
    assert res.trace is not None
    assert 0.0 <= res.score <= 1.0


def test_fail_with_impossible_threshold():
    req = {"constraints": [{"type": "transmission_angle_min", "value": 89.0, "hard": True}]}
    res = verify(load(), req)
    assert res.status == "fail"
    assert res.score == 0.0


def test_straightline_example_passes():
    mech = Mechanism.from_json_file(EX_STRAIGHT)
    with open(REQ_STRAIGHT, encoding="utf-8") as f:
        req = json.load(f)
    res = verify(mech, req)
    assert res.status == "pass"
    st = next(c for c in res.checks if c["name"] == "straightness")
    assert st["value"] < 0.005


def test_soft_straightness_gives_score():
    req = {
        "constraints": [
            {"type": "straightness", "value": 1.0, "window_frac": 0.25, "hard": False, "weight": 1.0}
        ]
    }
    res = verify(load(), req)
    assert res.status == "pass"
    assert 0.0 <= res.score <= 1.0
