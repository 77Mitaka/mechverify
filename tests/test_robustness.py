import json
import os

import numpy as np

from mechverify import Mechanism
from mechverify.robustness import perturb_mechanism, verify_robust

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "fourbar_crank_rocker.json")
STRAIGHT = os.path.join(HERE, "..", "examples", "fourbar_straightline.json")
REQ_STRAIGHT = os.path.join(HERE, "..", "examples", "req_straightline.json")


def test_perturb_changes_geometry():
    mech = Mechanism.from_json_file(EX)
    rng = np.random.default_rng(0)
    m2 = perturb_mechanism(mech, 0.01, rng)
    assert m2.dimensions != mech.dimensions


def test_robust_pass_for_good_mechanism():
    mech = Mechanism.from_json_file(STRAIGHT)
    with open(REQ_STRAIGHT, encoding="utf-8") as f:
        req = json.load(f)
    res = verify_robust(mech, req, samples=60, rel_tol=0.001, seed=0, min_pass_rate=0.9)
    assert res.nominal.status == "pass"
    assert res.pass_rate >= 0.9
    assert res.status == "pass"
    assert "straightness" in res.metrics


def test_robust_fails_when_margin_too_small():
    mech = Mechanism.from_json_file(EX)
    req = {"constraints": [{"type": "transmission_angle_min", "value": 34.9, "hard": True}]}
    res = verify_robust(mech, req, samples=80, rel_tol=0.01, seed=0, min_pass_rate=0.99)
    assert res.status == "fail"
