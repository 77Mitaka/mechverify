import json
import os

import numpy as np

from mechverify.ir import Mechanism
from mechverify.multibody import GeneralPlanarMechanism
from mechverify.verifier import verify

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDER = os.path.join(HERE, "..", "examples", "slider_crank.json")
SLIDER_REQ = os.path.join(HERE, "..", "examples", "req_slider_crank.json")


def load():
    return Mechanism.from_json_file(SLIDER)


def test_residual_zero_at_solution():
    gpm = GeneralPlanarMechanism(load())
    q = gpm.solve_multistart(np.radians(90.0), seed=0)
    assert q is not None
    assert np.linalg.norm(gpm.residuals(q, np.radians(90.0))) < 1e-8


def test_slider_position_matches_analytic():
    gpm = GeneralPlanarMechanism(load())
    a, b = 1.0, 2.5
    sw = gpm.sweep(90.0, 360.0, 360, seed=0)
    assert sw.completed
    th = sw.theta_driver
    x_analytic = a * np.cos(th) + np.sqrt(b * b - a * a * np.sin(th) ** 2)
    err = float(np.nanmax(np.abs(sw.points[:, 0] - x_analytic)))
    assert err < 1e-6


def test_slider_crank_verify_passes():
    with open(SLIDER_REQ, encoding="utf-8") as f:
        req = json.load(f)
    res = verify(load(), req)
    assert res.status == "pass"
    assert res.trace is not None
    assert "sigma_min" in res.trace


SIXBAR = os.path.join(HERE, "..", "examples", "sixbar_watt.json")
SIXBAR_REQ = os.path.join(HERE, "..", "examples", "req_sixbar.json")


def test_sixbar_multiloop_full_rotation():
    mech = Mechanism.from_json_file(SIXBAR)
    with open(SIXBAR_REQ, encoding="utf-8") as f:
        req = json.load(f)
    res = verify(mech, req)
    assert res.status == "pass"
    assert res.trace["completed"] is True
    assert res.trace["lost_at"] is None
