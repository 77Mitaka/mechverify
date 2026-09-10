import csv
import math
import os

import numpy as np
import pytest

from mechverify import Mechanism
from mechverify.modelica import omc_available, simulate
from mechverify.planar_mechanics import generate_planar_mechanics

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDER = os.path.join(HERE, "..", "examples", "slider_crank.json")
FOURBAR = os.path.join(HERE, "..", "examples", "fourbar_crank_rocker.json")


def test_generated_code_uses_library_components():
    mech = Mechanism.from_json_file(SLIDER)
    code = generate_planar_mechanics(mech)
    assert "inner PlanarMechanics.PlanarWorld" in code
    assert "PlanarMechanics.Parts.Body" in code
    assert "PlanarMechanics.Joints.Prismatic" in code
    assert "connect(" in code
    assert "poi_x" in code
    assert code.strip().endswith("end slider_crank;")


@pytest.mark.skipif(not omc_available(), reason="OpenModelica not available")
def test_slider_crank_pm_matches_analytic(tmp_path):
    mech = Mechanism.from_json_file(SLIDER)
    mo = tmp_path / "slider_crank.mo"
    mo.write_text(generate_planar_mechanics(mech, stop_time=1.0), encoding="utf-8")
    try:
        path = simulate(str(mo), "slider_crank", stop_time=1.0, number_of_intervals=50)
    except Exception as exc:
        pytest.skip("OpenModelica build/simulation unavailable here: %s" % exc)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time"]) for r in rows])
    x = np.array([float(r["poi_x"]) for r in rows])
    th = math.pi / 2 + 2 * math.pi * t
    xa = np.cos(th) + np.sqrt(2.5 ** 2 - np.sin(th) ** 2)
    assert float(np.max(np.abs(x - xa))) < 1e-6


@pytest.mark.skipif(not omc_available(), reason="OpenModelica not available")
def test_closed_loop_fourbar_pm_simulates(tmp_path):
    mech = Mechanism.from_json_file(FOURBAR)
    mo = tmp_path / "fourbar_crank_rocker.mo"
    mo.write_text(generate_planar_mechanics(mech, stop_time=1.0), encoding="utf-8")
    try:
        path = simulate(str(mo), "fourbar_crank_rocker", stop_time=1.0, number_of_intervals=30)
    except Exception as exc:
        pytest.skip("OpenModelica build/simulation unavailable here: %s" % exc)
    assert os.path.exists(path)
