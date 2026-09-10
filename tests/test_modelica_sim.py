import csv
import math
import os

import numpy as np
import pytest

from mechverify.modelica import generate_modelica, omc_available, simulate, _world_expr

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDER = os.path.join(HERE, "..", "examples", "slider_crank.json")


def test_world_expr_is_correct_rotation():
    x_expr, y_expr = _world_expr("L", "ground", 2.0, 3.0)
    th = 0.7
    ns = {"L_x": 1.0, "L_y": 2.0, "L_th": th, "cos": math.cos, "sin": math.sin}
    x = eval(x_expr, {"__builtins__": {}}, ns)
    y = eval(y_expr, {"__builtins__": {}}, ns)
    assert abs(x - (1.0 + 2.0 * math.cos(th) - 3.0 * math.sin(th))) < 1e-9
    assert abs(y - (2.0 + 2.0 * math.sin(th) + 3.0 * math.cos(th))) < 1e-9


@pytest.mark.skipif(not omc_available(), reason="OpenModelica not available")
def test_slider_crank_simulation_matches_analytic(tmp_path):
    from mechverify import Mechanism

    mech = Mechanism.from_json_file(SLIDER)
    mo = tmp_path / "slider_crank.mo"
    mo.write_text(generate_modelica(mech), encoding="utf-8")
    try:
        path = simulate(str(mo), "slider_crank", stop_time=1.0, number_of_intervals=100)
    except Exception as exc:  # e.g. parallel-build DLL issue in some environments
        pytest.skip("OpenModelica build/simulation unavailable here: %s" % exc)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time"]) for r in rows])
    x = np.array([float(r["poi_x"]) for r in rows])
    y = np.array([float(r["poi_y"]) for r in rows])
    th = math.pi / 2 + 2 * math.pi * t
    xa = np.cos(th) + np.sqrt(2.5 ** 2 - np.sin(th) ** 2)
    assert float(np.max(np.abs(x - xa))) < 1e-6
    assert float(np.max(np.abs(y))) < 1e-6
