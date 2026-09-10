import math
import os

import pytest

from mechverify.ir import Mechanism
from mechverify.modelica import (
    ensure_general,
    generate_modelica,
    initial_state,
    omc_available,
    simulate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "fourbar_crank_rocker.json")
SLIDER = os.path.join(HERE, "..", "examples", "slider_crank.json")


def test_generate_self_contained_model():
    mech = Mechanism.from_json_file(SLIDER)
    code = generate_modelica(mech)
    assert "model slider_crank" in code
    assert code.strip().endswith("end slider_crank;")
    assert "driver_angle" in code
    assert "poi_x" in code and "poi_y" in code
    assert "annotation(experiment" in code


def test_ensure_general_converts_four_bar():
    mech = Mechanism.from_json_file(EX)
    gen = ensure_general(mech)
    assert any(j.anchors for j in gen.joints)
    assert gen.point_of_interest is not None


def test_initial_state_four_bar():
    mech = Mechanism.from_json_file(EX)
    state = initial_state(mech, 0.0)
    # driver link (crank) angle should equal the driver start angle
    assert abs(state["crank"][2] - 0.0) < 1e-6
    # crank pivot is at the ground origin
    assert abs(state["crank"][0]) < 1e-6 and abs(state["crank"][1]) < 1e-6


def test_initial_state_slider_crank():
    mech = Mechanism.from_json_file(SLIDER)
    state = initial_state(mech, 90.0)
    assert abs(state["crank"][2] - math.pi / 2) < 1e-6
    # slider sits on the x axis
    assert abs(state["slider"][1]) < 1e-6


@pytest.mark.skipif(omc_available(), reason="omc present; missing-tool error not raised")
def test_simulate_without_omc_raises():
    with pytest.raises(RuntimeError, match="omc"):
        simulate("nonexistent.mo", "m")
