import os
import re

from mechverify import Mechanism, verify
from mechverify.modelica import generate_modelica
from mechverify.convert import modelica_to_mechanism

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "fourbar_crank_rocker.json")
SIX = os.path.join(HERE, "..", "examples", "sixbar_watt.json")


def _strip_ir(mo):
    return re.sub(r"\n\s*// mechverify-ir:.*", "", mo)


def test_embedded_roundtrip_fourbar():
    mech = Mechanism.from_json_file(EX)
    mo = generate_modelica(mech)
    back = modelica_to_mechanism(mo)
    assert len(back.joints) == 4
    req = {"constraints": [{"type": "crank_rotatability", "hard": True}]}
    a = verify(mech, req)
    b = verify(back, req)
    assert a.status == b.status == "pass"


def test_equation_parse_fourbar():
    mech = Mechanism.from_json_file(EX)
    mo = _strip_ir(generate_modelica(mech))
    back = modelica_to_mechanism(mo)
    assert len(back.joints) == 4
    assert back.driver is not None
    res = verify(back, {"constraints": [{"type": "crank_rotatability", "hard": True}]})
    assert res.status == "pass"


def test_equation_parse_sixbar():
    mech = Mechanism.from_json_file(SIX)
    mo = _strip_ir(generate_modelica(mech))
    back = modelica_to_mechanism(mo)
    assert len(back.joints) == 7
    res = verify(back, {"constraints": [{"type": "crank_rotatability", "hard": False}]})
    assert res.trace is not None
    assert res.trace["completed"] is True
