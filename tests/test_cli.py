import json
import os

from mechverify.cli import main

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "fourbar_straightline.json")
REQ = os.path.join(HERE, "..", "examples", "req_straightline.json")
SLIDER = os.path.join(HERE, "..", "examples", "slider_crank.json")


def test_cli_verify_pass():
    assert main(["verify", EX, "--req", REQ]) == 0


def test_cli_verify_robust():
    crank = os.path.join(HERE, "..", "examples", "fourbar_crank_rocker.json")
    req = os.path.join(HERE, "..", "examples", "req_crank_rocker.json")
    assert main(["verify", crank, "--req", req, "--robust", "--samples", "30"]) == 0


def test_cli_export(tmp_path):
    out = tmp_path / "m.mo"
    assert main(["export", SLIDER, "-o", str(out)]) == 0
    assert out.exists()
    assert "model slider_crank" in out.read_text(encoding="utf-8")


def test_cli_import(tmp_path):
    from mechverify import Mechanism
    from mechverify.modelica import write_model

    mo = tmp_path / "s.mo"
    write_model(Mechanism.from_json_file(SLIDER), str(mo))
    out = tmp_path / "s.ir.json"
    assert main(["import", str(mo), "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["id"] == "slider_crank"
