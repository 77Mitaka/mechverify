from mechverify.ir import Mechanism
from mechverify.structural import grashof, mobility, structural_checks


def make_crank_rocker():
    return Mechanism.from_dict(
        {
            "id": "fb",
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
                {"id": "D", "type": "revolute", "links": ["rocker", "ground"], "position": [2.5, 0.0]},
            ],
            "dimensions": {"crank": 1.0, "coupler": 2.5, "rocker": 2.5},
        }
    )


def test_mobility_four_bar():
    assert mobility(4, 4) == 1


def test_grashof_true_and_false():
    assert grashof(1.0, 2.5, 2.5, 2.5)["is_grashof"]
    assert not grashof(2.0, 3.0, 4.0, 6.0)["is_grashof"]


def test_structural_checks_pass():
    checks, diags = structural_checks(make_crank_rocker())
    by_name = {c["name"]: c for c in checks}
    assert by_name["mobility"]["passed"]
    assert by_name["grashof"]["passed"]
    assert by_name["closure_feasible"]["passed"]
    assert all(c["passed"] for c in checks)
