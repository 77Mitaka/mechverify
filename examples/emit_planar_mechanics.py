"""Emit PlanarMechanics models for all example mechanisms.

Run:  python examples/emit_planar_mechanics.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechverify.ir import Mechanism
from mechverify.planar_mechanics import generate_planar_mechanics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "generated")

MECHS = [
    "fourbar_crank_rocker.json",
    "fourbar_straightline.json",
    "slider_crank.json",
    "sixbar_watt.json",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in MECHS:
        mech = Mechanism.from_json_file(os.path.join(HERE, name))
        path = os.path.join(OUT, "%s_pm.mo" % mech.id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(generate_planar_mechanics(mech))
        print("wrote", path)


if __name__ == "__main__":
    main()
