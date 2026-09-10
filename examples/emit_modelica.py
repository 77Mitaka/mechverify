"""Emit self-contained Modelica models for all example mechanisms.

Run:  python examples/emit_modelica.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechverify.ir import Mechanism
from mechverify.modelica import omc_available, write_model

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
        path = os.path.join(OUT, "%s.mo" % mech.id)
        write_model(mech, path)
        print("wrote", path)
    print("omc available:", omc_available())


if __name__ == "__main__":
    main()
