"""Search for a four-bar with a straight coupler-path segment.

This is a small demonstration of the *verifier-as-oracle* idea: the fast
Layer-1 sweep is used as the scoring function inside a random search.

Run:  python examples/search_straightline.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechverify.kinematics import FourBarGeometry
from mechverify.requirement import path_straightness


def evaluate(a, b, c, d, p, phi, steps=240):
    try:
        fb = FourBarGeometry(a, b, c, d, coupler_p=p, coupler_phi=phi)
    except ValueError:
        return None
    sw = fb.sweep(0.0, 360.0, steps, branch=1)
    if not sw.completed:
        return None
    mu = float(np.nanmin(sw.mu_deg))
    if mu < 25.0:
        return None
    st = path_straightness(sw.valid_points(), window_frac=0.3)
    if not math.isfinite(st):
        return None
    return st, mu


def main(n_iter=4000, seed=7):
    random.seed(seed)
    best = None
    for _ in range(n_iter):
        a = random.uniform(0.5, 2.0)
        b = random.uniform(1.0, 5.0)
        c = random.uniform(1.0, 5.0)
        d = random.uniform(1.0, 5.0)
        if a > min(b, c, d) + 1e-9:  # crank must be the shortest link
            continue
        ls = sorted([a, b, c, d])
        if ls[0] + ls[3] > ls[1] + ls[2]:  # Grashof
            continue
        p = random.uniform(0.2 * b, 1.3 * b)
        phi = random.uniform(-math.pi, math.pi)
        r = evaluate(a, b, c, d, p, phi)
        if r is None:
            continue
        st, mu = r
        if best is None or st < best["straightness"]:
            best = {
                "straightness": st,
                "min_transmission_angle": mu,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "p": p,
                "phi": phi,
            }
    return best


if __name__ == "__main__":
    best = main()
    print(json.dumps(best, indent=2))
    if best:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fourbar_straightline.json")
        mech = {
            "id": "fourbar_straightline",
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
                {"id": "D", "type": "revolute", "links": ["rocker", "ground"], "position": [best["d"], 0.0]},
            ],
            "dimensions": {"crank": best["a"], "coupler": best["b"], "rocker": best["c"]},
            "coupler_point": {"link": "coupler", "p": best["p"], "phi": best["phi"]},
            "driver": {"joint": "A", "link": "crank", "start": 0.0, "span": 360.0, "steps": 720, "branch": 1},
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(mech, f, indent=2, ensure_ascii=False)
        print("saved:", out)
