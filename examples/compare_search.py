"""Quantitative comparison: verifier-scored random search vs differential
evolution, on the task "find a straight-line four-bar".

Both methods share the same objective (best relative straightness of the
coupler path over a window), the same box bounds, and the same feasibility
constraints (Grashof crank-rocker, min transmission angle >= 30 deg).  We fix
the evaluation budget and report, over several seeds, the best straightness,
the success rate (straightness <= 0.005) and the evaluations to reach it.

Run:  python examples/compare_search.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechverify.kinematics import FourBarGeometry
from mechverify.requirement import path_straightness

BOUNDS = [
    (0.5, 2.0),   # a (crank)
    (1.0, 5.0),   # b (coupler)
    (1.0, 5.0),   # c (rocker)
    (1.0, 6.0),   # d (ground)
    (0.0, 6.0),   # p (coupler point distance)
    (-math.pi, math.pi),  # phi
]
THRESH = 0.005
BUDGET = 2000
STEPS = 180
INFEASIBLE = 1e3


def objective(x):
    a, b, c, d, p, phi = x
    if min(a, b, c, d) <= 0:
        return INFEASIBLE
    if a > min(b, c, d) + 1e-9:  # crank must be the shortest link
        return INFEASIBLE
    ls = sorted([a, b, c, d])
    if ls[0] + ls[3] > ls[1] + ls[2]:  # Grashof
        return INFEASIBLE
    try:
        fb = FourBarGeometry(a, b, c, d, coupler_p=p, coupler_phi=phi)
    except ValueError:
        return INFEASIBLE
    sw = fb.sweep(0.0, 360.0, STEPS, branch=1)
    if not sw.completed:
        return INFEASIBLE
    if float(np.nanmin(sw.mu_deg[sw.valid])) < 30.0:
        return INFEASIBLE
    return path_straightness(sw.valid_points(), 0.3)


def random_search(rng):
    best = INFEASIBLE
    curve = []
    for _ in range(BUDGET):
        x = [rng.uniform(lo, hi) for lo, hi in BOUNDS]
        v = objective(x)
        if v < best:
            best = v
        curve.append(best)
    return best, curve


def differential_evolution():
    from scipy.optimize import differential_evolution as de

    curve = []

    def cb(xk, convergence=None):
        curve.append(objective(xk))

    res = de(objective, BOUNDS, maxiter=40, popsize=10, tol=1e-9, seed=0,
             polish=False, callback=cb)
    return float(res.fun), curve


def main(runs=12):
    rng = np.random.default_rng(0)
    rs_best, de_best = [], []
    for _ in range(runs):
        rs_best.append(random_search(rng)[0])
        de_best.append(differential_evolution()[0])
    rs = np.array(rs_best)
    de_ = np.array(de_best)
    print("budget=%d evals, runs=%d, threshold=%.3f" % (BUDGET, runs, THRESH))
    print("random+verifier : median_best=%.5f  success=%.2f" % (np.median(rs), np.mean(rs <= THRESH)))
    print("differential_ev : median_best=%.5f  success=%.2f" % (np.median(de_), np.mean(de_ <= THRESH)))


if __name__ == "__main__":
    main()
