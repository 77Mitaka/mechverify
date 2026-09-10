"""Compute the numbers for the error-convergence and sensitivity tables."""

import os
import sys
import csv
import math
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = None
for _cand in (os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "mechverify")):
    if os.path.isdir(os.path.join(_cand, "mechverify")):
        PROJ = _cand
        break
if PROJ is None:
    raise RuntimeError("cannot locate the mechverify package")
sys.path.insert(0, PROJ)

from mechverify import Mechanism
from mechverify.kinematics import FourBarGeometry, four_bar_from_mechanism
from mechverify.modelica import simulate
from mechverify.planar_mechanics import generate_planar_mechanics
from mechverify.requirement import path_straightness

EX = os.path.join(PROJ, "examples")


def convergence():
    mech = Mechanism.from_json_file(os.path.join(EX, "fourbar_straightline.json"))
    fb = four_bar_from_mechanism(mech)
    ref = fb.sweep(0.0, 360.0, 5760, branch=1)
    ref_st = path_straightness(ref.valid_points(), 0.3)
    ref_mu = float(np.nanmin(ref.mu_deg[ref.valid]))
    print("=== discretization convergence (straight-line four-bar) ===")
    print("reference steps=5760  straightness=%.6e  min_mu=%.4f" % (ref_st, ref_mu))
    for steps in [45, 90, 180, 360, 720, 1440]:
        sw = fb.sweep(0.0, 360.0, steps, branch=1)
        st = path_straightness(sw.valid_points(), 0.3)
        mu = float(np.nanmin(sw.mu_deg[sw.valid]))
        print("%5d  straightness=%.6e (err=%.2e)  min_mu=%.4f (err=%.2e)"
              % (steps, st, abs(st - ref_st), mu, abs(mu - ref_mu)))


def simulation_accuracy():
    mech = Mechanism.from_json_file(os.path.join(EX, "slider_crank.json"))
    d = tempfile.mkdtemp(prefix="conv_")
    mo = os.path.join(d, "slider_crank.mo")
    with open(mo, "w", encoding="utf-8") as f:
        f.write(generate_planar_mechanics(mech, stop_time=1.0))
    print("=== simulation accuracy (slider crank) ===")
    for tol in [1e-3, 1e-5, 1e-7, 1e-9]:
        path = simulate(mo, "slider_crank", stop_time=1.0, number_of_intervals=200, tolerance=tol)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        t = np.array([float(r["time"]) for r in rows])
        x = np.array([float(r["poi_x"]) for r in rows])
        th = math.pi / 2 + 2 * math.pi * t
        xa = np.cos(th) + np.sqrt(2.5 ** 2 - np.sin(th) ** 2)
        print("tol=%.0e  max_err=%.3e" % (tol, float(np.max(np.abs(x - xa)))))


def sensitivity():
    mech = Mechanism.from_json_file(os.path.join(EX, "fourbar_straightline.json"))
    fb = four_bar_from_mechanism(mech)
    print("=== sensitivity (straight-line four-bar, +0.5%) ===")

    def eval_fb(a, b, c, d, p):
        g = FourBarGeometry(a, b, c, d, coupler_p=p, coupler_phi=fb.coupler_phi)
        sw = g.sweep(0.0, 360.0, 720, branch=1)
        st = path_straightness(sw.valid_points(), 0.3)
        mu = float(np.nanmin(sw.mu_deg[sw.valid]))
        return st, mu

    base = eval_fb(fb.a, fb.b, fb.c, fb.d, fb.coupler_p)
    print("base    straightness=%.5f  min_mu=%.2f" % base)
    for name in ["a", "b", "c", "d", "p"]:
        a, b, c, d, p = fb.a, fb.b, fb.c, fb.d, fb.coupler_p
        if name == "a":
            a *= 1.005
        elif name == "b":
            b *= 1.005
        elif name == "c":
            c *= 1.005
        elif name == "d":
            d *= 1.005
        elif name == "p":
            p *= 1.005
        st, mu = eval_fb(a, b, c, d, p)
        print("%-6s  straightness=%.5f (%+.1f%%)  min_mu=%.2f (%+.1f%%)"
              % (name, st, 100 * (st - base[0]) / base[0], mu, 100 * (mu - base[1]) / base[1]))


if __name__ == "__main__":
    convergence()
    sensitivity()
