"""Measure performance and determinism for the paper's experiment table."""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = None
for _cand in (os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "mechverify")):
    if os.path.isdir(os.path.join(_cand, "mechverify")):
        PROJ = _cand
        break
sys.path.insert(0, PROJ)

from mechverify import Mechanism, verify
from mechverify.modelica import simulate
from mechverify.planar_mechanics import generate_planar_mechanics
from mechverify.robustness import verify_robust

EX = os.path.join(PROJ, "examples")

REQ4 = {"constraints": [{"type": "crank_rotatability", "hard": True},
                        {"type": "transmission_angle_min", "value": 30.0, "hard": True}]}
REQ_SC = {"constraints": [{"type": "crank_rotatability", "hard": True},
                          {"type": "singularity_margin_min", "value": 0.05, "hard": False}]}


def timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    fb = Mechanism.from_json_file(os.path.join(EX, "fourbar_crank_rocker.json"))
    sc = Mechanism.from_json_file(os.path.join(EX, "slider_crank.json"))

    print("L0+L1 four-bar analytic : %.3f ms/verify" % (1000 * timeit(lambda: verify(fb, REQ4), 300)))
    print("L1 general multibody    : %.3f ms/verify" % (1000 * timeit(lambda: verify(sc, REQ_SC), 30)))
    print("L3 robustness(100)      : %.3f s/verify" % timeit(lambda: verify_robust(fb, REQ4, samples=100), 3))

    import tempfile
    d = tempfile.mkdtemp(prefix="perf_")
    mo = os.path.join(d, "slider_crank.mo")
    with open(mo, "w", encoding="utf-8") as f:
        f.write(generate_planar_mechanics(sc, stop_time=1.0))
    t0 = time.perf_counter()
    simulate(mo, "slider_crank", stop_time=1.0, number_of_intervals=200)
    print("L2 omc compile+simulate : %.3f s (first run)" % (time.perf_counter() - t0))

    scores = [round(verify(fb, REQ4).score, 12) for _ in range(50)]
    print("determinism (50 runs)   : unique scores = %d" % len(set(scores)))


if __name__ == "__main__":
    main()
