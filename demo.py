"""End-to-end demo: load a four-bar, verify it against requirements, plot.

Run:  python demo.py
"""

from __future__ import annotations

import json
import math
import os

from mechverify import Mechanism, verify, verify_robust

HERE = os.path.dirname(os.path.abspath(__file__))


def ascii_plot(points, width=64, height=24):
    pts = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if not pts:
        return "(no path)"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    if maxx - minx < 1e-9:
        maxx = minx + 1.0
    if maxy - miny < 1e-9:
        maxy = miny + 1.0
    grid = [[" "] * width for _ in range(height)]
    for x, y in pts:
        c = int((x - minx) / (maxx - minx) * (width - 1))
        r = int((y - miny) / (maxy - miny) * (height - 1))
        grid[height - 1 - r][c] = "."
    return "\n".join("".join(row) for row in grid)


CASES = [
    ("fourbar_crank_rocker.json", "req_crank_rocker.json"),
    ("fourbar_straightline.json", "req_straightline.json"),
    ("slider_crank.json", "req_slider_crank.json"),
    ("sixbar_watt.json", "req_sixbar.json"),
]


def main():
    for mech_file, req_file in CASES:
        mech = Mechanism.from_json_file(os.path.join(HERE, "examples", mech_file))
        with open(os.path.join(HERE, "examples", req_file), encoding="utf-8") as f:
            req = json.load(f)

        res = verify(mech, req)
        print("=== verify: %s ===" % mech.id)
        print(res.summary())

        if res.trace:
            print()
            print("coupler path (ascii):")
            print(ascii_plot(res.trace["coupler_path"]))
            mu = res.trace.get("mu_deg")
            if mu:
                vals = [m for m in mu if math.isfinite(m)]
                print("min transmission angle: %.2f deg" % min(vals))
            sig = res.trace.get("sigma_min")
            if sig:
                vals = [s for s in sig if math.isfinite(s)]
                print("min singular value: %.4g" % min(vals))
            print("crank full revolution: %s" % res.trace["completed"])
        print()

    print("=== robust verification: fourbar_straightline (rel_tol=0.002, 100 samples) ===")
    mech = Mechanism.from_json_file(os.path.join(HERE, "examples", "fourbar_straightline.json"))
    with open(os.path.join(HERE, "examples", "req_straightline.json"), encoding="utf-8") as f:
        req = json.load(f)
    rres = verify_robust(mech, req, samples=100, rel_tol=0.002, seed=0)
    print(rres.summary())


if __name__ == "__main__":
    main()
