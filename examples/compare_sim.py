import csv, math
import numpy as np
from mechverify.modelica import simulate

path = simulate("examples/generated/slider_crank.mo", "slider_crank", stop_time=1.0, number_of_intervals=200)
print("CSV:", path)
with open(path) as f:
    rows = list(csv.DictReader(f))
print("columns:", list(rows[0].keys())[:6], "... n=", len(rows))
t = np.array([float(r["time"]) for r in rows])
x = np.array([float(r["poi_x"]) for r in rows])
y = np.array([float(r["poi_y"]) for r in rows])
th = math.pi / 2 + 2 * math.pi * t
xa = np.cos(th) + np.sqrt(2.5 ** 2 - np.sin(th) ** 2)
print("max |poi_x - analytic_x| =", float(np.max(np.abs(x - xa))))
print("max |poi_y| =", float(np.max(np.abs(y))))
