"""Build a multi-loop six-bar that a full-rotating crank can drive, and verify.

Construction:
    loop 1 : A - B - C - D    (a Grashof crank-rocker)
    loop 2 : D - E - F - G    (a dyad attached to a point E on link C-D)
The dyad links are sized so that the distance |EG| is always reachable, which
guarantees the crank can rotate a full revolution.

Run:  python examples/build_sixbar.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechverify.kinematics import FourBarGeometry
from mechverify.sketch import build_mechanism
from mechverify.verifier import verify

# --- loop 1: known Grashof crank-rocker a=1, b=2.5, c=2.5, d=2.5 ---------
A = np.array([0.0, 0.0])
D = np.array([2.5, 0.0])
fb = FourBarGeometry(1.0, 2.5, 2.5, 2.5)
sw = fb.sweep(0.0, 360.0, 720, branch=1)

k0 = int(round(45.0 / 0.5))  # reference at crank angle 45 deg
t2_0 = math.radians(45.0)
t4_0 = float(sw.theta4[k0])
B0 = A + 1.0 * np.array([math.cos(t2_0), math.sin(t2_0)])
C0 = D + 2.5 * np.array([math.cos(t4_0), math.sin(t4_0)])
E0 = C0 + 0.6 * (D - C0)  # point on link L3 (C-D)

# --- loop 2: dyad E - F - G with G chosen and L sized for reachability ----
G = np.array([1.25, -2.5])
Cx = D[0] + 2.5 * np.cos(sw.theta4)
Cy = D[1] + 2.5 * np.sin(sw.theta4)
Ex = Cx + 0.6 * (D[0] - Cx)
Ey = Cy + 0.6 * (D[1] - Cy)
max_d = float(np.max(np.hypot(Ex - G[0], Ey - G[1])))
L = 0.5 * max_d * 1.05  # equal dyad links, always reachable

EG = G - E0
dEG = float(np.linalg.norm(EG))
mid = (E0 + G) / 2.0
h = math.sqrt(max(L * L - (dEG / 2.0) ** 2, 0.0))
perp = np.array([-EG[1], EG[0]]) / dEG
F0 = mid + h * perp

link_joints = {
    "ground": ["A", "D", "G"],
    "L1": ["A", "B"],
    "L2": ["B", "C"],
    "L3": ["C", "D", "E"],
    "L4": ["E", "F"],
    "L5": ["F", "G"],
}
joint_pos = {
    "A": A.tolist(),
    "D": D.tolist(),
    "G": G.tolist(),
    "B": B0.tolist(),
    "C": C0.tolist(),
    "E": E0.tolist(),
    "F": F0.tolist(),
}
joint_types = {jid: "revolute" for jid in joint_pos}

driver = {"joint": "A", "start": 45.0, "span": 360.0, "steps": 721, "seed": 0}
poi = {"link": "L4", "local": [L, 0.0]}  # traces joint F

mech = build_mechanism("sixbar_watt", "ground", link_joints, joint_pos, joint_types, driver, poi)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sixbar_watt.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(mech.to_dict(), f, indent=2)
print("saved:", out)

req = {
    "driver_sweep": {"start": 45.0, "span": 360.0, "steps": 721, "seed": 0},
    "constraints": [
        {"type": "crank_rotatability", "hard": False},
        {"type": "singularity_margin_min", "value": 0.05, "hard": False, "weight": 1.0},
    ],
}
res = verify(mech, req)
print(res.summary())
if res.trace:
    print("lost_at:", res.trace["lost_at"])
