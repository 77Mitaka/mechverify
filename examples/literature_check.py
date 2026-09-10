"""Independent calibration against classical (textbook) closed-form results.

No third-party simulation tool is used; the reference values are classical
exact results from the mechanism literature:

  (1) Slider-crank displacement (e.g., Norton, "Design of Machinery"):
          x(theta) = r*cos(theta) + sqrt(l^2 - (r*sin(theta))^2)
  (2) Four-bar limit (toggle) positions: the rocker's extreme angles occur when
      the crank and coupler are collinear; their span follows from the law of
      cosines (standard textbook result, e.g., Erdman & Sandor):
          span = acos((c^2+d^2-(a+b)^2)/(2cd)) - acos((c^2+d^2-(b-a)^2)/(2cd))

We compare the outputs of our own solvers to these closed-form references and
report the maximum deviation.

Run:  python examples/literature_check.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechverify import Mechanism
from mechverify.kinematics import FourBarGeometry
from mechverify.modelica import ensure_general
from mechverify.multibody import GeneralPlanarMechanism

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def slider_crank_check():
    """Constraint-based multibody solver vs the textbook displacement formula."""
    mech = Mechanism.from_json_file(os.path.join(EX, "slider_crank.json"))
    gpm = GeneralPlanarMechanism(ensure_general(mech))
    start = float((mech.driver or {}).get("start", 0.0))
    sw = gpm.sweep(start, 360.0, 720)
    theta = sw.theta_driver
    x = sw.points[:, 0]
    a, b = 1.0, 2.5
    x_ref = a * np.cos(theta) + np.sqrt(b * b - (a * np.sin(theta)) ** 2)
    return float(np.nanmax(np.abs(x - x_ref)))


def fourbar_limit_check():
    """Rocker span of a four-bar vs the law-of-cosines limit positions."""
    a, b, c, d = 1.0, 2.5, 2.5, 2.5
    fb = FourBarGeometry(a, b, c, d)
    sw = fb.sweep(0.0, 360.0, 3600, branch=1)
    t4 = np.degrees(sw.theta4[sw.valid])
    span = float(t4.max() - t4.min())
    ext = math.degrees(math.acos((c * c + d * d - (a + b) ** 2) / (2 * c * d)))
    fold = math.degrees(math.acos((c * c + d * d - (b - a) ** 2) / (2 * c * d)))
    return abs(span - (ext - fold))


def main():
    print("slider-crank vs textbook formula   : max_err = %.3e" % slider_crank_check())
    print("four-bar rocker span vs law-cosines : max_err = %.3e deg" % fourbar_limit_check())


if __name__ == "__main__":
    main()
