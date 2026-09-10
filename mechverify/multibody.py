"""Layer 1 (general): constraint-based planar kinematics for lower-pair
mechanisms with revolute and prismatic joints.

Model
-----
Each movable link is a rigid body with three generalized coordinates
``(x, y, theta)`` (the ground link is fixed).  Joints contribute scalar
constraint equations:

* revolute  : the two anchor points coincide                -> 2 equations
* prismatic : relative orientation fixed, and the anchor
              offset stays perpendicular to the axis        -> 2 equations

The driver adds one equation (relative angle of its joint).  For a 1-DOF
mechanism the system is square: 3*(n-1) unknowns vs 2*joints + 1 equations.

Solution
--------
A damped Newton iteration with a numerical Jacobian solves the position
problem; the driver is swept by continuation (warm start from the previous
configuration), which also fixes the assembly branch.  The smallest singular
value of the constraint Jacobian is reported as a distance-to-singularity
metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .ir import Mechanism


def _R(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass
class GeneralSweepResult:
    theta_driver: np.ndarray
    points: np.ndarray  # (N, 2) point-of-interest trajectory
    sigma_min: np.ndarray
    valid: np.ndarray
    completed: bool
    lost_at: Optional[int]
    n: int
    mu_deg: Optional[np.ndarray] = None  # not defined for general mechanisms
    configs: Optional[np.ndarray] = None  # (N, n_unknowns) solved configurations

    def valid_points(self) -> np.ndarray:
        return self.points[self.valid]


class GeneralPlanarMechanism:
    def __init__(self, mech: Mechanism):
        self.mech = mech
        self.ground = mech.ground
        self.movable = [l.id for l in mech.links if l.id != self.ground]
        self.index = {lid: i for i, lid in enumerate(self.movable)}
        self.joints = mech.joints
        if not mech.driver or "joint" not in mech.driver:
            raise ValueError("general solver requires a driver joint")
        self.driver_joint = next(j for j in mech.joints if j.id == mech.driver["joint"])
        self._poi = mech.point_of_interest
        if self._poi is None:
            raise ValueError("general solver requires 'point_of_interest'")

    # ---------------------------------------------------------- dimensions
    def n_unknowns(self) -> int:
        return 3 * len(self.movable)

    def n_equations(self) -> int:
        return 2 * len(self.joints) + 1

    def _pose(self, q: np.ndarray, link: str):
        if link == self.ground:
            return np.zeros(2), 0.0
        i = self.index[link] * 3
        return q[i : i + 2], float(q[i + 2])

    def world_point(self, q: np.ndarray, link: str, local) -> np.ndarray:
        p, th = self._pose(q, link)
        return p + _R(th) @ np.asarray(local, dtype=float)

    # ------------------------------------------------------------ residual
    def residuals(self, q: np.ndarray, driver_value: float) -> np.ndarray:
        res: List[float] = []
        for j in self.joints:
            la, lb = j.links[0], j.links[1]
            if j.type == "revolute":
                pa = self.world_point(q, la, j.anchors[la])
                pb = self.world_point(q, lb, j.anchors[lb])
                res.extend((pa - pb).tolist())
            elif j.type == "prismatic":
                aa = np.asarray(j.axis[la], dtype=float)
                ab = np.asarray(j.axis[lb], dtype=float)
                _, tha = self._pose(q, la)
                _, thb = self._pose(q, lb)
                ang_a = math.atan2(aa[1], aa[0])
                ang_b = math.atan2(ab[1], ab[0])
                res.append((thb + ang_b) - (tha + ang_a))
                pa = self.world_point(q, la, j.anchors[la])
                pb = self.world_point(q, lb, j.anchors[lb])
                axis_world = _R(tha) @ aa
                d = pb - pa
                res.append(axis_world[0] * d[1] - axis_world[1] * d[0])
            else:
                raise ValueError("unsupported joint type: %s" % j.type)

        la, lb = self.driver_joint.links
        _, tha = self._pose(q, la)
        _, thb = self._pose(q, lb)
        res.append((thb - tha) - driver_value)
        return np.asarray(res, dtype=float)

    def jacobian(self, q: np.ndarray, driver_value: float, h: float = 1e-6):
        m = self.n_unknowns()
        F0 = self.residuals(q, driver_value)
        J = np.zeros((len(F0), m))
        for k in range(m):
            qp = q.copy()
            qp[k] += h
            J[:, k] = (self.residuals(qp, driver_value) - F0) / h
        return J, F0

    # -------------------------------------------------------------- solve
    def _base_guess(self) -> np.ndarray:
        q = np.zeros(self.n_unknowns())
        for l in self.mech.links:
            if l.id == self.ground:
                continue
            i = self.index[l.id] * 3
            pose = l.pose or [0.0, 0.0, 0.0]
            q[i : i + 3] = pose
        return q

    def solve(
        self, driver_value: float, guess: Optional[np.ndarray] = None, tol: float = 1e-11,
        max_iter: int = 120,
    ):
        q = self._base_guess() if guess is None else guess.copy()
        for _ in range(max_iter):
            J, F = self.jacobian(q, driver_value)
            if np.linalg.norm(F) < tol:
                return q, True
            try:
                dq = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, -F, rcond=None)
            # simple backtracking line search
            step = 1.0
            for _ in range(12):
                qn = q + step * dq
                if np.linalg.norm(self.residuals(qn, driver_value)) < np.linalg.norm(F):
                    break
                step *= 0.5
            q = q + step * dq
        _, F = self.jacobian(q, driver_value)
        return q, bool(np.linalg.norm(F) < 1e-7)

    def solve_multistart(self, driver_value: float, restarts: int = 60, seed: int = 0):
        rng = np.random.default_rng(seed)
        base = self._base_guess()
        best = None
        for k in range(restarts + 1):
            g = base.copy() if k == 0 else base + rng.normal(0.0, 2.0, size=base.shape)
            q, ok = self.solve(driver_value, g)
            if not ok:
                continue
            r = float(np.linalg.norm(self.residuals(q, driver_value)))
            if best is None or r < best[0]:
                best = (r, q)
        return None if best is None else best[1]

    # --------------------------------------------------------------- sweep
    def poi(self, q: np.ndarray) -> np.ndarray:
        return self.world_point(q, self._poi["link"], self._poi["local"])

    def sigma_min(self, q: np.ndarray, driver_value: float) -> float:
        J, _ = self.jacobian(q, driver_value)
        sv = np.linalg.svd(J, compute_uv=False)
        return float(sv[-1])

    def sweep(
        self,
        start_deg: float = 0.0,
        span_deg: float = 360.0,
        steps: int = 360,
        seed: int = 0,
    ) -> GeneralSweepResult:
        start = math.radians(start_deg)
        span = math.radians(span_deg)
        vals = np.linspace(start, start + span, steps + 1)

        q = self.solve_multistart(vals[0], seed=seed)
        if q is None:
            return GeneralSweepResult(
                vals, np.full((steps + 1, 2), np.nan), np.full(steps + 1, np.nan),
                np.zeros(steps + 1, dtype=bool), False, 0, steps + 1,
            )

        pts = np.full((steps + 1, 2), np.nan)
        sig = np.full(steps + 1, np.nan)
        valid = np.zeros(steps + 1, dtype=bool)
        configs = np.full((steps + 1, self.n_unknowns()), np.nan)
        lost_at = None
        for k, val in enumerate(vals):
            if k > 0:
                q, ok = self.solve(val, q)
                if not ok:
                    lost_at = k
                    break
            pts[k] = self.poi(q)
            sig[k] = self.sigma_min(q, val)
            configs[k] = q
            valid[k] = True

        completed = bool(valid.all() and span_deg >= 360.0 - 1e-6)
        return GeneralSweepResult(
            theta_driver=vals,
            points=pts,
            sigma_min=sig,
            valid=valid,
            completed=completed,
            lost_at=lost_at,
            n=steps + 1,
            configs=configs,
        )
