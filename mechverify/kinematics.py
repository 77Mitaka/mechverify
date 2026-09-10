"""Layer 1: analytic kinematics for planar four-bar linkages.

The four-bar loop is solved in closed form via the Freudenstein equation.
A vectorised sweep provides the fast, deterministic "filter" layer of the
verifier: for a given driver angle range it returns joint angles, the coupler
point trajectory, the transmission angle and singularity information.

Loop convention (ground frame):
    A = (0, 0)                       ground pivot (crank)
    D = (d, 0)                       ground pivot (rocker)
    B = A + a * e^{i theta2}         crank-coupler joint
    C = D + c * e^{i theta4}         rocker-coupler joint
    B + b * e^{i theta3} = C         loop closure
where a = |AB| (crank), b = |BC| (coupler), c = |DC| (rocker), d = |AD| (ground).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .ir import Mechanism


def wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angdist(a: float, b: float) -> float:
    return abs(wrap(a - b))


@dataclass
class FourBarGeometry:
    a: float  # crank AB
    b: float  # coupler BC
    c: float  # rocker DC
    d: float  # ground AD
    coupler_p: float = 0.0
    coupler_phi: float = 0.0  # radians

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d"):
            if getattr(self, name) <= 0:
                raise ValueError("link length %s must be positive" % name)

    # -------------------------------------------------- scalar closed-form
    def theta4_candidates(self, theta2: float) -> List[float]:
        """Return the (0, 1 or 2) assembly solutions for theta4."""
        a, b, c, d = self.a, self.b, self.c, self.d
        k1 = d / a
        k2 = d / c
        k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)
        A = k1 - math.cos(theta2)
        B = -math.sin(theta2)
        C = k2 * math.cos(theta2) - k3
        R = math.hypot(A, B)
        if R < 1e-12:
            return []
        ratio = C / R
        if ratio < -1.0 - 1e-9 or ratio > 1.0 + 1e-9:
            return []
        ratio = max(-1.0, min(1.0, ratio))
        base = math.atan2(B, A)
        delta = math.acos(ratio)
        return [base + delta, base - delta]

    def theta3(self, theta2: float, theta4: float) -> float:
        a, b, c, d = self.a, self.b, self.c, self.d
        x = d + c * math.cos(theta4) - a * math.cos(theta2)
        y = c * math.sin(theta4) - a * math.sin(theta2)
        return math.atan2(y, x)

    def solve_initial(self, theta2: float, branch: int = 1) -> Optional[Tuple[float, float]]:
        cands = self.theta4_candidates(theta2)
        if not cands:
            return None
        theta4 = cands[0] if branch >= 0 else cands[-1]
        return self.theta3(theta2, theta4), theta4

    def solve_continuation(
        self, theta2: float, theta4_prev: float, max_jump: float = math.radians(40.0)
    ) -> Optional[Tuple[float, float]]:
        """Pick the assembly branch closest to the previous solution."""
        cands = self.theta4_candidates(theta2)
        if not cands:
            return None
        best = min(cands, key=lambda t4: angdist(t4, theta4_prev))
        if angdist(best, theta4_prev) > max_jump:
            return None
        return self.theta3(theta2, best), best

    # ------------------------------------------------------- observables
    def coupler_point(self, theta2: float, theta3: float) -> Tuple[float, float]:
        bx = self.a * math.cos(theta2)
        by = self.a * math.sin(theta2)
        ang = theta3 + self.coupler_phi
        return (bx + self.coupler_p * math.cos(ang), by + self.coupler_p * math.sin(ang))

    @staticmethod
    def transmission_angle(theta3: float, theta4: float) -> float:
        """Acute angle between coupler and rocker, in [0, pi/2]."""
        mu = abs(wrap(theta4 - theta3))
        if mu > math.pi / 2:
            mu = math.pi - mu
        return mu

    @staticmethod
    def singularity_metric(theta3: float, theta4: float) -> float:
        """|sin(theta4 - theta3)|; zero at a dead point."""
        return abs(math.sin(theta4 - theta3))

    # --------------------------------------------------------- fast sweep
    def sweep(
        self,
        start_deg: float = 0.0,
        span_deg: float = 360.0,
        steps: int = 720,
        branch: int = 1,
    ) -> "SweepResult":
        """Vectorised driver sweep; the fast-filter primitive of the verifier."""
        t2 = np.radians(np.linspace(start_deg, start_deg + span_deg, steps + 1))
        a, b, c, d = self.a, self.b, self.c, self.d
        k1 = d / a
        k2 = d / c
        k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)

        A = k1 - np.cos(t2)
        B = -np.sin(t2)
        C = k2 * np.cos(t2) - k3
        R = np.hypot(A, B)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(R > 1e-12, C / np.where(R > 1e-12, R, 1.0), np.nan)

        valid = np.isfinite(ratio) & (np.abs(ratio) <= 1.0 + 1e-9)
        ratio_c = np.clip(np.nan_to_num(ratio, nan=2.0), -1.0, 1.0)
        delta = np.arccos(ratio_c)
        base = np.arctan2(B, A)
        t4 = base + branch * delta
        t3 = np.arctan2(c * np.sin(t4) - a * np.sin(t2), d + c * np.cos(t4) - a * np.cos(t2))

        bx = a * np.cos(t2)
        by = a * np.sin(t2)
        ang = t3 + self.coupler_phi
        px = bx + self.coupler_p * np.cos(ang)
        py = by + self.coupler_p * np.sin(ang)

        diff = np.arctan2(np.sin(t4 - t3), np.cos(t4 - t3))
        mu = np.abs(diff)
        mu = np.where(mu > np.pi / 2, np.pi - mu, mu)

        # mask invalid samples
        for arr in (t3, t4, px, py, mu):
            arr[~valid] = np.nan

        completed = bool(valid.all() and span_deg >= 360.0 - 1e-6)
        lost_at = None if valid.all() else int(np.argmax(~valid))

        return SweepResult(
            theta2=t2,
            theta3=t3,
            theta4=t4,
            px=px,
            py=py,
            mu_deg=np.degrees(mu),
            valid=valid,
            completed=completed,
            lost_at=lost_at,
            n=steps + 1,
        )

    # -------------------------------------------------- loop residual
    def loop_residual(self, theta2: float, theta3: float, theta4: float) -> float:
        a, b, c, d = self.a, self.b, self.c, self.d
        zx = a * math.cos(theta2) + b * math.cos(theta3) - d - c * math.cos(theta4)
        zy = a * math.sin(theta2) + b * math.sin(theta3) - c * math.sin(theta4)
        return math.hypot(zx, zy)


@dataclass
class SweepResult:
    theta2: np.ndarray
    theta3: np.ndarray
    theta4: np.ndarray
    px: np.ndarray
    py: np.ndarray
    mu_deg: np.ndarray
    valid: np.ndarray
    completed: bool
    lost_at: Optional[int]
    n: int

    def points(self) -> np.ndarray:
        return np.column_stack([self.px, self.py])

    def valid_points(self) -> np.ndarray:
        return self.points()[self.valid]


# --------------------------------------------------------------- extraction
def four_bar_from_mechanism(mech: Mechanism) -> FourBarGeometry:
    """Extract a :class:`FourBarGeometry` from a four-bar mechanism IR."""
    links = mech.link_ids()
    if len(links) != 4:
        raise ValueError("MVP supports four-bar linkages only (got %d links)" % len(links))
    if len(mech.joints) != 4:
        raise ValueError("four-bar requires exactly 4 joints (got %d)" % len(mech.joints))
    if any(j.type != "revolute" for j in mech.joints):
        raise ValueError("MVP supports revolute joints only")

    ground = mech.ground
    adj = []
    for j in mech.ground_joints():
        other = [x for x in j.links if x != ground]
        if len(other) != 1:
            raise ValueError("ground joint %s must connect ground to one link" % j.id)
        adj.append(other[0])
    if len(adj) != 2:
        raise ValueError("ground must connect to exactly two links")

    driver_link = (mech.driver or {}).get("link")
    if driver_link not in adj:
        driver_link = adj[0]
    rocker = [x for x in adj if x != driver_link][0]
    coupler = [x for x in links if x not in (ground, driver_link, rocker)]
    if len(coupler) != 1:
        raise ValueError("could not identify a unique coupler link")
    coupler = coupler[0]

    try:
        a = float(mech.dimensions[driver_link])
        b = float(mech.dimensions[coupler])
        c = float(mech.dimensions[rocker])
    except KeyError as exc:
        raise ValueError("missing dimension for link %s" % exc) from exc

    d = _ground_length(mech)
    cp = mech.coupler_point
    if cp is None:
        coupler_p, coupler_phi = 0.0, 0.0
    else:
        if cp.link not in (coupler, ""):
            raise ValueError("coupler_point.link must reference the coupler link")
        coupler_p, coupler_phi = float(cp.p), float(cp.phi)

    return FourBarGeometry(a=a, b=b, c=c, d=d, coupler_p=coupler_p, coupler_phi=coupler_phi)


def _ground_length(mech: Mechanism) -> float:
    gj = mech.ground_joints()
    if len(gj) != 2:
        raise ValueError("expected exactly two ground joints")
    p0, p1 = gj[0].position, gj[1].position
    if p0 is None or p1 is None:
        raise ValueError("ground joints must carry a 'position'")
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
