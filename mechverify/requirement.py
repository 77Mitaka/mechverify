"""Requirement schema and evaluation / scoring.

A requirement is a list of constraints over the *behaviour* of a mechanism
(the sweep produced by Layer 1).  Each constraint yields a check dict:

    {layer, name, value, threshold, passed, hard, weight, score, detail}

Hard constraints decide feasibility (pass/fail); soft constraints contribute
to a weighted score in [0, 1] that can be used as a dense reward.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .kinematics import FourBarGeometry, SweepResult

try:  # shapely is optional; only needed for target_path
    from shapely.geometry import LineString, Point
    _HAS_SHAPELY = True
except Exception:  # pragma: no cover
    _HAS_SHAPELY = False


def evaluate(requirement: Dict, sweep, fb: FourBarGeometry = None) -> List[dict]:
    checks: List[dict] = []
    for con in requirement.get("constraints", []):
        ctype = con["type"]
        if ctype == "transmission_angle_min":
            checks.append(_transmission_angle_min(con, sweep))
        elif ctype == "crank_rotatability":
            checks.append(_crank_rotatability(con, sweep))
        elif ctype == "singularity_margin_min":
            checks.append(_singularity_margin_min(con, sweep))
        elif ctype == "straightness":
            checks.append(_straightness(con, sweep))
        elif ctype == "target_path":
            checks.append(_target_path(con, sweep))
        else:
            raise ValueError("unknown constraint type: %s" % ctype)
    return checks


# --------------------------------------------------------------- metrics
def path_straightness(points: np.ndarray, window_frac: float = 0.25) -> float:
    """Best relative straightness over any window of the path.

    For each contiguous window the maximum perpendicular deviation from the
    chord is divided by the chord length; the minimum over all windows is
    returned.  Lower is straighter.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = len(pts)
    if n < 3:
        return float("inf")
    w = max(3, int(round(window_frac * n)))
    if w > n:
        w = n
    best = float("inf")
    for s in range(0, n - w + 1):
        seg = pts[s : s + w]
        p0, p1 = seg[0], seg[-1]
        chord = float(np.linalg.norm(p1 - p0))
        if chord < 1e-12:
            continue
        d = (p1 - p0) / chord
        normal = np.array([-d[1], d[0]])
        dev = np.abs((seg - p0) @ normal)
        best = min(best, float(dev.max()) / chord)
    return best


def path_deviation(points: np.ndarray, target: List[List[float]]) -> float:
    """Max distance from the sampled path to a target polyline."""
    if not _HAS_SHAPELY:
        raise RuntimeError("target_path requires shapely")
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return float("inf")
    line = LineString([tuple(t) for t in target])
    return max(line.distance(Point(float(x), float(y))) for x, y in pts)


# ----------------------------------------------------------- constraint evals
def _singularity_margin_min(con: Dict, sweep) -> dict:
    sig = getattr(sweep, "sigma_min", None)
    if sig is None:
        raise ValueError("singularity_margin_min requires the general solver")
    thr = float(con["value"])
    vals = np.asarray(sig)[np.asarray(sweep.valid)]
    val = float(np.nanmin(vals)) if vals.size else 0.0
    passed = val >= thr - 1e-12
    score = min(1.0, val / thr) if thr > 0 else 1.0
    return _mk(
        "singularity_margin_min",
        value=val,
        threshold=thr,
        passed=passed,
        hard=con.get("hard", True),
        weight=con.get("weight", 1.0),
        score=score if passed else 0.0,
        detail="min singular value %.4g (>= %.4g required)" % (val, thr),
    )


def _transmission_angle_min(con: Dict, sweep: SweepResult) -> dict:
    if getattr(sweep, "mu_deg", None) is None:
        raise ValueError("transmission_angle_min is only defined for four-bar linkages")
    thr = float(con["value"])
    valid_mu = sweep.mu_deg[sweep.valid]
    if valid_mu.size == 0:
        val = 0.0
    else:
        val = float(np.nanmin(valid_mu))
    passed = val >= thr - 1e-9
    score = min(1.0, val / thr) if thr > 0 else 1.0
    return _mk(
        "transmission_angle_min",
        value=val,
        threshold=thr,
        passed=passed,
        hard=con.get("hard", True),
        weight=con.get("weight", 1.0),
        score=score if passed else 0.0,
        detail="min transmission angle %.2f deg (>= %.2f required)" % (val, thr),
    )


def _crank_rotatability(con: Dict, sweep: SweepResult) -> dict:
    passed = bool(sweep.completed)
    detail = (
        "crank rotates a full revolution"
        if passed
        else "assembly lost at sample %s; crank cannot fully rotate" % sweep.lost_at
    )
    return _mk(
        "crank_rotatability",
        value=1.0 if passed else 0.0,
        threshold=1.0,
        passed=passed,
        hard=con.get("hard", True),
        weight=con.get("weight", 1.0),
        score=1.0 if passed else 0.0,
        detail=detail,
    )


def _straightness(con: Dict, sweep: SweepResult) -> dict:
    thr = float(con["value"])
    window = float(con.get("window_frac", 0.25))
    val = path_straightness(sweep.valid_points(), window_frac=window)
    passed = val <= thr + 1e-12
    score = 1.0 if val <= 0 else min(1.0, thr / val) if val < float("inf") else 0.0
    return _mk(
        "straightness",
        value=val,
        threshold=thr,
        passed=passed,
        hard=con.get("hard", False),
        weight=con.get("weight", 1.0),
        score=score,
        detail="best relative straightness %.5f (<= %.5f required)" % (val, thr),
    )


def _target_path(con: Dict, sweep: SweepResult) -> dict:
    thr = float(con["tolerance"])
    val = path_deviation(sweep.valid_points(), con["points"])
    passed = val <= thr + 1e-12
    score = 1.0 if val <= 0 else min(1.0, thr / val) if val < float("inf") else 0.0
    return _mk(
        "target_path",
        value=val,
        threshold=thr,
        passed=passed,
        hard=con.get("hard", False),
        weight=con.get("weight", 1.0),
        score=score,
        detail="max deviation from target path %.5f (<= %.5f required)" % (val, thr),
    )


def _mk(
    name: str,
    value: float,
    threshold: float,
    passed: bool,
    hard: bool,
    weight: float,
    score: float,
    detail: str,
) -> dict:
    return {
        "layer": "behavior",
        "name": name,
        "value": float(value),
        "threshold": float(threshold),
        "passed": bool(passed),
        "hard": bool(hard),
        "weight": float(weight),
        "score": float(max(0.0, min(1.0, score))),
        "detail": detail,
    }
