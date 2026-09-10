"""The verifier orchestrator.

    verify(mechanism, requirement) -> VerifyResult

Pipeline:
    Layer 0  structural checks            (structural.py)
    Layer 1  analytic four-bar sweep      (kinematics.py)
             requirement evaluation       (requirement.py)
    aggregate -> status + score + diagnostics (+ optional trace)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .ir import Mechanism
from .kinematics import four_bar_from_mechanism
from .requirement import evaluate
from .structural import structural_checks


@dataclass
class VerifyResult:
    status: str  # 'pass' | 'fail' | 'unknown'
    score: float
    checks: List[dict] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    trace: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "checks": self.checks,
            "diagnostics": self.diagnostics,
            "trace": self.trace,
        }

    def summary(self) -> str:
        lines = ["status: %s   score: %.3f" % (self.status, self.score)]
        for c in self.checks:
            mark = "OK " if c["passed"] else "XX "
            lines.append(
                "  [%s] %-24s %s" % (mark, c["name"], c.get("detail", ""))
            )
        for d in self.diagnostics:
            lines.append("  (diag) %s" % d)
        return "\n".join(lines)


def verify(mech: Mechanism, requirement: Optional[Dict[str, Any]] = None) -> VerifyResult:
    requirement = requirement or {}
    checks: List[dict] = []
    diags: List[str] = []

    # ---- Layer 0: structural -------------------------------------------
    l0, l0_diags = structural_checks(mech)
    checks.extend(l0)
    diags.extend(l0_diags)
    if _has_hard_failure(checks):
        return _finalize(checks, diags, trace=None)

    # ---- Layer 1: sweep -------------------------------------------------
    sw = dict(requirement.get("driver_sweep") or mech.driver or {})
    is_general = any(j.anchors for j in mech.joints)

    try:
        if is_general:
            from .multibody import GeneralPlanarMechanism

            solver = GeneralPlanarMechanism(mech)
            sweep = solver.sweep(
                start_deg=float(sw.get("start", 0.0)),
                span_deg=float(sw.get("span", 360.0)),
                steps=int(sw.get("steps", 360)),
                seed=int(sw.get("seed", 0)),
            )
            trace = {
                "theta_driver_deg": np.degrees(sweep.theta_driver).tolist(),
                "coupler_path": sweep.points.tolist(),
                "sigma_min": sweep.sigma_min.tolist(),
                "completed": sweep.completed,
                "lost_at": sweep.lost_at,
            }
        else:
            fb = four_bar_from_mechanism(mech)
            sweep = fb.sweep(
                start_deg=float(sw.get("start", 0.0)),
                span_deg=float(sw.get("span", 360.0)),
                steps=int(sw.get("steps", 720)),
                branch=int(sw.get("branch", 1)),
            )
            trace = {
                "theta2_deg": np.degrees(sweep.theta2).tolist(),
                "theta3_deg": np.degrees(sweep.theta3).tolist(),
                "theta4_deg": np.degrees(sweep.theta4).tolist(),
                "coupler_path": sweep.points().tolist(),
                "mu_deg": sweep.mu_deg.tolist(),
                "completed": sweep.completed,
                "lost_at": sweep.lost_at,
            }
    except Exception as exc:
        diags.append("kinematic solve failed: %s" % exc)
        return _finalize(checks, diags, trace=None)

    # ---- behaviour checks ----------------------------------------------
    try:
        checks.extend(evaluate(requirement, sweep))
    except Exception as exc:  # pragma: no cover - defensive
        diags.append("requirement evaluation failed: %s" % exc)
        return _finalize(checks, diags, trace=None)

    return _finalize(checks, diags, trace=trace)


# --------------------------------------------------------------------- utils
def _has_hard_failure(checks: List[dict]) -> bool:
    return any(c["hard"] and not c["passed"] for c in checks)


def _finalize(
    checks: List[dict], diags: List[str], trace: Optional[Dict[str, Any]]
) -> VerifyResult:
    hard_fail = _has_hard_failure(checks)
    status = "fail" if hard_fail else "pass"
    if hard_fail:
        score = 0.0
    else:
        total_w = sum(c["weight"] for c in checks) or 1.0
        score = sum(c["weight"] * c["score"] for c in checks) / total_w
    return VerifyResult(status=status, score=score, checks=checks, diagnostics=diags, trace=trace)
