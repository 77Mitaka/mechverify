"""Layer 3: robustness analysis.

A nominal "pass" is not enough: real links have manufacturing errors.  This
layer perturbs the geometry (link lengths, joint positions/anchors) with a
Monte-Carlo sample and reports how often the requirement still holds, plus
percentiles of every behavioural metric.

    verify_robust(mechanism, requirement, samples=..., rel_tol=..., seed=...)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .ir import Mechanism
from .verifier import verify, VerifyResult


def _characteristic_length(mech: Mechanism) -> float:
    vals: List[float] = [abs(v) for v in mech.dimensions.values()]
    for j in mech.joints:
        if j.position:
            vals.extend(abs(p) for p in j.position)
        if j.anchors:
            for a in j.anchors.values():
                vals.extend(abs(p) for p in a)
    return max(vals + [1.0])


def perturb_mechanism(mech: Mechanism, rel_tol: float, rng: np.random.Generator) -> Mechanism:
    d = copy.deepcopy(mech.to_dict())
    scale = _characteristic_length(mech)
    for k, v in d.get("dimensions", {}).items():
        d["dimensions"][k] = v * (1.0 + rng.normal(0.0, rel_tol))
    for j in d.get("joints", []):
        if j.get("position"):
            j["position"] = [p + rng.normal(0.0, rel_tol * scale) for p in j["position"]]
        if j.get("anchors"):
            for link, a in j["anchors"].items():
                j["anchors"][link] = [p + rng.normal(0.0, rel_tol * scale) for p in a]
    return Mechanism.from_dict(d)


@dataclass
class RobustResult:
    status: str
    score: float
    pass_rate: float
    samples: int
    rel_tol: float
    min_pass_rate: float
    nominal: VerifyResult
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "robust status: %s   score: %.3f   pass_rate: %.3f (%d samples, rel_tol=%.3g)"
            % (self.status, self.score, self.pass_rate, self.samples, self.rel_tol)
        ]
        lines.append("  nominal: %s (score %.3f)" % (self.nominal.status, self.nominal.score))
        for name, m in self.metrics.items():
            lines.append(
                "  %-24s nominal=%.5g  p05=%.5g  p50=%.5g  p95=%.5g"
                % (name, m["nominal"], m["p05"], m["p50"], m["p95"])
            )
        for d in self.diagnostics:
            lines.append("  (diag) %s" % d)
        return "\n".join(lines)


def verify_robust(
    mech: Mechanism,
    requirement: Optional[Dict] = None,
    samples: int = 200,
    rel_tol: float = 0.002,
    seed: int = 0,
    min_pass_rate: float = 0.99,
) -> RobustResult:
    requirement = requirement or {}
    nominal = verify(mech, requirement)
    rng = np.random.default_rng(seed)

    passes = 0
    metric_vals: Dict[str, List[float]] = {}
    for _ in range(samples):
        m = perturb_mechanism(mech, rel_tol, rng)
        try:
            res = verify(m, requirement)
        except Exception:
            continue
        if res.status == "pass":
            passes += 1
        for c in res.checks:
            if c["layer"] == "behavior":
                metric_vals.setdefault(c["name"], []).append(c["value"])

    pass_rate = passes / float(samples)
    metrics: Dict[str, Dict[str, float]] = {}
    for name, vals in metric_vals.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        nom = next((c["value"] for c in nominal.checks if c["name"] == name), float("nan"))
        metrics[name] = {
            "nominal": float(nom),
            "mean": float(np.mean(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    diagnostics: List[str] = []
    if nominal.status != "pass":
        diagnostics.append("nominal verification failed")
    if pass_rate < min_pass_rate:
        diagnostics.append(
            "pass rate %.3f below required %.3f under tolerance" % (pass_rate, min_pass_rate)
        )

    status = "pass" if (nominal.status == "pass" and pass_rate >= min_pass_rate) else "fail"
    score = nominal.score * pass_rate
    return RobustResult(
        status=status,
        score=score,
        pass_rate=pass_rate,
        samples=samples,
        rel_tol=rel_tol,
        min_pass_rate=min_pass_rate,
        nominal=nominal,
        metrics=metrics,
        diagnostics=diagnostics,
    )
