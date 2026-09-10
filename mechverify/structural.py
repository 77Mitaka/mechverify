"""Layer 0: structural / combinatorial checks (microsecond, no simulation).

These are the "compiler-level" checks of the verifier: they reject
structurally invalid mechanisms before any kinematics are attempted.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .ir import Mechanism
from .kinematics import four_bar_from_mechanism


def mobility(n_links: int, n_lower_pairs: int, n_higher_pairs: int = 0) -> int:
    """Planar Grubler/Kutzbach mobility."""
    return 3 * (n_links - 1) - 2 * n_lower_pairs - n_higher_pairs


def grashof(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    s, p, q, l = sorted([a, b, c, d])
    return {
        "s": s,
        "p": p,
        "q": q,
        "l": l,
        "is_grashof": (s + l) <= (p + q) + 1e-9,
    }


def structural_checks(mech: Mechanism) -> Tuple[List[dict], List[str]]:
    """Return (checks, diagnostics) for the structural layer."""
    checks: List[dict] = []
    diags: List[str] = []

    n_links = len(mech.link_ids())
    n_joints = len(mech.joints)

    # 1) mobility -----------------------------------------------------
    m = mobility(n_links, n_joints)
    checks.append(
        _check(
            name="mobility",
            value=float(m),
            threshold=1.0,
            passed=(m == 1),
            detail="Grubler mobility M=%d (expected 1 for a single-input mechanism)" % m,
        )
    )

    # 2) positive dimensions -----------------------------------------
    bad = [k for k, v in mech.dimensions.items() if v <= 0]
    checks.append(
        _check(
            name="positive_dimensions",
            value=float(len(bad)),
            threshold=0.0,
            passed=(len(bad) == 0),
            detail="non-positive lengths: %s" % (bad or "none"),
        )
    )

    # 3) four-bar specific checks ------------------------------------
    is_general = any(j.anchors for j in mech.joints)
    if is_general:
        diags.append("general mechanism: four-bar specific checks skipped")
        return checks, diags

    try:
        fb = four_bar_from_mechanism(mech)
    except ValueError as exc:
        diags.append("not a four-bar: %s" % exc)
        checks.append(
            _check(
                name="four_bar_structure",
                value=0.0,
                threshold=1.0,
                passed=False,
                detail=str(exc),
            )
        )
        return checks, diags

    g = grashof(fb.a, fb.b, fb.c, fb.d)
    checks.append(
        _check(
            name="grashof",
            value=1.0 if g["is_grashof"] else 0.0,
            threshold=1.0,
            passed=bool(g["is_grashof"]),
            detail="s+l=%.4g <= p+q=%.4g -> %s"
            % (g["s"] + g["l"], g["p"] + g["q"], "Grashof" if g["is_grashof"] else "non-Grashof"),
        )
    )

    # closure feasibility: longest link must not exceed the sum of others
    lengths = [fb.a, fb.b, fb.c, fb.d]
    longest, rest = max(lengths), sum(lengths) - max(lengths)
    checks.append(
        _check(
            name="closure_feasible",
            value=float(longest),
            threshold=float(rest),
            passed=(longest <= rest + 1e-9),
            detail="longest=%.4g, sum of others=%.4g" % (longest, rest),
        )
    )

    return checks, diags


def _check(
    name: str,
    value: float,
    threshold: float,
    passed: bool,
    detail: str = "",
    hard: bool = True,
    weight: float = 1.0,
) -> dict:
    return {
        "layer": "structural",
        "name": name,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
        "hard": hard,
        "weight": weight,
        "score": 1.0 if passed else 0.0,
        "detail": detail,
    }
