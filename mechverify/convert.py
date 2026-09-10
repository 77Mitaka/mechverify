"""Modelica -> mechanism IR conversion.

Lets the project reuse existing graphical Modelica tools (OMEdit, Dymola,
MapleSim) instead of building a front-end: a model exported from such a tool is
read back into our IR and then verified.

Two paths:
1. If the model carries the source IR (our generator embeds it as
   ``// mechverify-ir: {...}``), it is recovered exactly.
2. Otherwise the flattened planar form is parsed from the declarations and
   constraint equations (revolute joints).
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Tuple

from .ir import Joint, Link, Mechanism

_GROUND = "ground"


def modelica_to_mechanism(mo_text: str) -> Mechanism:
    m = re.search(r"//\s*mechverify-ir:\s*(\{.*\})\s*$", mo_text, re.MULTILINE)
    if m:
        return Mechanism.from_dict(json.loads(m.group(1)))
    return _parse_flattened_revolute(mo_text)


def _split_wrapped(s):
    s = s.strip()
    if s[0] != "(":
        raise ValueError("bad expression: " + s)
    depth = 0
    end = -1
    for i in range(len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError("unbalanced parentheses: " + s)
    left = s[1:end]
    tail = s[end + 1:].strip()
    if tail[0] != "-":
        raise ValueError("bad separator: " + s)
    tail = tail[1:].strip()
    right = tail[1:-1]
    return left, right


def _parse_anchor(expr, coord):
    expr = expr.strip()
    m = re.fullmatch(r"\(([-\d.eE+]+)\)", expr)
    if m:
        return None, float(m.group(1)), None
    if coord == "x":
        pat = r"(\w+)_x \+ \(([-\d.eE+]+)\)\*cos\(\1_th\) - \(([-\d.eE+]+)\)\*sin\(\1_th\)"
        m = re.fullmatch(pat, expr)
        if m:
            return m.group(1), float(m.group(2)), float(m.group(3))
    else:
        pat = r"(\w+)_y \+ \(([-\d.eE+]+)\)\*sin\(\1_th\) \+ \(([-\d.eE+]+)\)\*cos\(\1_th\)"
        m = re.fullmatch(pat, expr)
        if m:
            return m.group(1), float(m.group(2)), float(m.group(3))
    raise ValueError("cannot parse anchor expression: " + expr)


def _parse_flattened_revolute(mo_text: str) -> Mechanism:
    name = re.search(r"\bmodel\s+(\w+)", mo_text)
    if not name:
        raise ValueError("no model declaration found")
    model_name = name.group(1)

    poses: Dict[str, Dict[str, float]] = {}
    for coord in ("x", "y", "th"):
        pat = r"Real\s+(\w+)_" + coord + r"\(start=([-\d.eE+]+)(?:,\s*fixed=\w+)?\);"
        for m in re.finditer(pat, mo_text):
            poses.setdefault(m.group(1), {})[coord] = float(m.group(2))
    if not poses:
        raise ValueError("no movable link states found")

    body = mo_text.split("equation", 1)[1]
    body = re.split(r"annotation\s*\(", body)[0]
    lines = [ln.strip() for ln in body.splitlines() if ln.strip().endswith(";")]

    driver_links = None
    driver_start = 0.0
    for ln in lines:
        md = re.search(r"driver_angle\s*=\s*([-\d.eE+]+)\s*\+\s*driver_omega\*time;", ln)
        if md:
            driver_start = float(md.group(1))
        mc = re.match(r"\((\w+_th|0\.0)\)\s*-\s*\((\w+_th|0\.0)\)\s*=\s*driver_angle;", ln)
        if mc:
            lb = _GROUND if mc.group(1) == "0.0" else mc.group(1)[:-3]
            la = _GROUND if mc.group(2) == "0.0" else mc.group(2)[:-3]
            driver_links = [la, lb]

    const_lines = [ln for ln in lines if ln[0:6] == "0.0 = "]
    if len(const_lines) % 2 != 0:
        raise ValueError("unexpected number of constraint equations")

    joints: List[Joint] = []
    for k in range(0, len(const_lines), 2):
        lx = const_lines[k]
        ly = const_lines[k + 1]
        ax_str, bx_str = _split_wrapped(lx[6:len(lx) - 1])
        ay_str, by_str = _split_wrapped(ly[6:len(ly) - 1])
        lA, axA, _ = _parse_anchor(ax_str, "x")
        lB, axB, _ = _parse_anchor(bx_str, "x")
        lAy, _, ayA = _parse_anchor(ay_str, "y")
        lBy, _, ayB = _parse_anchor(by_str, "y")
        la = _GROUND if lA is None else lA
        lb = _GROUND if lB is None else lB
        anchor_a = [
            float(axA if lA is not None else _coord_value(ax_str)),
            float(ayA if lAy is not None else _coord_value(ay_str)),
        ]
        anchor_b = [
            float(axB if lB is not None else _coord_value(bx_str)),
            float(ayB if lBy is not None else _coord_value(by_str)),
        ]
        jid = "J" + str(len(joints) + 1)
        joints.append(
            Joint(id=jid, type="revolute", links=[la, lb], anchors={la: anchor_a, lb: anchor_b})
        )

    poi = None
    px = re.search(r"poi_x\s*=\s*(.+);", mo_text)
    py = re.search(r"poi_y\s*=\s*(.+);", mo_text)
    if px and py:
        link, lx, _ = _parse_anchor(px.group(1), "x")
        _, _, ly = _parse_anchor(py.group(1), "y")
        if link is not None:
            poi = {"link": link, "local": [float(lx), float(ly)]}

    driver = None
    if driver_links:
        for j in joints:
            if set(j.links) == set(driver_links):
                driver = {
                    "joint": j.id,
                    "start": math.degrees(driver_start),
                    "span": 360.0,
                    "steps": 721,
                }
                break
        if driver is None:
            raise ValueError("driver joint not found among constraints")

    links = [Link(id=_GROUND, kind="ground")]
    for lid, p in poses.items():
        links.append(Link(id=lid, pose=[p.get("x", 0.0), p.get("y", 0.0), p.get("th", 0.0)]))

    return Mechanism(
        id=model_name,
        links=links,
        joints=joints,
        ground=_GROUND,
        dimensions={},
        point_of_interest=poi,
        driver=driver,
    )


def _coord_value(expr):
    m = re.fullmatch(r"\(([-\d.eE+]+)\)", expr.strip())
    if not m:
        raise ValueError("expected a constant ground coordinate: " + expr)
    return float(m.group(1))
