"""IR -> PlanarMechanics code generation (library-based backend).

Generates idiomatic PlanarMechanics models (Body / FixedTranslation / Fixed /
Revolute / Prismatic + connect), so the result can be opened, animated and
simulated in OMEdit/Dymola.  The input joint of the mechanism is driven by a
rotational Position source.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

from .ir import Mechanism
from .modelica import OMEGA, ensure_general, initial_state


def _ident(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", s)


def _theta(state: Dict, link: str, ground: str) -> float:
    return 0.0 if link == ground else float(state[link][2])


def _pose(state: Dict, link: str, ground: str):
    if link == ground:
        return (0.0, 0.0, 0.0)
    return tuple(state[link])


def _world_point(pose, local):
    x, y, th = pose
    lx, ly = local
    return (x + lx * math.cos(th) - ly * math.sin(th), y + lx * math.sin(th) + ly * math.cos(th))


def _prismatic_s0(state, ground, la, lb, anchors, axis) -> float:
    pa = _world_point(_pose(state, la, ground), anchors[la])
    pb = _world_point(_pose(state, lb, ground), anchors[lb])
    e = axis[la]
    th = _theta(state, la, ground)
    ew = (e[0] * math.cos(th) - e[1] * math.sin(th), e[0] * math.sin(th) + e[1] * math.cos(th))
    return (pb[0] - pa[0]) * ew[0] + (pb[1] - pa[1]) * ew[1]


def generate_planar_mechanics(
    mech: Mechanism,
    driver_start_deg: float = None,
    driver_omega: float = OMEGA,
    stop_time: float = 1.0,
    interval: float = 0.002,
    animate: bool = False,
) -> str:
    gen = ensure_general(mech)
    ground = gen.ground
    movable = [l.id for l in gen.links if l.id != ground]
    sw = dict(gen.driver or {})
    if driver_start_deg is None:
        driver_start_deg = float(sw.get("start", 0.0))
    state = initial_state(gen, driver_start_deg, seed=int(sw.get("seed", 0)))
    dj = next(j for j in gen.joints if j.id == gen.driver["joint"])

    model = _ident(gen.id)
    L: List[str] = []
    L.append("model %s" % model)
    L.append(
        "  inner PlanarMechanics.PlanarWorld planarWorld(enableAnimation=%s);"
        % ("true" if animate else "false")
    )

    # ground anchors
    for j in gen.joints:
        for link in j.links:
            if link == ground:
                g = j.anchors[link]
                L.append(
                    "  PlanarMechanics.Parts.Fixed fixed_%s(r={%.12g,%.12g});"
                    % (_ident(j.id), g[0], g[1])
                )
    # bodies
    for link in movable:
        L.append(
            "  PlanarMechanics.Parts.Body body_%s(m=1, I=0.01, enableGravity=false);" % _ident(link)
        )
    # fixed translations for each movable-link anchor
    for j in gen.joints:
        for link in j.links:
            if link == ground:
                continue
            a = j.anchors[link]
            L.append(
                "  PlanarMechanics.Parts.FixedTranslation ft_%s_%s(r={%.12g,%.12g});"
                % (_ident(j.id), _ident(link), a[0], a[1])
            )
    # joints
    for j in gen.joints:
        jid = _ident(j.id)
        la, lb = j.links
        if j.type == "revolute":
            phi0 = _theta(state, lb, ground) - _theta(state, la, ground)
            if j.id == dj.id:
                L.append(
                    "  PlanarMechanics.Joints.Revolute rev_%s(useFlange=true, phi(start=%.12g, fixed=true), stateSelect=StateSelect.always);"
                    % (jid, phi0)
                )
            else:
                L.append("  PlanarMechanics.Joints.Revolute rev_%s(phi(start=%.12g));" % (jid, phi0))
        elif j.type == "prismatic":
            r = j.axis[la]
            s0 = _prismatic_s0(state, ground, la, lb, j.anchors, j.axis)
            L.append(
                "  PlanarMechanics.Joints.Prismatic pri_%s(r={%.12g,%.12g}, s(start=%.12g));"
                % (jid, r[0], r[1], s0)
            )
        else:
            raise ValueError("unsupported joint type: %s" % j.type)
    # driver
    dphi0 = _theta(state, dj.links[1], ground) - _theta(state, dj.links[0], ground)
    L.append("  Modelica.Mechanics.Rotational.Sources.Position drv(exact=true);")
    L.append(
        "  Modelica.Blocks.Sources.Ramp ramp(height=%.12g, duration=%.12g, offset=%.12g);"
        % (driver_omega * stop_time, stop_time, dphi0)
    )
    if gen.point_of_interest:
        L.append("  output Real poi_x;")
        L.append("  output Real poi_y;")
    L.append("equation")

    # connect joints
    for j in gen.joints:
        jid = _ident(j.id)
        comp = ("rev_" if j.type == "revolute" else "pri_") + jid
        for side, link in (("a", j.links[0]), ("b", j.links[1])):
            if link == ground:
                L.append("  connect(fixed_%s.frame, %s.frame_%s);" % (jid, comp, side))
            else:
                L.append("  connect(ft_%s_%s.frame_b, %s.frame_%s);" % (jid, _ident(link), comp, side))
                L.append(
                    "  connect(body_%s.frame_a, ft_%s_%s.frame_a);" % (_ident(link), jid, _ident(link))
                )
    # driver connections
    L.append("  connect(drv.flange, rev_%s.flange_a);" % _ident(dj.id))
    L.append("  connect(ramp.y, drv.phi_ref);")
    # poi
    if gen.point_of_interest:
        link = _ident(gen.point_of_interest["link"])
        lx, ly = gen.point_of_interest["local"]
        L.append(
            "  poi_x = body_%s.r[1] + (%.12g)*cos(body_%s.phi) - (%.12g)*sin(body_%s.phi);"
            % (link, lx, link, ly, link)
        )
        L.append(
            "  poi_y = body_%s.r[2] + (%.12g)*sin(body_%s.phi) + (%.12g)*cos(body_%s.phi);"
            % (link, lx, link, ly, link)
        )
    L.append("  annotation(experiment(StopTime=%.12g, Interval=%.12g));" % (stop_time, interval))
    L.append("end %s;" % model)
    return "\n".join(L) + "\n"
