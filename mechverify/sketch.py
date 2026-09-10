"""Build a general mechanism IR from a reference sketch.

A sketch gives, for a reference configuration, the world coordinates of every
joint and the connectivity (which joints belong to which link).  Local anchors
and link poses are then derived mechanically.  This is exactly what a graphical
schematic editor would emit, and it makes multi-loop mechanisms easy to define
correctly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .ir import Mechanism, Link, Joint


def build_mechanism(
    mech_id: str,
    ground: str,
    link_joints: Dict[str, List[str]],
    joint_pos: Dict[str, List[float]],
    joint_types: Dict[str, str],
    driver: Dict,
    point_of_interest: Dict,
    axes: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> Mechanism:
    links: List[Link] = []
    anchors: Dict[str, Dict[str, List[float]]] = {}

    for lid, jids in link_joints.items():
        links.append(Link(id=lid, kind=("ground" if lid == ground else "binary")))
        if lid == ground:
            for jid in jids:
                anchors.setdefault(jid, {})[lid] = list(joint_pos[jid])
            continue

        origin = np.asarray(joint_pos[jids[0]], dtype=float)
        if len(jids) >= 2:
            d = np.asarray(joint_pos[jids[1]], dtype=float) - origin
            theta = math.atan2(d[1], d[0])
        else:
            theta = 0.0
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        links[-1].pose = [float(origin[0]), float(origin[1]), float(theta)]
        for jid in jids:
            local = rot.T @ (np.asarray(joint_pos[jid], dtype=float) - origin)
            anchors.setdefault(jid, {})[lid] = [float(local[0]), float(local[1])]

    # joint -> the two links it connects
    joint_links: Dict[str, List[str]] = {}
    for lid, jids in link_joints.items():
        for jid in jids:
            joint_links.setdefault(jid, []).append(lid)

    joints: List[Joint] = []
    for jid, typ in joint_types.items():
        la, lb = joint_links[jid]
        j = Joint(id=jid, type=typ, links=[la, lb], anchors={la: anchors[jid][la], lb: anchors[jid][lb]})
        if typ == "prismatic":
            if not axes or jid not in axes:
                raise ValueError("prismatic joint %s requires an axis" % jid)
            j.axis = {la: list(axes[jid][la]), lb: list(axes[jid][lb])}
        joints.append(j)

    return Mechanism(
        id=mech_id,
        links=links,
        joints=joints,
        ground=ground,
        dimensions={},
        point_of_interest=point_of_interest,
        driver=driver,
    )
