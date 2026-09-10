"""Mechanism intermediate representation (IR).

A minimal, JSON-serializable schema for planar mechanisms.  For the MVP only
four-bar linkages (4 links, 4 revolute joints, single loop) are supported, but
the representation is kept general so that other lower-pair topologies can be
added later.

Geometry convention
-------------------
* ``joints`` of type ``revolute`` connect exactly two links.
* Joints attached to the ground link carry an explicit ``position`` in the
  ground frame; all other joint positions are derived by kinematics.
* Link lengths for movable links are given in ``dimensions`` (link id -> length).
  The ground link length is derived from the two ground joint positions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Link:
    id: str
    kind: str = "binary"  # 'ground' | 'binary' | 'ternary'
    pose: Optional[List[float]] = None  # initial guess [x, y, theta] for general solver


@dataclass
class Joint:
    id: str
    type: str  # 'revolute' | 'prismatic'
    links: List[str]
    position: Optional[List[float]] = None  # ground-frame coords (four-bar shortcut)
    anchors: Optional[Dict[str, List[float]]] = None  # link id -> local [x, y] anchor
    axis: Optional[Dict[str, List[float]]] = None  # prismatic: link id -> local [x, y] axis


@dataclass
class CouplerPoint:
    link: str
    p: float  # distance from the link's first joint (crank-side joint)
    phi: float = 0.0  # angle relative to the link direction, radians


@dataclass
class Mechanism:
    id: str
    links: List[Link]
    joints: List[Joint]
    ground: str
    dimensions: Dict[str, float]
    coupler_point: Optional[CouplerPoint] = None
    point_of_interest: Optional[Dict[str, Any]] = None  # {link, local:[x,y]}
    driver: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Mechanism":
        links = [Link(**l) for l in d["links"]]
        joints = [Joint(**j) for j in d["joints"]]
        cp = d.get("coupler_point")
        cp = CouplerPoint(**cp) if cp else None
        return Mechanism(
            id=d["id"],
            links=links,
            joints=joints,
            ground=d["ground"],
            dimensions=dict(d.get("dimensions", {})),
            coupler_point=cp,
            point_of_interest=d.get("point_of_interest"),
            driver=d.get("driver"),
            meta=d.get("meta", {}),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "Mechanism":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    # -------------------------------------------------------------- helpers
    def link_ids(self) -> List[str]:
        return [l.id for l in self.links]

    def joints_of(self, link_id: str) -> List[Joint]:
        return [j for j in self.joints if link_id in j.links]

    def other_link(self, joint: Joint, link_id: str) -> str:
        return [x for x in joint.links if x != link_id][0]

    def ground_joints(self) -> List[Joint]:
        return [j for j in self.joints if self.ground in j.links]
