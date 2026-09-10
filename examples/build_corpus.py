"""Close the generate -> verify -> persist loop on a small four-bar corpus.

Run:  python examples/build_corpus.py
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechverify import Mechanism
from mechverify.corpus import Corpus

REQ = {
    "driver_sweep": {"start": 0.0, "span": 360.0, "steps": 360, "branch": 1},
    "constraints": [
        {"type": "crank_rotatability", "hard": True},
        {"type": "transmission_angle_min", "value": 30.0, "hard": True},
        {"type": "straightness", "value": 0.02, "window_frac": 0.3, "hard": False, "weight": 1.0},
    ],
}


def random_fourbar(rng):
    a = rng.uniform(0.5, 1.5)
    b = rng.uniform(1.5, 4.0)
    c = rng.uniform(1.5, 4.0)
    d = rng.uniform(1.5, 4.0)
    if a > min(b, c, d) + 1e-9:
        return None
    ls = sorted([a, b, c, d])
    if ls[0] + ls[3] > ls[1] + ls[2]:
        return None
    p = rng.uniform(0.2 * b, 1.2 * b)
    phi = rng.uniform(-math.pi, math.pi)
    return Mechanism.from_dict({
        "id": "fb_rand",
        "ground": "ground",
        "links": [
            {"id": "ground", "kind": "ground"},
            {"id": "crank"},
            {"id": "coupler"},
            {"id": "rocker"},
        ],
        "joints": [
            {"id": "A", "type": "revolute", "links": ["ground", "crank"], "position": [0.0, 0.0]},
            {"id": "B", "type": "revolute", "links": ["crank", "coupler"]},
            {"id": "C", "type": "revolute", "links": ["coupler", "rocker"]},
            {"id": "D", "type": "revolute", "links": ["rocker", "ground"], "position": [d, 0.0]},
        ],
        "dimensions": {"crank": a, "coupler": b, "rocker": c},
        "coupler_point": {"link": "coupler", "p": p, "phi": phi},
        "driver": {"joint": "A", "link": "crank", "start": 0.0, "span": 360.0, "steps": 360, "branch": 1},
    })


def main(n=300, seed=0):
    here = os.path.dirname(os.path.abspath(__file__))
    db = os.path.join(here, "corpus.sqlite")
    jsonl = os.path.join(here, "corpus.jsonl")
    if os.path.exists(db):
        os.remove(db)
    corpus = Corpus(db)
    rng = random.Random(seed)
    added = 0
    while added < n:
        m = random_fourbar(rng)
        if m is None:
            continue
        corpus.verify_and_add(m, REQ, source="random_search")
        added += 1
    print("stats:", corpus.stats())
    corpus.export_jsonl(jsonl)
    print("exported:", jsonl)
    best = corpus.top(1)
    if best:
        print("best score:", best[0]["score"], "metrics:", best[0]["metrics"])
    corpus.close()


if __name__ == "__main__":
    main()
