"""Corpus: persist (requirement, mechanism, score) triples.

Closes the "generate -> verify -> persist" loop of the success criterion.
SQLite backed, with JSONL import/export.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional

from .ir import Mechanism
from .verifier import verify

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL,
    status TEXT,
    score REAL,
    requirement_json TEXT,
    mechanism_json TEXT,
    metrics_json TEXT,
    source TEXT
);
"""


class Corpus:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------- write
    def add(self, mechanism: Mechanism, requirement: Dict, result, source: Optional[str] = None) -> None:
        metrics = {c["name"]: c["value"] for c in result.checks if c["layer"] == "behavior"}
        self.conn.execute(
            "INSERT INTO samples (created_at, status, score, requirement_json, mechanism_json,"
            " metrics_json, source) VALUES (?,?,?,?,?,?,?)",
            (
                time.time(),
                result.status,
                float(result.score),
                json.dumps(requirement, ensure_ascii=False),
                json.dumps(mechanism.to_dict(), ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                source,
            ),
        )
        self.conn.commit()

    def verify_and_add(self, mechanism: Mechanism, requirement: Dict, source: Optional[str] = None):
        result = verify(mechanism, requirement)
        self.add(mechanism, requirement, result, source=source)
        return result

    # -------------------------------------------------------------- read
    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    def stats(self) -> Dict[str, Any]:
        rows = self.conn.execute("SELECT status, COUNT(*) FROM samples GROUP BY status").fetchall()
        best = self.conn.execute("SELECT MAX(score) FROM samples").fetchone()[0]
        return {
            "total": self.count(),
            "by_status": {s: n for s, n in rows},
            "best_score": best,
        }

    def top(self, n: int = 10) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT id, status, score, mechanism_json, metrics_json FROM samples"
            " ORDER BY score DESC LIMIT ?",
            (n,),
        )
        return [
            {
                "id": r[0],
                "status": r[1],
                "score": r[2],
                "mechanism": json.loads(r[3]),
                "metrics": json.loads(r[4]),
            }
            for r in cur.fetchall()
        ]

    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT requirement_json, mechanism_json, status, score, metrics_json FROM samples"
        )
        for req, mech, status, score, metrics in cur.fetchall():
            yield {
                "requirement": json.loads(req),
                "mechanism": json.loads(mech),
                "status": status,
                "score": score,
                "metrics": json.loads(metrics),
            }

    # ------------------------------------------------------------- io
    def export_jsonl(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            for s in self.iter_samples():
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path: str, db_path: str = ":memory:") -> "Corpus":
        corpus = cls(db_path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                corpus.conn.execute(
                    "INSERT INTO samples (created_at, status, score, requirement_json,"
                    " mechanism_json, metrics_json, source) VALUES (?,?,?,?,?,?,?)",
                    (
                        time.time(),
                        s.get("status"),
                        float(s.get("score", 0.0)),
                        json.dumps(s.get("requirement", {}), ensure_ascii=False),
                        json.dumps(s.get("mechanism", {}), ensure_ascii=False),
                        json.dumps(s.get("metrics", {}), ensure_ascii=False),
                        "jsonl",
                    ),
                )
        corpus.conn.commit()
        return corpus

    def close(self) -> None:
        self.conn.close()
