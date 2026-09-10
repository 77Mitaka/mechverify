import os

from mechverify import Mechanism
from mechverify.corpus import Corpus

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "fourbar_crank_rocker.json")

REQ = {"constraints": [{"type": "crank_rotatability", "hard": True}]}


def test_add_and_stats():
    mech = Mechanism.from_json_file(EX)
    corpus = Corpus(":memory:")
    corpus.verify_and_add(mech, REQ, source="test")
    assert corpus.count() == 1
    st = corpus.stats()
    assert st["total"] == 1
    assert st["by_status"].get("pass") == 1


def test_jsonl_roundtrip(tmp_path):
    mech = Mechanism.from_json_file(EX)
    corpus = Corpus(":memory:")
    corpus.verify_and_add(mech, REQ)
    p = tmp_path / "c.jsonl"
    corpus.export_jsonl(str(p))
    c2 = Corpus.from_jsonl(str(p))
    assert c2.count() == 1
    s = next(c2.iter_samples())
    assert s["status"] == "pass"
    assert s["mechanism"]["id"] == mech.id
