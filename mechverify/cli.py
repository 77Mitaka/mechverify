"""Command line interface.

Reuse existing Modelica tools instead of building a front-end:

    mechverify import   model.mo   -> ir.json     (from OMEdit/Dymola export)
    mechverify verify   ir.json    [--req req.json] [--robust ...]
    mechverify export   ir.json    -> model.mo     (open in OMEdit/Dymola)
    mechverify simulate ir.json                    (export + omc)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .convert import modelica_to_mechanism
from .ir import Mechanism
from .modelica import generate_modelica, simulate, write_model
from .robustness import verify_robust
from .verifier import verify


def _load_ir(path):
    return Mechanism.from_json_file(path)


def _load_req(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_verify(args):
    mech = _load_ir(args.ir)
    req = _load_req(args.req)
    if args.robust:
        res = verify_robust(mech, req, samples=args.samples, rel_tol=args.rel_tol, seed=args.seed)
        print(res.summary())
        return 0 if res.status == "pass" else 1
    res = verify(mech, req)
    print(res.summary())
    return 0 if res.status == "pass" else 1


def cmd_import(args):
    with open(args.model, "r", encoding="utf-8") as f:
        mo = f.read()
    mech = modelica_to_mechanism(mo)
    out = args.output or (os.path.splitext(args.model)[0] + ".ir.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(mech.to_dict(), f, indent=2)
    print("wrote", out)
    return 0


def _generate(mech, backend, **kw):
    if backend == "planarmechanics":
        from .planar_mechanics import generate_planar_mechanics

        return generate_planar_mechanics(mech, **kw)
    return generate_modelica(mech, **kw)


def cmd_export(args):
    mech = _load_ir(args.ir)
    out = args.output or (os.path.splitext(args.ir)[0] + ".mo")
    code = _generate(mech, args.backend)
    with open(out, "w", encoding="utf-8") as f:
        f.write(code)
    print("wrote", out, "(%s)" % args.backend)
    return 0


def cmd_simulate(args):
    mech = _load_ir(args.ir)
    import tempfile

    d = tempfile.mkdtemp(prefix="mechverify_")
    mo = os.path.join(d, mech.id + ".mo")
    code = _generate(mech, args.backend, stop_time=args.stop_time)
    with open(mo, "w", encoding="utf-8") as f:
        f.write(code)
    csv = simulate(mo, mech.id, stop_time=args.stop_time, number_of_intervals=args.intervals)
    print("CSV:", csv, "(%s)" % args.backend)
    return 0


def cmd_corpus(args):
    from .corpus import Corpus

    corpus = Corpus(args.db)
    if args.action == "stats":
        print(corpus.stats())
        return 0
    if args.action == "export":
        out = args.output or (os.path.splitext(args.db)[0] + ".jsonl")
        corpus.export_jsonl(out)
        print("wrote", out)
        return 0
    print("unknown action:", args.action)
    return 2


def build_parser():
    p = argparse.ArgumentParser(prog="mechverify", description="Planar mechanism verifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="verify a mechanism IR against requirements")
    v.add_argument("ir")
    v.add_argument("--req", default=None)
    v.add_argument("--robust", action="store_true")
    v.add_argument("--samples", type=int, default=200)
    v.add_argument("--rel-tol", dest="rel_tol", type=float, default=0.002)
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=cmd_verify)

    i = sub.add_parser("import", help="Modelica -> IR")
    i.add_argument("model")
    i.add_argument("-o", "--output", default=None)
    i.set_defaults(func=cmd_import)

    e = sub.add_parser("export", help="IR -> Modelica")
    e.add_argument("ir")
    e.add_argument("-o", "--output", default=None)
    e.add_argument("--backend", choices=["planarmechanics", "selfcontained"], default="planarmechanics")
    e.set_defaults(func=cmd_export)

    s = sub.add_parser("simulate", help="IR -> Modelica -> omc simulation")
    s.add_argument("ir")
    s.add_argument("--stop-time", dest="stop_time", type=float, default=1.0)
    s.add_argument("--intervals", type=int, default=200)
    s.add_argument("--backend", choices=["planarmechanics", "selfcontained"], default="planarmechanics")
    s.set_defaults(func=cmd_simulate)

    c = sub.add_parser("corpus", help="inspect/export a persisted corpus")
    c.add_argument("action", choices=["stats", "export"])
    c.add_argument("db")
    c.add_argument("-o", "--output", default=None)
    c.set_defaults(func=cmd_corpus)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
