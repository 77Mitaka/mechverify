#!/usr/bin/env bash
# One-click reproduction (Linux / macOS / Git Bash).
set -e
cd "$(dirname "$0")"

echo "== 1/6 install dependencies =="
python -m pip install -r requirements.txt

echo "== 2/6 unit + integration tests =="
python -m pytest -q

echo "== 3/6 end-to-end demo (nominal + robust) =="
python demo.py

echo "== 4/6 data loop: generate -> verify -> persist =="
python examples/build_corpus.py

echo "== 5/6 emit Modelica models (self-contained + PlanarMechanics) =="
python examples/emit_modelica.py
python examples/emit_planar_mechanics.py

echo "== 6/6 paper figures (optional) =="
if [ -f paper/make_figures.py ]; then (cd paper && python make_figures.py); fi

echo "DONE"
