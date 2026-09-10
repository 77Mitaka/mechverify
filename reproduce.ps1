# One-click reproduction (Windows / PowerShell).
# Usage:  powershell -ExecutionPolicy Bypass -File reproduce.ps1
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "== 1/6 install dependencies =="
python -m pip install -r requirements.txt

Write-Host "== 2/6 unit + integration tests =="
python -m pytest -q

Write-Host "== 3/6 end-to-end demo (nominal + robust) =="
python demo.py

Write-Host "== 4/6 data loop: generate -> verify -> persist =="
python examples/build_corpus.py

Write-Host "== 5/6 emit Modelica models (self-contained + PlanarMechanics) =="
python examples/emit_modelica.py
python examples/emit_planar_mechanics.py

Write-Host "== 6/6 paper figures (optional) =="
if (Test-Path "paper\make_figures.py") {
    Push-Location paper
    python make_figures.py
    Pop-Location
}

Write-Host "DONE"
