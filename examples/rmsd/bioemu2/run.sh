#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
for pdb in 1GGG 1WDN; do
    [ -s "$pdb.cif" ] || curl -fLsS "https://files.rcsb.org/download/$pdb.cif" -o "$pdb.cif"
done
uv run --no-project --python "$WS/bioemu2_restr/.venv/bin/python" bioemu2 predict-structure \
    --system "$HERE/qbp_2.65.yaml" --out "$HERE/out" \
    --num-samples 1 --num-steps 200 --sampler edm --seed 0
