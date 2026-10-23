#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
for pdb in 1GGG 1WDN; do
    [ -s "$pdb.cif" ] || curl -fLsS "https://files.rcsb.org/download/$pdb.cif" -o "$pdb.cif"
done
uv run --no-project --python "$WS/foundry_restr/.venv/bin/python" rf3 fold \
    inputs="$HERE/qbp_2.65.json" out_dir="$HERE/out" \
    diffusion_batch_size=1 num_steps=50 early_stopping_plddt_threshold=null seed=0
