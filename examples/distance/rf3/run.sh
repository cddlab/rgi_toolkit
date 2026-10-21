#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
uv run --no-project --python "$WS/foundry_restr/.venv/bin/python" rf3 fold \
    inputs="$HERE/qbp_25.00.json" out_dir="$HERE/out" \
    diffusion_batch_size=1 num_steps=50 early_stopping_plddt_threshold=null seed=0
