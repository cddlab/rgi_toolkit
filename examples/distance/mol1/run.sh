#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-qbp_25.00.json}"
test -f "$INPUT"
STEM="$(basename "$INPUT" .json)"
OUT="${OUT_DIR:-out/$STEM}"
export MOL_ROOT_DIR="${MOL_ROOT_DIR:-$WS/Mol-1_restr/data}"
uv run --no-project --python "$WS/Mol-1_restr/.venv/bin/python" mol pred \
    -i "$INPUT" -o "$OUT" --use_msa true --use_template false --use_rna_msa false \
    --seeds 0 --step 200 --sample 1
