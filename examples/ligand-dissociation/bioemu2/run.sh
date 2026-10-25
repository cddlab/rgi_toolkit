#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_30.00.yaml}"
test -f "$INPUT"
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"
OUT="out/$STEM"
uv run --no-project --python "$WS/bioemu2_restr/.venv/bin/python" bioemu2 predict-structure \
    --system "$INPUT" --out "$OUT" \
    --num-samples 1 --num-steps 200 --sampler edm --seed 0
