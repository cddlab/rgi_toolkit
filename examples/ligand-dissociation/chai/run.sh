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
uv run --no-project --python "$WS/chai-lab_restr/.venv/bin/python" python -m chai_lab.main fold \
    mdm2_p53.fasta "$OUT" --restraints-config-path "$INPUT" \
    --num-trunk-samples 1 --num-diffn-samples 1 --seed 0 --use-msa-server
