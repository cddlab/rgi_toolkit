#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_13.00.yaml}"
test -f "$INPUT"
STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"
OUT="out/$STEM"
bash "$HERE/../prepare_msa.sh"
uv run --no-project --python "$WS/chai-lab_restr/.venv/bin/python" python prepare_msa.py
uv run --no-project --python "$WS/chai-lab_restr/.venv/bin/python" python -m chai_lab.main fold \
    mdm2_p53.fasta "$OUT" --restraints-config-path "$INPUT" \
    --num-trunk-samples 1 --num-diffn-samples 1 --seed 0 --msa-directory ../msas/chai
