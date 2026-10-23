#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_13.00.json}"
test -f "$INPUT"
STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"
OUT="out/$STEM"
bash "$HERE/../prepare_msa.sh"
uv run --no-project --python "$WS/OpenDDE_restr/.venv/bin/python" opendde pred \
    -i "$INPUT" -o "$OUT" -n opendde_v1 --use_msa true --use_template false \
    --use_rna_msa false --seeds 0 --sample 1 --step 200 --cycle 4
