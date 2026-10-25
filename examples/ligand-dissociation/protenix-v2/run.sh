#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_30.00.json}"
test -f "$INPUT"
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"
OUT="out/$STEM"
uv run --no-project --python "$WS/protenix_restr/.venv/bin/python" protenix pred \
    -i "$INPUT" -o "$OUT" --model_name protenix-v2 --use_default_params true \
    --use_msa true --seeds 0 --step 200 --sample 1 --cycle 10
