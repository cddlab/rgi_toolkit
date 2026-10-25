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
uv run --no-project --python "$WS/boltz_restr/.venv/bin/python" boltz predict "$INPUT" \
    --out_dir "$OUT" --model boltz2 --use_msa_server --diffusion_samples 1 --seed 0
