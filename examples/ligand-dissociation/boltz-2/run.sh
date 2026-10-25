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
uv run --no-project --python "$WS/boltz_restr/.venv/bin/python" boltz predict "$INPUT" \
    --out_dir "$OUT" --model boltz2 --diffusion_samples 1 --seed 0
