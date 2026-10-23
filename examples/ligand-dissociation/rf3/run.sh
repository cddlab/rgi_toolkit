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
uv run --no-project --python "$WS/foundry_restr/.venv/bin/python" rf3 fold \
    inputs="$HERE/$INPUT" out_dir="$HERE/$OUT" diffusion_batch_size=1 \
    num_steps=50 early_stopping_plddt_threshold=null seed=0
