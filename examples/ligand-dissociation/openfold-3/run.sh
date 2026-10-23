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
PIXI="$WS/openfold-3_restr/.pixi-bin/pixi"
[ -x "$PIXI" ] || PIXI=pixi
"$PIXI" run --manifest-path "$WS/openfold-3_restr/pixi.toml" -e openfold3-cuda12 \
    run_openfold predict --query-json "$HERE/$INPUT" --output-dir "$HERE/$OUT" \
    --num-diffusion-samples 1 --use-msa-server false --use-templates false
