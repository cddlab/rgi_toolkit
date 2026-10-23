#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_13.00.json}"
test -f "$INPUT"
bash "$HERE/../prepare_msa.sh"
PIXI="$WS/esm_restr/.pixi-bin/pixi"
[ -x "$PIXI" ] || PIXI=pixi
"$PIXI" run --manifest-path "$WS/esm_restr/pyproject.toml" python \
    "$HERE/run_dissociation.py" "$INPUT"
