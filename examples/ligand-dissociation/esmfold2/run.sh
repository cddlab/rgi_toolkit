#!/usr/bin/env bash
set -euo pipefail
: "${MSA_A3M_A:?set MSA_A3M_A to the full ColabFold A3M for chain A}"
: "${MSA_A3M_B:?set MSA_A3M_B to the full ColabFold A3M for chain B}"
MSA_A3M_A="$(cd "$(dirname "$MSA_A3M_A")" && pwd)/$(basename "$MSA_A3M_A")"
MSA_A3M_B="$(cd "$(dirname "$MSA_A3M_B")" && pwd)/$(basename "$MSA_A3M_B")"
export MSA_A3M_A MSA_A3M_B
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_30.00.json}"
test -f "$INPUT"
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
PIXI="$WS/esm_restr/.pixi-bin/pixi"
[ -x "$PIXI" ] || PIXI=pixi
"$PIXI" run --manifest-path "$WS/esm_restr/pyproject.toml" python \
    "$HERE/run_dissociation.py" "$INPUT"
