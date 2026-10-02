#!/bin/bash
# esmfold2 RGI example -- dual-reference QBP RMSD targets: open 2.65 A, closed 2.65 A
# Supply the full ColabFold A3M via MSA_A3M; weights are downloaded when needed.
# GPU only: run on a GPU compute node (not a shared login node).
# Requires esm_restr on rgi-integration alongside RGI-toolkit.
set -euo pipefail
: "${MSA_A3M:?set MSA_A3M to the full ColabFold A3M for this protein}"
export MSA_A3M="$(cd "$(dirname "$MSA_A3M")" && pwd)/$(basename "$MSA_A3M")"
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
# Reference structures (1GGG open / 1WDN closed) are downloaded from RCSB at run time
# instead of being stored in the repo. The config's ref_cif uses the bare filename.
( cd "$HERE" && for pdb in 1GGG 1WDN; do
    [ -f "$pdb.cif" ] || wget -q "https://files.rcsb.org/download/$pdb.cif"
done )
PIXI="$WS/esm_restr/.pixi-bin/pixi"; [ -x "$PIXI" ] || PIXI=pixi
cd "$HERE"
"$PIXI" run --manifest-path "$WS/esm_restr/pyproject.toml" \
    env MSA_A3M="$MSA_A3M" python "$HERE/run_rmsd.py"
