#!/bin/bash
# esmfold2 RGI example -- dual-ref RMSD morph -> midpoint of 1GGG(open)/1WDN(closed), target 3.0 A
# ESMFold2 uses single-sequence input and downloads model weights when needed.
# GPU only: run on a GPU compute node (not a shared login node).
# Requires esm_restr on rgi-integration alongside RGI-toolkit.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
# Reference structures (1GGG open / 1WDN closed) are downloaded from RCSB at run time
# instead of being stored in the repo. The config's ref_cif uses the bare filename.
( cd "$HERE" && for pdb in 1GGG 1WDN; do
    [ -f "$pdb.cif" ] || wget -q "https://files.rcsb.org/download/$pdb.cif"
done )
PIXI="$WS/esm_restr/.pixi-bin/pixi"; [ -x "$PIXI" ] || PIXI=pixi
cd "$HERE"
# Use the native ESM model and its RGI sampling hook.
"$PIXI" run --manifest-path "$WS/esm_restr/pyproject.toml" \
    python "$HERE/run_rmsd.py"
