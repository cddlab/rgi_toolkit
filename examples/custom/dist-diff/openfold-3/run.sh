#!/bin/bash
# openfold-3 RGI example -- custom dist-diff: Delta D = D_in - D_out -> 0.8 A (DgoT)
# MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the openfold-3_restr checkout to exist as a sibling of rgi_toolkit/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
export OPENFOLD_CACHE="${OPENFOLD_CACHE:-$HOME/.openfold3}"
PIXI="$WS/openfold-3_restr/.pixi-bin/pixi"; [ -x "$PIXI" ] || PIXI=pixi
cd "$HERE"
"$PIXI" run --manifest-path "$WS/openfold-3_restr/pixi.toml" -e openfold3-cuda12 \
    run_openfold predict --query-json "$HERE/dgot_0.80.json" --output-dir "$HERE/out" \
    --num-model-seeds 1 --num-diffusion-samples 1 --use-msa-server true --use-templates false
