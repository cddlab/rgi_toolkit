#!/bin/bash
# esmfold2 RGI example -- centroid distance -> 25.0 A (QBP)
# ESMFold2 uses single-sequence input and downloads model weights when needed.
# GPU only: run on a GPU compute node (not a shared login node).
# Requires esm_restr on rgi-integration alongside RGI-toolkit.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
PIXI="$WS/esm_restr/.pixi-bin/pixi"; [ -x "$PIXI" ] || PIXI=pixi
cd "$HERE"
# Use the native ESM model and its RGI sampling hook.
"$PIXI" run --manifest-path "$WS/esm_restr/pyproject.toml" \
    python "$HERE/run_distance.py"
