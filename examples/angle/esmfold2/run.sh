#!/bin/bash
# esmfold2 RGI example -- group-centroid angle -> 72.85 deg (ADK NMP-CORE-LID)
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
    python "$HERE/run_angle.py"
