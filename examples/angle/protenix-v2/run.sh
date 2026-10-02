#!/bin/bash
# protenix-v2 RGI example -- group-centroid angle -> 72.85 deg (ADK NMP-CORE-LID)
# MSA is fetched from a server.
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the protenix_restr checkout to exist as a sibling of RGI-toolkit/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
source "$WS/protenix_restr/.venv/bin/activate"
cd "$HERE"
# protenix must run on sm_89 (e.g. RTX 4090); Blackwell emits silent all-NaN coords.
protenix pred -i adk_72.85.json -o out \
    --model_name protenix-v2 --use_default_params true --use_msa true --seeds 0 --step 200 --sample 1 --cycle 10
