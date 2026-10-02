#!/bin/bash
# chai RGI example -- group-centroid angle -> 72.85 deg (ADK NMP-CORE-LID)
# MSA is fetched from a server.
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the chai-lab_restr checkout to exist as a sibling of RGI-toolkit/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
source "$WS/chai-lab_restr/.venv/bin/activate"
cd "$HERE"
export CHAI_DOWNLOADS_DIR="${CHAI_DOWNLOADS_DIR:-$HOME/.cache/chai}"
python -m chai_lab.main fold adk_72.85.fasta out \
    --restraints-config-path adk_72.85.yaml \
    --num-trunk-samples 1 --num-diffn-samples 1 --seed 0 \
    --use-msa-server
