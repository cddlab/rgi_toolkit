#!/bin/bash
# boltz-2 RGI example -- group-centroid angle -> 72.85 deg (ADK NMP-CORE-LID)
# MSA is fetched from a server.
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the boltz_restr checkout to exist as a sibling of RGI-toolkit/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
source "$WS/boltz_restr/.venv/bin/activate"
cd "$HERE"
boltz predict adk_72.85.yaml --out_dir out --model boltz2 --use_msa_server --seed 0
