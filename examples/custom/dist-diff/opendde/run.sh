#!/bin/bash
# OpenDDE RGI example -- custom dist-diff: Delta D = D_in - D_out -> 0.8 A (DgoT)
# External feature searches are disabled; the OpenDDE checkpoint and common runtime
# files must already be installed. Run on a GPU compute node, not a login node.
# Requires the OpenDDE_restr checkout to exist as a sibling of RGI-toolkit/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
source "$WS/OpenDDE_restr/.venv/bin/activate"
cd "$HERE"
opendde pred -i dgot_0.80.json -o out -n opendde_v1 \
    --use_msa false --use_template false --use_rna_msa false \
    --sample 1 --step 200 --cycle 4
