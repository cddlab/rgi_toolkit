#!/bin/bash
# boltz-2 RGI example -- custom dist-diff: Delta D = D_in - D_out -> 0.8 A (DgoT)
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the boltz_restr checkout to exist as a sibling of rgi_toolkit/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
source "$WS/boltz_restr/.venv/bin/activate"
cd "$HERE"
boltz predict dgot_0.80.yaml --out_dir out --model boltz2 --use_msa_server --seed 0
