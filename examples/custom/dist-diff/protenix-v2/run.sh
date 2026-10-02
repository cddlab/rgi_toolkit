#!/bin/bash
# protenix-v2 RGI example -- custom dist-diff: Delta D = D_in - D_out -> 0.8 A (DgoT)
# MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the protenix_restr checkout to exist as a sibling of rgi_toolkit/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
source "$WS/protenix_restr/.venv/bin/activate"
cd "$HERE"
# protenix must run on sm_89 (e.g. RTX 4090); Blackwell emits silent all-NaN coords.
protenix pred -i dgot_0.80.json -o out \
    --model_name protenix-v2 --use_default_params true --use_msa true --seeds 0 --step 200 --sample 1 --cycle 10
