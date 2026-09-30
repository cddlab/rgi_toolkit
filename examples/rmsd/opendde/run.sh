#!/bin/bash
# OpenDDE RGI example -- dual-reference QBP RMSD targets: open 2.65 A, closed 2.65 A
# External feature searches are disabled; the OpenDDE checkpoint and common runtime
# files must already be installed. Run on a GPU compute node, not a login node.
# Requires the OpenDDE_restr checkout to exist as a sibling of rgi_toolkit/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/RGI-toolkit" ]; do WS="$(dirname "$WS")"; done
# Reference structures are downloaded from RCSB at run time and are not stored here.
( cd "$HERE" && for pdb in 1GGG 1WDN; do
    [ -f "$pdb.cif" ] || wget -q "https://files.rcsb.org/download/$pdb.cif"
done )
source "$WS/OpenDDE_restr/.venv/bin/activate"
cd "$HERE"
opendde pred -i qbp_2.65.json -o out -n opendde_v1 \
    --use_msa false --use_template false --use_rna_msa false \
    --sample 1 --step 200 --cycle 4
