#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-qbp_2.65.json}"
test -f "$INPUT"
STEM="$(basename "$INPUT" .json)"
OUT="${OUT_DIR:-out/$STEM}"
for pdb in 1GGG 1WDN; do
    test -f "$pdb.cif" || curl -fLsS "https://files.rcsb.org/download/$pdb.cif" -o "$pdb.cif"
done
MODEL_DIR="${MODEL_DIR:-$WS/IntelliFold_restr/model_v2}"
PIPELINE_ARGS=(--norun_data_pipeline)
if [[ "${RUN_DATA_PIPELINE:-true}" == true ]]; then
    DB_DIR="${DB_DIR:?set DB_DIR to your sequence-database directory}"
    PIPELINE_ARGS=(--run_data_pipeline=true --db_dir="$DB_DIR")
fi
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${TMPDIR:-/tmp}/rgi-intellifold-jax}"
uv run --no-project --python "$WS/IntelliFold_restr/.venv/bin/python" intellifold predict \
    "$INPUT" --model-dir "$MODEL_DIR" --output-dir "$OUT" -- \
    --num_diffusion_samples=1 "${PIPELINE_ARGS[@]}"
