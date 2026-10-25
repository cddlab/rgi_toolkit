#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_30.00.json}"
test -f "$INPUT"
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"
OUT="out/$STEM"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to your AF3 model-parameters directory}"
DB_DIR="${DB_DIR:?set DB_DIR to your AF3 sequence-database directory}"
uv run --no-project --python "$WS/alphafold3_restr/.venv/bin/python" python \
    "$WS/alphafold3_restr/run_alphafold.py" --json_path="$INPUT" --output_dir="$OUT" \
    --model_dir="$MODEL_DIR" --db_dir="$DB_DIR" --run_data_pipeline=True --num_diffusion_samples=1
