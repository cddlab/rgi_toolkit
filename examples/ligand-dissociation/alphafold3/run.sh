#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_13.00.json}"
test -f "$INPUT"
STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"
OUT="out/$STEM"
bash "$HERE/../prepare_msa.sh"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to your AF3 model-parameters directory}"
uv run --no-project --python "$WS/alphafold3_restr/.venv/bin/python" python \
    "$WS/alphafold3_restr/run_alphafold.py" --json_path="$INPUT" --output_dir="$OUT" \
    --model_dir="$MODEL_DIR" --run_data_pipeline=False --num_diffusion_samples=1
