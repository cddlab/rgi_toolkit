#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
uv run --no-project --python "$WS/bioemu2_restr/.venv/bin/python" bioemu2 predict-structure \
    --system "$HERE/adk_72.85.yaml" --out "$HERE/out" \
    --num-samples 1 --num-steps 200 --sampler edm --seed 0
