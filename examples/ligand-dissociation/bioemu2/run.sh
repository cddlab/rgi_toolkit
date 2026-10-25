#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
INPUT="${1:-mdm2_p53_13.00.yaml}"
test -f "$INPUT"
bash "$HERE/../prepare_msa.sh"
uv run --no-project --python "$WS/bioemu2_restr/.venv/bin/python" python \
    "$HERE/run_dissociation.py" "$INPUT"
