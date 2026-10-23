#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
output_dir="$HERE/msas/a3m"
for filename in mdm2_p53.a3m p53.a3m; do
    if [ ! -s "$HERE/msas/$filename" ]; then
        printf 'Missing MSA: %s (see README.md)\n' "$HERE/msas/$filename" >&2
        exit 1
    fi
done
mkdir -p "$output_dir"
for filename in mdm2_p53.a3m p53.a3m; do
    prepared_tmp="$(mktemp "$output_dir/$filename.XXXXXX")"
    # Some native FASTA parsers reject ColabFold's metadata comments.
    sed '/^#/d' "$HERE/msas/$filename" > "$prepared_tmp"
    mv "$prepared_tmp" "$output_dir/$filename"
done
