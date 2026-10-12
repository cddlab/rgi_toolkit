"""Reproducible query-preserving sampling of A3M homolog records."""

import random
from pathlib import Path


def subsample_msa(source, destination, depth, seed):
    if depth < 1 or seed < 0:
        raise ValueError("MSA depth must be positive and seed non-negative")
    records = []
    for line in Path(source).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith(">"):
            records.append([line, ""])
        elif records:
            records[-1][1] += line.strip()
        else:
            raise ValueError("Invalid A3M record")
    if not records or any(not sequence for _, sequence in records):
        raise ValueError("Empty A3M record")
    homologs = random.Random(seed).sample(records[1:], min(depth, len(records) - 1))
    selected = [records[0], *homologs]
    Path(destination).write_text(
        "".join(f"{header}\n{sequence}\n" for header, sequence in selected)
    )
    return len(selected)
