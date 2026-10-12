"""Download public benchmark inputs with payload checks."""

from __future__ import annotations

import argparse
import csv
import lzma
import urllib.request
from pathlib import Path

from common import coordinates, msa_query, protocol, slug


def place(path, payload, size):
    if len(payload) != int(size):
        raise ValueError(f"Dataset size mismatch: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def conformer(root, limit):
    from huggingface_hub import hf_hub_download

    dataset = protocol("conformer")["dataset"]

    def fetch(name):
        return Path(
            hf_hub_download(
                dataset["repo"],
                name,
                repo_type="dataset",
                revision=dataset["revision"],
                local_dir=root / "hub",
                token=False,
            )
        )

    index = fetch("index.csv")
    with index.open() as handle:
        rows = list(csv.DictReader(handle))[:limit]
    output = []
    for row in rows:
        name = slug(row["sys_id"])
        msa, sdf = Path("msa", f"{name}.a3m"), Path("sdf", f"{name}.sdf")
        for relative, key in ((msa, "a3m"), (sdf, "sdf")):
            path = root / relative
            payload = fetch(
                f"msa/{name}.a3m.xz" if key == "a3m" else sdf.as_posix()
            ).read_bytes()
            place(
                path,
                lzma.decompress(payload) if key == "a3m" else payload,
                row[f"{key}_bytes"],
            )
        msa_query(root / msa)
        output.append(
            {
                "id": name,
                "smiles": row["smiles"],
                "msa": msa.as_posix(),
                "sdf": sdf.as_posix(),
            }
        )
    temporary = root / "index.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles", "msa", "sdf"])
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(root / "index.csv")
    print(f"Fetched {len(output)} systems: {root / 'index.csv'}")


def references(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("1GGG", "1WDN", "4AKE", "1AKE", "6E9O", "6E9N"):
        path = root / f"{name}.cif"
        if not path.exists():
            with urllib.request.urlopen(
                f"https://files.rcsb.org/download/{name}.cif", timeout=60
            ) as response:
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(response.read())
            coordinates(temporary)
            temporary.replace(path)
        coordinates(path)
    print(f"References ready: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["conformer", "references"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.output or Path("data") / args.dataset
    if args.dataset == "conformer":
        conformer(root, args.limit)
    else:
        references(root)


if __name__ == "__main__":
    main()
