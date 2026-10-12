"""Shared files, manifests, and coordinate validation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import yaml
from rgi_toolkit.pdb_ref import read_cif_atoms

MODELS = (
    "boltz_restr",
    "protenix_restr",
    "alphafold3_restr",
    "chai-lab_restr",
    "openfold-3_restr",
    "esm_restr",
)
BENCHMARKS = ("conformer", "distance", "angle", "rmsd", "custom")
ROOT = Path(__file__).resolve().parent
PREDICTION_PATTERNS = {
    "boltz_restr": "**/predictions/*/*_model_0.cif",
    "protenix_restr": "**/seed_*/predictions/*_sample_0.cif",
    "alphafold3_restr": "**/seed-*_sample-0/*model.cif",
    "chai-lab_restr": "**/pred.model_idx_0.cif",
    "openfold-3_restr": "**/*_model.cif",
    "esm_restr": "model.cif",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def slug(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def protocol(name):
    return yaml.safe_load((ROOT / "protocols" / f"{name}.yaml").read_text())


def msa_query(path):
    sequence = []
    started = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(">"):
            if started:
                break
            started = True
        elif started:
            sequence.append(line.strip())
        else:
            raise ValueError(f"Invalid A3M: {path}")
    query = "".join(c for c in "".join(sequence) if c.isupper())
    if not query:
        raise ValueError(f"Empty A3M query: {path}")
    return query


def coordinates(path):
    atoms = [
        a for a in read_cif_atoms(str(path)) if a.element.upper() not in {"H", "D"}
    ]
    xyz = np.array([[a.x, a.y, a.z] for a in atoms])
    if not len(atoms) or not np.isfinite(xyz).all():
        raise ValueError(f"Empty or non-finite coordinates: {path}")
    return atoms, xyz


def verify_inputs(run):
    manifest = read_json(Path(run) / "manifest.json")
    for relative in manifest["files"]:
        path = Path(run) / relative
        if not path.is_file():
            raise ValueError(f"Missing prepared input: {relative}")
    return manifest


def validate_prediction(path, job):
    import gemmi

    atoms, xyz = coordinates(path)
    ca = [a for a in atoms if a.chain == "A" and a.name == "CA"]
    sequence = "".join(
        gemmi.find_tabulated_residue(a.res_name).one_letter_code for a in ca
    )
    if sequence != job["sequence"] or len({a.resid for a in ca}) != len(ca):
        raise ValueError(
            "Prediction must contain the full, correctly ordered chain A sequence"
        )
    if job.get("smiles"):
        from rdkit import Chem

        expected = Chem.MolFromSmiles(job["smiles"]).GetNumHeavyAtoms()
        if sum(a.chain == "B" for a in atoms) != expected:
            raise ValueError("Prediction has missing or extra ligand atoms in chain B")
    return atoms, xyz
