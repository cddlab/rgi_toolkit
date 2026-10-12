"""Read confidence from the exact raw prediction retained by the runner."""

from __future__ import annotations

import hashlib

import gemmi
import numpy as np

from common import PREDICTION_PATTERNS, read_json


def plddt(path):
    block = gemmi.cif.read_file(str(path)).sole_block()
    values = list(block.find_values("_atom_site.B_iso_or_equiv"))
    if not values or any(v in {".", "?"} for v in values):
        raise ValueError("Prediction has missing atom pLDDT values")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError("Atom pLDDT must be finite and within [0, 100]")
    return {"plddt": float(values.mean())}


def ptm(folder, model):
    matches = list((folder / "raw").glob(PREDICTION_PATTERNS[model]))
    if len(matches) != 1:
        raise ValueError(
            f"Cannot pair confidence: found {len(matches)} raw predictions"
        )
    raw = matches[0]
    if (
        hashlib.sha256(raw.read_bytes()).digest()
        != hashlib.sha256((folder / "prediction.cif").read_bytes()).digest()
    ):
        raise ValueError("Raw prediction differs from the evaluated prediction")
    stem = raw.stem
    names = {
        "boltz_restr": [f"confidence_{stem}.json"],
        "protenix_restr": [
            stem.replace("_sample_", "_summary_confidence_sample_") + ".json"
        ],
        "alphafold3_restr": [stem.removesuffix("model") + "summary_confidences.json"],
        "openfold-3_restr": [
            stem.removesuffix("_model") + "_confidences_aggregated.json"
        ],
        "chai-lab_restr": ["scores.model_idx_0.npz"],
        "esm_restr": ["model.confidence.json"],
    }[model]
    paths = [raw.parent / name for name in names if (raw.parent / name).is_file()]
    if len(paths) != 1:
        raise ValueError(f"Missing paired {model} confidence: {names[0]}")
    if model == "chai-lab_restr":
        with np.load(paths[0], allow_pickle=False) as data:
            values = np.asarray(data["ptm"]).reshape(-1)
            if len(values) != 1:
                raise ValueError("Expected one Chai pTM")
            value = float(values[0])
    else:
        value = read_json(paths[0]).get("ptm")
        if value is None:
            raise ValueError("Paired confidence contains no pTM")
        value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("pTM must be finite and within [0, 1]")
    return {"ptm": value}
