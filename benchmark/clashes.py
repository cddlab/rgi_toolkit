"""PoseBusters clash criteria shared with the published benchmark."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import gemmi
import numpy as np
import yaml
from rdkit import Chem

INTER_FIELDS = ("clash_ok", "clash_rel", "clash_min")
INTRA_FIELDS = ("iclash_ok", "iclash_rel", "iclash_vdw_rel")


@cache
def buster(function):
    from posebusters import PoseBusters

    config = yaml.safe_load(Path(__file__).with_name("posebusters.yaml").read_text())
    return PoseBusters(
        config={"modules": [m for m in config["modules"] if m["function"] == function]}
    )


def protein_mol(path):
    structure = gemmi.read_structure(str(path))
    if not len(structure):
        raise ValueError("No structural model for intermolecular clash evaluation")
    if "B" in structure[0]:
        structure[0].remove_chain("B")
    structure.remove_hydrogens()
    mol = Chem.MolFromPDBBlock(
        structure.make_pdb_string(),
        sanitize=False,
        removeHs=False,
        proximityBonding=False,
    )
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("No protein atoms for intermolecular clash evaluation")
    return mol


def record(function, mol, protein=None):
    table = buster(function).bust(mol_pred=mol, mol_cond=protein, full_report=True)
    if len(table) != 1:
        raise ValueError("PoseBusters did not return exactly one result")
    return table.to_dict(orient="records")[0]


def number(value):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("PoseBusters returned a non-finite distance")
    return result


def boolean(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("PoseBusters returned no pass/fail result")
    return bool(value)


def intermolecular(mol, path):
    values = record("intermolecular_distance", mol, protein_mol(path))
    return {
        "clash_ok": boolean(values["minimum_distance_to_protein"]),
        "clash_rel": number(values["most_extreme_relative_distance_protein"]),
        "clash_min": number(values["smallest_distance_protein"]),
    }


def intramolecular(mol):
    heavy = Chem.RemoveAllHs(mol)
    i, j = np.where(np.triu(Chem.GetDistanceMatrix(heavy) > 2, k=1))
    if not len(i):
        return dict.fromkeys(INTRA_FIELDS)
    coordinates = np.asarray(heavy.GetConformer().GetPositions())
    radii = np.array(
        [Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in heavy.GetAtoms()]
    )
    relative = np.linalg.norm(coordinates[i] - coordinates[j], axis=-1) / (
        radii[i] + radii[j]
    )
    values = record("distance_geometry", mol)
    return {
        "iclash_ok": boolean(values["internal_steric_clash"]),
        "iclash_rel": number(values["shortest_noncovalent_relative_distance"]),
        "iclash_vdw_rel": number(np.min(relative)),
    }
