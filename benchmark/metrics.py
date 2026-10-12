"""Geometry and stereochemistry in the prepared protocol's atom selections."""

from __future__ import annotations

from collections import Counter
from functools import cache
from itertools import combinations

import numpy as np
from rdkit import Chem
from rgi_toolkit._mol_build import (
    _coordinate_stereo_labels,
    _graph_stereo_labels,
    ff_relax,
    generate_ideal_conformer,
    repair_stereo,
)
from rgi_toolkit.atom_context import candidate_dict
from rgi_toolkit.selection import AtomSelector

from common import coordinates


def selected(atoms, xyz, selection):
    selector = AtomSelector(selection)
    indices = [
        i
        for i, atom in enumerate(atoms)
        if selector.matches(candidate_dict(atom, "res_name"))
    ]
    if not indices:
        raise ValueError(f"Empty selection: {selection}")
    return xyz[indices], [atoms[i] for i in indices]


def angle(a, b, c):
    u, v = a - b, c - b
    scale = np.linalg.norm(u) * np.linalg.norm(v)
    if scale < 1e-12:
        raise ValueError("Degenerate angle")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / scale, -1, 1))))


def rmsd(a, b):
    if a.shape != b.shape or len(a) < 3:
        raise ValueError("RMSD needs matching sets of at least three atoms")
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rotation - b) ** 2, axis=1))))


def state_metrics(atoms, xyz, job, config, folder):
    kind, rc = job["benchmark"], config["restraints"]

    def center(selection):
        return selected(atoms, xyz, selection)[0].mean(0)

    if kind in {"distance", "angle"}:
        entry = rc[f"{kind}_restraints_config"][0]
        centers = [
            center(entry[f"atom_selection{i}"])
            for i in range(1, 3 if kind == "distance" else 4)
        ]
        value = (
            float(np.linalg.norm(centers[0] - centers[1]))
            if kind == "distance"
            else angle(*centers)
        )
        key = "distance_A" if kind == "distance" else "angle_deg"
        result = {key: value}
        if job["target"] is not None:
            result["absolute_error"] = abs(value - job["target"])
        return result
    if kind == "custom":
        groups = rc["custom_restraints_config"][0]["selections"]
        a, b, c, d = (center(groups[key]) for key in "ABCD")
        din, dout = float(np.linalg.norm(a - b)), float(np.linalg.norm(c - d))
        result = {
            "distance_in_A": din,
            "distance_out_A": dout,
            "difference_A": din - dout,
        }
        if job["target"] is not None:
            result["absolute_error"] = abs(din - dout - job["target"])
        return result
    result = {}
    for i, entry in enumerate(rc["rmsd_restraints_config"]):
        ref_atoms, ref_xyz = coordinates(folder / entry["ref_cif"])
        pred, pred_atoms = selected(atoms, xyz, entry["atom_selection_target_calc"])
        ref, reference_atoms = selected(
            ref_atoms, ref_xyz, entry["atom_selection_ref_calc"]
        )
        if len(pred) != 220 or [a.res_name for a in pred_atoms] != [
            a.res_name for a in reference_atoms
        ]:
            raise ValueError(
                "RMSD protocol requires the same 220 corresponding CA atoms"
            )
        value = rmsd(pred, ref)
        result[f"rmsd_{'open' if i == 0 else 'closed'}_A"] = value
        if job["target"] is not None:
            result[f"absolute_error_{i + 1}"] = abs(value - job["target"][i])
    return result


def ligand_names(mol, model):
    expanded = Chem.AddHs(mol)
    symbols = [atom.GetSymbol().upper() for atom in expanded.GetAtoms()]
    if model in {"boltz_restr", "esm_restr"}:
        ranks = list(Chem.CanonicalRankAtoms(expanded))
        names = [
            f"{symbol}{rank + 1}" for symbol, rank in zip(symbols, ranks, strict=True)
        ]
        if any(len(name) > 4 for name in names):
            counts = Counter()
            for i in sorted(range(len(symbols)), key=ranks.__getitem__):
                counts[symbols[i]] += 1
                names[i] = f"{symbols[i]}{counts[symbols[i]]}"
    else:
        counts = Counter()
        names = []
        for symbol in symbols:
            counts[symbol] += 1
            names.append(f"{symbol}{counts[symbol]}")
    return names[: mol.GetNumAtoms()]


@cache
def ideal(smiles):
    mol = Chem.MolFromSmiles(smiles)
    xyz = generate_ideal_conformer(mol)
    if xyz is None:
        raise ValueError("Reference conformer embedding failed")
    if any(
        b.GetIsAromatic() or b.GetBondType() == Chem.BondType.DOUBLE
        for b in mol.GetBonds()
    ):
        relaxed = ff_relax(mol, xyz, "uff", stereo_mol=mol)
        xyz = relaxed if relaxed is not None else repair_stereo(mol, xyz, mol)
    else:
        xyz = repair_stereo(mol, xyz, mol)
    return mol, np.asarray(xyz)


def geometry_difference(mol, a, b):
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    angles = [
        (i, atom.GetIdx(), k)
        for atom in mol.GetAtoms()
        for i, k in combinations([n.GetIdx() for n in atom.GetNeighbors()], 2)
    ]
    bd = [np.linalg.norm(a[i] - a[j]) - np.linalg.norm(b[i] - b[j]) for i, j in bonds]
    ad = [angle(a[i], a[j], a[k]) - angle(b[i], b[j], b[k]) for i, j, k in angles]
    return (
        float(np.sqrt(np.mean(np.square(bd)))) if bd else None,
        float(np.sqrt(np.mean(np.square(ad)))) if ad else None,
    )


def benchmark_ez(mol, labels):
    result = {}
    for (i, j), label in labels.items():
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond.IsInRing() or bond.GetIsAromatic():
            continue
        ends = [(mol.GetAtomWithIdx(i), j), (mol.GetAtomWithIdx(j), i)]
        for atom, other in ends:
            neighbors = [
                n
                for n in atom.GetNeighbors()
                if n.GetIdx() != other and n.GetAtomicNum() > 1
            ]
            if not neighbors or (atom.GetAtomicNum() == 7 and atom.GetTotalNumHs()):
                break
            if any(
                n.GetAtomicNum() in {7, 8, 16} and n.GetTotalNumHs() for n in neighbors
            ):
                break
        else:
            result[i, j] = label
    return result


def predicted_ligand(atoms, xyz, job):
    mol = Chem.MolFromSmiles(job["smiles"])
    ligand = [(a, p) for a, p in zip(atoms, xyz, strict=True) if a.chain == "B"]
    by_name = {a.name.upper(): (a, p) for a, p in ligand}
    names = ligand_names(mol, job["model"])
    if len(by_name) != len(ligand) or set(by_name) != set(names):
        raise ValueError(
            "Ligand atom names differ from the model's SMILES naming scheme"
        )
    ordered = [by_name[name] for name in names]
    if [a.element.upper() for a, _ in ordered] != [
        a.GetSymbol().upper() for a in mol.GetAtoms()
    ]:
        raise ValueError("Ligand element mapping mismatch")
    pred = np.array([p for _, p in ordered])
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, point in enumerate(pred):
        conformer.SetAtomPosition(i, point)
    mol.AddConformer(conformer)
    return mol


def conformer_metrics(atoms, xyz, job, folder):
    mol = predicted_ligand(atoms, xyz, job)
    pred = np.asarray(mol.GetConformer().GetPositions())
    ref_mol, ref = ideal(Chem.MolToSmiles(mol))
    match = ref_mol.GetSubstructMatch(mol, useChirality=True)
    if len(match) != len(pred):
        raise ValueError("Reference graph mapping failed")
    bond, angular = geometry_difference(mol, pred, ref[list(match)])
    expected, observed = _graph_stereo_labels(mol), _coordinate_stereo_labels(mol, pred)
    expected = expected[0], benchmark_ez(mol, expected[1])
    result = {"ideal_bond_rms_A": bond, "ideal_angle_rms_deg": angular}
    for label, reference_labels, prediction_labels in zip(
        ("chiral", "ez"), expected, observed, strict=True
    ):
        result[f"{label}_sites"] = len(reference_labels)
        result[f"{label}_match"] = (
            all(
                prediction_labels.get(i) == value
                for i, value in reference_labels.items()
            )
            if reference_labels
            else None
        )
    crystal = next(
        iter(
            Chem.SDMolSupplier(
                str(folder / "../../data" / f"{job['system']}.sdf"), removeHs=True
            )
        ),
        None,
    )
    if crystal is None:
        raise ValueError("Cannot read crystal ligand SDF")
    match = crystal.GetSubstructMatch(mol, useChirality=True)
    if len(match) != len(pred):
        raise ValueError("Crystal ligand graph differs from the input SMILES")
    crystal_xyz = crystal.GetConformer().GetPositions()[list(match)]
    bond, angular = geometry_difference(mol, pred, crystal_xyz)
    result.update(
        crystal_bond_rms_A=bond,
        crystal_angle_rms_deg=angular,
        ligand_rmsd_A=rmsd(pred, crystal_xyz),
    )
    return result
