"""Dictionary-free polymer chemistry and periodic reference torsions.

RDKit's standard-residue templates supply bond orders without network access.
They are used only for chemical classification and the new torsion terms; they
never replace the predictor's reference coordinates or protonate the model.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

import numpy as np
from rdkit import Chem

from rgi_toolkit import _geometry as G
from rgi_toolkit._array_ops import get_ops
from rgi_toolkit._atom_names import normalise_atom_name
from rgi_toolkit._monlib_records import GeometryTarget, deduplicate

logger = logging.getLogger(__name__)
_AA = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
        strict=True,
    )
)
# Standard heavy-atom chi paths. Every consecutive four-atom window is one chi.
_CHI_PATHS = {
    "ARG": "N CA CB CG CD NE CZ NH2",
    "ASN": "N CA CB CG OD1",
    "ASP": "N CA CB CG OD1",
    "CYS": "N CA CB SG",
    "GLN": "N CA CB CG CD OE1",
    "GLU": "N CA CB CG CD OE1",
    "HIS": "N CA CB CG CD2",
    "ILE": "N CA CB CG1 CD1",
    "LEU": "N CA CB CG CD1",
    "LYS": "N CA CB CG CD CE NZ",
    "MET": "N CA CB CG SD CE",
    "PHE": "N CA CB CG CD1",
    "PRO": "N CA CB CG CD",
    "SER": "N CA CB OG",
    "THR": "N CA CB OG1",
    "TRP": "N CA CB CG CD1",
    "TYR": "N CA CB CG CD1",
    "VAL": "N CA CB CG1",
}


def atom_name(name):
    name = normalise_atom_name(name)
    return {"O1P": "OP1", "O2P": "OP2", "O3P": "OP3"}.get(name, name)


@lru_cache(maxsize=256)
def standard_residue(resname, mol_type, previous=True, following=True):
    """Return a read-only complete template and its central residue's atom names."""
    resname = (resname or "").strip().upper()
    if mol_type == "protein":
        code = _AA.get(resname)
        if code is None:
            return None
        seq = ("G" if previous else "") + code + ("G" if following else "")
        center, flavor = 1 + int(previous), 0
    elif mol_type in ("rna", "dna"):
        code = resname[1:] if resname.startswith("D") else resname
        if code not in ("ACGT" if mol_type == "dna" else "ACGU") or len(code) != 1:
            return None
        # The central nucleotide includes its phosphate irrespective of terminal
        # naming conventions. Only atoms actually present in the model are mapped.
        seq, center, flavor = "A" + code + "A", 2, 6 if mol_type == "dna" else 2
    else:
        return None
    mol = Chem.MolFromSequence(seq, flavor=flavor)
    names = {
        atom_name(a.GetPDBResidueInfo().GetName()): a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetPDBResidueInfo().GetResidueNumber() == center
    }
    return mol, names


def _reference_target(indices, coords, period, esd):
    xyz = np.asarray(coords, dtype=float)[list(indices)]
    if not np.isfinite(xyz).all():
        return None
    left = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[1])
    right = np.cross(xyz[2] - xyz[1], xyz[3] - xyz[2])
    if min(float(left @ left), float(right @ right)) <= 1e-12:
        return None
    value = float(G.dihedral_points(get_ops("numpy"), *xyz))
    return GeometryTarget(tuple(indices), value, math.radians(esd), period)


def sp2_torsions(mol, mapping, coords, excluded_axes=(), include_double=True):
    """Resolve acyclic conjugated torsions, preserving the state of real double bonds."""
    result = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if j not in mapping or k not in mapping or bond.IsInRing():
            continue
        axis = tuple(sorted((mapping[j], mapping[k])))
        if axis in excluded_axes:
            continue
        first, second = mol.GetAtomWithIdx(j), mol.GetAtomWithIdx(k)
        if any(
            a.GetHybridization() != Chem.HybridizationType.SP2 for a in (first, second)
        ):
            continue
        double = bond.GetBondType() == Chem.BondType.DOUBLE
        if double and not include_double:
            continue
        if bond.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
            continue
        for a in first.GetNeighbors():
            for b in second.GetNeighbors():
                i, end = a.GetIdx(), b.GetIdx()
                if i == k or end == j or i not in mapping or end not in mapping:
                    continue
                indices = tuple(mapping[n] for n in (i, j, k, end))
                row = _reference_target(indices, coords, 1 if double else 2, 5.0)
                if row is not None:
                    result.append(row)
    return deduplicate(result)


def add_polymer_torsions(targets, residues, connections, coords):
    """Add approximate torsions only where the configured library has no coverage."""
    from rgi_toolkit.monlib_geom import PeptideChoice

    before = len(targets.terms["cistrans"])
    failures = []
    for meta in residues:
        if targets.atoms.issuperset(meta["names"].values()) and meta["names"]:
            continue
        template = standard_residue(meta["resname"], meta["mol_type"])
        context = f"{meta['chain']}:{meta['resname']}:{meta['uid']}"
        if template is None:
            failures.append(context + " (unknown residue)")
            continue
        mol, template_names = template
        names = {atom_name(n): g for n, g in meta["names"].items()}
        mapping = {i: names[n] for n, i in template_names.items() if n in names}
        axes = set()
        path = _CHI_PATHS.get((meta["resname"] or "").strip().upper(), "").split()
        if meta["mol_type"] != "protein":
            path = []
        for number in range(max(0, len(path) - 3)):
            quad = path[number : number + 4]
            if not all(n in names and n in template_names for n in quad):
                failures.append(context + f" chi{number + 1} (missing atoms)")
                continue
            indices = tuple(names[n] for n in quad)
            centers = [mol.GetAtomWithIdx(template_names[n]) for n in quad[1:3]]
            n_sp2 = sum(
                a.GetHybridization() == Chem.HybridizationType.SP2 for a in centers
            )
            period = (3, 6, 2)[n_sp2]
            row = _reference_target(
                indices, coords, period, 5.0 if n_sp2 == 2 else 10.0
            )
            if row is None:
                failures.append(context + f" chi{number + 1} (degenerate reference)")
            else:
                targets.terms["cistrans"].append(row)
                axes.add(tuple(sorted(indices[1:3])))
        targets.terms["cistrans"].extend(sp2_torsions(mol, mapping, coords, axes))
    for previous, current in connections:
        if (
            current["mol_type"] != "protein"
            or (previous["uid"], current["uid"]) in targets.covered_links
        ):
            continue
        indices = tuple(
            meta["names"].get(name)
            for meta, name in (
                (previous, "CA"),
                (previous, "C"),
                (current, "N"),
                (current, "CA"),
            )
        )
        if any(i is None for i in indices):
            failures.append(
                f"{previous['chain']}:{previous['uid']}->{current['uid']} omega (missing atoms)"
            )
            continue
        selector = len(targets.peptides)
        targets.peptides.append(PeptideChoice(indices, -math.pi, 0.0))
        for cis, value in ((0, -math.pi), (1, 0.0)):
            targets.terms["cistrans"].append(
                GeometryTarget(
                    indices,
                    value,
                    math.radians(5),
                    1,
                    conditions=((selector, cis),),
                )
            )
    targets.terms["cistrans"] = deduplicate(targets.terms["cistrans"])
    count = len(targets.terms["cistrans"]) - before
    if count:
        logger.info(
            "[rgi_toolkit] approximate polymer torsions: %d rows (including cis/trans alternatives)",
            count,
        )
    if failures:
        logger.warning(
            "[rgi_toolkit] omitted approximate torsions: %s%s",
            "; ".join(failures[:8]),
            f"; and {len(failures) - 8} more" if len(failures) > 8 else "",
        )


def ligand_sp2_torsions(ligands):
    """Add conjugated single-bond torsions alongside the existing E/Z restraints."""
    result = []
    for lc in ligands:
        mol = lc.stereo_mol if lc.stereo_mol is not None else lc.mol
        # Local coordinates avoid allocating an array up to a padded global index.
        mapping = {i: i for i in range(mol.GetNumAtoms())}
        for row in sp2_torsions(mol, mapping, lc.conf_coords, include_double=False):
            result.append(
                GeometryTarget(
                    tuple(int(lc.global_indices[i]) for i in row.atoms),
                    row.value,
                    row.esd,
                    row.period,
                )
            )
    return result
