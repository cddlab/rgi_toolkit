"""Host-side chemical typing and sparse topology for all VdW paths.

Contact formulas follow Servalcat's Geometry::set_vdw_values. A configured CCP4
library supplies exact energy types; otherwise RDKit supplies approximate local
chemistry, with built-in elemental parameters and no dictionary download.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDConfig

from rgi_toolkit._moltype import polymer_type
from rgi_toolkit._polymer_torsions import atom_name, standard_residue

logger = logging.getLogger(__name__)
# CCP4 ener_lib elemental defaults: hydrogen-inclusive VdW radius, ionic radius,
# hydrogen-bond type (N none, A acceptor, D donor, B both, H donor hydrogen).
# These constants are the offline approximation, not a generated monomer database.
_ELEMENTS = {
    "H": (1.2, 0, "N"),
    "C": (1.75, 0, "N"),
    "N": (1.6, 1.32, "N"),
    "O": (1.52, 1.28, "A"),
    "S": (1.88, 1.7, "A"),
    "P": (1.88, 1.79, "N"),
    "F": (1.47, 1.19, "B"),
    "CL": (1.75, 1.67, "A"),
    "BR": (1.85, 0.73, "N"),
    "I": (1.98, 0.56, "N"),
    "B": (0.85, 0.25, "N"),
    "SI": (2.1, 0.4, "N"),
    "SE": (1.9, 0.42, "N"),
    "LI": (1.82, 0.73, "N"),
    "NA": (2.27, 1.13, "N"),
    "K": (2.75, 1.51, "N"),
    "RB": (2.0, 1.48, "N"),
    "CS": (2.98, 1.81, "N"),
    "MG": (1.73, 0.71, "N"),
    "CA": (1.94, 1.14, "N"),
    "SR": (2.19, 1.32, "N"),
    "BA": (2.53, 1.49, "N"),
    "MN": (1.4, 0.46, "N"),
    "FE": (1.4, 0.68, "N"),
    "CO": (1.35, 0.54, "N"),
    "NI": (1.63, 0.63, "N"),
    "CU": (1.4, 0.71, "N"),
    "ZN": (1.39, 0.74, "N"),
    "CD": (1.58, 0.92, "N"),
    "HG": (1.55, 1.1, "N"),
    "AL": (1.25, 0.53, "N"),
}
_NONMETALS = {
    0,
    1,
    2,
    5,
    6,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    18,
    32,
    33,
    34,
    35,
    36,
    51,
    52,
    53,
    54,
    85,
    86,
    117,
    118,
}


@dataclass(frozen=True)
class AtomType:
    radius: float
    ion: float
    hb: str
    element: int
    dummy: bool = False


def elemental_type(element, dummy=False):
    if element <= 0:
        return AtomType(0, 0, "N", 0, dummy)
    table = Chem.GetPeriodicTable()
    symbol = table.GetElementSymbol(int(element)).upper()
    radius, ion, hb = _ELEMENTS.get(symbol, (table.GetRvdw(int(element)), 0, "N"))
    return AtomType(min(2.0, radius), ion, hb, int(element), dummy)


def library_type(atom, element, dummy=False):
    radius = atom.vdwh_radius if math.isfinite(atom.vdwh_radius) else atom.vdw_radius
    if not math.isfinite(radius) or radius <= 0:
        return None
    ion = atom.ion_radius if math.isfinite(atom.ion_radius) else 0.0
    return AtomType(min(2.0, radius), ion, atom.hb_type, int(element), dummy)


def pair_contact(first, second, one_four=False):
    """Return critical distance and ESD using Servalcat's contact-type priority."""
    r1, r2 = first.radius, second.radius
    if r1 <= 0 or r2 <= 0:
        return 0.0, 0.2
    if one_four:
        return r1 + r2 - sum(
            0.1 if a.element in (7, 8) else 0.15 for a in (first, second)
        ), 0.2
    a, b = first.hb, second.hb
    if (a in ("A", "B") and b in ("D", "B")) or (b in ("A", "B") and a in ("D", "B")):
        return r1 + r2 - 0.3, 0.2
    if a in ("A", "B") and b == "H":
        return r1 + 0.1, 0.2
    if b in ("A", "B") and a == "H":
        return r2 + 0.1, 0.2
    if (
        any(a.element not in _NONMETALS for a in (first, second))
        and first.ion > 0
        and second.ion > 0
    ):
        return first.ion + second.ion, 0.2
    if first.dummy != second.dummy:
        return max(0.7, r1 + r2 - 0.7), 0.3
    if first.dummy and second.dummy:
        return r1 + r2, 0.3
    return r1 + r2, 0.2


@lru_cache(maxsize=1)
def _feature_factory():
    from rdkit.Chem import ChemicalFeatures

    return ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )


def molecule_types(mol, mapping, elements, records):
    """Infer chemistry without changing source charge, hydrogen count or stereo."""
    result = {}
    for i, g in mapping.items():
        if mol.GetAtomWithIdx(i).GetAtomicNum() != int(elements[g]):
            raise ValueError(f"VdW element mismatch at atom {g}")
    try:
        features = _feature_factory().GetFeaturesForMol(mol)
        donors = {
            i for f in features if f.GetFamily() == "Donor" for i in f.GetAtomIds()
        }
        acceptors = {
            i for f in features if f.GetFamily() == "Acceptor" for i in f.GetAtomIds()
        }
        for i, g in mapping.items():
            a = mol.GetAtomWithIdx(i)
            base = elemental_type(
                int(elements[g]),
                (getattr(records.get(g), "name", "") or "").startswith("DUM"),
            )
            radius = base.radius
            nh = a.GetTotalNumHs()
            if a.GetAtomicNum() == 6:
                if a.GetHybridization() == Chem.HybridizationType.SP3 and nh:
                    radius = {1: 1.95, 2: 1.92, 3: 1.94}.get(nh, 1.94)
                elif a.GetIsAromatic():
                    radius = 1.82 if nh else 1.74
                elif a.GetHybridization() == Chem.HybridizationType.SP2 and nh:
                    radius = 1.82 if nh == 1 else 1.8
            elif a.GetAtomicNum() == 16 and nh:
                radius = 1.95
            hb = (
                "B"
                if i in donors and i in acceptors
                else "D"
                if i in donors
                else "A"
                if i in acceptors
                else "N"
            )
            if a.GetAtomicNum() == 1:
                hb = "H" if any(n.GetIdx() in donors for n in a.GetNeighbors()) else "N"
            result[g] = AtomType(radius, base.ion, hb, base.element, base.dummy)
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "[rgi_toolkit] VdW chemical graph unavailable (%s); using elemental types",
            exc,
        )
        return {}
    return result


def _add_graph(mol, mapping, bonds, planes):
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in mapping and j in mapping:
            bonds.add(tuple(sorted((mapping[i], mapping[j]))))
    groups = [
        group
        for group in mol.GetRingInfo().AtomRings()
        if all(
            mol.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP2
            for i in group
        )
    ]
    for a in mol.GetAtoms():
        if a.GetHybridization() == Chem.HybridizationType.SP2:
            groups.append((a.GetIdx(), *(n.GetIdx() for n in a.GetNeighbors())))
    for group in groups:
        if len(group) >= 4 and all(i in mapping for i in group):
            planes.add(tuple(sorted(mapping[i] for i in group)))


def _residue_groups(records, reference_uids):
    groups = {}
    for r in records:
        kind = polymer_type(r.mol_type, r.resname) or r.mol_type
        key = (r.chain, r.resid, kind)
        if (
            reference_uids is not None
            and 0 <= r.index < len(reference_uids)
            and reference_uids[r.index] >= 0
        ):
            key = (r.chain, int(reference_uids[r.index]), kind)
        groups.setdefault(key, []).append(r)
    metas = []
    for uid, group in enumerate(groups.values()):
        names = {}
        duplicates = set()
        for r in group:
            name = atom_name(r.name)
            if name:
                if name in names:
                    duplicates.add(name)
                names[name] = int(r.index)
        for name in duplicates:
            del names[name]
        if duplicates:
            logger.warning(
                "[rgi_toolkit] ambiguous VdW atom names in %s:%s: %s; using graph/element fallback",
                group[0].chain,
                group[0].resid,
                sorted(duplicates),
            )
        metas.append(
            dict(
                uid=uid,
                chain=group[0].chain,
                resname=group[0].resname,
                mol_type=polymer_type(group[0].mol_type, group[0].resname)
                or group[0].mol_type,
                names=names,
                order=min(r.index for r in group),
                records=group,
            )
        )
    return metas


@dataclass
class VdwChemistry:
    types: list[AtomType]
    type_ids: np.ndarray
    molecules: np.ndarray
    bond_distances: dict[tuple[int, int], int]
    planes_by_atom: dict[int, set[int]]
    contact_table: np.ndarray
    inv_variance_table: np.ndarray
    one_four_table: np.ndarray

    @property
    def radii(self):
        return np.asarray([a.radius for a in self.types])[self.type_ids]

    def pair(self, first, second):
        key = tuple(sorted((int(first), int(second))))
        distance = self.bond_distances.get(key, 4)
        if distance < 3 or (
            distance == 3
            and self.planes_by_atom.get(key[0], set())
            & self.planes_by_atom.get(key[1], set())
        ):
            return None
        r, sigma = pair_contact(
            self.types[self.type_ids[first]],
            self.types[self.type_ids[second]],
            distance == 3,
        )
        return r, 1 / sigma**2

    def subset(self, query, target, moving, static_ligands, mode="both", active=False):
        """Pack O(N + sparse topology + T^2) constants, never a dense atom-pair matrix."""
        query, target = np.asarray(query), np.asarray(target)
        qmap = {int(g): i for i, g in enumerate(query)}
        tmap = {int(g): i for i, g in enumerate(target)}
        excluded, one_four = set(), set()
        size = len(target)
        if len(query) * size > np.iinfo(np.int32).max:
            raise ValueError("VdW topology pair codes exceed int32 capacity")
        for (a, b), distance in self.bond_distances.items():
            for first, second in ((a, b), (b, a)):
                if first not in qmap or second not in tmap:
                    continue
                code = qmap[first] * size + tmap[second]
                if self.pair(a, b) is None:
                    excluded.add(code)
                elif distance == 3:
                    one_four.add(code)
        return {
            "query_types": self.type_ids[query],
            "target_types": self.type_ids[target],
            "contacts": self.contact_table,
            "inv_variances": self.inv_variance_table,
            "one_four_contacts": self.one_four_table,
            "one_four_inv_variances": np.full_like(self.one_four_table, 1 / 0.2**2),
            "excluded": np.asarray(sorted(excluded), dtype=np.int64),
            "one_four": np.asarray(sorted(one_four), dtype=np.int64),
            "query_molecules": self.molecules[query],
            "target_molecules": self.molecules[target],
            "query_moving": np.isin(query, list(moving)),
            "target_moving": np.isin(target, list(moving)),
            "query_static": np.isin(query, list(static_ligands))
            if active
            else np.zeros(len(query), bool),
            "target_static": np.isin(target, list(static_ligands))
            if active
            else np.zeros(len(target), bool),
            "mode": np.asarray(
                {"both": 0, "intramolecular": 1, "intermolecular": 2}[mode],
                dtype=np.int64,
            ),
        }


def build_chemistry(
    ligands,
    elements,
    records=(),
    config=None,
    reference_uids=None,
    library=None,
    bonds=(),
    planes=(),
):
    """Build chemical parameters for moving atoms and the entire fixed background."""
    from rgi_toolkit import monlib_geom

    if elements is None:
        n = max((int(g) + 1 for lc in ligands for g in lc.global_indices), default=0)
        elements = np.zeros(n, dtype=np.int64)
        for lc in ligands:
            elements[lc.global_indices] = [a.GetAtomicNum() for a in lc.mol.GetAtoms()]
    elements = np.asarray(elements)
    records = [
        r for r in records if 0 <= r.index < len(elements) and elements[r.index] > 0
    ]
    by_index = {int(r.index): r for r in records}
    atom_types = {
        i: elemental_type(
            int(z), (getattr(by_index.get(i), "name", "") or "").startswith("DUM")
        )
        for i, z in enumerate(elements)
    }
    all_bonds = {tuple(sorted((int(a), int(b)))) for a, b, *_ in bonds}
    all_planes = {tuple(sorted(p)) for p in planes}
    molecules = np.arange(len(elements), dtype=np.int64)
    approximate, dictionary = set(), set()
    metas = _residue_groups(records, reference_uids)
    chain_groups = {}
    for meta in metas:
        for record in meta["records"]:
            molecules[record.index] = meta["order"]
        if meta["mol_type"] in ("protein", "rna", "dna"):
            chain_groups.setdefault((meta["chain"], meta["mol_type"]), []).append(meta)
    connections = []
    for group in chain_groups.values():
        group.sort(key=lambda m: m["order"])
        molecule = group[0]["order"]
        for pos, meta in enumerate(group):
            for record in meta["records"]:
                molecules[record.index] = molecule
            template = standard_residue(
                meta["resname"], meta["mol_type"], pos > 0, pos + 1 < len(group)
            )
            if template is not None:
                mol, names = template
                mapping = {
                    i: meta["names"][n] for n, i in names.items() if n in meta["names"]
                }
                inferred = molecule_types(mol, mapping, elements, by_index)
                atom_types.update(inferred)
                approximate.update(inferred)
                _add_graph(mol, mapping, all_bonds, all_planes)
        for previous, current in zip(group, group[1:]):
            connections.append((previous, current))
            name1, name2 = (
                ("C", "N") if current["mol_type"] == "protein" else ("O3'", "P")
            )
            a, b = previous["names"].get(name1), current["names"].get(name2)
            if a is not None and b is not None:
                all_bonds.add(tuple(sorted((a, b))))
            if current["mol_type"] == "protein":
                plane = [previous["names"].get(n) for n in ("CA", "C", "O")] + [b]
                if all(g is not None for g in plane):
                    all_planes.add(tuple(sorted(plane)))
    for lc in ligands:
        mol = lc.stereo_mol if lc.stereo_mol is not None else lc.mol
        mapping = {i: int(g) for i, g in enumerate(lc.global_indices)}
        inferred = molecule_types(mol, mapping, elements, by_index)
        atom_types.update(inferred)
        approximate.update(inferred)
        _add_graph(mol, mapping, all_bonds, all_planes)
        if mapping:
            molecules[list(mapping.values())] = min(mapping.values())
    lib_config = monlib_geom.parse_config(config)
    if lib_config is not None and metas:
        path, _ = lib_config
        names = {m["resname"] for m in metas if m["resname"]}
        if library is None or not all(library.covers(n) for n in names):
            library = monlib_geom.MonomerLibrary.load(
                library.path if library is not None else path, names
            )
        topology = monlib_geom.collect(
            library, metas, "fallback", connections, enabled=set()
        )
        covered_groups = [
            set(m["names"].values()) for m in metas if library.covers(m["resname"])
        ]
        group_index = {g: i for i, group in enumerate(covered_groups) for g in group}
        all_bonds = {
            p
            for p in all_bonds
            if not (p[0] in group_index and group_index.get(p[1]) == group_index[p[0]])
        }
        all_bonds.update(topology.bond_pairs)
        all_planes = {
            p
            for p in all_planes
            if not (
                p
                and p[0] in group_index
                and all(group_index.get(g) == group_index[p[0]] for g in p)
            )
        }
        all_planes.update(topology.plane_groups)
        missing_types = {}
        for g, name in topology.atom_types.items():
            atom = library._monlib.ener_lib.atoms.get(name)
            value = (
                library_type(atom, elements[g], atom_types[g].dummy)
                if atom is not None
                else None
            )
            if value is None:
                missing_types[name] = missing_types.get(name, 0) + 1
                symbol = (
                    Chem.GetPeriodicTable().GetElementSymbol(int(elements[g])).upper()
                )
                atom = library._monlib.ener_lib.atoms.get(symbol)
                value = (
                    library_type(atom, elements[g], atom_types[g].dummy)
                    if atom is not None
                    else None
                ) or elemental_type(int(elements[g]), atom_types[g].dummy)
                approximate.discard(g)
            else:
                dictionary.add(g)
            atom_types[g] = value
        if missing_types:
            logger.warning(
                "[rgi_toolkit] unknown VdW chemical types %s; using elemental types",
                missing_types,
            )
    # Explicit donor hydrogens take their parent's chemical class, as in Servalcat.
    for a, b in all_bonds:
        for h, parent in ((a, b), (b, a)):
            if elements[h] == 1:
                old = atom_types[h]
                atom_types[h] = AtomType(
                    old.radius,
                    old.ion,
                    "H" if atom_types[parent].hb in ("D", "B") else "N",
                    1,
                    old.dummy,
                )
    planes_by_atom = {}
    for index, group in enumerate(sorted(all_planes)):
        for g in group:
            planes_by_atom.setdefault(g, set()).add(index)
    adjacency = {}
    for a, b in all_bonds:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    distances = {}
    for start in adjacency:
        seen, frontier = {start}, {start}
        for distance in range(1, 4):
            frontier = {b for a in frontier for b in adjacency.get(a, ())} - seen
            for end in frontier:
                if end > start:
                    distances[start, end] = distance
            seen.update(frontier)
    type_map = {}
    type_ids = np.asarray(
        [
            type_map.setdefault(atom_types[g], len(type_map))
            for g in range(len(elements))
        ],
        dtype=np.int64,
    )
    types = list(type_map)
    contact = np.zeros((len(types), len(types)))
    inverse = np.zeros_like(contact)
    one_four = np.zeros_like(contact)
    for i, j in itertools.product(range(len(types)), repeat=2):
        contact[i, j], sigma = pair_contact(types[i], types[j])
        inverse[i, j] = 1 / sigma**2
        one_four[i, j] = pair_contact(types[i], types[j], True)[0]
    fallback = {i for i, z in enumerate(elements) if z > 0} - approximate - dictionary
    logger.info(
        "[rgi_toolkit] VdW typing: dictionary=%d approximate=%d elemental=%d; ESD=0.2 A (dummy=0.3 A)",
        len(dictionary),
        len(approximate - dictionary),
        len(fallback),
    )
    if fallback:
        logger.warning(
            "[rgi_toolkit] VdW elemental fallback for %d atoms (indices %s)",
            len(fallback),
            sorted(fallback)[:8],
        )
    return VdwChemistry(
        types,
        type_ids,
        molecules,
        distances,
        planes_by_atom,
        contact,
        inverse,
        one_four,
    )
