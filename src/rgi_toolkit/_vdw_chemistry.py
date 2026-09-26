"""RDKit elemental VdW radii and sparse covalent topology for all paths.

The unscaled contact is the sum of the two elemental radii, as in v0.1.0-a.
Source graphs and optional dictionaries supply topology, never radius overrides.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from rgi_toolkit._config_util import conformer_use_esd
from rgi_toolkit._moltype import polymer_type
from rgi_toolkit._polymer_torsions import atom_name, standard_residue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtomType:
    radius: float
    element: int


def elemental_type(element):
    """Use the same atomic-number lookup and invalid-element rule as v0.1.0-a."""
    if element is None or element < 1 or element > 118:
        return AtomType(0.0, 0)
    return AtomType(float(Chem.GetPeriodicTable().GetRvdw(int(element))), int(element))


def pair_contact(first, second):
    """Return the unscaled elemental radius sum and optional normalization ESD."""
    return first.radius + second.radius, 0.2


def molecule_types(mol, mapping, elements):
    """Validate source elements without inferring environment-dependent radii."""
    result = {}
    for i, g in mapping.items():
        if mol.GetAtomWithIdx(i).GetAtomicNum() != int(elements[g]):
            raise ValueError(f"VdW element mismatch at atom {g}")
        result[g] = elemental_type(int(elements[g]))
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
    use_esd: bool = False

    @property
    def radii(self):
        return np.asarray([a.radius for a in self.types])[self.type_ids]

    def pair(self, first, second):
        key = tuple(sorted((int(first), int(second))))
        distance = self.bond_distances.get(key, 4)
        if distance <= 3:
            return None
        r, sigma = pair_contact(
            self.types[self.type_ids[first]],
            self.types[self.type_ids[second]],
        )
        return r, 1 / sigma**2 if self.use_esd else 1.0

    def subset(self, query, target, moving, static_ligands, mode="both", active=False):
        """Pack O(N + sparse topology + T^2) constants, never a dense atom-pair matrix."""
        query, target = np.asarray(query), np.asarray(target)
        qmap = {int(g): i for i, g in enumerate(query)}
        tmap = {int(g): i for i, g in enumerate(target)}
        excluded = set()
        size = len(target)
        if len(query) * size > np.iinfo(np.int32).max:
            raise ValueError("VdW topology pair codes exceed int32 capacity")
        for a, b in self.bond_distances:
            for first, second in ((a, b), (b, a)):
                if first not in qmap or second not in tmap:
                    continue
                code = qmap[first] * size + tmap[second]
                if self.pair(a, b) is None:
                    excluded.add(code)
        return {
            "query_types": self.type_ids[query],
            "target_types": self.type_ids[target],
            "contacts": self.contact_table,
            "inv_variances": self.inv_variance_table,
            "one_four_contacts": self.one_four_table,
            "one_four_inv_variances": np.full_like(
                self.one_four_table, 1 / 0.2**2 if self.use_esd else 1.0
            ),
            "excluded": np.asarray(sorted(excluded), dtype=np.int64),
            # Retain the packed schema for compatibility with stored specifications.
            "one_four": np.empty(0, dtype=np.int64),
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
    """Build elemental contacts and topology for moving and background atoms."""
    from rgi_toolkit import monlib_geom

    use_esd = conformer_use_esd(config)
    if elements is None:
        n = max((int(g) + 1 for lc in ligands for g in lc.global_indices), default=0)
        elements = np.zeros(n, dtype=np.int64)
        for lc in ligands:
            elements[lc.global_indices] = [a.GetAtomicNum() for a in lc.mol.GetAtoms()]
    elements = np.asarray(elements)
    records = [
        r for r in records if 0 <= r.index < len(elements) and elements[r.index] > 0
    ]
    atom_types = {i: elemental_type(int(z)) for i, z in enumerate(elements)}
    all_bonds = {tuple(sorted((int(a), int(b)))) for a, b, *_ in bonds}
    all_planes = {tuple(sorted(p)) for p in planes}
    molecules = np.arange(len(elements), dtype=np.int64)
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
                inferred = molecule_types(mol, mapping, elements)
                atom_types.update(inferred)
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
        inferred = molecule_types(mol, mapping, elements)
        atom_types.update(inferred)
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
        inverse[i, j] = 1 / sigma**2 if use_esd else 1.0
        one_four[i, j] = contact[i, j]
    logger.info(
        "[rgi_toolkit] VdW typing: RDKit elemental radii, %d atom types; use_esd=%s",
        len(types),
        use_esd,
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
        use_esd=use_esd,
    )
