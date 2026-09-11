"""Build local polymer geometry restraints from per-residue reference coordinates.

Reference conformers in the supported predictors are residue-local: each
``ref_space_uid`` identifies one independently positioned CCD component. That makes
them suitable targets for intra-residue bonds, angles, chirality and planar groups
(aromatic side chains / nucleic-acid bases), but not for measuring inter-residue link
geometry. Canonical peptide and phosphodiester links (and the peptide plane) are
therefore supplied explicitly below.

Those reference conformers are approximate chemistry, not refinement geometry (AF3
ETKDG-embeds the free CCD component). Set
``conformer_restraints_config.monomer_library`` to take geometry targets and ESDs
from the CCP4 monomer library instead; see ``monlib_geom`` for chiral propagation,
the omega/sp2 torsion subset, and cis/trans link alternatives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from rgi_toolkit import monlib_geom
from rgi_toolkit._atom_names import normalise_atom_name as _normalise_name
from rgi_toolkit._config_util import conformer_weight
from rgi_toolkit._mol_build import build_ligand_mol
from rgi_toolkit._moltype import polymer_type
from rgi_toolkit.atom_context import LigandConf

logger = logging.getLogger(__name__)


@dataclass
class PolymerGeometry:
    """Reference residue conformers plus canonical inter-residue geometry."""

    residue_confs: list[LigandConf]
    atom_indices: np.ndarray
    link_bonds: list[tuple[int, int, float, float]]
    link_angles: list[tuple[int, int, int, float, float]]
    # Canonical inter-residue planar groups (e.g. the peptide plane): each a tuple of
    # global atom indices scored by the `plane` term (best-fit-plane flatness).
    link_planes: list[tuple[int, ...]]
    # Dictionary geometry stays separate from built-in fallback tolerances.
    library: monlib_geom.LibraryTargets = field(
        default_factory=monlib_geom.LibraryTargets
    )


# Each link atom explicitly selects the previous or current residue.
_PREV, _CURR = 0, 1


@dataclass(frozen=True)
class _LinkGeometry:
    """Canonical inter-residue link targets as (atom_name, side) references.

    ``bond`` is ``((name, side), (name, side), target_angstrom)``; each ``angles`` entry
    is three ``(name, side)`` atoms plus a target in DEGREES; each ``planes`` entry is an
    N-tuple of ``(name, side)`` atoms forming one planar group restrained to zero
    out-of-plane deviation (peptide-plane flatness, carried by the `plane` energy leaf
    so polymer chemistry stays bond/angle/chiral/plane/VdW).
    """

    bond: tuple
    angles: tuple
    bond_esd: float
    planes: tuple = ()


# Fallback link geometry: Engh-Huber peptide targets and CCP4 `p` phosphodiester
# targets. Used when no monomer library is configured.
_PEPTIDE_BOND = 1.329
_PEPTIDE_BOND_ESD = 0.011
_LINK_ANGLE_ESD = 1.5
_PHOSPHODIESTER_BOND = 1.607
_PHOSPHODIESTER_BOND_ESD = 0.010
_PROTEIN_LINK = _LinkGeometry(
    bond=(("C", _PREV), ("N", _CURR), _PEPTIDE_BOND),
    bond_esd=_PEPTIDE_BOND_ESD,
    angles=(
        (("CA", _PREV), ("C", _PREV), ("N", _CURR), 116.2),
        (("O", _PREV), ("C", _PREV), ("N", _CURR), 122.7),
        (("C", _PREV), ("N", _CURR), ("CA", _CURR), 121.7),
    ),
    # Refmac/servalcat TRANS plan-1. Exclude the current residue's CA so the
    # plane does not constrain omega, which has its own torsion restraint.
    # The second plane has only three heavy atoms and adds no flatness constraint.
    planes=((("C", _PREV), ("CA", _PREV), ("O", _PREV), ("N", _CURR)),),
)
# Phosphodiester geometry from the CCP4 `p` link.
_NUCLEIC_LINK = _LinkGeometry(
    bond=(("O3'", _PREV), ("P", _CURR), _PHOSPHODIESTER_BOND),
    bond_esd=_PHOSPHODIESTER_BOND_ESD,
    angles=(
        (("C3'", _PREV), ("O3'", _PREV), ("P", _CURR), 121.082),
        (("OP1", _CURR), ("P", _CURR), ("O3'", _PREV), 109.493),
        (("OP2", _CURR), ("P", _CURR), ("O3'", _PREV), 109.493),
        (("O5'", _CURR), ("P", _CURR), ("O3'", _PREV), 100.661),
    ),
)


def _is_enabled_polymer(record) -> bool:
    """A polymer atom record whose chain opted into conformer restraints."""
    return polymer_type(record.mol_type, record.resname) is not None and bool(
        getattr(record, "conformer_restraints", False)
    )


def _link_geometry(previous, current, mol_type: str):
    """Built-in fallback geometry; its historical ESD-as-slack behavior is retained."""
    names = (previous["names"], current["names"])
    link = _PROTEIN_LINK if mol_type == "protein" else _NUCLEIC_LINK

    def resolve(atom):
        name, side = atom
        return names[side].get(name)

    b0, b1, bond_target = link.bond
    g0, g1 = resolve(b0), resolve(b1)
    bonds = [] if g0 is None or g1 is None else [(g0, g1, bond_target, link.bond_esd)]
    angles = []
    for a0, a1, a2, degrees in link.angles:
        idx = tuple(resolve(a) for a in (a0, a1, a2))
        if all(i is not None for i in idx):
            angles.append(
                (*idx, float(np.deg2rad(degrees)), float(np.deg2rad(_LINK_ANGLE_ESD)))
            )
    planes = []
    for group in link.planes:
        idx = tuple(resolve(a) for a in group)
        if all(i is not None for i in idx):
            planes.append(idx)
    return bonds, angles, planes


def build_polymer_geometry(
    adapter, conformer_config: dict | None, elements=None
) -> PolymerGeometry | None:
    """Build polymer-local reference conformers through the framework adapter.

    Requested adapters must expose reference positions in addition to ordinary atom
    records and elements. ``elements`` may be passed in when the caller already resolved
    it (avoids a second ``get_elements`` call); otherwise it is read from the adapter.
    Reference-space UIDs are used when available; otherwise the grouping falls back to
    chain/residue/type records. Failing loudly is important: silently omitting polymer
    restraints would otherwise look like a successful run.
    """

    cfg_present = conformer_config is not None
    if not cfg_present:
        return None
    if not hasattr(adapter, "iter_atoms"):
        return None
    records = list(adapter.iter_atoms())
    if not any(_is_enabled_polymer(record) for record in records):
        return None

    required = ["get_reference_positions"]
    if elements is None:
        required.insert(0, "get_elements")
    missing = [name for name in required if not hasattr(adapter, name)]
    if missing:
        raise TypeError(
            "polymer conformer restraints require adapter method(s): "
            + ", ".join(missing)
        )

    if elements is None:
        elements = adapter.get_elements()
    elements = np.asarray(elements)
    ref_pos = np.asarray(adapter.get_reference_positions(), dtype=np.float64)
    n = min(len(elements), len(ref_pos))
    if ref_pos.ndim != 2 or ref_pos.shape[-1] != 3:
        raise ValueError(
            f"adapter reference positions must have shape (n_atom, 3), got {ref_pos.shape}"
        )

    ref_uid = None
    if hasattr(adapter, "get_reference_space_uid"):
        try:
            ref_uid = np.asarray(adapter.get_reference_space_uid()).reshape(-1)
        except (AttributeError, KeyError):
            ref_uid = None
        if ref_uid is not None and len(ref_uid) < n:
            raise ValueError(
                "adapter reference-space UID array is shorter than reference positions"
            )
    if ref_uid is None:
        # Derive residue groups when the adapter provides no reference-space UID.
        ref_uid = np.full(n, -1, dtype=np.int64)
        uid_for_key = {}
        for record in records:
            g = int(record.index)
            if polymer_type(record.mol_type, record.resname) is not None and 0 <= g < n:
                ptype = polymer_type(record.mol_type, record.resname)
                key = (record.chain, int(record.resid), ptype)
                ref_uid[g] = uid_for_key.setdefault(key, len(uid_for_key))

    # Preserve adjacency in the complete structure before excluding disabled residues.
    # Token ordinals cannot encode this: modified residues may use one token per atom.
    source_order = {}
    for record in records:
        g = int(record.index)
        if not 0 <= g < n or polymer_type(record.mol_type, record.resname) is None:
            continue
        uid = int(ref_uid[g])
        if uid >= 0:
            order = source_order.setdefault(record.chain, {})
            order[uid] = min(order.get(uid, g), g)
    adjacent = set()
    for order in source_order.values():
        uids = sorted(order, key=order.get)
        adjacent.update(zip(uids, uids[1:]))

    by_uid: dict[int, list] = {}
    for record in records:
        g = int(record.index)
        if not _is_enabled_polymer(record) or g < 0 or g >= n:
            continue
        if int(elements[g]) <= 0 or int(ref_uid[g]) < 0:
            continue
        by_uid.setdefault(int(ref_uid[g]), []).append(record)

    residue_confs: list[LigandConf] = []
    residue_meta = []
    atom_indices: set[int] = set()
    for uid, group in by_uid.items():
        group = sorted(group, key=lambda r: int(r.index))
        ptypes = {polymer_type(r.mol_type, r.resname) for r in group}
        chains = {r.chain for r in group}
        if len(ptypes) != 1 or len(chains) != 1:
            raise ValueError(
                f"ref_space_uid {uid} spans multiple polymer residues: "
                f"types={ptypes}, chains={chains}"
            )
        gidx = np.asarray([int(r.index) for r in group], dtype=np.int64)
        coords = ref_pos[gidx]
        mol = build_ligand_mol(elements[gidx], coords, [], perceive_bonds=True)
        residue_confs.append(
            LigandConf(
                mol=mol,
                conf_coords=coords,
                global_indices=gidx,
                conformer_restraints=True,
            )
        )
        names = {
            _normalise_name(record.name): int(record.index)
            for record in group
            if _normalise_name(record.name)
        }
        residue_meta.append(
            {
                "uid": uid,
                "chain": group[0].chain,
                "order": int(gidx.min()),
                "mol_type": next(iter(ptypes)),
                "resname": group[0].resname,
                "names": names,
            }
        )
        atom_indices.update(int(g) for g in gidx)

    by_chain: dict[str, list[dict]] = {}
    for meta in residue_meta:
        by_chain.setdefault(meta["chain"], []).append(meta)
    # Resolve adjacency once, before applying link-specific residue modifications.
    # Keeping actual edges also distinguishes a residue's incoming/outgoing proline
    # links: one shared "second residue" marker cannot represent both neighbours.
    connections = []
    for residues in by_chain.values():
        residues.sort(key=lambda x: (x["order"], x["uid"]))
        for previous, current in zip(residues, residues[1:]):
            if (
                current["mol_type"] == previous["mol_type"]
                and (previous["uid"], current["uid"]) in adjacent
            ):
                connections.append((previous, current))

    targets = _load_library(conformer_config, residue_meta, connections)
    if conformer_weight(conformer_config, "cistrans") > 0:
        from rgi_toolkit._polymer_torsions import add_polymer_torsions

        add_polymer_torsions(targets, residue_meta, connections, ref_pos)
    link_bonds, link_angles, link_planes = [], [], []
    for previous, current in connections:
        if (previous["uid"], current["uid"]) in targets.covered_links:
            continue
        bonds, angles, planes = _link_geometry(previous, current, current["mol_type"])
        link_bonds.extend(bonds)
        link_angles.extend(angles)
        link_planes.extend(planes)

    return PolymerGeometry(
        residue_confs=residue_confs,
        atom_indices=np.asarray(sorted(atom_indices), dtype=np.int64),
        link_bonds=link_bonds,
        link_angles=link_angles,
        link_planes=link_planes,
        library=targets,
    )


def _load_library(conformer_config, residue_meta, connections):
    """Load dictionary targets only when an enabled geometry term needs them."""
    spec = monlib_geom.parse_config(conformer_config)
    enabled = {
        k for k in monlib_geom.KINDS if conformer_weight(conformer_config, k) > 0
    }
    if spec is None or not enabled:
        return monlib_geom.LibraryTargets()
    path, on_missing = spec
    library = monlib_geom.MonomerLibrary.load(
        path, {m["resname"] for m in residue_meta if m["resname"]}
    )
    targets = monlib_geom.collect(
        library, residue_meta, on_missing, connections, enabled
    )
    n_covered = sum(library.covers(m["resname"]) for m in residue_meta)
    counts = ", ".join(f"{k}={len(rows)}" for k, rows in targets.terms.items())
    logger.info(
        "[rgi_toolkit] monomer library: %d/%d residues; components=%s; %s; "
        "peptide choices=%d (state-dependent rows are alternatives); fallback=%s",
        n_covered,
        len(residue_meta),
        list(targets.covered),
        counts,
        len(targets.peptides),
        list(targets.missing),
    )
    return targets
