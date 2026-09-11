"""Build a backend-agnostic ``RestraintSpec`` from ligand conformers + distances.

This is the single place where conformer restraints (bond/angle/chiral/cistrans/
plane) are derived from RDKit mols. Each ligand supplies its own
``global_indices``, so multiple ligands produce non-colliding restraints.

Flow:
  1. extract bond/angle/chiral/cistrans/plane restraints per ligand in GLOBAL atom
     indices,
  2. collect distance restraint atom groups (already resolved to global indices),
  3. active_sites = union of all referenced global atoms (sorted, unique),
  4. remap every index to a LOCAL index into active_sites and pack padded arrays.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import replace

import numpy as np
from rdkit import Chem

from rgi_toolkit._config_util import (
    VDW_MAX_ATOM_STEP_DEFAULT,
    VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT,
    VDW_NEIGHBOR_SKIN_DEFAULT,
    VDW_SCALE_DEFAULT,
    conformer_weight,
    validate_vdw_config,
)
from rgi_toolkit._conformer_planes import prefer_cistrans
from rgi_toolkit._mol_build import ff_relax, parse_relax_force_field, repair_stereo
from rgi_toolkit._monlib_records import validate_target
from rgi_toolkit._monlib_spec import append_library_arrays, used_peptides
from rgi_toolkit.atom_context import LigandConf
from rgi_toolkit.monlib_geom import LibraryTargets, missing_geometry
from rgi_toolkit.spec import (
    DIST_TYPE_CODES,
    ActiveVdwConfig,
    AngleArrays,
    BondArrays,
    ChiralArrays,
    CisTransArrays,
    DistanceArrays,
    GroupAngleArrays,
    GroupDihedralArrays,
    GroupImproperArrays,
    GroupPlaneArrays,
    PlaneArrays,
    RestraintSpec,
    RmsdArrays,
    VdwArrays,
    VdwConfig,
)

logger = logging.getLogger(__name__)

_ANGLE_PATT = Chem.MolFromSmarts("*~*~*")
_CHIRAL_TAGS = (
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
# Maximum reference out-of-plane deviation (Angstrom). The 0.1 A threshold
# separates flat aromatic/conjugated rings from puckered saturated rings
# without relying on RDKit aromaticity flags.
_PLANE_TOL = 0.1


def _max_plane_dev(crds, idxs):
    """Max out-of-plane distance (Angstrom) of atoms ``idxs`` from their own best-fit
    plane, in the reference conformer ``crds``. The plane normal is the smallest-
    eigenvalue eigenvector of the centred covariance (build-time numpy, no autodiff)."""
    pts = crds[list(idxs)]
    x0 = pts - pts.mean(axis=0)
    _w, vecs = np.linalg.eigh(
        x0.T @ x0
    )  # ascending eigenvalues; columns = eigenvectors
    return float(np.max(np.abs(x0 @ vecs[:, 0])))


def _bond_length(crds: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(crds[i] - crds[j]))


def _angle_rad(crds: np.ndarray, i: int, j: int, k: int) -> float:
    rij = crds[i] - crds[j]
    rkj = crds[k] - crds[j]
    cos = np.dot(rij, rkj) / (np.linalg.norm(rij) * np.linalg.norm(rkj) + 1e-12)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _chiral_vol(crds: np.ndarray, c: int, n1: int, n2: int, n3: int) -> float:
    v1 = crds[n1] - crds[c]
    v2 = crds[n2] - crds[c]
    v3 = crds[n3] - crds[c]
    return float(np.dot(v1, np.cross(v2, v3)))


def _cistrans_rad(crds: np.ndarray, i: int, j: int, k: int, ll: int) -> float:
    """Signed torsion angle (radians) for atoms i-j-k-ll about the j-k axis.

    Identical formula to the energy backends' ``cistrans_energy`` so the target
    computed here equals the value the energy sees at the reference conformer
    (residual starts at zero before any perturbation).
    """
    b1 = crds[j] - crds[i]
    b2 = crds[k] - crds[j]
    b3 = crds[ll] - crds[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / np.sqrt(np.dot(b2, b2) + 1e-12)
    m1 = np.cross(n1, b2n)
    # Mirror the energy backends' dihedral exactly (incl. the atan2(0,0) guard) so
    # phi0 == the value the energy sees at the reference geometry.
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    if x == 0.0 and y == 0.0:
        x = 1e-12
    return float(np.arctan2(y, x))


def _extract_conformer(
    ligand_confs: list[LigandConf],
    *,
    relax: bool = True,
    force_field: str = "uff",
    extra_torsions: list | None = None,
):
    """Return bond/angle/chiral/cistrans restraint tuples and plane groups in GLOBAL atom
    indices.

    ``relax`` is the STRUCTURAL switch (the polymer call site passes False: monomer-library
    residues are never force-field relaxed); ``force_field`` is the user's
    ``conformer_restraints_config.relax_force_field.ligand`` choice, applied to LIGANDS
    only.
    """
    bonds = []  # (g0, g1, r0)
    angles = []  # (g0, g1, g2, th0)
    chirals = []  # (g0, g1, g2, g3, vol0)
    cistrans = []  # (g0, g1, g2, g3, phi0)
    planes = []  # tuple(global idx, ...) — a planar atom group (ring or sp2 group)
    ff = str(force_field).lower()
    do_relax = relax and ff != "none"

    for li, lc in enumerate(ligand_confs):
        mol = lc.mol
        stereo_mol = getattr(lc, "stereo_mol", None)
        stereo_topology = stereo_mol if stereo_mol is not None else mol
        crds = np.asarray(lc.conf_coords, dtype=np.float64)
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        # Relax locally to avoid using distorted cached geometry as the target.
        # Only aromatic/double bonds establish reliable bond orders; relaxing an
        # all-single perceived graph would distort aromatic bond lengths.
        # UFF may fall back to cached coordinates; an explicit MMFF request raises.
        has_orders = any(
            b.GetIsAromatic() or b.GetBondType() == Chem.BondType.DOUBLE
            for b in mol.GetBonds()
        )
        if do_relax and not has_orders and ff != "uff":
            _at = f"global atom index {int(gidx[0])}, " if len(gidx) else ""
            raise ValueError(
                "conformer_restraints_config.relax_force_field."
                f"ligand={force_field!r}: ligand #{li} "
                f"({_at}{mol.GetNumAtoms()} atoms) has "
                "no aromatic or double bond, so the relax is skipped and the force field "
                "would never run. Either the tool supplied no real bond orders (chai / "
                "esmfold2 without SMILES -- relaxing an all-single mol would collapse "
                "aromatic rings to ~1.5 A), or the ligand is genuinely saturated. Supply "
                "the ligand as SMILES/CCD, or set relax_force_field: {ligand: uff} "
                "(same skip, no error) or {ligand: none}."
            )
        if do_relax and has_orders:
            _relaxed = ff_relax(mol, crds, ff, stereo_mol=stereo_mol)
            if _relaxed is not None and len(_relaxed) == len(crds):
                crds = _relaxed
        elif stereo_mol is not None:
            # Stereo validation is independent of the ordinary relax guard. This
            # covers saturated chiral ligands (no aromatic/double bond) and explicit
            # relax_force_field=none without changing already-correct coordinates.
            crds = repair_stereo(
                mol,
                crds,
                stereo_mol,
                force_field=ff if do_relax else "none",
            )

        if extra_torsions is not None:
            from rgi_toolkit._polymer_torsions import ligand_sp2_torsions

            extra_torsions.extend(ligand_sp2_torsions([replace(lc, conf_coords=crds)]))

        for b in mol.GetBonds():
            ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bonds.append(
                (int(gidx[ai]), int(gidx[aj]), _bond_length(crds, ai, aj), None)
            )

        for ai, aj, ak in mol.GetSubstructMatches(_ANGLE_PATT):
            angles.append(
                (
                    int(gidx[ai]),
                    int(gidx[aj]),
                    int(gidx[ak]),
                    _angle_rad(crds, ai, aj, ak),
                    None,
                )
            )

        for atom in stereo_topology.GetAtoms():
            if atom.GetChiralTag() not in _CHIRAL_TAGS:
                continue
            ci = atom.GetIdx()
            nei = [b.GetOtherAtom(atom).GetIdx() for b in atom.GetBonds()]
            for cand in itertools.combinations(nei, 3):
                vol = _chiral_vol(crds, ci, cand[0], cand[1], cand[2])
                if lc.invert_chirality:
                    vol = -vol
                chirals.append(
                    (
                        int(gidx[ci]),
                        int(gidx[cand[0]]),
                        int(gidx[cand[1]]),
                        int(gidx[cand[2]]),
                        vol,
                    )
                )

        # Acyclic double bonds supply E/Z torsions; ring bonds cannot isomerise.
        try:
            Chem.FastFindRings(stereo_topology)  # ensure IsInRing() has ring info
        except Exception:
            pass
        for b in stereo_topology.GetBonds():
            if b.GetBondType() != Chem.BondType.DOUBLE:
                continue
            if b.GetIsAromatic() or b.IsInRing():
                continue
            aj, ak = b.GetBeginAtom(), b.GetEndAtom()
            j, k = aj.GetIdx(), ak.GetIdx()
            nbr_j = [n.GetIdx() for n in aj.GetNeighbors() if n.GetIdx() != k]
            nbr_k = [n.GetIdx() for n in ak.GetNeighbors() if n.GetIdx() != j]
            if not nbr_j or not nbr_k:
                continue  # terminal double bond (e.g. C=O) — no dihedral
            # Enumerate every (i, l) substituent pair across the bond, like the
            # chiral combination enumeration: an order-independent restraint set.
            for i in nbr_j:
                for ll in nbr_k:
                    phi0 = _cistrans_rad(crds, i, j, k, ll)
                    cistrans.append(
                        (
                            int(gidx[i]),
                            int(gidx[j]),
                            int(gidx[k]),
                            int(gidx[ll]),
                            phi0,
                        )
                    )

        # Plane candidates are whole rings and non-ring sp2 centres with their
        # heavy neighbours. Require reference coplanarity and at least four atoms;
        # ring membership also works for geometry-perceived, all-single graphs.
        candidates = [tuple(r) for r in mol.GetRingInfo().AtomRings()]
        for b in mol.GetBonds():
            if (
                b.GetBondType() != Chem.BondType.DOUBLE
                or b.GetIsAromatic()
                or b.IsInRing()
            ):
                continue
            for atom in (b.GetBeginAtom(), b.GetEndAtom()):
                candidates.append(
                    tuple([atom.GetIdx()] + [n.GetIdx() for n in atom.GetNeighbors()])
                )
        seen_planes: set[frozenset] = set()
        for cand in candidates:
            key = frozenset(cand)
            if len(key) < 4 or key in seen_planes:
                continue  # a plane needs >=4 atoms (3 are trivially coplanar)
            local = sorted(key)
            if _max_plane_dev(crds, local) >= _PLANE_TOL:
                continue  # not coplanar in the reference (e.g. saturated ring)
            seen_planes.add(key)
            planes.append(tuple(int(gidx[i]) for i in local))

    return bonds, angles, chirals, cistrans, planes


def _pad_groups(restraints, n_groups, g2l):
    """Pad each restraint's ``n_groups`` atom groups into (n, max_grp) local-index and
    {0,1} mask arrays. Reads ``restraints[k].target_sites{1..n_groups}`` (global
    indices). ``max_grp`` spans every group of every restraint so one padded width
    covers them all. Returns (idx_arrays, mask_arrays): each a list of ``n_groups``
    arrays in group order. Used for group-centroid angle (3) and dihedral (4)."""
    n = len(restraints)
    max_grp = max(
        len(getattr(r, f"target_sites{g + 1}"))
        for r in restraints
        for g in range(n_groups)
    )
    idx_arrays = [np.zeros((n, max_grp), dtype=np.int64) for _ in range(n_groups)]
    mask_arrays = [np.zeros((n, max_grp)) for _ in range(n_groups)]
    for ri, r in enumerate(restraints):
        for g in range(n_groups):
            local = [g2l[int(s)] for s in getattr(r, f"target_sites{g + 1}")]
            idx_arrays[g][ri, : len(local)] = local
            mask_arrays[g][ri, : len(local)] = 1.0
    return idx_arrays, mask_arrays


def _group_geom_params(restraints):
    """(geom_type codes, target1, target2, move_free) arrays for group angle/dihedral
    restraints. The string type maps to the SAME int code as the distance restraint
    (``DIST_TYPE_CODES``: harmonic=0 / flat-bottomed=1 / flat-bottomed1=2 (lower) /
    flat-bottomed2=3 (upper)); target1/target2 are radians (0.0 where unused). move_free
    is an (n, n_groups) {0,1} mask (1 = that group is free to move)."""
    codes = np.array([DIST_TYPE_CODES[r.geom_type] for r in restraints], dtype=np.int64)
    t1 = np.array([float(r.target1) for r in restraints])
    t2 = np.array([float(r.target2) for r in restraints])
    move_free = np.array([[1.0 if f else 0.0 for f in r.move_free] for r in restraints])
    return codes, t1, t2, move_free


def _window_arrays(restraints, default_start_sigma):
    """Return the four common per-entry sigma/step window arrays."""
    start_sigma = np.array(
        [
            float(restraint.start_sigma)
            if getattr(restraint, "start_sigma", None) is not None
            else default_start_sigma
            for restraint in restraints
        ]
    )
    stop_sigma = np.array(
        [float(getattr(restraint, "stop_sigma", -1.0)) for restraint in restraints]
    )
    start_step = np.array(
        [
            float(getattr(restraint, "start_step", float("-inf")))
            for restraint in restraints
        ]
    )
    stop_step = np.array(
        [
            float(getattr(restraint, "stop_step", float("inf")))
            for restraint in restraints
        ]
    )
    return start_sigma, stop_sigma, start_step, stop_step


def _build_group_geom_arrays(
    restraints, n_groups, array_type, g2l, default_start_sigma
):
    """Build one padded angle/dihedral/improper array family."""
    indices, masks = _pad_groups(restraints, n_groups, g2l)
    codes, target1, target2, move_free = _group_geom_params(restraints)
    start_sigma, stop_sigma, start_step, stop_step = _window_arrays(
        restraints, default_start_sigma
    )
    groups = {}
    for group, (idx, group_mask) in enumerate(zip(indices, masks), start=1):
        groups[f"grp{group}_idx"] = idx
        groups[f"grp{group}_mask"] = group_mask
    return array_type(
        **groups,
        target1=target1,
        target2=target2,
        geom_type=codes,
        move_free=move_free,
        weight=np.array([float(restraint.weight) for restraint in restraints]),
        mask=np.ones(len(restraints)),
        start_sigma=start_sigma,
        stop_sigma=stop_sigma,
        start_step=start_step,
        stop_step=stop_step,
    )


def _dist_params(dr) -> tuple[int, float, float]:
    """Map a DistanceData (string type + targets) to (code, target1, target2)."""
    t = dr.distance_restraint_type
    code = DIST_TYPE_CODES[t]
    if t == "harmonic":
        return code, float(dr.target_distance), 0.0
    if t == "flat-bottomed":
        return code, float(dr.target_distance1), float(dr.target_distance2)
    if t == "flat-bottomed1":
        return code, float(dr.target_distance1), 0.0
    if t == "flat-bottomed2":
        return code, 0.0, float(dr.target_distance2)
    raise ValueError(f"unknown distance type {t!r}")


def _build_distance_arrays(restraints, g2l, default_start_sigma):
    """Build padded distance arrays using the shared group/window encoders."""
    indices, masks = _pad_groups(restraints, 2, g2l)
    params = [_dist_params(restraint) for restraint in restraints]
    start_sigma, stop_sigma, start_step, stop_step = _window_arrays(
        restraints, default_start_sigma
    )
    return DistanceArrays(
        grp1_idx=indices[0],
        grp2_idx=indices[1],
        grp1_mask=masks[0],
        grp2_mask=masks[1],
        target1=np.array([param[1] for param in params]),
        target2=np.array([param[2] for param in params]),
        dist_type=np.array([param[0] for param in params], dtype=np.int64),
        move_mode=np.array(
            [int(getattr(restraint, "move_mode", 0)) for restraint in restraints],
            dtype=np.int64,
        ),
        weight=np.array(
            [float(getattr(restraint, "weight", 1.0)) for restraint in restraints]
        ),
        mask=np.ones(len(restraints)),
        start_sigma=start_sigma,
        stop_sigma=stop_sigma,
        start_step=start_step,
        stop_step=stop_step,
    )


def _vdw_radius(z: int) -> float:
    """VdW radius (A) for atomic number ``z`` from RDKit's periodic table."""
    if z is None or z < 1 or z > 118:
        return 0.0
    return float(Chem.GetPeriodicTable().GetRvdw(int(z)))


def _conf_weight(conformer_config: dict | None, key: str) -> float:
    """Delegate conformer defaults to the backend-independent configuration helper."""
    return conformer_weight(conformer_config, key)


def _conf_slack(conformer_config: dict | None, key: str, default: float) -> float:
    """Slack (flat-bottom half-width) of a conformer sub-term, with uniform null handling.

    Shared by all five conformer terms (bond/angle/chiral/cistrans/plane) so the
    omitted / explicit-0 / null cases can't drift between them: an OMITTED ``slack`` ->
    the per-term ``default``; an explicit ``slack: 0`` -> 0.0 (a hard, zero-width
    restraint, NOT the default); a ``slack: null`` -> the ``default`` (null is treated as
    omitted, matching ``apply_window_params``). The truthiness trap (``slack or default``
    would silently turn an explicit 0 into the default) is what this avoids.
    """
    v = (conformer_config or {}).get(key)
    v = (v or {}).get("slack", default)
    return float(default if v is None else v)


def _build_vdw_config(
    ligand_confs: list[LigandConf],
    polymer_atoms: np.ndarray,
    conformer_config: dict,
    active_sites: np.ndarray,
    g2l: dict,
    elements: np.ndarray | None,
    chemistry=None,
) -> VdwConfig | None:
    """Build the dynamic fixed-background VdW config (torch + jax optimizers).

    The fixed-background half of the ``intermolecular`` category. Ligand atoms (the
    moving set) come from the RDKit mols; the background is every non-padding atom NOT
    in ``active_sites`` (i.e. not optimised) — protein, DNA/RNA, and any non-restrained
    ligand — so the VdW term pushes the ligand out of the fixed pocket. Returns
    ``None`` when VdW is disabled (weight<=0), there are no ligands, or element info
    is unavailable.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if (
        weight <= 0.0
        or (not ligand_confs and len(polymer_atoms) == 0)
        or elements is None
    ):
        return None

    lig_radius: dict[int, float] = {}
    for lc in ligand_confs:
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        for i, atom in enumerate(lc.mol.GetAtoms()):
            lig_radius[int(gidx[i])] = _vdw_radius(atom.GetAtomicNum())
    elements = np.asarray(elements)
    for g in polymer_atoms:
        if 0 <= int(g) < len(elements) and int(elements[int(g)]) > 0:
            lig_radius[int(g)] = _vdw_radius(int(elements[int(g)]))
    ligand_global = np.array(sorted(lig_radius), dtype=np.int64)
    # every ligand atom is in active_sites (added in build_spec when VdW is on)
    ligand_local = np.array([g2l[int(g)] for g in ligand_global], dtype=np.int64)
    ligand_radii = np.array(
        [lig_radius[int(g)] for g in ligand_global], dtype=np.float64
    )

    # fixed background = all non-padding atoms NOT optimised (not in active_sites):
    # protein / DNA/RNA / non-restrained ligand. element code 0 is the padding sentinel.
    active_set = {int(a) for a in active_sites}
    background_global = np.array(
        [
            a
            for a in range(len(elements))
            if int(elements[a]) > 0 and a not in active_set
        ],
        dtype=np.int64,
    )
    if len(background_global) == 0:
        return None
    background_radii = np.array(
        [_vdw_radius(int(elements[a])) for a in background_global], dtype=np.float64
    )
    typed = None
    if chemistry is not None:
        ligand_radii = chemistry.radii[ligand_global]
        background_radii = chemistry.radii[background_global]
        typed = chemistry.subset(
            ligand_global,
            background_global,
            set(ligand_global),
            set(),
            vcfg.get("mode", "both"),
        )
    max_neighbors = int(vcfg.get("max_neighbors", 32))
    if max_neighbors < 1:
        raise ValueError("conformer vdw max_neighbors must be >= 1")

    return VdwConfig(
        weight=weight,
        ligand_local=ligand_local,
        ligand_radii=ligand_radii,
        background_global=background_global,
        background_radii=background_radii,
        scale=float(vcfg.get("scale", VDW_SCALE_DEFAULT)),
        dmax=float(vcfg.get("dmax", 5.0)),
        max_neighbors=max_neighbors,
        chemistry=typed,
    )


def _build_active_vdw_config(
    polymer_atoms: np.ndarray,
    conformer_config: dict,
    active_sites: np.ndarray,
    elements: np.ndarray | None,
    bonds,
    angles,
    chemistry=None,
    ligand_confs=(),
) -> ActiveVdwConfig | None:
    """Build dynamic polymer-involving VdW metadata in local active-site space."""

    weight = _conf_weight(conformer_config, "vdw")
    static_ligands = {int(g) for lc in ligand_confs for g in lc.global_indices}
    moving = set(map(int, polymer_atoms)) | static_ligands
    extra_active = set(map(int, active_sites)) - static_ligands
    if (
        weight <= 0.0
        or elements is None
        or (
            len(polymer_atoms) == 0
            and (chemistry is None or not extra_active or not moving)
        )
    ):
        return None
    elements = np.asarray(elements)
    g2l = {int(g): i for i, g in enumerate(active_sites)}
    radii = np.array(
        [
            _vdw_radius(int(elements[g])) if int(elements[g]) > 0 else 0.0
            for g in active_sites
        ],
        dtype=np.float64,
    )
    if chemistry is not None:
        radii = chemistry.radii[active_sites]
    polymer_set = {int(g) for g in polymer_atoms}
    polymer_mask = np.array([int(g) in polymer_set for g in active_sites], dtype=bool)
    n_active = len(active_sites)
    excluded = set()

    def exclude(global_i, global_j):
        if global_i not in g2l or global_j not in g2l:
            return
        i, j = sorted((g2l[int(global_i)], g2l[int(global_j)]))
        if i != j:
            excluded.add(i * n_active + j)

    adjacency: dict[int, set[int]] = {}
    for g0, g1, *_ in bonds:
        g0, g1 = int(g0), int(g1)
        adjacency.setdefault(g0, set()).add(g1)
        adjacency.setdefault(g1, set()).add(g0)

    # Exclude every pair separated by at most three covalent bonds (1-2/1-3/1-4),
    # including paths that cross peptide or phosphodiester links.
    for start in adjacency if chemistry is None else ():
        seen = {start}
        frontier = {start}
        for _distance in range(3):
            frontier = {
                neighbour
                for atom in frontier
                for neighbour in adjacency.get(atom, ())
                if neighbour not in seen
            }
            for end in frontier:
                exclude(start, end)
            seen.update(frontier)

    # Keep explicit angle exclusions even if an incomplete external geometry source
    # supplied an angle without both constituent bonds.
    for g0, g1, g2, *_ in angles:
        exclude(g0, g1)
        exclude(g1, g2)
        exclude(g0, g2)
    typed = None
    if chemistry is not None:
        excluded.clear()
        typed = chemistry.subset(
            active_sites,
            active_sites,
            moving,
            static_ligands,
            (conformer_config.get("vdw") or {}).get("mode", "both"),
            active=True,
        )

    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    max_neighbors = int(vcfg.get("max_neighbors", 32))
    if max_neighbors < 1:
        raise ValueError("conformer vdw max_neighbors must be >= 1")
    return ActiveVdwConfig(
        weight=weight,
        radii=radii,
        polymer_mask=polymer_mask,
        excluded_codes=np.asarray(sorted(excluded), dtype=np.int64),
        scale=float(vcfg.get("scale", VDW_SCALE_DEFAULT)),
        dmax=float(vcfg.get("dmax", 5.0)),
        max_neighbors=max_neighbors,
        chemistry=typed,
    )


def _build_intramolecular_vdw(
    ligand_confs: list[LigandConf],
    conformer_config: dict,
    g2l: dict,
    chemistry=None,
) -> VdwArrays | None:
    """Static intramolecular VdW repulsion within each ligand (all backends).

    Excludes 1-2/1-3 and same-plane 1-4 pairs. Every other pair uses its chemical
    contact distance and inverse-ESD-squared weight, including eligible 1-4 contacts.
    Reference distance is deliberately not a build filter. Unlike the
    dynamic fixed-background ``VdwConfig``, the pair list is fixed, so this term also
    works in the jax/numpy backends via ``VdwArrays``. Enabled when
    ``conformer_config['vdw']['mode']`` is ``'intramolecular'`` or ``'both'`` (the
    DEFAULT); ``'intermolecular'`` leaves it off.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if weight <= 0.0 or not ligand_confs:
        return None
    if chemistry is None:
        from rgi_toolkit._vdw_chemistry import build_chemistry

        chemistry = build_chemistry(ligand_confs, None)

    scale = float(vcfg.get("scale", VDW_SCALE_DEFAULT))
    idx_pairs: list[list[int]] = []
    r_min_list: list[float] = []
    weights = []
    for lc in ligand_confs:
        mol = lc.mol
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        n = mol.GetNumAtoms()
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                pair = chemistry.pair(int(gidx[i]), int(gidx[j]))
                if pair is None:
                    continue
                idx_pairs.append([g2l[int(gidx[i])], g2l[int(gidx[j])]])
                r_min_list.append(scale * pair[0])
                weights.append(weight * pair[1])
    if not idx_pairs:
        return None
    n_pair = len(idx_pairs)
    return VdwArrays(
        idx=np.array(idx_pairs, dtype=np.int64),
        r_min=np.array(r_min_list, dtype=np.float64),
        weight=np.asarray(weights),
        mask=np.ones(n_pair),
    )


def _build_interligand_vdw(
    ligand_confs: list[LigandConf],
    conformer_config: dict,
    g2l: dict,
    chemistry=None,
) -> VdwArrays | None:
    """Static inter-ligand VdW repulsion BETWEEN distinct ligands (all backends).

    The restrained-ligand half of the ``intermolecular`` category (the other half is
    the dynamic fixed-background ``VdwConfig``). Both endpoints of every pair live in
    ``active_sites`` (each ligand moves under its own conformer restraint), so
    autodiff drives BOTH ligands apart — exactly like ``vdw_energy``'s
    intramolecular pairs, only the pair list crosses molecules. Unlike
    ``_build_intramolecular_vdw`` there is NO topological skip (different mols have
    no shared bonds) and NO reference-distance ``dmax`` cutoff (two ligands'
    ``conf_coords`` live in independent frames, so a build-time distance is
    meaningless); every cross pair is listed and the ``clamp(d - r_min, max=0)``
    penalty contributes zero beyond contact. Ligands are H-removed and small, so
    the all-pairs list stays cheap. Built only when VdW is on, ``mode`` is
    ``'intermolecular'`` or ``'both'``, and at least two ligands opted in.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if weight <= 0.0 or len(ligand_confs) < 2:
        return None

    scale = float(vcfg.get("scale", VDW_SCALE_DEFAULT))
    if chemistry is None:
        from rgi_toolkit._vdw_chemistry import build_chemistry

        chemistry = build_chemistry(ligand_confs, None)
    idx_pairs: list[list[int]] = []
    r_min_list: list[float] = []
    weights = []
    for a in range(len(ligand_confs)):
        gA = np.asarray(ligand_confs[a].global_indices, dtype=np.int64)
        for b in range(a + 1, len(ligand_confs)):
            gB = np.asarray(ligand_confs[b].global_indices, dtype=np.int64)
            for i in range(len(gA)):
                li = g2l[int(gA[i])]
                for j in range(len(gB)):
                    pair = chemistry.pair(int(gA[i]), int(gB[j]))
                    if pair is None:
                        continue
                    idx_pairs.append([li, g2l[int(gB[j])]])
                    r_min_list.append(scale * pair[0])
                    weights.append(weight * pair[1])
    if not idx_pairs:
        return None
    n_pair = len(idx_pairs)
    return VdwArrays(
        idx=np.array(idx_pairs, dtype=np.int64),
        r_min=np.array(r_min_list, dtype=np.float64),
        weight=np.asarray(weights),
        mask=np.ones(n_pair),
    )


def _concat_vdw_arrays(*parts: VdwArrays | None) -> VdwArrays | None:
    """Stack several ``VdwArrays`` into one (drops ``None``s). They are scored by the
    same ``vdw_energy`` term under the same conformer gate, so concatenating their
    ``idx``/``r_min``/``weight``/``mask`` rows composes the flavours with no extra
    wiring. Returns ``None`` when every part is empty."""
    present = [p for p in parts if p is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return VdwArrays(
        idx=np.concatenate([p.idx for p in present], axis=0),
        r_min=np.concatenate([p.r_min for p in present]),
        weight=np.concatenate([p.weight for p in present]),
        mask=np.concatenate([p.mask for p in present]),
    )


def build_spec(
    ligand_confs: list[LigandConf] | None = None,
    distance_restraints: list | None = None,
    conformer_config: dict | None = None,
    elements: np.ndarray | None = None,
    # Omitted start_sigma activates restraints at every diffusion step.
    conf_start_sigma: float = float("inf"),
    conf_stop_sigma: float = -1.0,
    conf_start_step: float = float("-inf"),
    conf_stop_step: float = float("inf"),
    rmsd_restraints: list | None = None,
    angle_restraints: list | None = None,
    dihedral_restraints: list | None = None,
    custom_restraints: list | None = None,
    polymer_geometry=None,
    plane_restraints: list | None = None,
    improper_restraints: list | None = None,
    atom_records=(),
    reference_uids=None,
) -> RestraintSpec:
    """Build a RestraintSpec. ``distance_restraints`` are DistanceData with
    ``target_sites1``/``target_sites2`` already resolved to global indices;
    ``rmsd_restraints`` are RmsdData with fit/calc target sites and paired reference
    coordinates resolved;
    ``angle_restraints``/``dihedral_restraints``/``improper_restraints`` carry
    resolved per-group global indices (N=3 for angle, N=4 for
    dihedral/improper). ``plane_restraints`` are
    PlaneRestraintData with ``target_sites`` (a LIST of per-group global-index lists)
    resolved — the standalone ``plane_restraints_config`` term, which is independent of
    the conformer ``plane`` sub-block (its own weight/type/gate per entry). The base-pair
    coplanarity macro also arrives here as pre-resolved PlaneRestraintData.
    """
    ligand_confs = ligand_confs or []
    cfg = conformer_config or {}
    validate_vdw_config(cfg)
    # Both a conformer config and per-chain opt-in are required. An empty config
    # requests the five default-on terms.
    cfg_present = conformer_config is not None
    if not cfg_present:
        ligand_confs = []
    all_ligand_confs = list(ligand_confs)
    _n_before = len(ligand_confs)
    ligand_confs = [
        lc for lc in ligand_confs if getattr(lc, "conformer_restraints", False)
    ]
    if cfg_present and _n_before and not ligand_confs and polymer_geometry is None:
        # Print as well as log so the warning survives host logging configurations.
        msg = (
            "conformer_restraints_config present but no ligand opted in "
            "(set conformer_restraints: true on the ligand) -- no conformer restraints built"
        )
        logger.warning(msg)
        print(f"[rgi_toolkit] WARNING: {msg}", flush=True)
    distance_restraints = [
        dr for dr in (distance_restraints or []) if getattr(dr, "run_restr", False)
    ]
    rmsd_restraints = [
        rr for rr in (rmsd_restraints or []) if getattr(rr, "run_restr", False)
    ]
    angle_restraints = [
        ar for ar in (angle_restraints or []) if getattr(ar, "run_restr", False)
    ]
    dihedral_restraints = [
        dr for dr in (dihedral_restraints or []) if getattr(dr, "run_restr", False)
    ]
    improper_restraints = [
        ir for ir in (improper_restraints or []) if getattr(ir, "run_restr", False)
    ]
    plane_restraints = [
        pr for pr in (plane_restraints or []) if getattr(pr, "run_restr", False)
    ]
    custom_restraints = [
        c for c in (custom_restraints or []) if getattr(c, "run_restr", False)
    ]
    # Resolve weights from the original value: None and {} have different meanings.
    bw = _conf_weight(conformer_config, "bond")
    bsl = _conf_slack(cfg, "bond", 0.0)
    aw = _conf_weight(conformer_config, "angle")
    asl = _conf_slack(cfg, "angle", 0.0)
    cw = _conf_weight(conformer_config, "chiral")
    csl = _conf_slack(cfg, "chiral", 0.05)
    dw = _conf_weight(conformer_config, "cistrans")
    dsl = _conf_slack(cfg, "cistrans", 0.0)
    vdw_weight = _conf_weight(conformer_config, "vdw")
    pw = _conf_weight(conformer_config, "plane")
    psl = _conf_slack(cfg, "plane", 0.0)

    # Force-field relaxation applies to ligands; polymer calls use relax=False.
    relax_ff = parse_relax_force_field(cfg)
    ligand_torsions = [] if dw > 0 else None
    bonds, angles, chirals, cistrans, planes = _extract_conformer(
        ligand_confs, force_field=relax_ff, extra_torsions=ligand_torsions
    )
    polymer_atoms = np.empty(0, dtype=np.int64)
    library = LibraryTargets()
    if polymer_geometry is not None:
        pb, pa, pc, _pd, pp = _extract_conformer(
            polymer_geometry.residue_confs, relax=False
        )
        # Library targets replace reference-derived tuples within each covered residue.
        # Keep uncovered residues' targets without duplicating covered geometry.
        library = polymer_geometry.library
        lib_atoms = library.atoms
        if lib_atoms:
            pb = [t for t in pb if not lib_atoms.issuperset(t[:2])]
            pa = [t for t in pa if not lib_atoms.issuperset(t[:3])]
            pp = [t for t in pp if not lib_atoms.issuperset(t)]
        bonds.extend(pb)
        bonds.extend(polymer_geometry.link_bonds)
        angles.extend(pa)
        angles.extend(polymer_geometry.link_angles)
        if cw > 0:
            uncovered_chirals = {
                t[0]
                for t in pc
                if t[0] in lib_atoms
                and t[0] not in library.chiral_centers
                and t[0] not in library.chiral_fallback
            }
            if uncovered_chirals:
                missing_geometry(
                    f"no chiral definition for atom(s) {sorted(uncovered_chirals)}",
                    library.on_missing,
                )
        pc = [t for t in pc if t[0] not in library.chiral_centers]
        chirals.extend(pc)  # residue-local stereocentres (Calpha) only
        # Library planes may cover fused rings as one group. Append inter-residue
        # link planes directly; residue-local coplanarity checks do not apply to them.
        planes.extend(pp)
        planes.extend(polymer_geometry.link_planes)
        polymer_atoms = np.asarray(polymer_geometry.atom_indices, dtype=np.int64)
    if dw > 0:
        library = replace(library, terms={k: list(v) for k, v in library.terms.items()})
        library.terms["cistrans"].extend(ligand_torsions)
    # VdW covalent exclusions must survive even when bond/angle energy blocks are off.
    exclusion_bonds = list(bonds)
    exclusion_angles = list(angles)
    # Covalent exclusions survive disabled dictionary energies (including ESD <= 0).
    exclusion_bonds.extend((*idx, 0.0, None) for idx in library.bond_pairs)
    exclusion_angles.extend((*idx, 0.0, None) for idx in library.angle_tuples)
    exclusion_planes = list(planes)
    # Drop disabled terms before collecting active atoms.
    if bw <= 0:
        bonds = []
    if aw <= 0:
        angles = []
    if cw <= 0:
        chirals = []
    if dw <= 0:
        cistrans = []
    if pw <= 0:
        planes = []

    # Disabled energies must not add active atoms; topology survives for VdW.
    library = replace(
        library,
        terms={
            key: [row for row in rows if validate_target(row, key)]
            if _conf_weight(conformer_config, key) > 0
            else []
            for key, rows in library.terms.items()
        },
    )
    planes, plane_conditions, library = prefer_cistrans(planes, cistrans, library)

    active: set[int] = set()
    for g0, g1, *_ in bonds:
        active.update((g0, g1))
    for g0, g1, g2, *_ in angles:
        active.update((g0, g1, g2))
    for g0, g1, g2, g3, _ in chirals:
        active.update((g0, g1, g2, g3))
    for g0, g1, g2, g3, _ in cistrans:
        active.update((g0, g1, g2, g3))
    for grp in planes:
        active.update(grp)
    for rows in library.terms.values():
        for row in rows:
            active.update(row.atoms)
    for selector in used_peptides(library, plane_conditions):
        active.update(library.peptides[selector].atoms)
    resolved_restraints = itertools.chain(
        distance_restraints,
        rmsd_restraints,
        angle_restraints,
        dihedral_restraints,
        improper_restraints,
        plane_restraints,
        custom_restraints,
    )
    for restraint in resolved_restraints:
        active.update(int(site) for site in restraint.iter_global_sites())
    # VdW pushes the whole ligand, so every ligand atom must be optimisable even
    # if it carries no bond/angle/chiral term (e.g. a monatomic ion).
    if vdw_weight > 0:
        for lc in ligand_confs:
            active.update(int(g) for g in lc.global_indices)
        active.update(int(g) for g in polymer_atoms)

    active_sites = np.array(sorted(active), dtype=np.int64)
    g2l = {int(g): i for i, g in enumerate(active_sites)}
    custom_specs = [cr.build_spec(g2l) for cr in custom_restraints]
    # Intramolecular and inter-ligand pairs use static energy arrays.
    # Intermolecular contacts against fixed background use dynamic optimizer lists.
    vdw_mode = (cfg.get("vdw", {}) or {}).get("mode", "both")
    if vdw_mode == "ligand_protein":
        raise ValueError(
            "conformer vdw mode 'ligand_protein' was renamed to 'intermolecular', which "
            "now repels the ligand off EVERY other molecule (protein/DNA/RNA/non-restrained "
            "ligand background AND other restrained ligands), not just the fixed background "
            "-- update your config to mode: intermolecular"
        )
    if vdw_mode not in ("intramolecular", "intermolecular", "both"):
        raise ValueError(
            "conformer vdw mode must be 'intramolecular', 'intermolecular', or "
            f"'both', got {vdw_mode!r}"
        )
    chemistry = None
    if vdw_weight > 0 and (ligand_confs or len(polymer_atoms)):
        from rgi_toolkit._vdw_chemistry import build_chemistry

        chemistry = build_chemistry(
            all_ligand_confs,
            elements,
            atom_records,
            cfg,
            reference_uids,
            library.source,
            exclusion_bonds,
            exclusion_planes,
        )
    vdw_intra = (
        _build_intramolecular_vdw(ligand_confs, conformer_config, g2l, chemistry)
        if vdw_mode in ("intramolecular", "both")
        else None
    )
    vdw_inter = (
        _build_interligand_vdw(ligand_confs, conformer_config, g2l, chemistry)
        if vdw_mode in ("intermolecular", "both")
        else None
    )
    vdw_arrays = _concat_vdw_arrays(vdw_intra, vdw_inter)
    vdw_config = (
        _build_vdw_config(
            ligand_confs,
            polymer_atoms,
            conformer_config,
            active_sites,
            g2l,
            elements,
            chemistry,
        )
        if vdw_mode in ("intermolecular", "both")
        or (chemistry is not None and len(polymer_atoms))
        else None
    )
    active_vdw_config = _build_active_vdw_config(
        polymer_atoms,
        conformer_config,
        active_sites,
        elements,
        exclusion_bonds,
        exclusion_angles,
        chemistry,
        ligand_confs,
    )

    bond = None
    if bonds:
        idx = np.array([[g2l[g0], g2l[g1]] for g0, g1, *_ in bonds], dtype=np.int64)
        bond = BondArrays(
            idx=idx,
            r0=np.array([r for _, _, r, _ in bonds]),
            # Built-in link tolerances are flat-bottom slack; dictionary ESDs instead
            # enter the inverse-variance weights below.
            slack=np.array([bsl if e is None else float(e) for *_, e in bonds]),
            weight=np.full(len(bonds), bw),
            half=np.zeros(len(bonds)),
            mask=np.ones(len(bonds)),
        )
    angle = None
    if angles:
        idx = np.array(
            [[g2l[g0], g2l[g1], g2l[g2]] for g0, g1, g2, *_ in angles],
            dtype=np.int64,
        )
        angle = AngleArrays(
            idx=idx,
            th0=np.array([t for _, _, _, t, _ in angles]),
            slack=np.array([asl if e is None else float(e) for *_, e in angles]),
            weight=np.full(len(angles), aw),
            mask=np.ones(len(angles)),
        )
    chiral = None
    if chirals:
        idx = np.array(
            [[g2l[g0], g2l[g1], g2l[g2], g2l[g3]] for g0, g1, g2, g3, _ in chirals],
            dtype=np.int64,
        )
        chiral = ChiralArrays(
            idx=idx,
            vol0=np.array([v for _, _, _, _, v in chirals]),
            slack=np.full(len(chirals), csl),
            weight=np.full(len(chirals), cw),
            mask=np.ones(len(chirals)),
        )
    cistrans_arr = None
    if cistrans:
        idx = np.array(
            [[g2l[g0], g2l[g1], g2l[g2], g2l[g3]] for g0, g1, g2, g3, _ in cistrans],
            dtype=np.int64,
        )
        cistrans_arr = CisTransArrays(
            idx=idx,
            phi0=np.array([p for _, _, _, _, p in cistrans]),
            slack=np.full(len(cistrans), dsl),
            weight=np.full(len(cistrans), dw),
            mask=np.ones(len(cistrans)),
        )
    plane = None
    # Conformer planes share a weight, slack and gate. Selection-driven planes
    # use the separate group_plane arrays below.
    plane_groups = list(planes)
    if plane_groups:
        # variable group size -> pad to the widest group; padding columns hold local
        # index 0 (a valid atom) and are zeroed in grp_mask (same layout as distance).
        n_plane = len(plane_groups)
        max_atoms = max(len(grp) for grp in plane_groups)
        idx = np.zeros((n_plane, max_atoms), dtype=np.int64)
        grp_mask = np.zeros((n_plane, max_atoms), dtype=np.float64)
        for r, grp in enumerate(plane_groups):
            for c, g in enumerate(grp):
                idx[r, c] = g2l[g]
                grp_mask[r, c] = 1.0
        plane = PlaneArrays(
            idx=idx,
            grp_mask=grp_mask,
            slack=np.full(n_plane, psl),
            weight=np.full(n_plane, pw),
            mask=np.ones(n_plane),
        )

    distance = (
        _build_distance_arrays(distance_restraints, g2l, conf_start_sigma)
        if distance_restraints
        else None
    )

    rmsd = None
    if rmsd_restraints:
        n = len(rmsd_restraints)
        max_fit = max(len(rr.fit_target_sites) for rr in rmsd_restraints)
        max_calc = max(len(rr.calc_target_sites) for rr in rmsd_restraints)
        fit_idx = np.zeros((n, max_fit), dtype=np.int64)
        fit_mask = np.zeros((n, max_fit))
        fit_ref = np.zeros((n, max_fit, 3))
        calc_idx = np.zeros((n, max_calc), dtype=np.int64)
        calc_mask = np.zeros((n, max_calc))
        calc_ref = np.zeros((n, max_calc, 3))
        target1 = np.zeros(n)
        target2 = np.zeros(n)
        geom_type = np.zeros(n, dtype=np.int64)
        rmsd_weight = np.zeros(n)
        (
            rmsd_start_sigma,
            rmsd_stop_sigma,
            rmsd_start_step,
            rmsd_stop_step,
        ) = _window_arrays(rmsd_restraints, conf_start_sigma)
        for ri, rr in enumerate(rmsd_restraints):
            f_local = [g2l[int(s)] for s in rr.fit_target_sites]
            kf = len(f_local)
            fit_idx[ri, :kf] = f_local
            fit_mask[ri, :kf] = 1.0
            fit_ref[ri, :kf] = np.asarray(rr.fit_ref_coords, dtype=np.float64)
            c_local = [g2l[int(s)] for s in rr.calc_target_sites]
            kc = len(c_local)
            calc_idx[ri, :kc] = c_local
            calc_mask[ri, :kc] = 1.0
            calc_ref[ri, :kc] = np.asarray(rr.calc_ref_coords, dtype=np.float64)
            target1[ri] = float(rr.target1)
            target2[ri] = float(rr.target2)
            geom_type[ri] = DIST_TYPE_CODES[rr.rmsd_type]
            # set_config normalizes the default weight; preserve an explicit zero.
            rmsd_weight[ri] = float(rr.weight)
        rmsd = RmsdArrays(
            fit_idx=fit_idx,
            fit_mask=fit_mask,
            fit_ref=fit_ref,
            calc_idx=calc_idx,
            calc_mask=calc_mask,
            calc_ref=calc_ref,
            target1=target1,
            target2=target2,
            geom_type=geom_type,
            weight=rmsd_weight,
            start_sigma=rmsd_start_sigma,
            stop_sigma=rmsd_stop_sigma,
            start_step=rmsd_start_step,
            stop_step=rmsd_stop_step,
            mask=np.ones(n),
        )

    group_angle = (
        _build_group_geom_arrays(
            angle_restraints, 3, GroupAngleArrays, g2l, conf_start_sigma
        )
        if angle_restraints
        else None
    )
    group_dihedral = (
        _build_group_geom_arrays(
            dihedral_restraints, 4, GroupDihedralArrays, g2l, conf_start_sigma
        )
        if dihedral_restraints
        else None
    )
    group_improper = (
        _build_group_geom_arrays(
            improper_restraints, 4, GroupImproperArrays, g2l, conf_start_sigma
        )
        if improper_restraints
        else None
    )

    group_plane = None
    if plane_restraints:
        n = len(plane_restraints)
        # Each entry POOLS all of its groups into one plane, so a row is the concatenated
        # atom list (padded to the widest entry). `free` is therefore per-ATOM: it repeats
        # each group's `move_free` flag across that group's atoms.
        pooled = [
            [int(s) for grp in pr.target_sites for s in grp] for pr in plane_restraints
        ]
        free_flags = [
            [
                1.0 if pr.move_free[gi] else 0.0
                for gi, grp in enumerate(pr.target_sites)
                for _ in grp
            ]
            for pr in plane_restraints
        ]
        max_atoms = max(len(row) for row in pooled)
        idx = np.zeros((n, max_atoms), dtype=np.int64)
        grp_mask = np.zeros((n, max_atoms))
        free = np.zeros((n, max_atoms))
        for r, (row, flags) in enumerate(zip(pooled, free_flags)):
            local = [g2l[g] for g in row]
            idx[r, : len(local)] = local
            grp_mask[r, : len(local)] = 1.0
            free[r, : len(flags)] = flags
        start_sigma, stop_sigma, start_step, stop_step = _window_arrays(
            plane_restraints, conf_start_sigma
        )
        group_plane = GroupPlaneArrays(
            idx=idx,
            grp_mask=grp_mask,
            free=free,
            target1=np.array([float(r.target1) for r in plane_restraints]),
            target2=np.array([float(r.target2) for r in plane_restraints]),
            geom_type=np.array(
                [DIST_TYPE_CODES[r.geom_type] for r in plane_restraints], dtype=np.int64
            ),
            weight=np.array([float(r.weight) for r in plane_restraints]),
            mask=np.ones(n),
            start_sigma=start_sigma,
            stop_sigma=stop_sigma,
            start_step=start_step,
            stop_step=stop_step,
        )

    spec = RestraintSpec(
        n_active=len(active_sites),
        active_sites=active_sites,
        bond=bond,
        angle=angle,
        chiral=chiral,
        plane=plane,
        cistrans=cistrans_arr,
        distance=distance,
        rmsd=rmsd,
        group_angle=group_angle,
        group_dihedral=group_dihedral,
        group_plane=group_plane,
        vdw=vdw_arrays,
        vdw_config=vdw_config,
        group_improper=group_improper,
        active_vdw_config=active_vdw_config,
        vdw_max_atom_step=float(
            (cfg.get("vdw", {}) or {}).get("max_atom_step", VDW_MAX_ATOM_STEP_DEFAULT)
        ),
        vdw_neighbor_rebuild_interval=int(
            (cfg.get("vdw", {}) or {}).get(
                "neighbor_rebuild_interval", VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT
            )
        ),
        vdw_neighbor_skin=float(
            (cfg.get("vdw", {}) or {}).get("neighbor_skin", VDW_NEIGHBOR_SKIN_DEFAULT)
        ),
        conf_start_sigma=conf_start_sigma,
        conf_stop_sigma=conf_stop_sigma,
        conf_start_step=conf_start_step,
        conf_stop_step=conf_stop_step,
        custom=custom_specs,
    )
    append_library_arrays(
        spec,
        library,
        conformer_config,
        g2l,
        reference_plane_conditions=plane_conditions,
    )
    vdw_parts = []
    if vdw_intra is not None:
        vdw_parts.append(f"{len(vdw_intra.idx)}intra")
    if vdw_inter is not None:
        vdw_parts.append(f"{len(vdw_inter.idx)}inter")
    if vdw_config is not None:
        vdw_parts.append(
            f"{len(vdw_config.ligand_local)}lig/"
            f"{len(vdw_config.background_global)}bg/{vdw_config.max_neighbors}nn"
        )
    if active_vdw_config is not None:
        vdw_parts.append(
            f"{int(active_vdw_config.polymer_mask.sum())}poly/"
            f"{active_vdw_config.max_neighbors}nn"
        )
    vdw_desc = "+".join(vdw_parts) if vdw_parts else "off"
    logger.info(
        "built spec: n_active=%d bonds=%d angles=%d chirals=%d plane=%d cistrans=%d "
        "distances=%d rmsd=%d group_angle=%d group_dihedral=%d "
        "group_improper=%d group_plane=%d "
        "vdw=%s custom=%d relax_ff=%s",
        spec.n_active,
        len(spec.bond.idx) if spec.bond is not None else 0,
        len(spec.angle.idx) if spec.angle is not None else 0,
        len(spec.chiral.idx) if spec.chiral is not None else 0,
        len(spec.plane.idx) if spec.plane is not None else 0,
        len(spec.cistrans.idx) if spec.cistrans is not None else 0,
        len(distance_restraints),
        len(rmsd_restraints),
        len(angle_restraints),
        len(dihedral_restraints),
        len(improper_restraints),
        len(plane_restraints),
        vdw_desc,
        len(custom_specs),
        relax_ff,
    )
    return spec
