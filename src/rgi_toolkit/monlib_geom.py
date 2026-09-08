"""CCP4 dictionary geometry, uncertainties, and local peptide alternatives.

Library targets replace covered polymer reference geometry. ESDs remain distinct
from user slack and are converted to inverse-variance weights when packing the spec.
Gemmi is loaded only when a structure actually requests dictionary geometry.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from rgi_toolkit._monlib_records import (
    KINDS,
    GeometryTarget,
    chiral_volume_esd,
    deduplicate,
    merge_conditions,
    modified_restraints,
    read_restraints,
    resolve,
    validate_target,
)

logger = logging.getLogger(__name__)
_LINK_ID = {"protein": "TRANS", "dna": "p", "rna": "p"}
_PEPTIDE_LINK_BY_GROUP = {"PPeptide": "PTRANS", "MPeptide": "NMTRANS"}


@dataclass(frozen=True)
class PeptideChoice:
    atoms: tuple[int, int, int, int]
    trans: float
    cis: float


@dataclass
class LibraryTargets:
    terms: dict[str, list[GeometryTarget]] = field(
        default_factory=lambda: {k: [] for k in KINDS}
    )
    atoms: set[int] = field(default_factory=set)
    chiral_centers: set[int] = field(default_factory=set)
    chiral_fallback: set[int] = field(default_factory=set)
    covered_links: set[tuple[int, int]] = field(default_factory=set)
    bond_pairs: set[tuple[int, int]] = field(default_factory=set)
    angle_tuples: set[tuple[int, int, int]] = field(default_factory=set)
    peptides: list[PeptideChoice] = field(default_factory=list)
    covered: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    on_missing: str = "fallback"
    source: object | None = None
    atom_types: dict[int, str] = field(default_factory=dict)
    plane_groups: set[tuple[int, ...]] = field(default_factory=set)


def parse_config(conformer_config: dict | None) -> tuple[str | None, str] | None:
    """Parse only: a None path requests the cache; no filesystem/network I/O here."""
    spec = (conformer_config or {}).get("monomer_library")
    if spec is None or spec is False:
        return None
    on_missing = "fallback"
    if spec is True:
        path = None
    elif isinstance(spec, str):
        path = spec
    elif isinstance(spec, dict):
        unknown = set(spec) - {"path", "on_missing"}
        if unknown:
            raise ValueError(
                "conformer_restraints_config.monomer_library: unknown key(s) "
                f"{sorted(unknown)}. Known keys: ['on_missing', 'path']"
            )
        path = spec.get("path")
        if "path" in spec and (not isinstance(path, str) or not path.strip()):
            raise ValueError(
                "conformer_restraints_config.monomer_library.path must be a nonempty path string"
            )
        on_missing = spec.get("on_missing", on_missing)
    else:
        raise ValueError(
            "conformer_restraints_config.monomer_library must be a boolean, path string or dict"
        )
    if path is not None and not path.strip():
        raise ValueError(
            "conformer_restraints_config.monomer_library: path must be nonempty"
        )
    if on_missing not in ("fallback", "error"):
        raise ValueError(
            f"conformer_restraints_config.monomer_library.on_missing: unknown value {on_missing!r}"
        )
    return (
        None if path is None else os.path.expanduser(os.path.expandvars(path))
    ), on_missing


def missing_geometry(message, on_missing, fallback="kept reference fallback"):
    if on_missing == "error":
        raise ValueError(f"monomer library: {message}")
    logger.warning("[rgi_toolkit] monomer library: %s; %s", message, fallback)


class MonomerLibrary:
    def __init__(self, monlib, path: str):
        self._monlib = monlib
        self.path = path

    @classmethod
    def load(cls, path: str | None, resnames):
        import gemmi

        from rgi_toolkit._monlib_cache import (
            ensure_cached_library,
            revision,
            validate_library_directory,
        )

        path = ensure_cached_library() if path is None else path
        validate_library_directory(Path(path))
        monlib = gemmi.MonLib()
        monlib.read_monomer_lib(os.path.join(path, ""), sorted(set(resnames)))
        sha = revision(Path(path))
        logger.info(
            "[rgi_toolkit] monomer library source: %s; Git SHA: %s",
            path,
            sha or "not a Git checkout",
        )
        return cls(monlib, path)

    def covers(self, resname):
        return bool(resname) and resname in self._monlib.monomers

    def link_id(self, mol_type, resname2):
        base = _LINK_ID.get(mol_type)
        if base is None or mol_type != "protein" or not self.covers(resname2):
            return base
        group = self._monlib.monomers[resname2].group
        specialised = _PEPTIDE_LINK_BY_GROUP.get(getattr(group, "name", ""))
        return specialised or base

    def get_link(self, link_id):
        return (
            self._monlib.links[link_id]
            if link_id and link_id in self._monlib.links
            else None
        )

    def link_mods(self, mol_type, side1, side2, resname2=None):
        link = self.get_link(self.link_id(mol_type, resname2))
        if link is None:
            return []
        return self.mods_for_link(
            link, [side for side, yes in ((1, side1), (2, side2)) if yes]
        )

    def mods_for_link(self, link, sides):
        mods = []
        for side in sides:
            mod_id = getattr(link, f"side{side}").mod
            if mod_id:
                if mod_id not in self._monlib.modifications:
                    raise ValueError(
                        f"monomer library link {link.id}: missing modification {mod_id}"
                    )
                mods.append(self._monlib.modifications[mod_id])
        return mods


def _link_options(library, previous, current, targets, enabled):
    link_id = library.link_id(current["mol_type"], current["resname"])
    trans = library.get_link(link_id)
    if trans is None:
        missing_geometry(
            f"no library entry for link {link_id}",
            targets.on_missing,
            "using built-in link geometry",
        )
        return []
    targets.covered_links.add((previous["uid"], current["uid"]))
    sides = {1: previous["names"], 2: current["names"]}
    trans_rows = resolve(read_restraints(trans.rt), sides)
    options = [(trans, trans_rows, ())]
    if current["mol_type"] != "protein":
        return options
    cis = library.get_link(link_id.replace("TRANS", "CIS"))
    trans_omega = next((r for r in trans_rows["cistrans"] if r.label == "omega"), None)
    if cis is None:
        if trans_omega is not None and "cistrans" in enabled:
            missing_geometry(
                f"{link_id} has no cis counterpart; omega omitted",
                targets.on_missing,
                "kept trans link geometry without omega",
            )
            trans_rows["cistrans"] = [
                r for r in trans_rows["cistrans"] if r.label != "omega"
            ]
        return options
    cis_rows = resolve(read_restraints(cis.rt), sides)
    cis_omega = next((r for r in cis_rows["cistrans"] if r.label == "omega"), None)
    if trans_omega is None or cis_omega is None:
        return (
            options  # incomplete backbone: retain trans geometry for the atoms present
        )
    if trans_omega.atoms not in (cis_omega.atoms, cis_omega.atoms[::-1]):
        raise ValueError(
            f"monomer library {link_id}: cis/trans omega atom orders disagree"
        )
    if not all(math.isfinite(r.value) for r in (trans_omega, cis_omega)):
        raise ValueError(f"monomer library {link_id}: nonfinite omega target")
    selector = len(targets.peptides)
    targets.peptides.append(
        PeptideChoice(
            trans_omega.atoms,
            -math.radians(trans_omega.value),
            -math.radians(cis_omega.value),
        )
    )
    return [(trans, trans_rows, ((selector, 0),)), (cis, cis_rows, ((selector, 1),))]


def _add_geometry(targets, records, conditions, enabled, source, peptide=False):
    targets.bond_pairs.update(tuple(sorted(r.atoms)) for r in records["bond"])
    targets.angle_tuples.update(r.atoms for r in records["angle"])
    targets.plane_groups.update(
        r.atoms for r in records["plane"] if math.isfinite(r.esd) and r.esd > 0
    )
    for kind in ("bond", "angle", "plane", "cistrans"):
        if kind not in enabled:
            continue
        for r in records[kind]:
            if kind == "cistrans" and not (
                r.label == "omega"
                or r.label.startswith("sp2_sp2")
                or (peptide and r.label.startswith("chi"))
            ):
                continue
            if not validate_target(r, f"{source} {kind}"):
                continue
            angular = kind in ("angle", "cistrans")
            value = math.radians(r.value) if angular else r.value
            if kind == "cistrans":
                value = -value  # RGI's signed dihedral is the negative of Gemmi's
            targets.terms[kind].append(
                GeometryTarget(
                    r.atoms,
                    value,
                    math.radians(r.esd) if angular else r.esd,
                    max(1, r.period),
                    conditions=conditions,
                )
            )


def _add_chirals(targets, records, bonds, angles, conditions, enabled, source):
    if "chiral" not in enabled:
        return
    for r in records["chiral"]:
        center = r.atoms[0]
        targets.chiral_centers.add(center)
        result = chiral_volume_esd(r, bonds, angles)
        if result is None:
            if center not in targets.chiral_fallback:
                missing_geometry(
                    f"{source}: cannot derive chiral volume/ESD at atom {center}",
                    targets.on_missing,
                )
            targets.chiral_fallback.add(center)
            continue
        volume, esd = result
        targets.terms["chiral"].append(
            GeometryTarget(
                r.atoms,
                -volume if r.sign == "Negative" else volume,
                esd,
                both=r.sign == "Both",
                conditions=conditions,
            )
        )


def collect(library, residues, on_missing, connections=(), enabled=None):
    """Resolve each residue/link with only its adjacent peptide state combinations."""
    enabled = set(KINDS) if enabled is None else set(enabled)
    targets = LibraryTargets(on_missing=on_missing, source=library)
    covered = {m["resname"] for m in residues if library.covers(m["resname"])}
    missing = {str(m["resname"]) for m in residues if not library.covers(m["resname"])}
    if missing:
        missing_geometry(
            f"no library entry for residue(s) {sorted(missing)}", on_missing
        )
    targets.covered, targets.missing = tuple(sorted(covered)), tuple(sorted(missing))
    edges = []
    neighbours = {m["uid"]: [] for m in residues}
    for previous, current in connections:
        options = _link_options(library, previous, current, targets, enabled)
        if options:
            edges.append((previous, current, options))
            neighbours[previous["uid"]].append((options, 1))
            neighbours[current["uid"]].append((options, 2))
            for _link, records, conditions in options:
                _add_geometry(targets, records, conditions, enabled, "link")

    variants = {}
    for meta in residues:
        uid = meta["uid"]
        variants[uid] = []
        if not library.covers(meta["resname"]):
            variants[uid].append(({k: [] for k in KINDS}, ()))
            continue
        targets.atoms.update(meta["names"].values())
        comp = library._monlib.monomers[meta["resname"]]
        adjoining = neighbours[uid]
        original_chirals = read_restraints(comp.rt)["chiral"]
        for choices in itertools.product(*(options for options, _side in adjoining)):
            conditions = merge_conditions(*(choice[2] for choice in choices))
            if conditions is None:
                continue
            mods = list(
                itertools.chain.from_iterable(
                    library.mods_for_link(choice[0], [side])
                    for choice, (_options, side) in zip(choices, adjoining)
                )
            )
            named = modified_restraints(comp.rt, mods)
            # Chemical typing and exclusions are needed even when the associated
            # geometry energy is disabled. Link modifications can change both.
            atom_types = {a.id: a.chem_type for a in comp.atoms}
            for mod in mods:
                for atom_mod in mod.atom_mods:
                    old = atom_mod.old_id
                    op = (
                        chr(atom_mod.func)
                        if isinstance(atom_mod.func, int)
                        else atom_mod.func
                    )
                    if op == "d":
                        atom_types.pop(old, None)
                    else:
                        name = atom_mod.new_id or old
                        value = atom_mod.chem_type or atom_types.get(old, "")
                        if name != old:
                            atom_types.pop(old, None)
                        atom_types[name] = value
            from rgi_toolkit._atom_names import normalise_atom_name

            targets.atom_types.update(
                {
                    meta["names"][normalise_atom_name(n)]: t
                    for n, t in atom_types.items()
                    if normalise_atom_name(n) in meta["names"]
                }
            )
            # Explicit deletions must not resurrect a reference-conformer chiral.
            final_atoms = {r.atoms for r in named["chiral"]}
            for r in original_chirals:
                if r.atoms not in final_atoms:
                    center = meta["names"].get(r.atoms[0][1])
                    if center is not None:
                        targets.chiral_centers.add(center)
            records = resolve(named, {1: meta["names"]})
            variants[uid].append((records, conditions))
            _add_geometry(
                targets,
                records,
                conditions,
                enabled,
                meta["resname"],
                peptide=meta["mol_type"] == "protein",
            )
            _add_chirals(
                targets,
                records,
                records["bond"],
                records["angle"],
                conditions,
                enabled,
                meta["resname"],
            )

    # Link chirals (notably phosphodiester P) depend on both residue dictionaries
    # after modifications as well as the link's own bonds/angles.
    for previous, current, options in edges:
        for (_link, link_rows, link_cond), (prev, pc), (curr, cc) in itertools.product(
            options, variants[previous["uid"]], variants[current["uid"]]
        ):
            conditions = merge_conditions(link_cond, pc, cc)
            if conditions is not None:
                _add_chirals(
                    targets,
                    link_rows,
                    prev["bond"] + curr["bond"] + link_rows["bond"],
                    prev["angle"] + curr["angle"] + link_rows["angle"],
                    conditions,
                    enabled,
                    "link",
                )
    targets.terms["chiral"] = [
        r for r in targets.terms["chiral"] if r.atoms[0] not in targets.chiral_fallback
    ]
    targets.chiral_centers.difference_update(targets.chiral_fallback)
    targets.terms = {kind: deduplicate(rows) for kind, rows in targets.terms.items()}
    return targets
