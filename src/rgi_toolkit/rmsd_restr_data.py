"""RMSD restraint config + atom-site resolution.

One ``rmsd_restraints_config`` entry restrains the **Kabsch-superposed RMSD**
between a moving group in the diffusion structure and a fixed group from a
reference structure -- given as EITHER ``ref_pdb`` (PDB) OR ``ref_cif`` (mmCIF),
mutually exclusive; both parse to the same atom records via
``read_pdb_atoms`` / ``read_cif_atoms`` -- shaped by a restraint-type block
(``harmonic`` / ``flat-bottomed`` /
``flat-bottomed1`` / ``flat-bottomed2``) on the RMSD value and optimised by the CG solver. The
superposition ("fit") atoms and the measured ("calc") atoms can differ:

  atom_selection_ref_fit / atom_selection_target_fit   -> Kabsch superposition
  atom_selection_ref_calc / atom_selection_target_calc -> RMSD measured here

The shorthand ``atom_selection_ref`` / ``atom_selection_target`` sets both fit and calc.

Selections are OPTIONAL. Omit them (no ``atom_selection*`` at all) to fit + measure
RMSD over the WHOLE structure: the whole diffusion structure is superposed onto the
whole reference and the RMSD is taken over everything, BEST-EFFORT -- atoms matched to
the reference by identity (chain, resid, name) are used and any structure atom missing
from the reference (e.g. the ref has no hydrogens) is skipped, so an incomplete ref
still works (pymol-align-like). Only a reference (``ref_pdb`` or ``ref_cif``) and a
restraint-type block are required.

Reference and target atoms are paired by IDENTITY (chain, resid, atom-name) when
both sides expose atom names, so the reference PDB's atom order need not match the
tool's internal order. If names are unavailable on either side it falls back to
selection-order pairing.

Pairing is BEST-EFFORT by default (PyMOL align/super-like): a target atom with no
matching (chain, resid, name) in the reference is SKIPPED, so a partially-overlapping
reference (missing hydrogens/side chains, an incomplete model) still fits + measures
over whatever overlaps. It still raises if NOTHING overlaps, so a wholly-wrong
selection is not silent. Set ``best_effort: false`` on the entry for STRICT pairing
that raises on the first unmatched atom (catch a mistyped selection loudly). The
order-fallback path (no atom names) always requires equal counts regardless.

``pairing`` is **"align" by DEFAULT** (set ``pairing: "identity"`` for the pure
ordinal pairing above). align matches a **homolog** reference (different sequence,
substitutions and indels): each polymer chain is sequence-aligned (``_align``:
BLOSUM62 for protein, residue-name identity for nucleic acids, semi-global with free
end gaps) so target residues map onto the corresponding ref residues regardless of
numbering or register, then atoms pair by name within each aligned residue pair. align
**only engages when the structure has polymer atoms** -- a ligand-only structure (or
any atom lacking a polymer type) falls back to ordinal identity, so the default is safe
on non-polymer inputs and never demands a sequence where there is none. NOTE align
pairs ALL shared atom names -- backbone (N/CA/C/O) PLUS CB and any side-chain names
that coincide -- so for a substituted residue the prediction's matching side-chain
atoms are pinned onto the REFERENCE's side-chain coordinates. To avoid that pinning,
restrict the selection to the backbone with a ``backbone`` / ``name CA`` atom_selection,
so only those atoms are superposed; this is PyMOL-align without the outlier-rejection
cycles. align defaults to best-effort (gap/unshared atoms skipped), but ``best_effort:
false`` with an EXPLICIT selection is honoured -- a residue aligned to a gap then raises.
align needs residue names on both sides (the reference
always has them; the target needs an adapter that fills ``AtomRecord.resname`` -- it
raises loudly otherwise). Ligand / non-polymer atoms stay on ordinal (chain, resid,
name) identity even under align.

``start_sigma`` / ``stop_sigma`` bound the NOISE WINDOW in which the restraint acts:
active when ``stop_sigma <= sigma <= start_sigma`` (``start_sigma`` defaults to +inf =
on from the first step; ``stop_sigma`` defaults to -1 = never released). Setting
``stop_sigma > 0`` RELEASES the restraint for the final low-sigma steps so the model's
own denoising can repair strained geometry, such as a peptide bond between a restrained
residue and a free tail. Choose ``stop_sigma`` in the model's sigma units.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from rgi_toolkit._align import pair_residues
from rgi_toolkit._atom_names import normalise_atom_name
from rgi_toolkit._config_util import (
    apply_window_params,
    coerce_bool,
    parse_geom_type,
    warn_unknown_keys,
)
from rgi_toolkit._moltype import polymer_type
from rgi_toolkit.atom_context import FrameworkAdapter, candidate_dict
from rgi_toolkit.pdb_ref import read_cif_atoms, read_pdb_atoms
from rgi_toolkit.selection import AtomSelector

logger = logging.getLogger(__name__)


def build_resid_map(atoms, ref_atoms, ref_path) -> dict:
    """Sequence-align each polymer chain present on both sides and return
    ``{(chain, target_resid): ref_resid}``. Polymer typing prefers an explicit
    ``mol_type`` (boltz/esm set it) and otherwise derives it from the residue name
    (protenix/of3/chai don't set mol_type), so only resname must be plumbed. Shared by the
    built-in RMSD restraint and the custom ``rmsd()`` primitive, so both align identically."""

    def seqs(records, resname_attr, side):
        by_chain: dict = {}
        mtype: dict = {}
        for a in records:
            rn = getattr(a, resname_attr, None)
            ptype = polymer_type(getattr(a, "mol_type", None), rn)
            if ptype is None:
                continue  # ligand/water/unknown -> not sequence-aligned
            if not rn:  # a polymer residue with no name can't be aligned
                raise ValueError(
                    f"rmsd pairing='align' needs residue names on the {side} side, "
                    f"but a polymer atom (chain {a.chain}) has none"
                    + (
                        ""
                        if side == "reference"
                        else " (adapter not plumbed for resname)"
                    )
                )
            d = by_chain.setdefault(a.chain, {})
            d.setdefault(a.resid, rn)
            mtype.setdefault(a.chain, ptype)
        return {c: sorted(d.items()) for c, d in by_chain.items()}, mtype

    t_seq, t_mt = seqs(atoms, "resname", "target")
    r_seq, _ = seqs(ref_atoms, "res_name", "reference")
    resid_map: dict = {}
    matched_chains = 0
    for ch, t_res in t_seq.items():
        r_res = r_seq.get(ch)
        if not r_res:
            continue
        matched_chains += 1
        for t_rid, r_rid in pair_residues(t_res, r_res, t_mt.get(ch)):
            resid_map[(ch, t_rid)] = r_rid
    if not resid_map:
        raise ValueError(
            "rmsd pairing='align' aligned no residues (no common polymer chain, or "
            f"chain ids differ between prediction and ref {ref_path!r}); check chain naming"
        )
    logger.info(
        "rmsd align: %d chain(s), %d residue pairs", matched_chains, len(resid_map)
    )
    return resid_map


def pair_target_to_ref(
    atoms,
    ref_atoms,
    sel_target,
    sel_ref,
    tag,
    *,
    ref_path,
    best_effort=False,
    align=False,
    resid_map=None,
):
    """Resolve one (target, ref) selection pair -> (target_global_indices, ref_coords
    aligned to the target order). A ``None`` selection means the WHOLE structure on that
    side (no filter). Pairing is by IDENTITY (chain, resid, name) when both sides expose
    names; else selection-order (counts must match). With ``best_effort`` (the no-selection
    whole-structure default) a target atom missing from the reference is SKIPPED rather than
    raising, so an incomplete ref still fits 'as much as possible'; with an explicit
    selection it stays strict (a missing match raises). With ``align`` a polymer target
    atom's resid is first translated to the aligned ref resid via ``resid_map`` (ligands stay
    on ordinal identity). Shared by the built-in RMSD restraint and the custom ``rmsd()``
    primitive, so both pair identically."""
    if sel_target is None:
        tgt = list(atoms)
    else:
        st = AtomSelector(sel_target)
        tgt = [a for a in atoms if st.matches(candidate_dict(a))]
    if sel_ref is None:
        ref = list(ref_atoms)
    else:
        sr = AtomSelector(sel_ref)
        ref = [
            r
            for r in ref_atoms
            if sr.matches(candidate_dict(r, resname_attr="res_name"))
        ]
    if not tgt:
        raise ValueError(
            f"rmsd {tag} target selection matched no atoms: {sel_target!r}"
        )
    if not ref:
        raise ValueError(
            f"rmsd {tag} ref selection matched no atoms: {sel_ref!r} in {ref_path!r}"
        )
    tgt_named = all(a.name for a in tgt)
    ref_named = all(r.name for r in ref)
    if align and not (tgt_named and ref_named):
        raise ValueError(
            f"rmsd {tag} pairing='align' needs atom names on both sides to pair "
            "atoms within aligned residues"
        )
    logger.debug(
        "rmsd %s pairing=%s target=%d ref=%d; target names[:4]=%s ref names[:4]=%s",
        tag,
        "identity" if (tgt_named and ref_named) else "order",
        len(tgt),
        len(ref),
        [a.name for a in tgt[:4]],
        [r.name for r in ref[:4]],
    )
    if tgt_named and ref_named:
        # Duplicate identity keys would collapse to the last atom and mispair the reference.
        refmap = {}
        for r in ref:
            k = (r.chain, r.resid, normalise_atom_name(r.name))
            if k in refmap:
                raise ValueError(
                    f"rmsd {tag}: duplicate reference atom {k} in {ref_path!r} — "
                    f"ambiguous identity pairing; disambiguate the ref selection"
                )
            refmap[k] = (r.x, r.y, r.z)
        sites, coords, skipped = [], [], 0
        for a in tgt:
            if align and polymer_type(a.mol_type, a.resname) is not None:
                # Map polymer residues through the alignment; gaps obey best_effort.
                # Ligands use ordinal identity below.
                mapped = resid_map.get((a.chain, a.resid))
                if mapped is None:
                    if best_effort:
                        skipped += 1
                        continue
                    raise ValueError(
                        f"rmsd {tag}: target polymer residue (chain {a.chain}, "
                        f"resid {a.resid}) aligned to a gap in ref {ref_path!r} (no "
                        f"corresponding residue); set best_effort:true to skip gaps"
                    )
                key = (a.chain, mapped, normalise_atom_name(a.name))
            else:
                key = (a.chain, a.resid, normalise_atom_name(a.name))
            if key not in refmap:
                if best_effort:
                    skipped += 1
                    continue
                raise ValueError(
                    f"rmsd {tag}: target atom {key} has no matching "
                    f"(chain, resid, name) in ref {ref_path!r}"
                )
            sites.append(int(a.index))
            coords.append(refmap[key])
        if not sites:
            raise ValueError(
                f"rmsd {tag}: no target atom matched the reference by "
                f"(chain, resid, name) in {ref_path!r}"
            )
        if skipped:
            logger.info(
                "rmsd %s (best-effort): matched %d / %d atoms "
                "(%d unmatched in ref skipped)",
                tag,
                len(sites),
                len(tgt),
                skipped,
            )
        return sites, np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    # order fallback (no atom names): pair by selection order, counts must match
    if len(tgt) != len(ref):
        raise ValueError(
            f"rmsd {tag} atom-count mismatch (order pairing): target={len(tgt)} "
            f"vs ref={len(ref)}; provide atom names or matching selections"
        )
    sites = [int(a.index) for a in tgt]
    coords = np.asarray([(r.x, r.y, r.z) for r in ref], dtype=np.float64)
    return sites, coords.reshape(-1, 3)


_KNOWN_RMSD_KEYS = {
    "ref_pdb",
    "ref_cif",
    "harmonic",
    "flat-bottomed",
    "flat-bottomed1",
    "flat-bottomed2",
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
    "atom_selection_ref",
    "atom_selection_target",
    "atom_selection_ref_fit",
    "atom_selection_target_fit",
    "atom_selection_ref_calc",
    "atom_selection_target_calc",
    "best_effort",
    "pairing",
}


@dataclass
class RmsdData:
    # Mutually exclusive PDB/mmCIF inputs; both readers return PdbAtom records.
    ref_pdb: str = None
    ref_cif: str = None
    ref_path: str = None
    # restraint type on the Kabsch RMSD value, mirroring the distance restraint:
    # harmonic / flat-bottomed / flat-bottomed1 (lower bound) / flat-bottomed2 (upper
    # bound). target1/target2 are the (Angstrom) bounds; the unused one is 0.0.
    rmsd_type: str = None
    target1: float = None
    target2: float = None
    weight: float = None
    start_sigma: float = None  # per-restraint; from_dict defaults None -> +inf
    # Release below stop_sigma so late denoising can repair strained boundary
    # geometry. -1 keeps the restraint active down to sigma=0.
    stop_sigma: float = -1.0
    # Inclusive step window, mutually exclusive with an explicit sigma window.
    # Step counts differ across predictors.
    start_step: float = float("-inf")
    stop_step: float = float("inf")
    # selection strings (fit = superposition atoms, calc = measured atoms)
    sel_ref_fit: str = None
    sel_target_fit: str = None
    sel_ref_calc: str = None
    sel_target_calc: str = None
    # Skip unmatched atoms unless strict pairing is requested. No matches always raises.
    best_effort: bool = True
    # Identity pairs by (chain, resid, name); the default align mode first aligns
    # polymer sequences. Non-polymer atoms always use identity.
    pairing: str = None
    resid_map: dict = field(default=None)  # (chain, target_resid) -> ref_resid (align)
    # resolved: global target atom indices + paired reference coords (n_atoms, 3)
    fit_target_sites: list = field(default=None)
    fit_ref_coords: np.ndarray = field(default=None)
    calc_target_sites: list = field(default=None)
    calc_ref_coords: np.ndarray = field(default=None)
    run_restr: bool = None

    def set_config(self, config: dict):
        if "atom_selection" in config:
            raise ValueError(
                "rmsd_restraints_config entry: bare 'atom_selection' is not a valid key "
                "-- use 'atom_selection_ref'/'atom_selection_target' (both-sides "
                "shorthand) or the '_fit'/'_calc' suffixed keys "
                "(e.g. atom_selection_target_fit)."
            )
        warn_unknown_keys(
            config, _KNOWN_RMSD_KEYS, "rmsd_restraints_config entry", logger
        )
        self.ref_pdb = config.get("ref_pdb", None)
        self.ref_cif = config.get("ref_cif", None)
        if self.ref_pdb is not None and self.ref_cif is not None:
            raise ValueError(
                "rmsd_restraints_config entry: ref_pdb and ref_cif are mutually "
                "exclusive -- give exactly one reference structure"
            )
        self.ref_path = self.ref_pdb if self.ref_pdb is not None else self.ref_cif
        # RMSD targets are in Angstroms, so no angular conversion is needed.
        self.rmsd_type, self.target1, self.target2 = parse_geom_type(
            config, "target_rmsd", float
        )
        apply_window_params(self, config, "rmsd_restraints_config entry")
        # explicit _fit / _calc override the shared ref/target shorthand. A selection
        # left None means "the whole structure on that side" (resolved best-effort).
        ref = config.get("atom_selection_ref")
        tgt = config.get("atom_selection_target")
        self.sel_ref_fit = config.get("atom_selection_ref_fit", ref)
        self.sel_target_fit = config.get("atom_selection_target_fit", tgt)
        self.sel_ref_calc = config.get("atom_selection_ref_calc", ref)
        self.sel_target_calc = config.get("atom_selection_target_calc", tgt)
        self.best_effort = coerce_bool(config.get("best_effort"), True)
        self.pairing = config.get("pairing") or "align"
        if self.pairing not in ("identity", "align"):
            raise ValueError(
                f"rmsd pairing must be 'identity' or 'align', got {self.pairing!r}"
            )
        self.run_restr = self.ref_path is not None and self.rmsd_type is not None
        if not self.run_restr:
            raise ValueError(
                "rmsd_restraints_config entry requires ref_pdb or ref_cif and a "
                "restraint-type block: harmonic{target_rmsd} / "
                "flat-bottomed{target_rmsd1,target_rmsd2} "
                "/ flat-bottomed1{target_rmsd1} / flat-bottomed2{target_rmsd2} (atom "
                "selections are optional: omit them to fit + measure RMSD over the whole "
                "structure, best-effort over atoms matched to the reference)"
            )
        logger.info(
            "rmsd restraint configured: type=%s target1=%.3f target2=%.3f",
            self.rmsd_type,
            self.target1,
            self.target2,
        )

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        if not self.run_restr:
            return
        atoms = list(adapter.iter_atoms())
        reader = read_cif_atoms if self.ref_cif is not None else read_pdb_atoms
        ref_atoms = reader(self.ref_path)
        # Build polymer correspondence once for both fit and calc. Ligand-only
        # structures use identity without attempting sequence alignment.
        has_polymer = any(
            polymer_type(a.mol_type, a.resname) is not None for a in atoms
        )
        align = self.pairing == "align" and has_polymer
        if align:
            self.resid_map = self._build_resid_map(atoms, ref_atoms)
        # Omitted target selections use whole-structure best-effort pairing;
        # explicit selections obey best_effort, including alignment gaps.
        self.fit_target_sites, self.fit_ref_coords = self._pair(
            atoms,
            ref_atoms,
            self.sel_target_fit,
            self.sel_ref_fit,
            "fit",
            best_effort=self.sel_target_fit is None or self.best_effort,
            align=align,
        )
        self.calc_target_sites, self.calc_ref_coords = self._pair(
            atoms,
            ref_atoms,
            self.sel_target_calc,
            self.sel_ref_calc,
            "calc",
            best_effort=self.sel_target_calc is None or self.best_effort,
            align=align,
        )
        logger.info(
            "rmsd restraint resolved: fit=%d calc=%d atoms, type=%s",
            len(self.fit_target_sites),
            len(self.calc_target_sites),
            self.rmsd_type,
        )

    def _build_resid_map(self, atoms, ref_atoms) -> dict:
        """Sequence-align polymer chains -> ``{(chain, target_resid): ref_resid}`` (thin
        wrapper over the module-level ``build_resid_map`` shared with custom ``rmsd()``)."""
        return build_resid_map(atoms, ref_atoms, self.ref_path)

    def _pair(
        self, atoms, ref_atoms, sel_target, sel_ref, tag, best_effort=False, align=False
    ):
        """Resolve one (target, ref) selection pair -> (target_global_indices, ref_coords
        aligned to the target order) — thin wrapper over the module-level
        ``pair_target_to_ref`` shared with the custom ``rmsd()`` primitive."""
        return pair_target_to_ref(
            atoms,
            ref_atoms,
            sel_target,
            sel_ref,
            tag,
            ref_path=self.ref_path,
            best_effort=best_effort,
            align=align,
            resid_map=self.resid_map,
        )

    def iter_global_sites(self):
        """Yield resolved fit and calculation indices used by this restraint."""
        yield from self.fit_target_sites or ()
        yield from self.calc_target_sites or ()

    def is_valid(self) -> bool:
        return self.run_restr
