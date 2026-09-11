"""Adapter from a chai-lab ``AllAtomStructureContext`` to the rgi_toolkit protocols.

chai exposes ``AllAtomStructureContext`` (per-atom token index / element / ideal
reference coords + a covalent bond index pair) — the chai analogue of a biotite
AtomArray. This adapter reads it the same way the protenix adapter reads its
AtomArray: rebuild the ligand mol from the atom subset (ideal ref coords + bonds +
atomic numbers). Chain ids come from the per-token ``subchain_id`` tensorcode
(label_asym_id); the per-chain 1-based resid is computed like every other tool.

The tensorcode decode is reimplemented here (a trivial uint8->str with pad token
255) so rgi_toolkit never imports chai_lab — keeping the dependency direction
rgi_toolkit -> nothing, like the boltz/protenix adapters.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from rgi_toolkit._mol_build import build_ligand_mol as _build_ligand_mol
from rgi_toolkit._mol_build import generate_ideal_conformer as _generate_ideal_conformer
from rgi_toolkit.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)

_LIGAND_ENTITY = 3  # chai EntityType.LIGAND
_TENSORCODE_PAD = 255  # chai TENSORCODE_PAD_TOKEN


# Chai uses RNA=1 and DNA=2, unlike the shared boltz/ESM enum. Entity types
# preserve polymer identity for modified residues.
_MOLTYPE_BY_ID_CHAI = {0: "protein", 1: "rna", 2: "dna", 3: "ligand", 7: "ligand"}


def _decode_tensorcode(codes) -> str:
    """Inverse of chai's string_to_tensorcode: uint8 codes -> ASCII string,
    dropping the pad token (255)."""
    return "".join(chr(int(c)) for c in codes if int(c) != _TENSORCODE_PAD)


class ChaiStructureAdapter:
    """rgi_toolkit adapter over a chai ``AllAtomStructureContext``.

    ``num_atoms`` is the padded coordinate length in the diffusion loop
    (``atom_single_mask.shape[-1]``), passed in from the build site so the global
    flat index matches the ``atom_pos`` tensor exactly.
    """

    def __init__(
        self,
        structure_context,
        num_atoms: int,
        smiles_by_subchain=None,
        conf_restraints_by_subchain=None,
    ) -> None:
        self.sc = structure_context
        self._n_atom = int(num_atoms)
        # Source SMILES provide bond orders missing from the structure context.
        # _mol_from_smiles aligns the reordered heavy atoms by name.
        self._smiles_by_subchain = dict(smiles_by_subchain or {})
        # {subchain_id -> bool} per-chain conformer-restraints opt-in. Chai's FASTA
        # cannot carry this flag, so it comes from the sidecar map keyed by chain id.
        self._conf_restraints_by_subchain = dict(conf_restraints_by_subchain or {})

    def _token_chains(self) -> list[str]:
        """Per-token chain id string decoded from the subchain_id tensorcode."""
        sub = np.asarray(self.sc.subchain_id)  # (n_tokens, 4) uint8
        return [_decode_tensorcode(sub[t]) for t in range(sub.shape[0])]

    def iter_atoms(self) -> Iterator[AtomRecord]:
        sc = self.sc
        if sc is None:
            return
        atom_token = np.asarray(sc.atom_token_index)
        exists = np.asarray(sc.atom_exists_mask, dtype=bool)
        names = getattr(sc, "atom_ref_name", None)  # list[str], per-atom
        token_chain = self._token_chains()
        # residue_names is a cached property, not a method.
        resn = getattr(sc, "residue_names", None)
        ent = getattr(sc, "token_entity_type", None)
        ent = np.asarray(ent) if ent is not None else None
        # Per-chain token ordinals: one per standard residue, one per atom for
        # ligands and non-standard residues.
        chain_seen: dict[str, dict[int, int]] = {}
        chain_counter: dict[str, int] = {}
        for i in range(len(atom_token)):
            if not bool(exists[i]):
                continue
            tok = int(atom_token[i])
            ch = token_chain[tok]
            seen = chain_seen.setdefault(ch, {})
            if tok not in seen:
                chain_counter[ch] = chain_counter.get(ch, 0) + 1
                seen[tok] = chain_counter[ch]
            nm = str(names[i]).strip() if names is not None else None
            rnm = str(resn[tok]).strip() if resn is not None else None
            mt = _MOLTYPE_BY_ID_CHAI.get(int(ent[tok])) if ent is not None else None
            yield AtomRecord(
                chain=ch,
                resid=seen[tok],
                index=int(i),
                name=nm,
                resname=rnm,
                mol_type=mt,
                conformer_restraints=bool(
                    self._conf_restraints_by_subchain.get(ch, False)
                ),
            )

    def num_atoms(self) -> int:
        return self._n_atom

    def get_elements(self) -> np.ndarray:
        """(num_atoms,) atomic numbers; padding atoms are 0. chai stores
        ``atom_ref_element`` as the atomic number directly."""
        elements = np.zeros(self._n_atom, dtype=np.int64)
        sc = self.sc
        if sc is not None:
            z = np.asarray(sc.atom_ref_element).astype(np.int64)
            exists = np.asarray(sc.atom_exists_mask, dtype=bool)
            n = min(len(z), self._n_atom)
            elements[:n] = np.where(exists[:n], z[:n], 0)
        return elements

    def get_reference_positions(self) -> np.ndarray:
        positions = np.zeros((self._n_atom, 3), dtype=np.float64)
        if self.sc is not None:
            ref = np.asarray(self.sc.atom_ref_pos, dtype=np.float64)
            n = min(len(ref), self._n_atom)
            positions[:n] = ref[:n]
        return positions

    def get_reference_space_uid(self) -> np.ndarray:
        uid = np.full(self._n_atom, -1, dtype=np.int64)
        if self.sc is not None:
            ref_uid = np.asarray(self.sc.atom_ref_space_uid, dtype=np.int64)
            n = min(len(ref_uid), self._n_atom)
            uid[:n] = ref_uid[:n]
        return uid

    def _mol_from_smiles(self, smiles, idxs, elements, coords):
        """Map the complete source SMILES graph into chai coordinate order.

        chai names a SMILES ligand's atoms
        ``element+counter`` over the AddHs atom order, uppercased; we replicate that
        naming on a fresh ``MolFromSmiles`` and map it to chai's atoms by name because
        Chai may reorder ligand atoms.

        Renumber the source graph itself: rebuilding from elements and bond orders
        loses formal charges, isotopes and explicit H counts. That can make charged
        nitrogen invalid or change an aromatic [nH] tautomer before stereo alignment.

        Returns ``(mol, stereo_mol)``. ``stereo_mol`` retains the source graph's
        stereochemistry in chai atom order. An incomplete name match returns
        ``(None, None)`` so an opted-out ligand may still use the geometry fallback.
        """
        from collections import defaultdict

        from rdkit import Chem

        base = Chem.MolFromSmiles(smiles)
        if base is None or base.GetNumAtoms() != len(idxs):
            return None, None
        nbase = base.GetNumAtoms()
        cnt: dict = defaultdict(int)
        base_name: dict[int, str] = {}
        for i, atom in enumerate(Chem.AddHs(base).GetAtoms()):
            s = atom.GetSymbol()
            cnt[s] += 1
            if i < nbase:  # heavy atoms come first (AddHs appends H)
                base_name[i] = (s + str(cnt[s])).upper()
        names = getattr(self.sc, "atom_ref_name", None)
        if names is None:
            return None, None

        def _norm(nm: object) -> str:
            # Drop Chai's copy suffix (C1_1 -> C1) before matching source atom names.
            s = str(nm).strip().upper()
            base, _, suf = s.rpartition("_")
            return base if (base and suf.isdigit()) else s

        name_to_local = {_norm(names[int(g)]): li for li, g in enumerate(idxs)}
        base_to_local = {
            bi: name_to_local[nm] for bi, nm in base_name.items() if nm in name_to_local
        }
        if len(base_to_local) != nbase or set(base_to_local.values()) != set(
            range(nbase)
        ):
            return None, None
        if any(
            base.GetAtomWithIdx(bi).GetAtomicNum() != int(elements[li])
            for bi, li in base_to_local.items()
        ):
            return None, None
        local_to_base = sorted(base_to_local, key=base_to_local.__getitem__)
        stereo_mol = Chem.RenumberAtoms(base, local_to_base)
        stereo_mol.RemoveAllConformers()
        mol = Chem.Mol(stereo_mol)
        conf = Chem.Conformer(nbase)
        for i, point in enumerate(coords):
            conf.SetAtomPosition(i, tuple(float(value) for value in point))
        mol.AddConformer(conf, assignId=True)
        # Geometry-derived tags belong to the coordinate mol; source stereo stays
        # independent so a wrong model reference cannot replace the user's labels.
        Chem.AssignStereochemistryFrom3D(mol)
        return mol, stereo_mol

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        sc = self.sc
        if sc is None:
            return
        atom_token = np.asarray(sc.atom_token_index)
        exists = np.asarray(sc.atom_exists_mask, dtype=bool)
        token_entity = np.asarray(sc.token_entity_type)
        ref_pos = np.asarray(sc.atom_ref_pos, dtype=np.float64)
        ref_elem = np.asarray(sc.atom_ref_element).astype(np.int64)
        token_chain = self._token_chains()
        per_atom_chain = np.array([token_chain[int(t)] for t in atom_token])
        is_lig = np.array(
            [int(token_entity[int(t)]) == _LIGAND_ENTITY for t in atom_token]
        )
        lig_mask = is_lig & exists
        # atom_covalent_bond_indices contains inter-residue links only. Without
        # source SMILES, infer intra-ligand connectivity from reference geometry.
        for ch in np.unique(per_atom_chain[lig_mask]):
            idxs = np.where((per_atom_chain == ch) & lig_mask)[0]
            coords = ref_pos[idxs]
            # Source SMILES retain the complete chemistry needed by force fields.
            # Fall back to perception only when the source is absent or an unmatched
            # ligand has not opted into conformer restraints.
            smiles = self._smiles_by_subchain.get(str(ch))
            mol = None
            stereo_mol = None
            stereo_required = bool(
                self._conf_restraints_by_subchain.get(str(ch), False)
            )
            if smiles is not None:
                mol, stereo_mol = self._mol_from_smiles(
                    smiles, idxs, ref_elem[idxs], coords
                )
                if stereo_mol is None and stereo_required:
                    raise ValueError(
                        f"chai ligand {ch}: cannot map source SMILES "
                        "stereochemistry to the model atom order"
                    )
            if mol is not None:
                # Embed source stereo to repair inverted model references. Featurizer
                # validation also covers fallback to model coordinates.
                ideal = (
                    _generate_ideal_conformer(stereo_mol)
                    if stereo_mol is not None
                    else None
                )
                if ideal is not None and len(ideal) == len(idxs):
                    coords = ideal
                    mol, rebuilt_stereo_mol = self._mol_from_smiles(
                        smiles, idxs, ref_elem[idxs], ideal
                    )
                    if rebuilt_stereo_mol is not None:
                        stereo_mol = rebuilt_stereo_mol
                    logger.info("chai ligand %s: stereo-correct ETKDG target", ch)
                else:
                    logger.info(
                        "chai ligand %s: ETKDG target unavailable, using ref_pos", ch
                    )
            if mol is None:
                mol = _build_ligand_mol(ref_elem[idxs], coords, [], perceive_bonds=True)
            yield LigandConf(
                mol=mol,
                conf_coords=coords,
                global_indices=idxs.astype(np.int64),
                # Per-chain opt-in from the sidecar map. Absent defaults to False.
                conformer_restraints=self._conf_restraints_by_subchain.get(
                    str(ch), False
                ),
                stereo_mol=stereo_mol,
            )
