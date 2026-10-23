"""Adapt BioEmu2's padded atom order and reference chemistry to RGI."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from rgi_toolkit._mol_build import mol_with_reference_conformer
from rgi_toolkit.atom_context import AtomRecord, LigandConf, decode_atom_name


class BioEmu2Adapter:
    """Receive plain arrays from one BioEmu2 structure before multiplicity expansion.

    Coordinates and reference positions are in angstroms. Each ligand source
    molecule carries ``atom_name`` properties in BioEmu2's naming convention.
    Padding remains in coordinate arrays but never participates in selections.
    """

    def __init__(
        self,
        *,
        chain_ids,
        atom_chain_indices,
        atom_to_token_map,
        atom_mask,
        atom_name_codes,
        residue_names,
        mol_types,
        elements,
        ref_pos,
        ref_space_uid,
        conformer_chains=(),
        ligand_mols=None,
    ) -> None:
        self._chain_ids = tuple(str(chain) for chain in chain_ids)
        self._chains = np.asarray(atom_chain_indices, dtype=np.int64)
        self._tokens = np.asarray(atom_to_token_map, dtype=np.int64)
        self._mask = np.asarray(atom_mask, dtype=bool)
        self._names = np.asarray(atom_name_codes, dtype=np.int64)
        self._resnames = np.asarray(residue_names, dtype=object)
        self._types = np.asarray(mol_types, dtype=object)
        self._elements = np.asarray(elements, dtype=np.int64).copy()
        self._positions = np.asarray(ref_pos, dtype=np.float64).copy()
        self._spaces = np.asarray(ref_space_uid, dtype=np.int64).copy()
        self._conformer_chains = frozenset(conformer_chains)
        self._ligand_mols = dict(ligand_mols or {})
        if self._mask.ndim != 1:
            raise ValueError("BioEmu2 atom_mask must be one-dimensional")
        n = len(self._mask)
        for name, array in (
            ("atom_chain_indices", self._chains),
            ("atom_to_token_map", self._tokens),
            ("residue_names", self._resnames),
            ("mol_types", self._types),
            ("elements", self._elements),
            ("ref_space_uid", self._spaces),
        ):
            if array.shape != (n,):
                raise ValueError(f"BioEmu2 {name} must follow the padded atom order")
        if self._names.shape != (n, 4):
            raise ValueError("BioEmu2 atom_name_codes must have shape (atoms, 4)")
        if self._positions.shape != (n, 3):
            raise ValueError("BioEmu2 ref_pos must have shape (atoms, 3)")
        if not np.isfinite(self._positions[self._mask]).all():
            raise ValueError("BioEmu2 reference positions must be finite")
        if len(set(self._chain_ids)) != len(self._chain_ids):
            raise ValueError("BioEmu2 chain IDs must be unique")
        chains = self._chains[self._mask]
        if np.any(chains < 0) or np.any(chains >= len(self._chain_ids)):
            raise ValueError("BioEmu2 atom chain indices do not match chain IDs")
        if np.any(self._tokens[self._mask] < 0):
            raise ValueError("BioEmu2 real atoms require nonnegative token indices")
        if not set(self._types[self._mask]).issubset(
            {None, "protein", "rna", "dna", "ligand"}
        ):
            raise ValueError("BioEmu2 mol_types must be normalized molecular types")
        if not self._conformer_chains.issubset(self._chain_ids):
            raise ValueError("BioEmu2 conformer flags refer to an unknown chain")
        self._elements[~self._mask] = 0
        self._positions[~self._mask] = 0
        self._spaces[~self._mask] = -1

    def iter_atoms(self) -> Iterator[AtomRecord]:
        chain_tokens: dict[str, dict[int, int]] = {}
        for index in np.flatnonzero(self._mask):
            chain = self._chain_ids[self._chains[index]]
            tokens = chain_tokens.setdefault(chain, {})
            token = int(self._tokens[index])
            if token not in tokens:
                tokens[token] = len(tokens) + 1
            yield AtomRecord(
                chain=chain,
                resid=tokens[token],
                index=int(index),
                name=decode_atom_name(self._names[index]),
                resname=self._resnames[index],
                mol_type=self._types[index],
                conformer_restraints=chain in self._conformer_chains,
            )

    def num_atoms(self) -> int:
        return len(self._mask)

    def get_elements(self) -> np.ndarray:
        return self._elements.copy()

    def get_reference_positions(self) -> np.ndarray:
        return self._positions.copy()

    def get_reference_space_uid(self) -> np.ndarray:
        return self._spaces.copy()

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        from rdkit import Chem

        for chain_index, chain in enumerate(self._chain_ids):
            indices = np.flatnonzero(
                self._mask & (self._chains == chain_index) & (self._types == "ligand")
            )
            if not len(indices):
                continue
            source = self._ligand_mols.get(chain)
            if source is None:
                raise ValueError(
                    f"BioEmu2 chain {chain}: missing source ligand chemistry"
                )
            source = Chem.RemoveAllHs(Chem.Mol(source))
            source_names = [
                atom.GetProp("atom_name") if atom.HasProp("atom_name") else None
                for atom in source.GetAtoms()
            ]
            target_names = [decode_atom_name(self._names[i]) for i in indices]
            if (
                None in source_names
                or None in target_names
                or len(set(source_names)) != len(source_names)
                or len(set(target_names)) != len(target_names)
                or not set(target_names).issubset(source_names)
            ):
                raise ValueError(f"BioEmu2 chain {chain}: ambiguous ligand atom names")
            subset = Chem.RWMol(source)
            for index in reversed(range(len(source_names))):
                if source_names[index] not in target_names:
                    subset.RemoveAtom(index)
            kept_names = [name for name in source_names if name in target_names]
            stereo = Chem.RenumberAtoms(
                subset.GetMol(), [kept_names.index(name) for name in target_names]
            )
            if [a.GetAtomicNum() for a in stereo.GetAtoms()] != self._elements[
                indices
            ].tolist():
                raise ValueError(
                    f"BioEmu2 chain {chain}: ligand element order mismatch"
                )
            stereo.RemoveAllConformers()
            coords = self._positions[indices]
            yield LigandConf(
                mol=mol_with_reference_conformer(stereo, coords),
                conf_coords=coords.copy(),
                global_indices=indices,
                conformer_restraints=chain in self._conformer_chains,
                stereo_mol=stereo,
            )
