"""Expose RF3's processed atom order and reference features to the shared engine."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from rgi_toolkit._biotite_adapter import biotite_get_elements, biotite_ligand_confs
from rgi_toolkit._mol_build import align_stereo_mol
from rgi_toolkit.atom_context import AtomRecord, LigandConf


class RF3Adapter:
    """Adapt plain arrays from Foundry without importing Foundry, AtomWorks or Torch.

    All arrays follow the final, filtered AtomArray, exactly as RF3's diffusion
    coordinates do. ``mol_types`` contains normalized entity types, supplied by
    the tool-side shim. ``ligand_mols`` preserves the original chemical graphs;
    named source atoms can be subset when a covalent link removes leaving atoms.
    """

    def __init__(
        self,
        atom_array,
        *,
        atom_to_token_map,
        ref_pos,
        ref_space_uid,
        mol_types,
        ligand_mols=None,
    ) -> None:
        self.atom_array = atom_array
        self._tokens = np.asarray(atom_to_token_map, dtype=np.int64)
        self._positions = np.asarray(ref_pos, dtype=np.float64)
        self._spaces = np.asarray(ref_space_uid, dtype=np.int64)
        self._mol_types = np.asarray(mol_types, dtype=object)
        self._ligand_mols = dict(ligand_mols or {})
        n_atom = len(atom_array)
        for name, value in (
            ("atom_to_token_map", self._tokens),
            ("ref_space_uid", self._spaces),
            ("mol_types", self._mol_types),
        ):
            if value.shape != (n_atom,):
                raise ValueError(f"RF3 {name} must follow the processed atom order")
        if self._positions.shape != (n_atom, 3):
            raise ValueError("RF3 ref_pos must have shape (number of atoms, 3)")
        if not np.isfinite(self._positions).all():
            raise ValueError("RF3 reference positions must be finite")
        if not set(self._mol_types).issubset({None, "protein", "dna", "rna", "ligand"}):
            raise ValueError("RF3 mol_types must contain normalized molecular types")

    def _conformer_flags(self) -> np.ndarray:
        aa = self.atom_array
        if "conformer_restraints" in aa.get_annotation_categories():
            return np.asarray(aa.conformer_restraints, dtype=bool)
        return np.zeros(len(aa), dtype=bool)

    def iter_atoms(self) -> Iterator[AtomRecord]:
        aa = self.atom_array
        flags = self._conformer_flags()
        chain_tokens: dict[str, dict[int, int]] = {}
        for index in range(len(aa)):
            chain = str(aa.chain_id[index])
            tokens = chain_tokens.setdefault(chain, {})
            token = int(self._tokens[index])
            if token not in tokens:
                tokens[token] = len(tokens) + 1
            yield AtomRecord(
                chain=chain,
                resid=tokens[token],
                index=index,
                name=str(aa.atom_name[index]).strip(),
                resname=str(aa.res_name[index]).strip(),
                mol_type=self._mol_types[index],
                conformer_restraints=bool(flags[index]),
            )

    def num_atoms(self) -> int:
        return len(self.atom_array)

    def get_elements(self) -> np.ndarray:
        return biotite_get_elements(self.atom_array, self.num_atoms())

    def get_reference_positions(self) -> np.ndarray:
        return self._positions.copy()

    def get_reference_space_uid(self) -> np.ndarray:
        return self._spaces.copy()

    def _source_in_atom_order(self, chain, target, indices):
        from rdkit import Chem

        source = self._ligand_mols.get(str(chain))
        if source is None:
            return None
        source = Chem.RemoveAllHs(Chem.Mol(source))
        target_names = [str(self.atom_array.atom_name[i]) for i in indices]
        source_names = [
            atom.GetProp("atom_name") if atom.HasProp("atom_name") else None
            for atom in source.GetAtoms()
        ]
        mapping = None
        if (
            all(name is not None for name in source_names)
            and len(set(source_names)) == len(source_names)
            and len(set(target_names)) == len(target_names)
        ):
            lookup = {name: i for i, name in enumerate(target_names)}
            if not set(target_names).issubset(source_names):
                raise ValueError(f"RF3 chain {chain}: source atom names do not match")
            subset = Chem.RWMol(source)
            for i in reversed(range(len(source_names))):
                if source_names[i] not in lookup:
                    subset.RemoveAtom(i)
            source = subset.GetMol()
            mapping = [lookup[name] for name in source_names if name in lookup]
        aligned = align_stereo_mol(source, target, source_to_target=mapping)
        if aligned is None:
            raise ValueError(f"RF3 chain {chain}: cannot map source ligand chemistry")
        return aligned

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        def post_build(chain, mol, coords, indices, elements, bonds):
            return mol, coords, self._source_in_atom_order(chain, mol, indices)

        yield from biotite_ligand_confs(
            self.atom_array,
            ligand_mask=self._mol_types == "ligand",
            chain_attr="chain_id",
            coords_all=self._positions,
            conf_rest_default=False,
            post_build=post_build,
        )
