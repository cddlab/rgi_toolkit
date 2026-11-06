"""Framework-free adapter for Mol-1's Protenix-shaped inference features."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

from rgi_toolkit.atom_context import AtomRecord
from rgi_toolkit.protenix.adapter import ProtenixAdapter


class Mol1Adapter(ProtenixAdapter):
    """Reuse shared chemistry while numbering the model's actual tokens per chain."""

    def __init__(self, input_feature_dict: dict) -> None:
        super().__init__(input_feature_dict)
        if self.atom_array is None:
            raise ValueError("Mol-1 restraints require the inference atom_array")
        if len(self.atom_array) > self.num_atoms():
            raise ValueError("Mol-1 atom_array exceeds the coordinate atom axis")
        self._atom_to_token = self._feature_numpy("atom_to_token_idx")

    def iter_atoms(self) -> Iterator[AtomRecord]:
        chain_tokens: dict[str, dict[int, int]] = {}
        for atom in super().iter_atoms():
            token = int(self._atom_to_token[atom.index])
            seen = chain_tokens.setdefault(atom.chain, {})
            ordinal = seen.setdefault(token, len(seen) + 1)
            yield replace(atom, resid=ordinal)
