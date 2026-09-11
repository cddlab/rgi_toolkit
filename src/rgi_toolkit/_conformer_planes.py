"""Give torsions priority over overlapping conformer planes, per local state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import chain


def _subtract(plane, torsion):
    """Subtract one condition conjunction without enumerating whole-chain states."""
    current = dict(plane)
    if any(key in current and current[key] != value for key, value in torsion):
        return [plane]
    remaining = []
    for key, value in torsion:
        if key not in current:
            remaining.append(tuple(sorted((*current.items(), (key, 1 - value)))))
            current[key] = value
    return remaining


def prefer_cistrans(planes, torsions, library):
    """Return reference planes, their conditions, and a private dictionary copy.

    Inputs contain only enabled energy rows. Topological planes in the library
    and in the caller's exclusion snapshot are deliberately not filtered.
    """
    if not planes and not library.terms["plane"]:
        return [], [], library
    by_atom = defaultdict(list)
    for atoms, conditions in chain(
        ((tuple(row[:4]), ()) for row in torsions),
        ((row.atoms, row.conditions) for row in library.terms["cistrans"]),
    ):
        atoms = frozenset(atoms)
        by_atom[min(atoms)].append((atoms, tuple(sorted(conditions))))

    def surviving(atoms, conditions):
        atoms = frozenset(atoms)
        conflicts = {
            condition
            for atom in atoms
            for quad, condition in by_atom[atom]
            if quad <= atoms
        }
        states = [tuple(sorted(conditions))]
        for condition in sorted(conflicts, key=lambda x: (len(x), x)):
            states = [rest for state in states for rest in _subtract(state, condition)]
            if not states:
                break
        return states

    kept, conditions = [], []
    for atoms in planes:
        for state in surviving(atoms, ()):
            kept.append(atoms)
            conditions.append(state)
    terms = {key: list(rows) for key, rows in library.terms.items()}
    terms["plane"] = [
        replace(row, conditions=state)
        for row in terms["plane"]
        for state in surviving(row.atoms, row.conditions)
    ]
    return kept, conditions, replace(library, terms=terms)
