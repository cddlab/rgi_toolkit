"""Framework-free dictionary records, link modifications, and chiral uncertainty.

Gemmi exposes ChemMod.rt in Python but not ChemMod.apply_to. Apply its add/change/
delete operations to private records so modifications never mutate shared monomers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from rgi_toolkit._atom_names import normalise_atom_name

KINDS = ("bond", "angle", "chiral", "plane", "cistrans")
_RT_FIELDS = {
    "bond": "bonds",
    "angle": "angles",
    "chiral": "chirs",
    "plane": "planes",
    "cistrans": "torsions",
}


@dataclass(frozen=True)
class NamedRestraint:
    atoms: tuple
    value: float = 0.0
    esd: float = 0.0
    period: int = 1
    sign: str = "Positive"
    label: str = ""


@dataclass(frozen=True)
class GeometryTarget:
    """One dictionary target; atoms are global until packed into RestraintSpec.

    Each condition is (peptide selector index, required state: 0 trans / 1 cis).
    Conditions are ANDed and contain at most the nearby links, never a whole-chain
    enumeration. Empty conditions mean that a target is independent of peptide state.
    """

    atoms: tuple[int, ...]
    value: float
    esd: float
    period: int = 1
    both: bool = False
    conditions: tuple[tuple[int, int], ...] = ()


def atom_ids(kind, restraint):
    if kind == "plane":
        return tuple(restraint.ids)
    if kind == "chiral":
        return (restraint.id_ctr, restraint.id1, restraint.id2, restraint.id3)
    count = {"bond": 2, "angle": 3, "cistrans": 4}[kind]
    return tuple(getattr(restraint, f"id{i}") for i in range(1, count + 1))


def named(kind, restraint, modification=False):
    atoms = tuple(
        (1 if modification else int(a.comp), normalise_atom_name(a.atom))
        for a in atom_ids(kind, restraint)
    )
    return NamedRestraint(
        atoms=atoms,
        value=float(getattr(restraint, "value", 0.0)),
        esd=float(getattr(restraint, "esd", 0.0)),
        period=int(getattr(restraint, "period", 1)),
        sign=getattr(getattr(restraint, "sign", None), "name", "Positive"),
        label=getattr(restraint, "label", ""),
    )


def read_restraints(rt):
    return {
        kind: [named(kind, r) for r in getattr(rt, field)]
        for kind, field in _RT_FIELDS.items()
    }


def canonical(kind, atoms):
    if kind == "plane":
        return tuple(sorted(set(atoms)))
    if kind == "chiral":
        # Only cyclic permutations preserve handedness. An odd permutation is a
        # different signed restraint, as in Gemmi's Restraints.find_chir.
        a, b, c = atoms[1:]
        return (atoms[0], *min((a, b, c), (b, c, a), (c, a, b)))
    return min(atoms, atoms[::-1])


def modified_restraints(rt, modifications):
    records = read_restraints(rt)
    for mod in modifications:
        # Atom deletion also removes its restraints, even without an explicit
        # _chem_mod_bond/angle/tor/chir deletion. Plane membership shrinks instead.
        deleted = {
            (1, normalise_atom_name(a.old_id))
            for a in mod.atom_mods
            if a.func in ("d", ord("d"))
        }
        if deleted:
            for kind, rows in records.items():
                records[kind] = [
                    replace(r, atoms=tuple(a for a in r.atoms if a not in deleted))
                    for r in rows
                    if kind == "plane" or not deleted.intersection(r.atoms)
                ]
        for kind, field in _RT_FIELDS.items():
            rows = records[kind]
            for raw in getattr(mod.rt, field):
                change = named(kind, raw, modification=True)
                if kind == "plane":
                    at = next(
                        (i for i, r in enumerate(rows) if r.label == change.label), None
                    )
                    for atom_id, atom in zip(raw.ids, change.atoms):
                        op = chr(atom_id.comp)
                        if at is None:
                            if op != "a":
                                continue
                            rows.append(replace(change, atoms=()))
                            at = len(rows) - 1
                        old = rows[at]
                        atoms = list(old.atoms)
                        if op == "a" and atom not in atoms:
                            atoms.append(atom)
                        elif op == "d" and atom in atoms:
                            atoms.remove(atom)
                        esd = old.esd
                        if (op == "c" or esd == 0) and not math.isnan(change.esd):
                            esd = change.esd
                        rows[at] = replace(old, atoms=tuple(atoms), esd=esd)
                    continue
                key = canonical(kind, change.atoms)
                at = next(
                    (i for i, r in enumerate(rows) if canonical(kind, r.atoms) == key),
                    None,
                )
                op = chr(raw.id1.comp)
                if op == "a" and at is None:
                    rows.append(change)
                elif op == "d" and at is not None:
                    rows.pop(at)
                elif op == "c" and at is not None:
                    old = rows[at]
                    rows[at] = replace(
                        old,
                        atoms=change.atoms if kind == "chiral" else old.atoms,
                        value=old.value if math.isnan(change.value) else change.value,
                        esd=old.esd if math.isnan(change.esd) else change.esd,
                        period=old.period if change.period == -1 else change.period,
                        sign=change.sign,
                        label=change.label or old.label,
                    )
    return records


def resolve(records, sides):
    """Resolve modeled atoms; absent hydrogens/terminal atoms are expected."""
    out = {kind: [] for kind in KINDS}
    for kind, rows in records.items():
        for row in rows:
            atoms = tuple(sides.get(side, {}).get(name) for side, name in row.atoms)
            if kind == "plane":
                atoms = tuple(sorted({i for i in atoms if i is not None}))
                if len(atoms) < 4:
                    continue
            elif any(i is None for i in atoms):
                continue
            out[kind].append(replace(row, atoms=atoms))
    return out


def merge_conditions(*conditions):
    result = {}
    for rows in conditions:
        for selector, state in rows:
            if selector in result and result[selector] != state:
                return None
            result[selector] = state
    return tuple(sorted(result.items()))


def validate_target(row, description):
    if not math.isfinite(row.esd):
        raise ValueError(
            f"monomer library {description}: nonfinite ESD for {row.atoms}"
        )
    if row.esd <= 0:
        return False
    if not math.isfinite(row.value):
        raise ValueError(
            f"monomer library {description}: nonfinite target for {row.atoms}"
        )
    return True


def chiral_volume_esd(chiral, bonds, angles):
    """Ideal scalar triple-product magnitude and propagated ESD (Servalcat).

    The six dictionary measurements are treated as independent, using first-order
    propagation through V = r1*r2*r3*sqrt(det(cosine matrix)). Angles arrive in degrees.
    Coordinate gradients are still computed exclusively by backend autodiff.
    """
    center, a, b, c = chiral.atoms
    by_bond = {canonical("bond", r.atoms): r for r in bonds}
    by_angle = {canonical("angle", r.atoms): r for r in angles}
    bs = [by_bond.get(canonical("bond", (center, x))) for x in (a, b, c)]
    ans = [
        by_angle.get(canonical("angle", (x, center, y)))
        for x, y in ((a, b), (b, c), (c, a))
    ]
    if any(r is None for r in bs + ans):
        return None
    if any(not math.isfinite(r.esd) for r in bs + ans):
        raise ValueError(f"monomer library chiral {chiral.atoms}: nonfinite input ESD")
    if any(r.esd <= 0 for r in bs + ans):
        return None
    if any(not math.isfinite(r.value) for r in bs + ans):
        raise ValueError(
            f"monomer library chiral {chiral.atoms}: nonfinite input target"
        )
    if any(r.value <= 0 for r in bs):
        return None
    lengths = [r.value for r in bs]
    cosines = [math.cos(math.radians(r.value)) for r in ans]
    x, y, z = cosines
    det = 1 + 2 * x * y * z - x * x - y * y - z * z
    if det <= 0:
        return None
    product = math.prod(lengths)
    volume = product * math.sqrt(det)
    variance = sum((volume / r.value * r.esd) ** 2 for r in bs)
    for i, r in enumerate(ans):
        other = cosines[(i + 1) % 3] * cosines[(i + 2) % 3]
        deriv = (
            product
            / math.sqrt(det)
            * (cosines[i] - other)
            * math.sin(math.radians(r.value))
        )
        variance += (deriv * math.radians(r.esd)) ** 2
    sigma = math.sqrt(variance)
    return (
        (volume, sigma)
        if math.isfinite(volume) and math.isfinite(sigma) and sigma > 0
        else None
    )


def deduplicate(targets):
    """Collapse equal targets across complementary local peptide alternatives."""
    grouped = {}
    for row in targets:
        base = replace(row, conditions=())
        grouped.setdefault(base, set()).add(frozenset(row.conditions))

    def disjoint_cover(cubes):
        if not cubes:
            return ()
        if frozenset() in cubes:
            return ((),)
        selector = min(s for cube in cubes for s, _ in cube)
        branches = []
        for state in (0, 1):
            rest = {
                frozenset((s, v) for s, v in cube if s != selector)
                for cube in cubes
                if (selector, 1 - state) not in cube
            }
            branches.append(disjoint_cover(rest))
        if branches[0] == branches[1]:
            return branches[0]
        return tuple(
            ((selector, state), *cube) for state in (0, 1) for cube in branches[state]
        )

    out = []
    for base, cubes in grouped.items():
        # A disjoint decision tree avoids scoring an overlap twice (e.g. the
        # equal-target states 00, 01, 10 must not become overlapping A=0 OR B=0).
        out.extend(replace(base, conditions=cube) for cube in disjoint_cover(cubes))
    return out
