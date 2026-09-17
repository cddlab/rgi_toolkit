"""Complete residue-reference link angles with one consistent local direction.

Independently positioned residue references cannot supply a link vector. Fallback
ideal angles guide its direction, but measuring the resulting angles against one
local reference frame avoids incompatible bond-angle and peptide-plane targets.
Dictionary centers retain their library geometry and link modifications.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import replace

import numpy as np

from rgi_toolkit._monlib_records import GeometryTarget, merge_conditions

logger = logging.getLogger(__name__)


def _complete_angles(vectors, ideals, planar):
    """Return angles to a unit partner vector in the residue's local frame."""
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(lengths <= 1e-10):
        raise ValueError("polymer link: nonfinite or coincident reference atoms")
    directions = vectors / lengths[:, None]
    if planar:
        if len(directions) != 2:
            raise ValueError(
                "polymer link: a planar completion requires two neighbours"
            )
        theta = np.arccos(np.clip(directions[0] @ directions[1], -1.0, 1.0))
        if min(theta, np.pi - theta) <= 1e-8:
            raise ValueError(
                "polymer link: collinear reference atoms cannot define a plane"
            )
        # The exterior sector is split nearest to the two ideal angles. These
        # angles and the measured residue angle then sum to exactly 2*pi.
        available = 2 * np.pi - theta
        first = (available + ideals[0] - ideals[1]) / 2
        result = np.array([first, available - first])
        if np.any(result <= 0) or np.any(result >= np.pi):
            raise ValueError(
                "polymer link: reference angle cannot support a planar link"
            )
        return result
    # Fit the desired direction cosines to the local reference bond directions.
    # A complete tetrahedral center generally has no exact solution for a unit
    # vector when independently sourced ideal angles are imposed on its reference.
    partner, _, rank, _ = np.linalg.lstsq(directions, np.cos(ideals), rcond=None)
    norm = np.linalg.norm(partner)
    if rank < 3 and norm <= 1:
        # The missing perpendicular component completes a unit vector without
        # changing the fitted direction cosines. This also handles coplanar
        # references whose ideal angles have no exact simultaneous solution.
        return np.arccos(np.clip(directions @ partner, -1.0, 1.0))
    if not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError("polymer link: reference directions cannot define a partner")
    partner /= norm
    return np.arccos(np.clip(directions @ partner, -1.0, 1.0))


def cohere_reference_links(angles, planes, residues, coords, covered=()):
    """Adjust only cross-residue angles centered on an uncovered reference atom.

    Each angle row is ``(i, center, k, target_radians, esd_radians)``. Local
    geometry, link lengths, ESDs, topology and row order are preserved. A single
    link angle has no redundant local condition and keeps its ideal value.
    """
    owner = {atom: meta["uid"] for meta in residues for atom in meta["names"].values()}
    covered = set(covered)
    groups = defaultdict(list)
    for row, (i, center, k, target, _esd) in enumerate(angles):
        if center in covered or center not in owner:
            continue
        local = owner[center]
        if owner.get(i) == local and owner.get(k) != local:
            neighbour, partner = i, k
        elif owner.get(k) == local and owner.get(i) != local:
            neighbour, partner = k, i
        else:
            continue
        groups[center, partner].append((row, neighbour, target))
    plane_sets = [frozenset(p) for p in planes]
    result = list(angles)
    for (center, partner), group in groups.items():
        if len(group) < 2:
            continue
        neighbours = [neighbour for _row, neighbour, _target in group]
        if len(set(neighbours)) != len(neighbours):
            raise ValueError("polymer link: duplicate reference angle definitions")
        atoms = frozenset((center, partner, *neighbours))
        planar = len(group) == 2 and any(atoms <= p for p in plane_sets)
        ideals = np.array([target for _row, _neighbour, target in group])
        values = _complete_angles(coords[neighbours] - coords[center], ideals, planar)
        for (row, _neighbour, _target), value in zip(group, values, strict=True):
            i, j, k, _old, esd = result[row]
            result[row] = (i, j, k, float(value), esd)
    return result


def cohere_mixed_library_links(targets, residues, coords):
    """Complete library links where a missing residue uses reference geometry.

    Local cis/trans variants are separate objectives. Keep their conditions and
    ESDs while completing the fallback side of each variant independently.
    """
    by_condition = defaultdict(list)
    for i, row in enumerate(targets.terms["angle"]):
        if row.atoms[1] not in targets.atoms:
            by_condition[row.conditions].append((i, row))
    rows = list(targets.terms["angle"])
    for group in by_condition.values():
        angles = [(*row.atoms, row.value, row.esd) for _i, row in group]
        coherent = cohere_reference_links(
            angles, targets.plane_groups, residues, coords, targets.atoms
        )
        for (i, row), value in zip(group, coherent, strict=True):
            rows[i] = replace(row, value=value[3])
    targets.terms["angle"] = rows


def cohere_dictionary_fallback_links(angles, planes, targets):
    """Complete built-in links against covered dictionary-local angle targets.

    Only generated fallback angles are changed. Existing dictionary rows remain
    intact; their local state conditions also apply to the completed link rows.
    """
    groups = defaultdict(list)
    for row, (i, center, k, _value, _esd) in enumerate(angles):
        if center in targets.atoms:
            groups[center].append((row, i, k))
    removed = set()
    for center, group in groups.items():
        if len(group) < 2:
            continue
        shared = set(group[0][1:]).intersection(*(set(g[1:]) for g in group[1:]))
        if len(shared) != 1:
            continue
        partner = shared.pop()
        neighbours = [k if i == partner else i for _row, i, k in group]
        pairs = list(itertools.combinations(range(len(neighbours)), 2))
        candidates = []
        for a, b in pairs:
            pair = {neighbours[a], neighbours[b]}
            rows = [
                r
                for r in targets.terms["angle"]
                if r.atoms[1] == center
                and {r.atoms[0], r.atoms[2]} == pair
                and r.esd > 0
            ]
            if not rows:
                break
            candidates.append(rows)
        if len(candidates) != len(pairs):
            # Missing local angles leave no complete redundant constraint set.
            continue
        completed = []
        atoms = frozenset((center, partner, *neighbours))
        planar = len(group) == 2 and any(atoms <= set(p) for p in planes)
        for combination in itertools.product(*candidates):
            conditions = merge_conditions(*(r.conditions for r in combination))
            if conditions is None:
                continue
            gram = np.eye(len(neighbours))
            for (a, b), r in zip(pairs, combination, strict=True):
                gram[a, b] = gram[b, a] = np.cos(r.value)
            values, vectors = np.linalg.eigh(gram)
            if values.min() < -1e-8:
                logger.warning(
                    "polymer link: dictionary angles at atom %d do not define a "
                    "consistent local frame; keeping built-in fallback angles",
                    center,
                )
                completed = []
                break
            directions = vectors * np.sqrt(np.maximum(values, 0))
            directions = np.pad(directions, ((0, 0), (0, 3 - len(neighbours))))
            ideals = np.array([angles[row][3] for row, _i, _k in group])
            result = _complete_angles(directions, ideals, planar)
            for (row, _i, _k), value in zip(group, result, strict=True):
                i, j, k, _old, esd = angles[row]
                completed.append(
                    GeometryTarget((i, j, k), float(value), esd, conditions=conditions)
                )
        if completed:
            targets.terms["angle"].extend(completed)
            targets.angle_tuples.update(r.atoms for r in completed)
            removed.update(row for row, _i, _k in group)
    return [row for i, row in enumerate(angles) if i not in removed]
