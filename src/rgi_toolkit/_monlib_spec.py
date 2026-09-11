"""Pack dictionary targets separately from reference-conformer tolerances."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from rgi_toolkit._config_util import conformer_weight
from rgi_toolkit.spec import (
    AngleArrays,
    BondArrays,
    ChiralArrays,
    CisTransArrays,
    PeptideStateArrays,
    PlaneArrays,
)


def used_peptides(targets, extra_conditions=()):
    return sorted(
        {i for rows in targets.terms.values() for r in rows for i, _ in r.conditions}
        | {i for condition in extra_conditions for i, _ in condition}
    )


def append_library_arrays(spec, targets, config, g2l, *, reference_plane_conditions=()):
    """Append dictionary rows with inverse-variance weights, preserving existing rows."""
    chosen = used_peptides(targets, reference_plane_conditions)
    selector_map = {g: i for i, g in enumerate(chosen)}
    condition_rows = {"plane": list(reference_plane_conditions)}
    for kind, rows in targets.terms.items():
        if not rows:
            continue
        block = (config or {}).get(kind) or {}
        weight = conformer_weight(config, kind)
        if weight <= 0:
            continue
        n = len(rows)
        sigma = np.asarray([r.esd for r in rows])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            weights = weight / np.square(sigma)
        if not np.isfinite(weights).all():
            raise ValueError(f"monomer library {kind}: ESD produces a nonfinite weight")
        common = dict(
            slack=np.full(n, float(block.get("slack") or 0.0)),
            weight=weights,
            mask=np.ones(n),
        )
        idx = [[g2l[g] for g in r.atoms] for r in rows]
        value = np.asarray([r.value for r in rows])
        if kind == "plane":
            width = max(map(len, idx))
            padded = np.zeros((n, width), dtype=np.int64)
            group_mask = np.zeros((n, width))
            for i, atoms in enumerate(idx):
                padded[i, : len(atoms)] = atoms
                group_mask[i, : len(atoms)] = 1
            # N * RMS^2 equals the sum of per-atom squared residuals; slack stays in RMS units.
            common["weight"] = weights * group_mask.sum(axis=-1)
            array = PlaneArrays(idx=padded, grp_mask=group_mask, **common)
        else:
            idx = np.asarray(idx, dtype=np.int64)
            if kind == "bond":
                array = BondArrays(idx=idx, r0=value, half=np.zeros(n), **common)
            elif kind == "angle":
                array = AngleArrays(idx=idx, th0=value, **common)
            elif kind == "chiral":
                array = ChiralArrays(
                    idx=idx,
                    vol0=value,
                    both=np.asarray([r.both for r in rows], dtype=float),
                    **common,
                )
            else:
                array = CisTransArrays(
                    idx=idx,
                    phi0=value,
                    period=np.asarray([r.period for r in rows], dtype=np.int64),
                    **common,
                )
        old = getattr(spec, kind)
        n_old = 0 if old is None else len(old.idx)
        if old is not None:
            packed = {}
            for f in fields(array):
                a, b = getattr(old, f.name), getattr(array, f.name)
                if kind == "plane" and a.ndim == 2:
                    width = max(a.shape[-1], b.shape[-1])
                    a = np.pad(a, ((0, 0), (0, width - a.shape[-1])))
                    b = np.pad(b, ((0, 0), (0, width - b.shape[-1])))
                packed[f.name] = np.concatenate((a, b), axis=0)
            array = type(array)(**packed)
        setattr(spec, kind, array)
        prior = condition_rows.get(kind) or [()] * n_old
        condition_rows[kind] = [*prior, *(r.conditions for r in rows)]
    conditions = {}
    for kind, rows in condition_rows.items():
        width = max((len(row) for row in rows), default=0)
        if width:
            cidx = np.full((len(rows), width), -1, dtype=np.int64)
            cis = np.zeros((len(rows), width))
            for row, target in enumerate(rows):
                for col, (selector, state) in enumerate(target):
                    cidx[row, col] = selector_map[selector]
                    cis[row, col] = state
            conditions[kind] = (cidx, cis)
    if conditions:
        peptides = [targets.peptides[i] for i in chosen]
        spec.peptide_states = PeptideStateArrays(
            idx=np.asarray(
                [[g2l[g] for g in p.atoms] for p in peptides], dtype=np.int64
            ),
            trans=np.asarray([p.trans for p in peptides]),
            cis=np.asarray([p.cis for p in peptides]),
            term_conditions=conditions,
        )
