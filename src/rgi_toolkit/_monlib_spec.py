"""Pack dictionary targets separately from reference-conformer tolerances."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from rgi_toolkit.spec import (
    AngleArrays,
    BondArrays,
    ChiralArrays,
    CisTransArrays,
    PeptideStateArrays,
    PlaneArrays,
)


def used_peptides(targets):
    return sorted(
        {i for rows in targets.terms.values() for r in rows for i, _ in r.conditions}
    )


def append_library_arrays(spec, targets, config, g2l):
    """Append effective inverse-variance weights, preserving every legacy row."""
    chosen = used_peptides(targets)
    selector_map = {g: i for i, g in enumerate(chosen)}
    conditions = {}
    for kind, rows in targets.terms.items():
        if not rows:
            continue
        block = config.get(kind) or {}
        weight = float(block.get("weight", 1.0) or 0)
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
            # N * RMS^2 = sum of per-atom squared plane residuals. User slack
            # still applies to the RMS, so the standalone plane APIs do not change.
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
        width = max(len(r.conditions) for r in rows)
        if width:
            cidx = np.full((n_old + n, width), -1, dtype=np.int64)
            cis = np.zeros((n_old + n, width))
            for row, target in enumerate(rows, n_old):
                for col, (selector, state) in enumerate(target.conditions):
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
