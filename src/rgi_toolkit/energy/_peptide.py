"""Bind peptide dictionary alternatives without changing the cached base spec."""

from __future__ import annotations

from rgi_toolkit import _geometry as G
from rgi_toolkit._array_ops import EPS


def bind_peptide_states(ops, positions, prepared):
    """Freeze the closest cis/trans state at this coordinate snapshot, per sample.

    Optimizers call this once before any line-search trial or VdW block. Energy-only
    callers may pass an unbound prepared spec; its current coordinates are then the
    snapshot. The returned dictionary omits metadata so binding is never repeated.
    """
    states = prepared.get("_peptide_states")
    if states is None:
        return prepared
    xyz = ops.stop_gradient(positions)
    idx = states["idx"]
    a, b, c, d = (xyz[..., idx[:, i], :] for i in range(4))
    phi = G.dihedral_points(ops, a, b, c, d)
    n1, n2 = ops.cross(b - a, c - b), ops.cross(c - b, d - c)
    valid = (ops.vdot(n1, n1) > EPS) & (ops.vdot(n2, n2) > EPS)
    cis_delta = ops.abs(G.wrap(ops, phi - states["cis"]))
    trans_delta = ops.abs(G.wrap(ops, phi - states["trans"]))
    # Strict comparison chooses trans on ties, including the degenerate fallback.
    is_cis = ops.stop_gradient(valid & (cis_delta < trans_delta - 1e-7))
    result = {k: v for k, v in prepared.items() if k != "_peptide_states"}
    for kind, condition in states["conditions"].items():
        if kind not in result:
            continue
        cond_idx = condition["idx"]
        selected = is_cis[..., ops.maximum(cond_idx, 0)]
        matches = (selected == (condition["cis"] > 0.5)) | (cond_idx < 0)
        count = ops.sum(ops.astype_like(matches, positions), axis=-1)
        keep = count == cond_idx.shape[-1]
        params = result[kind]
        result[kind] = {
            **params,
            "mask": params["mask"] * ops.astype_like(keep, positions),
        }
    return result
