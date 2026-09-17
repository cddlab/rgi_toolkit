"""Shared typed contact parameters for neighbour ranking and VdW energies."""

from __future__ import annotations

from typing import NamedTuple


class PairParameters(NamedTuple):
    """Contact data resolved for one fixed neighbour-index array."""

    contact: object
    inverse: object
    valid: object


def prepare_chemistry(ops, chemistry, like):
    """Convert small type tables and sparse topology alongside the optimizer cache."""
    if chemistry is None:
        return None
    import numpy as np

    # Pack each atom's short topology row on the host. Already prepared or
    # traced inputs keep their existing representation without a host transfer.
    chemistry = dict(chemistry)
    n_query = len(chemistry["query_types"])
    n_target = len(chemistry["target_types"])
    for key in ("excluded", "one_four"):
        if key + "_rows" in chemistry or not isinstance(chemistry[key], np.ndarray):
            continue
        codes = chemistry[key]
        first, second = codes // n_target, codes % n_target
        counts = np.bincount(first, minlength=n_query)
        width = int(counts.max(initial=0))
        rows = np.full((n_query, width), n_target, dtype=codes.dtype)
        offsets = np.cumsum(counts) - counts
        column = np.arange(len(codes)) - offsets[first]
        rows[first, column] = second
        chemistry[key + "_rows"] = rows
    return {
        key: ops.asint(value)
        if value.dtype.kind in "biu"
        else ops.const_like(value, like)
        for key, value in chemistry.items()
    }


def _contains(ops, sorted_values, values):
    if sorted_values.shape[0] == 0:
        return values < 0
    at = ops.searchsorted(sorted_values, values)
    at = ops.minimum(at, sorted_values.shape[0] - 1)
    return sorted_values[at] == values


def _topology_contains(ops, chemistry, key, source, target, codes):
    rows = chemistry.get(key + "_rows")
    if (
        rows is not None
        and getattr(source, "ndim", 0)
        and getattr(target, "ndim", 0) >= source.ndim
        and source.shape[-1] == 1
    ):
        return ops.contains_rows(rows[source[..., 0]], target)
    return _contains(ops, chemistry[key], codes)


def pair_parameters(ops, chemistry, source, target):
    """Resolve chemistry, or reuse parameters for the supplied neighbour array."""
    if isinstance(chemistry, PairParameters):
        return chemistry
    first = chemistry["query_types"][source]
    second = chemistry["target_types"][target]
    codes = source * chemistry["target_types"].shape[0] + target
    one_four = _topology_contains(ops, chemistry, "one_four", source, target, codes)
    contact = ops.where(
        one_four,
        chemistry["one_four_contacts"][first, second],
        chemistry["contacts"][first, second],
    )
    inverse = ops.where(
        one_four,
        chemistry["one_four_inv_variances"][first, second],
        chemistry["inv_variances"][first, second],
    )
    valid = ~_topology_contains(ops, chemistry, "excluded", source, target, codes)
    same = chemistry["query_molecules"][source] == chemistry["target_molecules"][target]
    mode = chemistry["mode"]
    valid = valid & ((mode == 0) | ((mode == 1) & same) | ((mode == 2) & ~same))
    valid = valid & (
        (chemistry["query_moving"][source] > 0)
        | (chemistry["target_moving"][target] > 0)
    )
    valid = valid & ~(
        (chemistry["query_static"][source] > 0)
        & (chemistry["target_static"][target] > 0)
    )
    return contact, inverse, valid & (contact > 0)
