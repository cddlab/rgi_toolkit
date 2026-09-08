"""Shared typed contact parameters for neighbour ranking and VdW energies."""

from __future__ import annotations


def prepare_chemistry(ops, chemistry, like):
    """Convert small type tables and sparse topology alongside the optimizer cache."""
    if chemistry is None:
        return None
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


def pair_parameters(ops, chemistry, source, target):
    """Gather contact distance, inverse ESD squared, and chemical eligibility."""
    first = chemistry["query_types"][source]
    second = chemistry["target_types"][target]
    codes = source * chemistry["target_types"].shape[0] + target
    one_four = _contains(ops, chemistry["one_four"], codes)
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
    valid = ~_contains(ops, chemistry["excluded"], codes)
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
