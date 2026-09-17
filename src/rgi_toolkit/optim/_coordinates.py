"""Fixed symmetric coordinate maps for ill-conditioned mixed centroid objectives."""

from __future__ import annotations

import numpy as np

from rgi_toolkit._array_ops import get_ops


class CentroidCoordinates:
    """Whiten relative translations without changing the physical scalar energy.

    For a centroid-difference row w, q=w/|w| and s=max(1, 1/|w|).
    S=I+sum((s-1) q q.T) is symmetric positive definite, including overlapping
    rows. CG minimizes E(x0+S(u-x0)); its gradient is S.T grad(E). Internal
    deformations retain unit scale. A free pair preserves its centre of mass.
    """

    def __init__(self, spec):
        rows, scales, entries = [], [], []
        terms = spec.distance
        if terms is not None and spec.has_conformer():
            for i in np.flatnonzero((terms.mask > 0) & (terms.weight > 0)):
                values = {}
                for group, sign in ((1, 1.0), (2, -1.0)):
                    if terms.move_mode[i] not in (0, group):
                        continue
                    mask = getattr(terms, f"grp{group}_mask")[i]
                    indices = getattr(terms, f"grp{group}_idx")[i]
                    for index, weight in zip(indices, mask, strict=True):
                        if weight:
                            values[int(index)] = values.get(int(index), 0.0) + (
                                sign * weight / mask.sum()
                            )
                norm = np.sqrt(sum(value**2 for value in values.values()))
                if not 0 < norm < 1:
                    continue
                rows.append([(index, value / norm) for index, value in values.items()])
                scales.append(1 / norm - 1)
                entries.append(i)
        width = max((len(row) for row in rows), default=0)
        self.indices = np.zeros((len(rows), width), dtype=np.int64)
        self.weights = np.zeros_like(self.indices, dtype=float)
        for i, row in enumerate(rows):
            for j, (index, weight) in enumerate(row):
                self.indices[i, j], self.weights[i, j] = index, weight
        self.scales = np.asarray(scales)
        self.entries = np.asarray(entries, dtype=np.int64)
        self.terms = terms

    def bind(self, backend, like, sigma=None, step=None, enabled=True):
        return bind_coordinates(backend, like, self.parameters(), sigma, step, enabled)

    def parameters(self):
        """Return the fixed map and its windows as a tree of numeric arrays."""
        if not len(self.entries):
            return None
        return dict(
            indices=self.indices,
            weights=self.weights,
            scales=self.scales,
            windows=np.stack(
                [
                    getattr(self.terms, key)[self.entries]
                    for key in ("start_sigma", "stop_sigma", "start_step", "stop_step")
                ],
                axis=-1,
            ),
        )


def bind_coordinates(backend, like, parameters, sigma=None, step=None, enabled=True):
    """Bind the same affine map using host arrays or traced JAX arguments."""
    if parameters is None:
        return None
    ops = get_ops(backend)
    indices = ops.asint(ops.const_like(parameters["indices"], like))
    weights = ops.const_like(parameters["weights"], like)
    scales = ops.const_like(parameters["scales"], like)
    windows = ops.const_like(parameters["windows"], like)
    active = enabled
    if sigma is not None:
        active = active & (sigma <= windows[:, 0]) & (sigma >= windows[:, 1])
    if step is not None:
        active = active & (step >= windows[:, 2]) & (step <= windows[:, 3])
    scales = scales * active

    def transform(value, origin=None):
        delta = value if origin is None else value - origin
        projected = ops.sum(delta[..., indices, :] * weights[..., None], axis=-2)
        correction = (
            projected[..., :, None, :] * scales[:, None, None] * weights[..., None]
        ).reshape(*value.shape[:-2], -1, 3)
        if backend == "torch":
            return value.index_add(-2, indices.reshape(-1), correction)
        return value.at[..., indices.reshape(-1), :].add(correction)

    return transform
