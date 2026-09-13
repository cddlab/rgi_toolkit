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
        if not len(self.entries):
            return None
        ops = get_ops(backend)
        indices = ops.asint(ops.const_like(self.indices, like))
        weights = ops.const_like(self.weights, like)
        scales = ops.const_like(self.scales, like)
        active = enabled
        for value, start, stop in (
            (sigma, "start_sigma", "stop_sigma"),
            (step, "stop_step", "start_step"),
        ):
            if value is not None:
                upper = ops.const_like(getattr(self.terms, start)[self.entries], like)
                lower = ops.const_like(getattr(self.terms, stop)[self.entries], like)
                active = active & (value <= upper) & (value >= lower)
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
