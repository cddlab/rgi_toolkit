"""Exact dynamic VdW evaluation with reusable, bounded-memory neighbour caches."""

from __future__ import annotations

import logging
from math import prod
from typing import NamedTuple

from rgi_toolkit._array_ops import get_ops
from rgi_toolkit.energy._nonbonded import PairParameters, pair_parameters
from rgi_toolkit.optim._cg_config import EPS

logger = logging.getLogger(__name__)
_TORCH_DENSE_CVG = {}


def _torch_dense_cvg(moving):
    import torch

    from rgi_toolkit.optim._torch_cg_gpu import _COMPILE_DISABLED

    if _COMPILE_DISABLED:
        return None
    if moving not in _TORCH_DENSE_CVG:

        def energy(a, runtime, overflow, offset, order):
            return runtime._chunk_energy(a, overflow, offset, moving, order)

        _TORCH_DENSE_CVG[moving] = torch.compile(
            torch.func.grad_and_value(energy), fullgraph=True, dynamic=False
        )
    return _TORCH_DENSE_CVG[moving]


class PairCache(NamedTuple):
    reference: object
    neighbours: object
    mask: object
    overflow: object
    valid: object
    overflow_order: object
    parameters: object


class VdwRuntime:
    """Validate every trial; capacity overflow falls back to complete pair sums.

    Cache changes preserve the objective and therefore do not restart CG. Fixed
    partners remain constant for this invocation. Moving pairs use two directed
    half-weight rows, including rows evaluated by the dense fallback.
    """

    chunk_size = 256

    def __init__(
        self, backend, like, *, fixed=None, moving=None, background=None, skin=1.0
    ):
        self.backend = backend
        self.ops = get_ops(backend)
        self.xp = self.ops.t if backend == "torch" else self.ops.xp
        self.fixed, self.moving = fixed, moving
        self.background = background
        self.skin = skin
        self.mode = int(fixed is not None) + 2 * int(moving is not None)
        if backend == "torch":
            import torch

            from rgi_toolkit.optim._torch_cg_gpu import (
                _vdw_pair_energy,
                active_vdw_pair_energy,
                build_active_vdw_pairs,
                build_fixed_vdw_pairs,
            )

            self.array = lambda v: torch.as_tensor(v, device=like.device)
            self.grad_value = torch.func.grad_and_value
        else:
            import jax

            from rgi_toolkit.optim.jax_optim import (
                _active_vdw_pair_energy as active_vdw_pair_energy,
            )
            from rgi_toolkit.optim.jax_optim import (
                _build_active_vdw_pairs as build_active_vdw_pairs,
            )
            from rgi_toolkit.optim.jax_optim import (
                _build_fixed_vdw_pairs as build_fixed_vdw_pairs,
            )
            from rgi_toolkit.optim.jax_optim import (
                _vdw_pair_energy,
            )

            self.array = self.xp.asarray

            def grad_value(fun):
                vg = jax.value_and_grad(fun)

                def call(*args):
                    f, g = vg(*args)
                    return g, f

                return call

            self.grad_value = grad_value
        self.fixed_energy, self.moving_energy = _vdw_pair_energy, active_vdw_pair_energy
        self.fixed_builder, self.moving_builder = (
            build_fixed_vdw_pairs,
            build_active_vdw_pairs,
        )
        self.chunk_indices = self.array(list(range(self.chunk_size)))
        self._compiled_dense = {}

    def cond(self, predicate, yes, no, operand):
        if self.backend == "torch":
            return yes(operand) if bool(predicate) else no(operand)
        import jax

        return jax.lax.cond(predicate, yes, no, operand)

    def _reference(self, a, moving):
        return a if moving else a[..., self.fixed["lig_local"], :]

    def _empty(self, a, moving):
        v = self.moving if moving else self.fixed
        if v is None:
            return None
        ref = self._reference(a, moving)
        n_target = a.shape[-2] if moving else self.background.shape[-2]
        shape = (
            prod(a.shape[:-2]),
            ref.shape[-2],
            min(v["max_neighbors"], n_target - int(moving)),
        )
        indices = self.ops.asint(self.xp.zeros_like(ref[..., 0])).reshape(shape[:2])[
            ..., None
        ]
        indices = self.xp.broadcast_to(indices, shape)
        mask = self.ops.astype_like(indices, a)
        chemistry = v["chemistry"]
        parameters = (
            None
            if chemistry is None
            else PairParameters(
                self.ops.astype_like(mask, chemistry["contacts"]),
                self.ops.astype_like(mask, chemistry["inv_variances"]),
                mask > 0,
            )
        )
        return PairCache(
            ref,
            indices,
            mask,
            self.xp.any(mask != 0, axis=-1),
            self.array(False),
            self.ops.asint(self.xp.zeros_like(ref[..., 0])).reshape(shape[:2]),
            parameters,
        )

    def empty(self, a):
        return self._empty(a, False), self._empty(a, True)

    def _build(self, a, moving):
        v = self.moving if moving else self.fixed
        cutoff = self.xp.maximum(v["dmax"], v["contact"] + self.skin)
        if moving:
            neighbours, mask = self.moving_builder(
                a,
                v["radii"],
                v["polymer_mask"],
                v["excluded_codes"],
                cutoff,
                v["max_neighbors"] + 1,
                v["scale"],
                v["chemistry"],
                pair_factors=False,
            )
        else:
            neighbours, mask = self.fixed_builder(
                a,
                self.background,
                v["lig_local"],
                cutoff,
                v["max_neighbors"] + 1,
                v["lig_r"],
                v["bg_r"],
                v["scale"],
                v["chemistry"],
            )
        # Ranking is by contact clearance. Only pairs within their own contact
        # distance plus the Verlet skin can enter the energy before a rebuild.
        # Filtering here avoids dense fallback for harmless, more distant pairs.
        batch = a.reshape(-1, a.shape[-2], 3)
        target = (
            batch
            if moving
            else self.background.reshape(-1, self.background.shape[-2], 3)
        )
        query = batch if moving else batch[:, v["lig_local"], :]
        batch_idx = self.array(list(range(batch.shape[0]))).reshape(-1, 1, 1)
        source = self.array(list(range(query.shape[-2]))).reshape(1, -1, 1)
        diff = query[:, :, None, :] - target[batch_idx, neighbours]
        distance = self.xp.sqrt(self.ops.sum(diff * diff, axis=-1) + EPS)
        parameters = None
        if v["chemistry"] is not None:
            contact, inverse, allowed = pair_parameters(
                self.ops, v["chemistry"], source, neighbours
            )
            # Chemistry is constant until the neighbour indices change. Store it
            # beside those indices instead of searching topology at every trial.
            k = v["max_neighbors"]
            parameters = PairParameters(
                contact[..., :k], inverse[..., :k], allowed[..., :k]
            )
        elif moving:
            contact = v["radii"][source] + v["radii"][neighbours]
        else:
            contact = v["lig_r"][source] + v["bg_r"][neighbours]
        mask = mask * (distance - v["scale"] * contact <= self.skin)
        k = v["max_neighbors"]
        overflow = self.xp.any(mask[..., k:] > 0, axis=-1)
        mask = self.ops.astype_like(mask[..., :k] > 0, a) * (~overflow)[..., None]
        if moving:
            mask = mask * 0.5
        reference = self.ops.stop_gradient(self._reference(a, moving))
        if self.backend == "torch":
            reference = reference.clone()
        return PairCache(
            reference,
            neighbours[..., :k],
            mask,
            overflow,
            self.array(True),
            self.ops.asint(self.xp.argsort(self.ops.asint(~overflow), stable=True)),
            parameters,
        )

    def prepare(self, a, cache, enabled=True):
        def update(old, moving):
            if old is None:
                return None
            delta = self._reference(a, moving) - old.reference
            threshold = self.skin * (0.5 if moving else 1.0)
            stale = (~old.valid) | self.xp.any(
                self.ops.sum(delta * delta, axis=-1) > threshold**2
            )
            need = enabled & stale & self.xp.all(self.xp.isfinite(a))
            return self.cond(
                need, lambda _: self._build(a, moving), lambda _: old, None
            )

        return update(cache[0], False), update(cache[1], True)

    def args(self, cache):
        args = ()
        if self.fixed is not None:
            v, c = self.fixed, cache[0]
            args += (
                self.background,
                v["lig_local"],
                c.neighbours,
                c.mask,
                v["lig_r"],
                v["bg_r"],
                v["scale"],
                v["weight"],
                c.parameters,
            )
        if self.moving is not None:
            v, c = self.moving, cache[1]
            args += (
                c.neighbours,
                c.mask,
                v["radii"],
                v["scale"],
                v["weight"],
                c.parameters,
            )
        return args

    def sparse_energy(self, a, cache):
        args = self.args(cache)
        e = self.ops.sum(a) * 0.0
        if self.fixed is not None:
            e = e + self.fixed_energy(a, *args[:9])
            args = args[9:]
        if self.moving is not None:
            e = e + self.moving_energy(a, *args)
        return e

    def _chunk_energy(self, a, overflow, offset, moving, order):
        v = self.moving if moving else self.fixed
        n_target = a.shape[-2] if moving else self.background.shape[-2]
        if moving:
            # Pack overflow queries once at rebuild time. A chunk now visits only
            # those rows, against every target, with the same O(N * chunk) memory.
            positions = self.chunk_indices + offset
            valid = positions < n_target
            positions = self.xp.minimum(positions, self.array(n_target - 1))
            source = order[:, positions][..., None]
            batch = self.array(list(range(order.shape[0]))).reshape(-1, 1, 1)
            selected = overflow[batch, source] & valid.reshape(1, -1, 1)
            target = self.array(list(range(n_target))).reshape(1, 1, -1)
            neighbours = source * 0 + target
            mask = selected
        else:
            target = self.chunk_indices + offset
            valid = target < n_target
            target = self.xp.minimum(target, self.array(n_target - 1))
            source = self.array(list(range(v["lig_local"].shape[0]))).reshape(1, -1, 1)
            neighbours = source * 0 + target.reshape(1, 1, -1)
            neighbours = neighbours + self.ops.asint(overflow[..., None]) * 0
            mask = overflow[..., None] & valid
        if moving:
            # Self-pairs are not chemical contacts, including in overflow rows.
            mask = mask & (source != neighbours)
        if v["chemistry"] is None:
            radii = v["radii"] if moving else v["lig_r"]
            other_radii = v["radii"] if moving else v["bg_r"]
            mask = mask & (radii[source] > 0) & (other_radii[neighbours] > 0)
            if moving:
                mask = mask & (
                    v["polymer_mask"][source] | v["polymer_mask"][neighbours]
                )
                excluded = v["excluded_codes"]
                if excluded.shape[0]:
                    codes = self.xp.minimum(
                        source, neighbours
                    ) * n_target + self.xp.maximum(source, neighbours)
                    locations = self.ops.searchsorted(excluded, codes)
                    locations = self.xp.minimum(
                        locations, self.array(excluded.shape[0] - 1)
                    )
                    mask = mask & (excluded[locations] != codes)
        mask = self.ops.astype_like(mask, a)
        if moving:
            return self.moving_energy(
                a,
                neighbours,
                mask * 0.5,
                v["radii"],
                v["scale"],
                v["weight"],
                v["chemistry"],
                source=source,
            )
        return self.fixed_energy(
            a,
            self.background,
            v["lig_local"],
            neighbours,
            mask,
            v["lig_r"],
            v["bg_r"],
            v["scale"],
            v["weight"],
            v["chemistry"],
        )

    def dense_value_grad(self, a, cache, *, gradient=True):
        """Accumulate each chunk's gradient immediately; never retain a dense graph."""
        zero = self.ops.sum(a) * 0.0
        if self.backend == "jax":
            import jax

            dtype = jax.eval_shape(lambda x: self.sparse_energy(x, cache), a).dtype
            zero = self.xp.zeros((), dtype=dtype)
        initial = (self.xp.zeros_like(a), zero)
        for moving, c in ((False, cache[0]), (True, cache[1])):
            if c is None:
                continue
            n_target = a.shape[-2] if moving else self.background.shape[-2]
            vg = self.grad_value(
                lambda x, offset: self._chunk_energy(
                    x, c.overflow, offset, moving, c.overflow_order
                )
            )
            if self.backend == "torch" and a.is_cuda and gradient:
                if moving not in self._compiled_dense:
                    self._compiled_dense[moving] = _torch_dense_cvg(moving)

            def dense(total):
                def body(i, total):
                    if gradient:
                        offset = self.array(i * self.chunk_size)
                        compiled = self._compiled_dense.get(moving)
                        if compiled is not None:
                            try:
                                g, f = compiled(
                                    a, self, c.overflow, offset, c.overflow_order
                                )
                            except Exception as exc:
                                logger.warning(
                                    "compiled VdW overflow objective failed (%s); eager",
                                    exc,
                                )
                                self._compiled_dense[moving] = None
                                _TORCH_DENSE_CVG[moving] = None
                                g, f = vg(a, offset)
                        else:
                            g, f = vg(a, offset)
                    else:
                        g = self.xp.zeros_like(a)
                        f = self._chunk_energy(
                            a, c.overflow, i * self.chunk_size, moving, c.overflow_order
                        )
                    return total[0] + g, total[1] + f

                count = (
                    self.xp.max(self.ops.sum(self.ops.asint(c.overflow), axis=-1))
                    if moving
                    else n_target
                )
                n_chunks = (count + self.chunk_size - 1) // self.chunk_size
                if self.backend == "torch":
                    for i in range(int(n_chunks)):
                        total = body(i, total)
                    return total
                import jax

                return jax.lax.fori_loop(0, n_chunks, body, total)

            initial = self.cond(
                self.xp.any(c.overflow), dense, lambda total: total, initial
            )
        return initial
