"""Reuse validated neighbour caches through JAXopt's accepted auxiliary state."""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxopt import LBFGS


@dataclass(eq=False)
class CachedLBFGS(LBFGS):
    """Keep the standard algorithm and pass the last accepted cache to each search.

    The objective returns ``((value, validated_cache), gradient)``. Every trial
    still validates its displacement against that cache before evaluating energy.
    JAXopt retains the accepted trial's auxiliary value alongside its gradient.
    """

    loss_tol: float | None = None

    def _cond_fun(self, inputs):
        if self.loss_tol is None:
            return super()._cond_fun(inputs)
        _, state = inputs[0]
        return (
            jnp.isfinite(state.value)
            & jnp.isfinite(state.error)
            & (jnp.abs(state.value) > self.loss_tol)
            & ~state.failed_linesearch
        )

    def update(self, params, state, _initial_cache):
        return super().update(params, state, state.aux)
