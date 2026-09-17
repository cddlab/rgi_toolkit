"""Reuse validated neighbour caches through JAXopt's accepted auxiliary state."""

from jaxopt import LBFGS


class CachedLBFGS(LBFGS):
    """Keep the standard algorithm and pass the last accepted cache to each search.

    The objective returns ``((value, validated_cache), gradient)``. Every trial
    still validates its displacement against that cache before evaluating energy.
    JAXopt retains the accepted trial's auxiliary value alongside its gradient.
    """

    def update(self, params, state, _initial_cache):
        return super().update(params, state, state.aux)
