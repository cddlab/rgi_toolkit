"""Accepted-cache reuse must preserve JAXopt's optimizer trajectory."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGS

from rgi_toolkit.optim._jax_lbfgs import CachedLBFGS


def test_lbfgs_reuses_accepted_cache_without_changing_algorithm():
    jax.config.update("jax_enable_x64", True)
    x0 = jnp.array([-1.2, 1.0])

    def energy(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def value_grad(x, accepted):
        value, grad = jax.value_and_grad(energy)(x)
        # The marker advances once per accepted optimizer iteration, not per trial.
        return (value, accepted + 1), grad

    settings = dict(maxiter=100, tol=1e-7, linesearch="zoom", implicit_diff=False)
    expected = LBFGS(energy, **settings).run(x0)
    actual = CachedLBFGS(value_grad, value_and_grad=True, has_aux=True, **settings).run(
        x0, jnp.int32(0)
    )
    np.testing.assert_allclose(actual.params, expected.params, rtol=1e-12, atol=1e-12)
    assert int(actual.state.iter_num) == int(expected.state.iter_num)
    assert int(actual.state.num_fun_eval) == int(expected.state.num_fun_eval)
    assert int(actual.state.aux) == int(actual.state.iter_num) + 1
