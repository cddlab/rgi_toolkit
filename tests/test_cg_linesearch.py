"""Independent SciPy checks of interpolation, search ordering and strict failure."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.optimize._dcsrch import dcstep as scipy_dcstep
from scipy.optimize._optimize import _line_search_wolfe12

from rgi_toolkit.optim._cg import JaxCG, TorchCG, jax_cg, torch_cg
from rgi_toolkit.optim._cg_linesearch import dcstep, strong_wolfe
from rgi_toolkit.optim._cg_scalar import HostScalars, JaxScalars
from rgi_toolkit.optim.info import CGStatus


@pytest.fixture(params=("torch", "jax"))
def backend(request):
    jax.config.update("jax_enable_x64", True)
    return request.param


def solve(backend, energy, initial, max_iter=100, state=None, **kwargs):
    if backend == "torch":
        return torch_cg(
            torch.func.grad_and_value(energy),
            torch.tensor(initial),
            max_iter,
            state=state,
            **kwargs,
        )
    return jax.jit(lambda x, st: jax_cg(energy, x, max_iter, state=st, **kwargs))(
        jnp.asarray(initial), state
    )


def array(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@pytest.mark.parametrize(
    "fp,dp,bracket",
    [
        (2.0, 0.5, False),
        (0.5, 0.5, False),
        (0.5, -0.5, False),
        (0.5, -0.5, True),
        (0.5, -1.5, False),
        (0.5, -1.5, True),
    ],
)
def test_dcstep_cases_match_scipy(backend, fp, dp, bracket):
    args = (0.0, 1.0, -1.0, 2.0, 0.7, 1.0, 0.5, fp, dp, bracket, 0.0, 4.0)
    expected = scipy_dcstep(*args)
    if backend == "jax":
        args = tuple(jnp.asarray(x) for x in args)
        out = jax.jit(lambda *a: dcstep(JaxScalars(a[0]), *a))(*args)
    else:
        s = HostScalars()
        args = tuple(s.boolean(x) if isinstance(x, bool) else s.scalar(x) for x in args)
        out = dcstep(s, *args)
    np.testing.assert_allclose(np.array(out[:7]), expected[:7], rtol=1e-12, atol=1e-12)
    assert bool(out[-1]) == expected[-1]


@pytest.mark.parametrize("reject_primary", [False, True])
def test_more_thuente_then_zoom_matches_scipy(backend, reject_primary):
    # More--Thuente finds alpha=3. Rejecting it makes Wolfe2 accept alpha=2.
    def energy(x):
        return (x[0] - 3.0) ** 2

    def run(x):
        b = TorchCG(x) if backend == "torch" else JaxCG(x, energy)
        s = b.s
        if backend == "torch":
            vg = torch.func.grad_and_value(energy)
            d = torch.ones_like(x)
        else:
            raw = jax.value_and_grad(energy)

            def vg(x):
                return tuple(reversed(raw(x)))

            d = jnp.ones_like(x)
        origin = b.evaluate(vg, x, x, d, s.scalar(0), s.integer(-1))
        origin = origin._replace(alpha=s.scalar(float("nan")))

        def evaluate(alpha, old):
            return s.cond(
                alpha == old.alpha,
                lambda _: old,
                lambda _: b.evaluate(
                    vg, x + b.cast(alpha, x) * d, x, d, alpha, old.nfev
                ),
                None,
            )

        return strong_wolfe(
            s,
            evaluate,
            lambda t: (t.alpha < 2.8) if reject_primary else s.boolean(True),
            origin,
            origin.f,
            origin.slope,
            s.scalar(12),
            s.scalar(1e-100),
            s.scalar(1e100),
        )

    x = np.zeros(1)
    out = run(torch.tensor(x)) if backend == "torch" else jax.jit(run)(jnp.asarray(x))
    extra = (lambda alpha, *_: alpha < 2.8) if reject_primary else (lambda *_: True)
    expected = _line_search_wolfe12(
        lambda x: float(energy(x)),
        lambda x: 2 * (x - 3),
        x,
        np.ones(1),
        np.array([-6.0]),
        9.0,
        12.0,
        c1=1e-4,
        c2=0.4,
        amin=1e-100,
        amax=1e100,
        extra_condition=extra,
    )
    trial, success, phase = out
    assert bool(success)
    assert int(phase) == (2 if reject_primary else 1)
    assert float(trial.alpha) == pytest.approx(expected[0], abs=1e-12)
    assert float(trial.f) <= 9 - 1e-4 * 6 * float(trial.alpha)
    assert abs(float(trial.slope)) <= 0.4 * 6


@pytest.mark.parametrize("diagonal", [(1.0, 1.0, 1.0), (16.0, 1.0, 1.0)])
def test_accepted_trajectory_and_conditions_match_scipy(backend, diagonal):
    diagonal = np.array(diagonal)
    target = np.array([0.5, -0.25, 0.75])
    initial = target + np.array([1.0, -1.0, 0.5])

    def energy(x):
        if isinstance(x, torch.Tensor):
            z = x - x.new_tensor(target)
            return 0.5 * (z * z * x.new_tensor(diagonal)).sum() + 0.25
        z = x - jnp.asarray(target)
        return 0.5 * jnp.sum(z * z * jnp.asarray(diagonal)) + 0.25

    def value(x):
        return 0.5 * np.sum((x - target) ** 2 * diagonal) + 0.25

    def gradient(x):
        return (x - target) * diagonal

    expected_trace = []
    ref = minimize(
        value,
        initial,
        jac=gradient,
        method="CG",
        callback=lambda x: expected_trace.append(x.copy()),
        options={"gtol": 1e-7, "maxiter": 100},
    )
    assert np.max(np.abs(gradient(ref.x))) <= 1e-7
    x, state = solve(backend, energy, initial, 0)
    if backend == "jax":
        advance = jax.jit(lambda x, st: jax_cg(energy, x, 1, state=st))
    else:

        def advance(x, st):
            return torch_cg(torch.func.grad_and_value(energy), x, 1, state=st)

    actual = []
    for _ in range(100):
        if not bool(state.valid):
            break
        before, old_state = array(x).copy(), state
        x, state = advance(x, state)
        if int(state.info.nit) == int(old_state.info.nit):
            break
        actual.append(array(x).copy())
        d, g = array(old_state.d), array(old_state.g)
        delta = array(x) - before
        alpha = np.dot(delta, d) / np.dot(d, d)
        gn = gradient(array(x))
        assert value(array(x)) <= value(before) + 1e-4 * alpha * np.dot(g, d) + 1e-14
        assert abs(np.dot(gn, d)) <= -0.4 * np.dot(g, d) + 1e-14
        if np.max(abs(gn)) > 1e-7:
            assert np.dot(array(state.d), gn) <= -0.01 * np.dot(gn, gn) + 1e-14
    assert int(state.info.status) == CGStatus.CONVERGED
    np.testing.assert_allclose(actual, expected_trace, rtol=1e-9, atol=1e-10)


def test_cap_without_wolfe_point_stops_and_caches_trial(backend):
    initial = np.zeros(3)
    out, state = solve(
        backend, lambda x: 0.5 * ((x - 1) ** 2).sum(), initial, max_atom_step=0.1
    )
    np.testing.assert_array_equal(array(out), initial)
    assert int(state.info.status) == CGStatus.LINE_SEARCH_FAILED
    assert int(state.info.nit) == 0
    # The primary and fallback see the same capped trial; it is evaluated once.
    assert int(state.info.nfev) == int(state.info.njev) == 2


def test_wolfe2_unverified_exhausted_trial_is_not_accepted(backend):
    initial = np.zeros(3)
    out, state = solve(
        backend,
        lambda x: ((x - 100) ** 2).sum(),
        initial,
        more_maxiter=0,
        wolfe_maxiter=1,
    )
    np.testing.assert_array_equal(array(out), initial)
    assert int(state.info.status) == CGStatus.LINE_SEARCH_FAILED
    assert int(state.info.nit) == 0


def test_nonfinite_initial_gradient_is_reported(backend):
    sqrt = torch.sqrt if backend == "torch" else jnp.sqrt
    initial = np.zeros(3)
    out, state = solve(backend, lambda x: sqrt(x[0]), initial)
    np.testing.assert_array_equal(array(out), initial)
    assert int(state.info.status) == CGStatus.NONFINITE
    assert int(state.info.nfev) == 1


def test_rounding_stagnation_retains_accepted_coordinates(backend):
    initial = np.full(3, 1e16)
    out, state = solve(backend, lambda x: x.sum(), initial)
    np.testing.assert_array_equal(array(out), initial)
    assert int(state.info.status) == CGStatus.NO_PROGRESS
    assert int(state.info.nit) == 0


def test_nonfinite_trial_never_becomes_an_accepted_point(backend):
    sqrt = torch.sqrt if backend == "torch" else jnp.sqrt
    initial = np.ones(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out, state = solve(backend, lambda x: (sqrt(x[0]) - 0.5) ** 2, initial)
    assert np.isfinite(array(out)).all()
    assert np.isfinite(array(state.g)).all()
    assert float(state.info.fun) <= 0.25
    if int(state.info.status) == CGStatus.CONVERGED:
        assert float(array(out)[0]) == pytest.approx(0.25, abs=1e-6)
