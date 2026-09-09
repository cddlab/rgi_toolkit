"""Independent SciPy/analytic oracles for the optimizer algorithms.

These objectives do not import RGI energy kernels. CPU dependencies are mandatory:
CI installs both backend extras, and missing SciPy/Torch/JAX must not skip validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import pytest
import torch
from scipy.optimize import minimize, rosen, rosen_der

from rgi_toolkit.optim._torch_cg_gpu import _cg_minimize_torch
from rgi_toolkit.optim.jax_optim import _cg_minimize
from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer
from rgi_toolkit.spec import RestraintSpec

CASES = ("isotropic", "diagonal16", "diagonal64", "coupled256", "rosenbrock", "solved")
CG_BACKENDS = ("torch", "torch_functional", "jax")
SOLVERS = [(b, "CG") for b in CG_BACKENDS] + [(b, "l-bfgs") for b in ("torch", "jax")]
MAX_ITER = 10_000


@dataclass
class Problem:
    name: str
    initial: np.ndarray
    solution: np.ndarray
    matrix: np.ndarray

    def value(self, x):
        if self.name == "rosenbrock":
            return float(rosen(x[:2]) + x[2] ** 2)
        delta = x - self.solution
        return float(0.5 * delta @ self.matrix @ delta + 0.25)

    def gradient(self, x):
        if self.name == "rosenbrock":
            return np.r_[rosen_der(x[:2]), 2 * x[2]]
        return self.matrix @ (x - self.solution)

    def energy(self, x):
        """Autodiff objective, independently expressed from the SciPy callbacks."""
        if self.name == "rosenbrock":
            return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + x[2] ** 2
        if isinstance(x, torch.Tensor):
            delta = x - x.new_tensor(self.solution)
            return 0.5 * delta @ x.new_tensor(self.matrix) @ delta + 0.25
        delta = x - jnp.asarray(self.solution)
        return 0.5 * delta @ jnp.asarray(self.matrix) @ delta + 0.25


def problem(name):
    solution = np.array([0.5, -0.25, 0.75])
    initial = solution + np.array([1.0, -1.0, 0.5])
    matrix = np.eye(3)
    if name.startswith("diagonal"):
        matrix[0, 0] = float(name.removeprefix("diagonal"))
    elif name == "coupled256":
        q, _ = np.linalg.qr(
            np.array([[1.0, 2.0, -1.0], [2.0, 1.0, 3.0], [3.0, -1.0, 2.0]])
        )
        matrix = q @ np.diag([1.0, 16.0, 256.0]) @ q.T
    elif name == "rosenbrock":
        initial, solution = np.array([-1.2, 1.0, 0.5]), np.array([1.0, 1.0, 0.0])
    elif name == "solved":
        initial = solution.copy()
    return Problem(name, initial, solution, matrix)


@pytest.fixture(scope="module", params=CASES)
def reference(request):
    p = problem(request.param)
    results = []
    for method in ("CG", "L-BFGS-B"):
        options = {"maxiter": MAX_ITER, "gtol": 1e-9}
        if method == "L-BFGS-B":
            options["ftol"] = 1e-14
        result = minimize(
            p.value, p.initial, jac=p.gradient, method=method, options=options
        )
        # Check the oracle itself, including a possible precision-loss termination.
        assert np.max(np.abs(p.gradient(result.x))) < 1e-6, result
        np.testing.assert_allclose(result.x, p.solution, rtol=0, atol=1e-6)
        results.append(result)
    return p, results


def run_cg(backend, energy, initial, max_iter=MAX_ITER, state=None, **kwargs):
    if backend == "jax":
        jax.config.update("jax_enable_x64", True)
        return jax.jit(
            lambda x, s: _cg_minimize(
                energy, x, max_iter, state=s, return_state=True, **kwargs
            )
        )(jnp.asarray(initial), state)
    active = torch.as_tensor(initial, dtype=torch.float64).detach().clone()
    if backend == "torch_functional":
        return _cg_minimize_torch(
            torch.func.grad_and_value(energy),
            active,
            max_iter,
            state=state,
            return_state=True,
            **kwargs,
        )
    optimizer = TorchRestraintOptimizer(
        RestraintSpec(n_active=1, active_sites=np.array([0]))
    )
    active.requires_grad_(True)
    result_state = optimizer._minimize_cg(
        active, lambda: energy(active), max_iter, state=state, **kwargs
    )
    return active.detach(), result_state


def as_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@pytest.mark.parametrize("backend,method", SOLVERS)
def test_solvers_match_scipy_and_analytic_solution(reference, backend, method):
    p, results = reference
    if method == "CG":
        out, _ = run_cg(backend, p.energy, p.initial)
    elif backend == "torch":
        out = torch.tensor(p.initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [out], max_iter=MAX_ITER, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = p.energy(out)
            loss.backward()
            return loss

        optimizer.step(closure)
    else:
        jax.config.update("jax_enable_x64", True)
        out = (
            jaxopt.LBFGS(
                fun=p.energy,
                maxiter=MAX_ITER,
                linesearch="backtracking",
                implicit_diff=False,
            )
            .run(jnp.asarray(p.initial))
            .params
        )
    out = as_numpy(out)
    assert np.isfinite(out).all()
    residual = float(np.max(np.abs(p.gradient(out))))
    assert residual < (1e-6 if method == "CG" else 1e-3), (
        p.name,
        backend,
        method,
        out,
        residual,
    )
    for result in results:
        assert abs(p.value(out) - result.fun) <= 1e-6 * (1 + abs(result.fun))
        np.testing.assert_allclose(out, result.x, rtol=0, atol=1e-3)
    if p.name == "solved":
        np.testing.assert_array_equal(out, p.initial)


@pytest.mark.parametrize("backend", CG_BACKENDS)
def test_small_change_restart_survives_block_boundaries(backend):
    p = problem("rosenbrock")
    # A large restart threshold enters the low-progress stretch immediately.
    whole, _ = run_cg(backend, p.energy, p.initial, 60, ftol=1e6)
    out, state = run_cg(backend, p.energy, p.initial, 1, ftol=1e6)
    assert state is not None
    assert bool(state[5]), "the restart latch must be part of resumable state"
    advance = (
        jax.jit(
            lambda x, s: _cg_minimize(
                p.energy, x, 1, state=s, ftol=1e6, return_state=True
            )
        )
        if backend == "jax"
        else lambda x, s: run_cg(backend, p.energy, x, 1, state=s, ftol=1e6)
    )
    for _ in range(59):
        out, state = advance(out, state)
    np.testing.assert_allclose(as_numpy(out), as_numpy(whole), rtol=0, atol=1e-9)


@pytest.mark.parametrize("backend", CG_BACKENDS)
def test_failed_conjugate_direction_retries_steepest_descent(backend):
    p = problem("isotropic")
    gradient = p.gradient(p.initial)
    # A huge, still descending conjugate direction exhausts two trials. Its SD retry
    # reaches the analytic solution in one trial, so a premature return is observable.
    if backend == "jax":
        jax.config.update("jax_enable_x64", True)
        g = jnp.asarray(gradient)
        state = (
            jnp.asarray(p.value(p.initial)),
            g,
            -1e6 * g,
            jnp.sum(g * g),
            jnp.asarray(1.0),
            jnp.asarray(False),
            jnp.asarray(True),
        )
    else:
        g = torch.tensor(gradient)
        state = (p.value(p.initial), g, -1e6 * g, torch.sum(g * g), 1.0, False)
    out, _ = run_cg(backend, p.energy, p.initial, 2, state=state, max_ls=2)
    np.testing.assert_allclose(as_numpy(out), p.solution, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", CG_BACKENDS)
def test_finite_energy_with_nonfinite_trial_gradient_is_rejected(backend):
    initial = np.array([1.0, 2.0, 3.0])
    sqrt = jnp.sqrt if backend == "jax" else torch.sqrt

    def energy(x):
        return sqrt(x[0])

    # The first trial reaches sqrt(0): its energy passes Armijo, but its gradient
    # is infinite. Backtracking must retain a finite-gradient accepted point.
    if backend == "jax":
        jax.config.update("jax_enable_x64", True)
        g = jnp.array([0.5, 0.0, 0.0])
        state = (
            jnp.asarray(1.0),
            g,
            -2 * g,
            jnp.sum(g * g),
            jnp.asarray(1.0),
            jnp.asarray(False),
            jnp.asarray(True),
        )
    else:
        g = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        state = (1.0, g, -2 * g, torch.sum(g * g), 1.0, False)
    out, state = run_cg(backend, energy, initial, 1, state=state)
    np.testing.assert_array_equal(as_numpy(out), [0.5, 2.0, 3.0])
    assert np.isfinite(as_numpy(state[1])).all()


@pytest.mark.parametrize("backend", CG_BACKENDS)
def test_unrepresentable_step_returns_last_point(backend):
    initial = np.full(3, 1e16)
    # The supplied derivative is finite and nonzero, but a unit update cannot move x.
    if backend == "jax":
        jax.config.update("jax_enable_x64", True)
        energy = jnp.sum
    else:
        energy = torch.sum
    out, state = run_cg(backend, energy, initial, 10)
    np.testing.assert_array_equal(as_numpy(out), initial)
    assert (state is None) if backend != "jax" else not bool(state[-1])


@pytest.mark.gpu
def test_cuda_compiled_cg_matches_scipy():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    p = problem("coupled256")
    reference = minimize(
        p.value, p.initial, jac=p.gradient, method="CG", options={"gtol": 1e-9}
    )
    compiled = torch.compile(torch.func.grad_and_value(p.energy), fullgraph=True)
    out = _cg_minimize_torch(compiled, torch.tensor(p.initial, device="cuda"), MAX_ITER)
    out = as_numpy(out)
    np.testing.assert_allclose(out, reference.x, rtol=0, atol=1e-3)
    assert np.max(np.abs(p.gradient(out))) < 1e-6
