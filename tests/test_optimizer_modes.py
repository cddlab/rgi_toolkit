"""Configured solvers and the historical Armijo stopping contract."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from rgi_toolkit import AtomRecord, CombinedRestraints
from rgi_toolkit.config import RestraintsConfig
from rgi_toolkit.optim._cg import jax_cg, torch_cg
from rgi_toolkit.optim.info import CGStatus

MODES = (
    {},
    {"method": "CG", "line_search": "armijo"},
    {"method": "CG", "line_search": "strong-wolfe"},
    {"method": "l-bfgs"},
)


@pytest.mark.parametrize("options", MODES)
def test_configured_mode(options):
    config = RestraintsConfig.from_dict(options)
    expected = (
        None
        if options.get("method") == "l-bfgs"
        else options.get("line_search", "strong-wolfe")
    )
    assert config.line_search == expected


@pytest.mark.parametrize("method", ["CG", "cg", "ncg", "nonlinear-cg", "nonlinearcg"])
def test_cg_aliases_default_to_strong_wolfe(method):
    assert RestraintsConfig.from_dict({"method": method}).line_search == "strong-wolfe"


@pytest.mark.parametrize(
    "options",
    [
        {"line_search": None},
        {"line_search": "wolfe"},
        {"line_search": 1},
        {"method": "l-bfgs", "line_search": "armijo"},
        {"method": "lbfgs", "line_search": "strong-wolfe"},
    ],
)
def test_invalid_line_search_is_rejected(options):
    with pytest.raises(ValueError, match="line_search"):
        RestraintsConfig.from_dict(options)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("options", MODES)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_public_solver_reduces_custom_objective_and_respects_gate(
    backend, options, device
):
    jax.config.update("jax_enable_x64", True)

    def energy(ctx):
        x = ctx.coords("index 1")
        return ctx.sum(8 * x[..., 0] ** 2 + x[..., 1] ** 2 + 2 * x[..., 2] ** 2)

    cr = CombinedRestraints()
    cr.setup(
        SimpleNamespace(
            iter_atoms=lambda: iter(AtomRecord("A", i + 1, i) for i in range(3))
        ),
        config={
            **options,
            "gpu": device == "gpu",
            "custom_restraints_config": [{"fn": energy, "start_sigma": 1.0}],
        },
    )
    initial = np.array([[20, 30, 40], [1, 2, 3], [40, 50, 60]], dtype=float)
    if backend == "torch":
        target = "cuda" if device == "gpu" else "cpu"

        def run(sigma):
            return (
                cr.minimize(torch.tensor(initial, device=target), sigma=sigma)
                .cpu()
                .numpy()
            )
    else:
        target = jax.devices(device)[0]
        fn = jax.jit(cr.get_minimizer())

        def run(sigma):
            return np.asarray(fn(jax.device_put(initial, target), sigma))

    np.testing.assert_array_equal(run(2.0), initial)
    out = run(0.0)
    np.testing.assert_array_equal(out[[0, 2]], initial[[0, 2]])
    np.testing.assert_allclose(out[1], 0, atol=2e-4)
    assert 8 * out[1, 0] ** 2 + out[1, 1] ** 2 + 2 * out[1, 2] ** 2 < 1e-7


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("options", MODES[1:])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_default_solver_converges_large_group_angle_gradient(backend, options, device):
    jax.config.update("jax_enable_x64", True)
    size = 512
    centers = np.array([[30.0, 0, 0], [0, 0, 0], [0, 30.0, 0]])
    initial = np.repeat(centers, size, axis=0)
    atoms = [AtomRecord("ABC"[i // size], 1, i) for i in range(len(initial))]
    restraint = {
        f"atom_selection{i + 1}": f"chain {chain}" for i, chain in enumerate("ABC")
    }
    restraint["harmonic"] = {"target_angle": 60.0}
    cr = CombinedRestraints()
    cr.setup(
        SimpleNamespace(iter_atoms=lambda: iter(atoms)),
        config={
            **options,
            "gpu": device == "gpu",
            "angle_restraints_config": [restraint],
        },
    )
    if backend == "torch":
        target = "cuda" if device == "gpu" else "cpu"
        out = cr.minimize(torch.tensor(initial, device=target)).cpu().numpy()
    else:
        out = np.asarray(
            jax.jit(cr.get_minimizer())(
                jax.device_put(initial, jax.devices(device)[0]), 0.0
            )
        )
    points = out.reshape(3, size, 3).mean(axis=1)
    first, second = points[0] - points[1], points[2] - points[1]
    cosine = first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    # Convergence bounds per-atom gradients, not the angular residual.
    assert abs(angle - 60.0) < abs(90.0 - 60.0)
    np.testing.assert_array_equal(out[size : 2 * size], initial[size : 2 * size])
    theta = np.radians(angle)
    residual = theta - np.pi / 3
    u, v = first / np.linalg.norm(first), second / np.linalg.norm(second)
    free_gradient = np.stack(
        (
            (cosine * u - v) / np.linalg.norm(first),
            (cosine * v - u) / np.linalg.norm(second),
        )
    ) * (2 * residual / (size * np.sin(theta)))
    if backend == "jax" and options["method"] == "l-bfgs":
        grad_norm = np.sqrt(size) * np.linalg.norm(free_gradient)
    else:
        grad_norm = np.max(np.abs(free_gradient))
    assert grad_norm <= 1e-5


def solve(backend, energy, initial, max_iter=100, **kwargs):
    jax.config.update("jax_enable_x64", True)
    kwargs.setdefault("line_search", "armijo")
    kwargs.setdefault("gtol", 1e-7)
    if backend == "torch":
        out, state = torch_cg(
            torch.func.grad_and_value(energy),
            torch.tensor(initial, dtype=torch.float64),
            max_iter,
            **kwargs,
        )
        return out.detach().numpy(), state.info
    out, state = jax.jit(lambda x: jax_cg(energy, x, max_iter, **kwargs))(
        jnp.asarray(initial)
    )
    return np.asarray(out), state.info


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_armijo_decrease_stop_is_distinct_from_gradient_convergence(backend):
    out, info = solve(backend, lambda x: 1e9 + 0.05 * (x**2).sum(), [1.0])
    np.testing.assert_allclose(out, [0.9], atol=1e-12)
    assert int(info.status) == CGStatus.FUNCTION_TOLERANCE
    assert int(info.nit) == 1
    assert float(info.grad_norm) == pytest.approx(0.09)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_armijo_backtracks_and_can_accept_without_wolfe_curvature(backend):
    out, info = solve(backend, lambda x: 4 * (x**2).sum(), [1.0])
    np.testing.assert_array_equal(out, [0.0])
    assert int(info.status) == CGStatus.CONVERGED
    assert int(info.nfev) == 5  # Initial point, then alpha=1, 1/2, 1/4, 1/8.
    out, info = solve(backend, lambda x: x.sum(), [0.0], max_iter=3)
    np.testing.assert_array_equal(out, [-7.0])
    assert int(info.status) == CGStatus.MAX_ITER
    assert float(info.grad_norm) == 1.0


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_failed_armijo_keeps_last_accepted_point(backend):
    out, info = solve(backend, lambda x: 4 * (x**2).sum(), [1.0], max_ls=2)
    np.testing.assert_array_equal(out, [1.0])
    assert int(info.status) == CGStatus.LINE_SEARCH_FAILED
    assert int(info.nit) == 0
    assert int(info.nfev) == 3


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_armijo_mean_objective_can_grow_beyond_a_unit_step(backend):
    initial = np.ones((1024, 3))
    out, info = solve(backend, lambda x: (x**2).mean(), initial)
    np.testing.assert_allclose(out, 0, atol=1e-4)
    assert int(info.status) == CGStatus.CONVERGED
    assert np.max(np.abs(2 * out / out.size)) <= 1e-7
    assert float(info.fun) == pytest.approx(np.mean(out**2))
    assert int(info.nit) < 100


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_implicit_cg_does_not_use_armijo_energy_change_stop(backend):
    out, info = solve(
        backend, lambda x: 1e9 + 0.05 * (x**2).sum(), [1.0], line_search=None
    )
    np.testing.assert_allclose(out, [0.0], atol=1e-6)
    assert int(info.status) == CGStatus.CONVERGED
    assert float(info.grad_norm) <= 1e-7


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_default_cg_accepts_the_scipy_gradient_tolerance(backend):
    jax.config.update("jax_enable_x64", True)

    def energy(x):
        return 1e-6 * (x**2).sum()

    if backend == "torch":
        out, state = torch_cg(
            torch.func.grad_and_value(energy), torch.ones(1, dtype=torch.float64), 100
        )
        out = out.numpy()
    else:
        out, state = jax.jit(lambda x: jax_cg(energy, x, 100))(jnp.ones(1))
        out = np.asarray(out)
    np.testing.assert_array_equal(out, [1.0])
    assert int(state.info.status) == CGStatus.CONVERGED
    assert float(state.info.grad_norm) == pytest.approx(2e-6)
    assert int(state.info.nit) == 0
