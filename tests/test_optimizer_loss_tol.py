"""Opt-in objective targets override small-gradient and small-change stops."""

import numpy as np
import pytest

from rgi_toolkit import AtomRecord
from rgi_toolkit.config import RestraintsConfig
from rgi_toolkit.optim.info import CGStatus

from .test_optimizer_gtol import MODES, run


@pytest.mark.parametrize(
    "value", [True, False, -1, "invalid", float("nan"), float("inf")]
)
def test_invalid_loss_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="loss_tol"):
        RestraintsConfig.from_dict({"loss_tol": value})


def test_loss_threshold_is_opt_in():
    assert RestraintsConfig.from_dict({}).loss_tol is None
    assert RestraintsConfig.from_dict({"loss_tol": None}).loss_tol is None
    assert RestraintsConfig.from_dict({"loss_tol": "1e-12"}).loss_tol == 1e-12
    assert RestraintsConfig.from_dict({"loss_tol": 0}).loss_tol == 0


def quadratic(ctx):
    return ctx.sum(ctx.coords("index 0") ** 2)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_small_gradient_does_not_stop_before_zero_loss(backend, mode, device):
    atoms = [AtomRecord("A", 1, 0)]
    initial = np.array([[1e-8, 0.0, 0.0]])
    config = {**mode, "max_iter": 10, "custom_restraints_config": [{"fn": quadratic}]}
    result = run({**config, "loss_tol": 0.0}, initial, atoms, backend, device)
    assert np.linalg.norm(result) < np.linalg.norm(initial) * 1e-6
    np.testing.assert_array_equal(
        run(config, initial, atoms, backend, device),
        run({**config, "loss_tol": None}, initial, atoms, backend, device),
    )
    if mode["method"] == "CG":
        _, info = run(
            {**config, "loss_tol": 0.0},
            initial,
            atoms,
            backend,
            device,
            return_info=True,
        )
        assert int(info.nit) > 0
        assert (int(info.status) == CGStatus.CONVERGED) == (float(info.fun) == 0.0)
        assert int(info.nit) <= config["max_iter"]


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("mode", MODES)
def test_absolute_loss_threshold_and_iteration_budget(backend, mode):
    atoms = [AtomRecord("A", 1, 0)]
    initial = np.array([[0.1, 0.0, 0.0]])
    config = {**mode, "custom_restraints_config": [{"fn": quadratic}]}
    np.testing.assert_array_equal(
        run({**config, "loss_tol": 0.02}, initial, atoms, backend), initial
    )
    np.testing.assert_array_equal(
        run({**config, "loss_tol": 0.0, "max_iter": 0}, initial, atoms, backend),
        initial,
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("offset", [1.0, -1.0])
def test_nonzero_stationary_objective_is_not_reported_as_converged(backend, offset):
    def shifted(ctx):
        return quadratic(ctx) + offset

    atoms = [AtomRecord("A", 1, 0)]
    initial = np.zeros((1, 3))
    config = {
        "loss_tol": 0.0,
        "max_iter": 3,
        "custom_restraints_config": [{"fn": shifted}],
    }
    result, info = run(config, initial, atoms, backend, return_info=True)
    np.testing.assert_array_equal(result, initial)
    assert float(info.fun) == offset
    assert int(info.status) != CGStatus.CONVERGED
    assert int(info.nit) <= 3


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_loss_target_disables_armijo_relative_change_stop(backend):
    def shallow(ctx):
        return 1e-8 * ctx.sum((ctx.coords("index 0") - 1.0) ** 2)

    atoms = [AtomRecord("A", 1, 0)]
    initial = np.zeros((1, 3))
    config = {
        "method": "CG",
        "line_search": "armijo",
        "gtol": 0.0,
        "max_iter": 3,
        "custom_restraints_config": [{"fn": shallow}],
    }
    _, ordinary = run(config, initial, atoms, backend, return_info=True)
    _, targeted = run(
        {**config, "loss_tol": 0.0}, initial, atoms, backend, return_info=True
    )
    assert int(ordinary.status) == CGStatus.FUNCTION_TOLERANCE
    assert int(targeted.status) == CGStatus.MAX_ITER and int(targeted.nit) == 3
    assert float(targeted.fun) < float(ordinary.fun)
