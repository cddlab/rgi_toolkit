"""Configured gradient thresholds reach the public Torch and JAX solver paths."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch

from rgi_toolkit import AtomRecord, CombinedRestraints
from rgi_toolkit.config import RestraintsConfig
from rgi_toolkit.optim.info import CGStatus

MODES = (
    {"method": "CG"},
    {"method": "CG", "line_search": "armijo"},
    {"method": "l-bfgs"},
)


@pytest.mark.parametrize(
    "value",
    [None, True, False, -1, "invalid", float("nan"), float("inf"), -float("inf")],
)
def test_invalid_gradient_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="gtol"):
        RestraintsConfig.from_dict({"gtol": value})


def test_gradient_threshold_default_and_yaml_scientific_notation():
    assert RestraintsConfig.from_dict({}).gtol == 1e-5
    assert RestraintsConfig.from_dict({"gtol": "1e-8"}).gtol == 1e-8
    assert RestraintsConfig.from_dict({"gtol": 0}).gtol == 0


def run(config, initial, atoms, backend, device="cpu", return_info=False):
    jax.config.update("jax_enable_x64", True)
    engine = CombinedRestraints()
    engine.setup(
        SimpleNamespace(iter_atoms=lambda: iter(atoms)),
        config={**config, "gpu": device == "gpu"},
    )
    if backend == "torch":
        target = "cuda" if device == "gpu" else "cpu"
        result = engine.minimize(
            torch.tensor(initial, device=target), sigma=0.0, return_info=return_info
        )
        if return_info:
            return result[0].cpu().numpy(), result[1]
        return result.cpu().numpy()
    result = jax.jit(engine.get_minimizer(return_info=return_info))(
        jax.device_put(initial, jax.devices(device)[0]), 0.0
    )
    return (np.asarray(result[0]), result[1]) if return_info else np.asarray(result)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("mode", MODES)
def test_solver_obeys_loose_and_tight_gradient_thresholds(backend, mode):
    def energy(ctx):
        x = ctx.coords("index 0")
        return ctx.sum(x[..., 0] ** 2 + 2 * x[..., 1] ** 2 + 3 * x[..., 2] ** 2)

    atoms = [AtomRecord("A", 1, 0)]
    initial = np.array([[0.1, 0.2, 0.3]])
    config = {**mode, "custom_restraints_config": [{"fn": energy}]}
    loose = run({**config, "gtol": 10.0}, initial, atoms, backend)
    tight = run({**config, "gtol": 1e-8}, initial, atoms, backend)
    if backend == "jax" and mode["method"] == "l-bfgs":
        # JAXopt always performs its first update before testing the gradient.
        assert np.linalg.norm(loose) > 1e-3
    else:
        np.testing.assert_array_equal(loose, initial)
    # L-BFGS can also stop on its independent change tolerance.
    np.testing.assert_allclose(tight, 0.0, atol=1e-4)
    np.testing.assert_array_equal(
        run(config, initial, atoms, backend),
        run({**config, "gtol": 1e-5}, initial, atoms, backend),
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_tighter_threshold_resolves_small_centroid_angle_error(backend, device):
    size = 256
    initial = np.repeat(
        np.array([[30.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 30.0, 0.0]]), size, axis=0
    )
    atoms = [AtomRecord("ABC"[i // size], 1, i) for i in range(len(initial))]
    entry = {
        f"atom_selection{i + 1}": f"chain {chain}" for i, chain in enumerate("ABC")
    }
    entry["harmonic"] = {"target_angle": 89.0}
    config = {"angle_restraints_config": [entry]}
    unchanged, info = run(config, initial, atoms, backend, device, return_info=True)
    np.testing.assert_array_equal(unchanged, initial)
    assert int(info.nit) == 0 and int(info.status) == CGStatus.CONVERGED
    config["gtol"] = 1e-8
    corrected, info = run(config, initial, atoms, backend, device, return_info=True)
    assert int(info.nit) > 0 and float(info.grad_norm) <= 1e-8
    points = corrected.reshape(3, size, 3).mean(axis=1)
    u, v = points[0] - points[1], points[2] - points[1]
    angle = np.degrees(np.arccos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))
    assert abs(angle - 89.0) < 0.01
    np.testing.assert_array_equal(corrected[size : 2 * size], initial[size : 2 * size])
    np.testing.assert_allclose(
        run(config, initial, atoms, backend, device), corrected, rtol=0.0, atol=0.0
    )
