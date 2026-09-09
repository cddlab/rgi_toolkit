"""Public termination diagnostics, strict step bounds, and neighbor-block lifetime."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from rdkit import Chem

from rgi_toolkit import AtomRecord, CGInfo, CGStatus, CombinedRestraints, LigandConf


class Adapter:
    def iter_atoms(self):
        return iter(AtomRecord(str(i), 1, i) for i in range(4))

    def num_atoms(self):
        return 4

    def get_elements(self):
        return np.array([0, 6, 0, 6])

    def iter_ligand_confs(self):
        mol = Chem.MolFromSmiles("C")
        mol.AddConformer(Chem.Conformer(1))
        return iter(
            LigandConf(
                Chem.Mol(mol),
                np.zeros((1, 3)),
                np.array([i]),
                conformer_restraints=(i == 1),
            )
            for i in (1, 3)
        )


@pytest.fixture(params=("torch", "numpy", "jax"))
def backend(request):
    jax.config.update("jax_enable_x64", True)
    return request.param


def native(points, backend):
    if backend == "torch":
        return torch.tensor(points, dtype=torch.float64)
    if backend == "jax":
        return jnp.asarray(points)
    return np.array(points, dtype=float)


def configured(*, max_iter=100, gate=None, dynamic=False, interval=2, skin=2.0):
    def quadratic(ctx):
        xyz = ctx.coords("index 1")
        return (
            ctx.sum(
                (16 if dynamic else 1) * xyz[..., 0] ** 2
                + xyz[..., 1] ** 2
                + xyz[..., 2] ** 2
            )
            / 2
        )

    config = {
        "gpu": False,
        "max_iter": max_iter,
        "custom_restraints_config": [{"fn": quadratic, **(gate or {})}],
    }
    if dynamic:
        config["conformer_restraints_config"] = {
            "relax_force_field": {"ligand": "none"},
            "vdw": {
                "mode": "intermolecular",
                "max_atom_step": 1.0,
                "neighbor_rebuild_interval": interval,
                "neighbor_skin": skin,
            },
        }
    cr = CombinedRestraints()
    cr.setup(Adapter(), config=config)
    return cr


def coords(point):
    x = np.full((4, 3), 50.0)
    x[1] = point
    return x


def apply(cr, x, backend, sigma=0.0, return_info=True):
    if backend == "jax":
        return jax.jit(cr.get_minimizer(return_info=return_info))(x, sigma)
    return cr.minimize(x, sigma=sigma, return_info=return_info)


@pytest.mark.parametrize(
    "point,max_iter,status",
    [
        ([1, 2, 3], 100, CGStatus.CONVERGED),
        ([0, 0, 0], 100, CGStatus.CONVERGED),
        ([1, 2, 3], 0, CGStatus.MAX_ITER),
        ([np.nan, 2, 3], 100, CGStatus.NONFINITE),
    ],
)
def test_public_info_and_coordinate_only_calls_agree(backend, point, max_iter, status):
    cr = configured(max_iter=max_iter)
    initial = coords(point)
    x = native(initial, backend)
    out, info = apply(cr, x, backend)
    plain = apply(cr, native(initial, backend), backend, return_info=False)
    assert isinstance(info, CGInfo)
    assert int(info.status) == status
    np.testing.assert_allclose(out, plain, rtol=0, atol=0, equal_nan=True)
    if backend != "jax":
        assert out is x
    np.testing.assert_array_equal(np.asarray(out)[[0, 2, 3]], initial[[0, 2, 3]])
    if status == CGStatus.CONVERGED:
        np.testing.assert_allclose(np.asarray(out)[1], 0, atol=1e-7)
        assert float(info.grad_norm) <= 1e-7
        assert float(info.fun) == pytest.approx(np.sum(np.asarray(out)[1] ** 2) / 2)
    else:
        np.testing.assert_array_equal(out, initial)
        assert int(info.nit) == 0
    assert int(info.nfev) == int(info.njev) >= 1
    if max_iter == 0 or point[0] == 0:
        assert int(info.nfev) == 1


def test_inactive_window_has_no_evaluations_and_reset_clears_both_factories(backend):
    cr = configured(gate={"start_sigma": 2.0, "stop_sigma": 1.0})
    initial = coords([1, 2, 3])
    for sigma, status in [
        (3.0, CGStatus.INACTIVE),
        (2.0, CGStatus.CONVERGED),
        (1.0, CGStatus.CONVERGED),
        (0.5, CGStatus.INACTIVE),
    ]:
        out, info = apply(cr, native(initial, backend), backend, sigma=sigma)
        assert int(info.status) == status
        if status == CGStatus.INACTIVE:
            np.testing.assert_array_equal(out, initial)
            assert tuple(map(float, info[1:])) == (0, 0, 0, 0, 0)
    cr.setup(Adapter(), config={})
    assert cr.get_minimizer() is None
    assert cr.get_minimizer(return_info=True) is None
    out, info = cr.minimize(native(initial, backend), return_info=True)
    np.testing.assert_array_equal(out, initial)
    assert info.status == CGStatus.INACTIVE


def test_info_is_one_record_for_the_entire_batch(backend):
    cr = configured(max_iter=0)
    points = np.stack([coords([1, 2, 3]), coords([4, 5, 6])])
    _, info = apply(cr, native(points, backend), backend)
    assert float(info.fun) == pytest.approx(np.sum(points[:, 1] ** 2) / 2)
    assert float(info.grad_norm) == 6.0
    assert np.shape(info.status) == ()
    assert int(info.nfev) == 1


def test_failure_after_accepted_step_ends_all_neighbor_blocks(backend):
    cr = configured(dynamic=True, interval=2)
    initial = coords([1, 5, 0])
    out, info = apply(cr, native(initial, backend), backend)
    # This quadratic accepts the first bounded step, but the next soft-axis
    # direction needs more than 1 A to reduce its slope to 40 percent.
    assert int(info.status) == CGStatus.LINE_SEARCH_FAILED
    assert int(info.nit) == 1
    assert int(info.nfev) == int(info.njev) == 3
    gradient = np.asarray(out)[1] * [16, 1, 1]
    previous_gradient = initial[1] * [16, 1, 1]
    beta = max(
        0,
        gradient
        @ (gradient - previous_gradient)
        / (previous_gradient @ previous_gradient),
    )
    direction = -gradient - beta * previous_gradient
    slope = gradient @ direction
    curvature = direction @ (direction * [16, 1, 1])
    assert 1 / np.linalg.norm(direction) < -0.6 * slope / curvature
    assert float(info.grad_norm) == pytest.approx(np.abs(gradient).max())
    assert float(info.grad_norm) > 1
    limited = configured(max_iter=2, dynamic=True, interval=2)
    short, short_info = apply(limited, native(initial, backend), backend)
    np.testing.assert_allclose(out, short, atol=1e-12)
    np.testing.assert_allclose(info, short_info, atol=1e-12)


def test_rebuild_preserves_aggregate_evaluation_counts(backend):
    cr = configured(dynamic=True, interval=1, skin=0.0)
    out, info = apply(cr, native(coords([1, 1, 0]), backend), backend)
    assert int(info.status) == CGStatus.CONVERGED
    assert int(info.nit) > 1
    # Every accepted move triggers a new block/list and a fresh initial f/g call.
    # Dropping the prior block's counters would report only the final block here.
    assert int(info.nfev) >= 2 * int(info.nit)
    assert int(info.nfev) == int(info.njev)
    np.testing.assert_allclose(np.asarray(out)[1], 0, atol=1e-6)


def test_jax_info_can_be_collected_inside_scan():
    jax.config.update("jax_enable_x64", True)
    cr = configured(gate={"start_sigma": 2.0, "stop_sigma": 1.0})
    minimize = cr.get_minimizer(return_info=True)

    def scan(x):
        return jax.lax.scan(
            lambda x, sigma: minimize(x, sigma), x, jnp.array([3.0, 2.0, 1.0, 0.5])
        )

    out, infos = jax.jit(scan)(jnp.asarray(coords([1, 2, 3])))
    np.testing.assert_array_equal(infos.status, [0, 1, 1, 0])
    np.testing.assert_array_equal(np.asarray(infos.nfev)[[0, 3]], [0, 0])
    assert int(infos.nit[1]) > 0 and int(infos.nit[2]) == 0
    np.testing.assert_allclose(out[1], 0, atol=1e-7)


def test_lbfgs_diagnostics_are_explicitly_unsupported():
    cr = CombinedRestraints()
    cr.setup(Adapter(), config={"method": "l-bfgs"})
    with pytest.raises(ValueError, match="only for method='cg'"):
        cr.get_minimizer(return_info=True)
    with pytest.raises(ValueError, match="only for method='cg'"):
        cr.minimize(coords([1, 2, 3]), return_info=True)
