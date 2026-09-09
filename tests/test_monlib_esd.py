"""Dictionary uncertainty, periodicity, and per-invocation peptide state contracts."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from rgi_toolkit._monlib_records import (
    GeometryTarget,
    NamedRestraint,
    chiral_volume_esd,
    deduplicate,
)
from rgi_toolkit._monlib_spec import append_library_arrays
from rgi_toolkit.energy import numpy_energy
from rgi_toolkit.monlib_geom import LibraryTargets, PeptideChoice
from rgi_toolkit.spec import RestraintSpec, VdwConfig


def _pack(rows, peptides=(), config=None):
    targets = LibraryTargets(peptides=list(peptides))
    targets.terms.update(rows)
    atoms = {a for rs in rows.values() for r in rs for a in r.atoms}
    atoms.update(a for p in peptides for a in p.atoms)
    active = np.asarray(sorted(atoms), dtype=np.int64)
    spec = RestraintSpec(len(active), active, conf_start_sigma=float("inf"))
    append_library_arrays(
        spec, targets, config or {}, {g: i for i, g in enumerate(active)}
    )
    return spec


def _torsion_coords(degrees):
    phi = math.radians(degrees)
    return np.array(
        [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, math.cos(phi), -math.sin(phi)]]
    )


def _energy_grad(spec, coords, backend):
    if backend == "numpy":
        return numpy_energy.total_energy(coords, numpy_energy.prepare_spec(spec)), None
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from rgi_toolkit.energy import torch_energy

        x = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
        value = torch_energy.total_energy(
            x, torch_energy.prepare_spec(spec, device=x.device, dtype=x.dtype)
        )
        (grad,) = torch.autograd.grad(value, x)
        return float(value.detach()), grad.detach().numpy()
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_toolkit.energy import jax_energy

    prepared = jax_energy.prepare_spec(spec)
    value, grad = jax.jit(
        jax.value_and_grad(lambda x: jax_energy.total_energy(x, prepared))
    )(jnp.asarray(coords))
    return float(value), np.asarray(grad)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize(
    "kind,target",
    [
        ("bond", GeometryTarget((0, 1), 1.2, 0.02)),
        ("angle", GeometryTarget((0, 1, 2), 1.8, 0.05)),
        ("chiral", GeometryTarget((0, 1, 2, 3), 0.8, 0.04, both=True)),
        ("plane", GeometryTarget((0, 1, 2, 3, 4), 0.0, 0.02)),
        ("cistrans", GeometryTarget((0, 1, 2, 3), -0.7, 0.08, period=3)),
    ],
)
def test_doubling_esd_quarters_energy_and_autodiff_gradient(backend, kind, target):
    coords = np.random.default_rng(731).normal(size=(len(target.atoms), 3))
    spec = _pack({kind: [target]})
    wider = _pack({kind: [replace(target, esd=2 * target.esd)]})
    e, grad = _energy_grad(spec, coords, backend)
    e2, grad2 = _energy_grad(wider, coords, backend)
    assert e > 1e-4
    assert e2 == pytest.approx(e / 4, rel=1e-10)
    reference, _ = _energy_grad(spec, coords, "numpy")
    assert e == pytest.approx(reference, rel=1e-9)
    if grad is not None:
        np.testing.assert_allclose(grad2, grad / 4, rtol=1e-9, atol=1e-9)
        # The plane fit minimizes this same sum, so envelope-theorem differentiation
        # permits a finite-difference check away from eigenspace degeneracy.
        numeric = np.zeros_like(coords)
        for idx in np.ndindex(coords.shape):
            plus, minus = coords.copy(), coords.copy()
            plus[idx] += 1e-6
            minus[idx] -= 1e-6
            numeric[idx] = (
                _energy_grad(spec, plus, "numpy")[0]
                - _energy_grad(spec, minus, "numpy")[0]
            ) / 2e-6
        np.testing.assert_allclose(grad, numeric, rtol=2e-5, atol=2e-5)


def test_plane_is_sum_of_squared_atom_residuals_and_slack_is_separate():
    coords = np.random.default_rng(981).normal(size=(5, 3))
    row = GeometryTarget(tuple(range(5)), 0, 0.04)
    centered = coords - coords.mean(axis=0)
    normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
    residuals = centered @ normal
    spec = _pack({"plane": [row]}, config={"plane": {"weight": 2}})
    assert spec.plane.weight[0] == pytest.approx(5 * 2 / 0.04**2)
    e, _ = _energy_grad(spec, coords, "numpy")
    assert e == pytest.approx(2 * np.square(residuals / 0.04).sum())
    slack = np.sqrt(np.mean(residuals**2)) / 2
    spec = _pack({"plane": [row]}, config={"plane": {"weight": 2, "slack": slack}})
    assert _energy_grad(spec, coords, "numpy")[0] == pytest.approx(e / 4)


@pytest.mark.parametrize("period", [0, 1, 2, 3])
def test_torsion_uses_periodic_wells_without_changing_angular_esd(period):
    n = max(period, 1)
    target = GeometryTarget(
        (0, 1, 2, 3), math.radians(30), math.radians(5), period=period
    )
    spec = _pack({"cistrans": [target]})
    for turn in range(n):
        coords = _torsion_coords(30 + 360 * turn / n + 10)
        assert _energy_grad(spec, coords, "numpy")[0] == pytest.approx((10 / 5) ** 2)


def test_chiral_volume_uncertainty_matches_independent_numerical_propagation():
    chiral = NamedRestraint((0, 1, 2, 3))
    bonds = [
        NamedRestraint((0, i), r, s)
        for i, r, s in [(1, 1.4, 0.01), (2, 1.5, 0.02), (3, 1.6, 0.03)]
    ]
    angles = [
        NamedRestraint(atoms, a, s)
        for atoms, a, s in [
            ((1, 0, 2), 105, 1.5),
            ((2, 0, 3), 109, 2.0),
            ((3, 0, 1), 113, 2.5),
        ]
    ]
    values = np.array([r.value for r in bonds + angles])
    sigmas = np.array([r.esd for r in bonds + angles])

    def determinant_volume(v):
        x, y, z = np.cos(np.deg2rad(v[3:]))
        gram = np.array([[1, x, z], [x, 1, y], [z, y, 1]])
        return np.prod(v[:3]) * np.sqrt(np.linalg.det(gram))

    derivatives = []
    for i in range(6):
        delta = np.eye(6)[i] * 1e-5
        derivatives.append(
            (determinant_volume(values + delta) - determinant_volume(values - delta))
            / 2e-5
        )
    volume, esd = chiral_volume_esd(chiral, bonds, angles)
    assert volume == pytest.approx(determinant_volume(values))
    assert esd == pytest.approx(np.linalg.norm(np.asarray(derivatives) * sigmas))


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_chiral_positive_negative_and_both(backend):
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.2]])
    for sign in (1, -1):
        spec = _pack({"chiral": [GeometryTarget((0, 1, 2, 3), sign, 0.1)]})
        good = coords.copy()
        good[-1, -1] *= sign
        assert _energy_grad(spec, good, backend)[0] == pytest.approx(4.0)
        assert _energy_grad(spec, -good, backend)[0] == pytest.approx(484.0)
    spec = _pack({"chiral": [GeometryTarget((0, 1, 2, 3), 1, 0.1, both=True)]})
    assert _energy_grad(spec, coords, backend)[0] == pytest.approx(4.0)
    assert _energy_grad(spec, -coords, backend)[0] == pytest.approx(4.0)


def _switching_spec():
    # An unconditional torsion moves every selector toward cis. The separate bond
    # must keep its STARTING state target during that entire minimization.
    return _pack(
        {
            "bond": [
                GeometryTarget((4, 5), distance, 0.1, conditions=((0, cis),))
                for cis, distance in [(0, 1), (1, 2)]
            ],
            "cistrans": [GeometryTarget((0, 1, 2, 3), 0, 0.2)],
        },
        [PeptideChoice((0, 1, 2, 3), -math.pi, 0)],
    )


def _switching_coords():
    return np.stack(
        [
            np.vstack([_torsion_coords(a), [[0, 0, 5], [1.4, 0, 5]], [[100, 100, 100]]])
            for a in (140, 40)
        ]
    )


def test_peptide_binding_is_per_sample_frozen_and_ties_choose_trans():
    spec = _switching_spec()
    base = numpy_energy.prepare_spec(spec)
    x = _switching_coords()[..., :6, :]
    bound = numpy_energy.bind_peptide_states(x, base)
    np.testing.assert_array_equal(bound["bond"]["mask"], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(base["bond"]["mask"], [1, 1])
    assert "_peptide_states" in base
    assert numpy_energy.bind_peptide_states(x[::-1], bound) is bound
    reverse = numpy_energy.bind_peptide_states(x[::-1], base)
    np.testing.assert_array_equal(reverse["bond"]["mask"], [[0, 1], [1, 0]])
    for coords in (_torsion_coords(90), np.zeros((4, 3))):
        probe = np.vstack([coords, x[0, 4:]])
        np.testing.assert_array_equal(
            numpy_energy.bind_peptide_states(probe, base)["bond"]["mask"], [1, 0]
        )
    spec.conf_start_sigma, spec.conf_stop_sigma = 10, 1
    prepared = numpy_energy.prepare_spec(spec)
    assert numpy_energy.total_energy(x, prepared, sigma=11) == 0
    assert numpy_energy.total_energy(x, prepared, sigma=0.5) == 0
    assert numpy_energy.total_energy(x, prepared, sigma=5) > 0


def test_equal_local_targets_form_a_disjoint_cover_of_peptide_states():
    rows = deduplicate(
        [
            GeometryTarget(
                (8, 9), 2 if a == b == 1 else 1, 0.1, conditions=((0, a), (1, b))
            )
            for a in (0, 1)
            for b in (0, 1)
        ]
    )
    spec = _pack(
        {"bond": rows},
        [PeptideChoice(tuple(range(i, i + 4)), -math.pi, 0) for i in (0, 4)],
    )
    base = numpy_energy.prepare_spec(spec)
    for a in (0, 1):
        for b in (0, 1):
            coords = np.vstack(
                [
                    _torsion_coords(0 if a else 180),
                    _torsion_coords(0 if b else 180),
                    [[0, 0, 5], [1.3, 0, 5]],
                ]
            )
            bound = numpy_energy.bind_peptide_states(coords, base)
            assert bound["bond"]["mask"].sum() == 1
            expected = ((1.3 - (2 if a == b == 1 else 1)) / 0.1) ** 2
            assert numpy_energy.total_energy(coords, bound) == pytest.approx(expected)


@pytest.mark.parametrize(
    "backend", ["torch", "jax", pytest.param("torch_cuda", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("method", ["CG", "l-bfgs"])
@pytest.mark.parametrize("custom", [False, True])
def test_minimize_freezes_state_across_searches_and_vdw_blocks_then_reselects(
    backend, method, custom
):
    spec = _switching_spec()
    if custom:
        from tests.test_custom import _spec_from_entries

        spec.cistrans = None
        spec.custom = _spec_from_entries(
            [
                {
                    "energy": "(dihedral(A,B,C,D) / 0.2)**2",
                    "selections": {name: f"index {i}" for i, name in enumerate("ABCD")},
                }
            ],
            n=6,
        ).custom
    spec.vdw_neighbor_rebuild_interval = 2
    # Allow Wolfe steps while testing frozen state selection across neighbor blocks.
    spec.vdw_max_atom_step = 2.0
    spec.vdw_config = VdwConfig(
        weight=1,
        ligand_local=np.arange(6),
        ligand_radii=np.ones(6),
        background_global=np.array([6]),
        background_radii=np.ones(1),
        max_neighbors=2,
    )
    coords = _switching_coords()
    if backend.startswith("torch"):
        torch = pytest.importorskip("torch")
        from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

        device = "cuda" if backend == "torch_cuda" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        opt = TorchRestraintOptimizer(spec, method=method, max_iter=200)

        def run(x):
            return (
                opt.minimize(
                    torch.tensor(x, dtype=torch.float64, device=device), sigma=5
                )
                .cpu()
                .numpy()
            )
    else:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        if method == "l-bfgs":
            pytest.importorskip("jaxopt")
        import jax.numpy as jnp

        from rgi_toolkit.optim.jax_optim import make_minimizer

        minimizer = make_minimizer(spec, method=method, max_iter=200)

        # Use the same compiled scan signature the AF3 sampling loop consumes.
        @jax.jit
        def scan_once(x):
            return jax.lax.scan(
                lambda carry, sigma: (minimizer(carry, sigma), None),
                x,
                jnp.array([5.0]),
            )[0]

        def run(x):
            return np.asarray(scan_once(jnp.asarray(x)))

    first = run(coords)
    np.testing.assert_allclose(
        np.linalg.norm(first[:, 4] - first[:, 5], axis=-1), [1, 2], atol=2e-3
    )
    phi = numpy_energy._dihedral_angle(*(first[:, i, :] for i in range(4)))
    assert np.max(np.abs(phi)) < math.radians(5)
    second = run(first)
    np.testing.assert_allclose(
        np.linalg.norm(second[:, 4] - second[:, 5], axis=-1), [2, 2], atol=2e-3
    )
    np.testing.assert_array_equal(second[:, 6], coords[:, 6])


@pytest.mark.parametrize("cuda", [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_peptide_state_never_pollutes_device_dtype_caches(cuda):
    torch = pytest.importorskip("torch")
    from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

    if cuda and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    opt = TorchRestraintOptimizer(_switching_spec(), max_iter=150)
    devices = ["cpu", "cuda", "cpu"] if cuda else ["cpu"] * 3
    for device, dtype in zip(devices, [torch.float64, torch.float32, torch.float64]):
        x = torch.tensor(_switching_coords()[..., :6, :], device=device, dtype=dtype)
        result = opt.minimize(x)
        np.testing.assert_allclose(
            torch.linalg.vector_norm(result[:, 4] - result[:, 5], dim=-1).cpu(),
            [1, 2],
            atol=2e-3,
        )
        assert opt._prepared["bond"]["mask"].shape == (2,)
        for prepared in opt._prepared_g.values():
            assert prepared["bond"]["mask"].shape == (2,)


@pytest.mark.gpu
def test_compiled_cuda_dictionary_energy_and_gradients_match_eager():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from rgi_toolkit.energy import torch_energy

    spec = _switching_spec()
    additions = LibraryTargets()
    additions.terms["plane"] = [GeometryTarget(tuple(range(6)), 0, 0.02)]
    additions.terms["chiral"] = [GeometryTarget((0, 1, 2, 3), 0.8, 0.1, both=True)]
    append_library_arrays(spec, additions, {}, {i: i for i in range(6)})
    spec.cistrans.period[:] = 3
    base = torch_energy.prepare_spec(spec, device="cuda", dtype=torch.float64)
    value_and_grad = torch.func.grad_and_value(torch_energy.total_energy, argnums=0)
    compiled = torch.compile(value_and_grad, fullgraph=True)
    for coords in (
        _switching_coords()[..., :6, :],
        _switching_coords()[::-1, :6, :].copy(),
    ):
        x = torch.tensor(coords, device="cuda", dtype=torch.float64)
        bound = torch_energy.bind_peptide_states(x, base)
        expected_grad, expected = value_and_grad(x, bound)
        grad, energy = compiled(x, bound)
        torch.testing.assert_close(energy, expected, rtol=1e-9, atol=1e-8)
        torch.testing.assert_close(grad, expected_grad, rtol=1e-8, atol=1e-7)
