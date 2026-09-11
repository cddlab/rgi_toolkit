"""Signed-volume restraints across selection, energy, custom and optimizer APIs."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rgi_toolkit import AtomRecord, CombinedRestraints
from rgi_toolkit import _geometry as G
from rgi_toolkit._array_ops import get_ops
from rgi_toolkit.config import RestraintsConfig
from rgi_toolkit.custom.closure import build_terms
from rgi_toolkit.custom.context import ResolveContext
from rgi_toolkit.custom.dsl import eval_formula, parse_formula
from rgi_toolkit.energy import numpy_energy
from rgi_toolkit.featurizer import build_spec
from rgi_toolkit.group_geom_restr_data import ChiralRestraintData

_CENTERS = np.array(
    [[0.0, 0.0, 0.0], [1.2, 0.1, 0.2], [0.1, 1.3, 0.3], [0.2, 0.1, -0.8]]
)
_SELECTIONS = {f"atom_selection{i}": f"chain {c}" for i, c in enumerate("ABCD", 1)}
_CUSTOM_SELECTIONS = {c: f"chain {c}" for c in "ABCD"}


def _system(sizes=(1, 1, 1, 1)):
    atoms, coords, groups = [], [], []
    for chain, center, size in zip("ABCD", _CENTERS, sizes, strict=True):
        coords.append([8.0, 9.0, 10.0])
        atoms.append(AtomRecord("Z", 1, len(atoms), name="C", mol_type="ligand"))
        indices = []
        for i in range(size):
            indices.append(len(atoms))
            atoms.append(
                AtomRecord(chain, 1, len(atoms), name=f"C{i}", mol_type="ligand")
            )
            coords.append(center + (i - (size - 1) / 2) * np.array([0.03, 0.02, 0.01]))
        groups.append(indices)
    return SimpleNamespace(iter_atoms=lambda: iter(atoms)), np.array(coords), groups


def _entry(**overrides):
    return {**_SELECTIONS, "harmonic": {"target_chiral": 0.8}, **overrides}


def _spec(adapter, entry):
    config = RestraintsConfig.from_dict({"chiral_restraints_config": [entry]})
    for data in itertools.chain(config.chiral_data, config.custom_data):
        data.resolve_sites(adapter)
    return build_spec(
        chiral_restraints=config.chiral_data, custom_restraints=config.custom_data
    )


def _volume(coords, groups):
    centers = np.stack(
        [coords[..., indices, :].mean(axis=-2) for indices in groups], axis=-2
    )
    return np.linalg.det(centers[..., 1:, :] - centers[..., :1, :])


def _penalty(volume, kind, low, high):
    if kind == "harmonic":
        residual = volume - low
    elif kind == "flat-bottomed":
        residual = volume - np.clip(volume, low, high)
    elif kind == "flat-bottomed1":
        residual = np.minimum(volume - low, 0)
    else:
        residual = np.maximum(volume - high, 0)
    return np.sum(residual**2)


_SHAPES = [
    ("harmonic", {"target_chiral": -0.6}, -0.6, 0.0),
    ("flat-bottomed", {"target_chiral1": -0.5, "target_chiral2": 0.5}, -0.5, 0.5),
    ("flat-bottomed1", {"target_chiral1": -0.5}, -0.5, 0.0),
    ("flat-bottomed2", {"target_chiral2": 0.5}, 0.0, 0.5),
]


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_volume_matches_independent_determinant_and_atom_order(order):
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)[
        list(order)
    ]
    expected = np.linalg.det(points[1:] - points[0])
    assert abs(expected) == pytest.approx(24)
    for transformed in (points, points + [3.2, -1.0, 4.5], points[:, [1, 2, 0]]):
        assert G.chiral_points(get_ops("numpy"), *transformed) == pytest.approx(
            expected
        )


@pytest.mark.parametrize("kind,params,low,high", _SHAPES)
def test_config_shapes_units_and_batched_padded_energy(kind, params, low, high):
    adapter, coords, groups = _system((1, 3, 2, 5))
    data = ChiralRestraintData()
    entry = {**_SELECTIONS, kind: params, "weight": 1.7}
    data.set_config(entry)
    assert (data.target1, data.target2) == (low, high)
    assert data.move_free == (True, True, True, True)
    spec = _spec(adapter, entry)
    assert spec.has_group_chiral() and spec.has_per_entry() and spec.is_active()
    assert not spec.has_conformer()
    assert spec.chiral is None and spec.max_start_sigma() == float("inf")
    np.testing.assert_array_equal(spec.active_sites, np.concatenate(groups))
    batch = np.stack([coords * scale for scale in (-1, 0.01, 1)])
    expected = 1.7 * _penalty(_volume(batch, groups), kind, low, high)
    breakdown = numpy_energy.energy_breakdown(
        batch[:, spec.active_sites], numpy_energy.prepare_spec(spec)
    )
    assert breakdown["group_chiral"] == pytest.approx(expected, abs=1e-9)
    assert breakdown["chiral"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"atom_selection4": None},
        {"unit": "degrees"},
        {"harmonic": {}},
        {"harmonic": {"target_chiral": float("nan")}},
        {"weight": float("inf")},
        {"flat-bottomed1": {"target_chiral1": 0}},
        {"move": []},
        {"move": 5},
        {"move": True},
        {"start_sigma": 1, "stop_sigma": 2},
        {"start_step": 2, "stop_step": 1},
        {"start_step": 1, "start_sigma": 2},
    ],
)
def test_invalid_config_fails_before_setup(overrides):
    with pytest.raises(ValueError):
        RestraintsConfig.from_dict({"chiral_restraints_config": [_entry(**overrides)]})


def test_missing_type_empty_selection_and_reversed_bounds():
    adapter, _, _ = _system()
    for entry in (
        _SELECTIONS,
        _entry(atom_selection4="chain M"),
        {**_SELECTIONS, "flat-bottomed": {"target_chiral1": 2, "target_chiral2": -2}},
    ):
        with pytest.raises(ValueError):
            _spec(adapter, entry)


@pytest.mark.parametrize(
    "move,expected",
    [
        (None, (True, True, True, True)),
        ("all", (True, True, True, True)),
        ("both", (True, True, True, True)),
        (1, (True, False, False, False)),
        ([2, 4], (False, True, False, True)),
        ("1,3", (True, False, True, False)),
    ],
)
def test_move_forms(move, expected):
    data = ChiralRestraintData()
    data.set_config(_entry(move=move))
    assert data.move_free == expected


@pytest.mark.parametrize("slack", [0.0, 0.05])
def test_single_atoms_match_conformer_energy_and_gradient(slack):
    torch = pytest.importorskip("torch")
    from rgi_toolkit.energy import torch_energy

    adapter, coords, _ = _system()
    target = -0.7
    block = (
        {"harmonic": {"target_chiral": target}}
        if not slack
        else {
            "flat-bottomed": {
                "target_chiral1": target - slack,
                "target_chiral2": target + slack,
            }
        }
    )
    spec = _spec(adapter, {**_SELECTIONS, **block})
    pos = torch.tensor(
        coords[spec.active_sites], dtype=torch.float64, requires_grad=True
    )
    group_energy = torch_energy.total_energy(
        pos, torch_energy.prepare_spec(spec, dtype=pos.dtype)
    )
    conformer_energy = torch_energy.chiral_energy(
        pos,
        torch.arange(4).reshape(1, 4),
        torch.tensor([target]),
        slack,
        torch.ones(1),
        torch.ones(1),
    )
    assert group_energy.item() == pytest.approx(conformer_energy.item(), rel=1e-6)
    actual = torch.autograd.grad(group_energy, pos)[0]
    expected = torch.autograd.grad(conformer_energy, pos)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_conformer_both_signs_still_accepts_absolute_volume():
    for sign in (-1, 1):
        coords = sign * _CENTERS
        vol = abs(np.linalg.det(coords[1:] - coords[0]))
        energy = numpy_energy.chiral_energy(
            coords,
            np.arange(4).reshape(1, 4),
            np.array([vol]),
            np.array([0.05]),
            np.ones(1),
            np.ones(1),
            both=np.ones(1),
        )
        assert energy == pytest.approx(0)


def _custom_spec(adapter, **options):
    data = RestraintsConfig.from_dict(
        {
            "custom_restraints_config": [
                {
                    "selections": _CUSTOM_SELECTIONS,
                    **options,
                }
            ]
        }
    ).custom_data[0]
    data.resolve_sites(adapter)
    return build_spec(custom_restraints=[data])


def _custom_fn(ctx):
    return ctx.harmonic(ctx.chiral("A", "B", "C", "D"), 0.8)


@pytest.mark.parametrize("move", ["all", [2, 4]])
def test_backend_gradients_and_custom_mean_derivatives(move):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_toolkit.energy import jax_energy, torch_energy

    sizes = (1, 3, 2, 5)
    adapter, coords, groups = _system(sizes)
    spec = _spec(adapter, _entry(move=move))
    pos = np.stack([coords[spec.active_sites], coords[spec.active_sites] * 1.1])
    tensor = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    e_torch = torch_energy.total_energy(
        tensor, torch_energy.prepare_spec(spec, dtype=tensor.dtype)
    )
    grad_torch = torch.autograd.grad(e_torch, tensor)[0].numpy()
    prepared = jax_energy.prepare_spec(spec)
    e_jax, grad_jax = jax.jit(
        jax.value_and_grad(lambda x: jax_energy.total_energy(x, prepared))
    )(jnp.asarray(pos))
    expected = _penalty(
        _volume(np.stack([coords, coords * 1.1]), groups), "harmonic", 0.8, 0
    )
    assert float(e_torch.detach()) == pytest.approx(expected, rel=1e-9)
    assert float(e_jax) == pytest.approx(expected, rel=1e-9)
    np.testing.assert_allclose(grad_torch, grad_jax, rtol=1e-9, atol=1e-9)
    names = "all" if move == "all" else ["ABCD"[i - 1] for i in move]
    custom = _custom_spec(adapter, energy="harmonic(chiral(A,B,C,D), 0.8)", move=names)
    code = _custom_spec(adapter, fn=_custom_fn, move=names)
    formula_torch = build_terms(custom.custom, "torch")[0][-1]
    formula_jax = build_terms(custom.custom, "jax")[0][-1]
    code_torch = build_terms(code.custom, "torch")[0][-1]
    e_custom = formula_torch(tensor)
    grad_custom = torch.autograd.grad(e_custom, tensor)[0].numpy()
    np.testing.assert_allclose(
        torch.autograd.grad(code_torch(tensor), tensor)[0], grad_custom, rtol=1e-10
    )
    np.testing.assert_allclose(
        jax.grad(formula_jax)(jnp.asarray(pos)), grad_custom, rtol=1e-9, atol=1e-9
    )
    assert e_custom.item() == pytest.approx(expected, rel=1e-9)
    local_groups = [np.searchsorted(spec.active_sites, indices) for indices in groups]
    for i, indices in enumerate(local_groups):
        np.testing.assert_allclose(
            grad_torch[:, indices],
            grad_custom[:, indices] * sizes[i],
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            grad_torch[:, indices],
            np.repeat(grad_torch[:, indices[:1]], len(indices), axis=1),
        )
        if move != "all" and i + 1 not in move:
            assert np.count_nonzero(grad_torch[:, indices]) == 0
    if move == "all":
        numpy_fn = build_terms(custom.custom, "numpy")[0][-1]
        eps = 1e-5
        fd = np.empty_like(pos)
        for index in np.ndindex(pos.shape):
            plus, minus = pos.copy(), pos.copy()
            plus[index] += eps
            minus[index] -= eps
            fd[index] = (numpy_fn(plus) - numpy_fn(minus)) / (2 * eps)
        np.testing.assert_allclose(grad_custom, fd, rtol=1e-6, atol=1e-7)


def test_custom_resolution_records_all_four_selections():
    context = ResolveContext()
    eval_formula(parse_formula("harmonic(chiral(A,B,C,D), 0.8)"), context)
    assert context.selections == list("ABCD")
    code_context = ResolveContext()
    _custom_fn(code_context)
    assert code_context.selections == list("ABCD")


@pytest.mark.parametrize("degenerate", ["plane", "coincident"])
def test_degenerate_volume_and_gradients_are_finite(degenerate):
    torch = pytest.importorskip("torch")
    coords = _CENTERS.copy()
    if degenerate == "plane":
        coords[:, 2] = 0
    else:
        coords[:] = 0
    tensor = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
    volume = G.chiral_points(get_ops("torch"), *tensor)
    assert volume.item() == 0
    gradient = torch.autograd.grad((volume - 1) ** 2, tensor)[0]
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize(
    "window,points",
    [
        (
            {"start_sigma": 2, "stop_sigma": 1},
            [(3, 0, False), (2, 0, True), (1, 0, True), (0.5, 0, False)],
        ),
        (
            {"start_step": 2, "stop_step": 3},
            [(1, 1, False), (1, 2, True), (1, 3, True), (1, 4, False)],
        ),
    ],
)
def test_windows_and_gpu_pregate(window, points):
    torch = pytest.importorskip("torch")
    from rgi_toolkit.energy import torch_energy
    from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

    adapter, coords, _ = _system()
    spec = _spec(adapter, _entry(**window))
    optimizer = TorchRestraintOptimizer(spec)
    pos = torch.tensor(coords[spec.active_sites], dtype=torch.float64)
    optimizer._ensure(pos.device, pos.dtype)
    for sigma, step, on in points:
        prepared = optimizer._gated_prepared(sigma, step)
        assert bool(prepared["group_chiral"]["mask"].sum()) == on
        direct = torch_energy.total_energy(
            pos, optimizer._prepared, sigma=sigma, step=step
        )
        pregated = torch_energy.total_energy(pos, prepared)
        assert float(direct) == pytest.approx(float(pregated))
        assert (float(direct) > 0) == on


@pytest.mark.parametrize(
    "backend,precision",
    [
        ("torch", "float64"),
        ("jax", "float64"),
        pytest.param("torch_cuda", "float64", marks=pytest.mark.gpu),
        pytest.param("jax_cuda", "float64", marks=pytest.mark.gpu),
        pytest.param("torch_cuda", "float32", marks=pytest.mark.gpu),
        pytest.param("jax_cuda", "float32", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("kind", ["builtin", "formula", "code", "reference"])
def test_public_setup_minimize_finalize(
    backend, precision, kind, tmp_path, capsys, monkeypatch
):
    adapter, coords, groups = _system((1, 3, 2, 5))
    window = {"start_step": 1, "stop_step": 2}
    entry = _entry(**window)
    if kind == "reference":
        reference = tmp_path / "center.pdb"
        reference.write_text(
            "HETATM    1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        entry.update(
            atom_selection1="ref1 and chain A",
            refs={"ref1": {"ref_pdb": str(reference)}},
        )
    cuda = backend.endswith("_cuda")
    config = {"verbose": True, "gpu": cuda, "max_iter": 300}
    if kind in ("builtin", "reference"):
        config["chiral_restraints_config"] = [entry]
    else:
        definition = (
            {"energy": "harmonic(chiral(A,B,C,D), 0.8)"}
            if kind == "formula"
            else {"fn": _custom_fn}
        )
        config["custom_restraints_config"] = [
            {"selections": _CUSTOM_SELECTIONS, **definition, **window}
        ]
    restraint = CombinedRestraints()
    restraint.setup(adapter, config=config)
    assert restraint.spec.is_active() and not restraint.spec.has_conformer()
    if kind == "builtin":
        assert "n_group_chiral=1" in capsys.readouterr().out
    elif kind == "reference":
        assert "ref_chiral=1" in capsys.readouterr().out
    batch = np.stack([coords, coords * 1.1]).astype(precision)
    if backend.startswith("torch"):
        torch = pytest.importorskip("torch")
        if cuda:
            if not torch.cuda.is_available():
                pytest.skip("no CUDA device")
            from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

            def reject_eager(*args, **kwargs):
                pytest.fail("CUDA chiral optimization fell back to eager CG")

            monkeypatch.setattr(TorchRestraintOptimizer, "_minimize_cg", reject_eager)
        value = torch.tensor(
            batch, dtype=getattr(torch, precision), device="cuda" if cuda else "cpu"
        )
        np.testing.assert_array_equal(
            restraint.minimize(value, istep=0, sigma=1).cpu().numpy(), batch
        )
        result = restraint.minimize(value, istep=1, sigma=1)
        after = result.cpu().numpy().copy()
        np.testing.assert_array_equal(
            restraint.minimize(result, istep=3, sigma=1).cpu().numpy(), after
        )
        if cuda and kind != "builtin":
            assert restraint._optimizer._custom_cvg
            assert all(
                fn is not False for fn in restraint._optimizer._custom_cvg.values()
            )
        elif cuda:
            from rgi_toolkit.optim import _torch_cg_gpu

            assert 0 in _torch_cg_gpu._CVG_BY_MODE
            assert not _torch_cg_gpu._compile_failed[0]
    else:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", precision == "float64")
        import jax.numpy as jnp

        if cuda and not any(device.platform == "gpu" for device in jax.devices()):
            pytest.skip("no JAX GPU device")
        minimize = restraint.get_minimizer()
        value = jnp.asarray(batch)
        if cuda:
            assert all(device.platform == "gpu" for device in value.devices())
        np.testing.assert_array_equal(minimize(value, 1.0, step=0), batch)

        def scan(carry, step):
            updated = minimize(carry, 1.0, step=step)
            return updated, updated

        result, _ = jax.jit(lambda x: jax.lax.scan(scan, x, jnp.array([1, 2])))(value)
        after = np.asarray(result)
        np.testing.assert_array_equal(minimize(result, 1.0, step=3), after)
    spectator = sorted(set(range(len(coords))) - set(restraint.spec.active_sites))
    np.testing.assert_array_equal(after[:, spectator], batch[:, spectator])
    np.testing.assert_allclose(_volume(after, groups), 0.8, atol=2e-4)
    for indices in groups:
        before_shape = batch[:, indices] - batch[:, indices[:1]]
        after_shape = after[:, indices] - after[:, indices[:1]]
        np.testing.assert_allclose(
            after_shape, before_shape, atol=1e-6 if precision == "float32" else 1e-9
        )
    restraint.finalize(result, istep=2)
    assert "group_chiral=0.00000" in capsys.readouterr().out


def test_reference_movement_and_native_units():
    entry = _entry(
        atom_selection1="ref1 and chain A", refs={"ref1": {"ref_cif": "center.cif"}}
    )
    config = RestraintsConfig.from_dict({"chiral_restraints_config": [entry]})
    reference = config.custom_data[0]
    assert reference.geom == "chiral" and reference.n_groups == 4
    assert reference.target1 == 0.8 and reference.move_free == (False, True, True, True)
    for extra in ({"move": 1}, {"unit": "radians"}):
        with pytest.raises(ValueError):
            RestraintsConfig.from_dict(
                {"chiral_restraints_config": [{**entry, **extra}]}
            )


@pytest.mark.parametrize("extension", ["pdb", "cif"])
def test_external_fitted_references_match_custom_formula(tmp_path, extension):
    gemmi = pytest.importorskip("gemmi")
    adapter, coords, groups = _system()
    refs, selections = {}, dict(_SELECTIONS)
    fit = "chain A or chain B or chain C"
    for i, chain in enumerate("ABC", 1):
        offset = np.array([i * 4, -i * 2, i])
        pdb = (
            "".join(
                f"HETATM{j:5d}  C0  LIG {name}   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                for j, (name, (x, y, z)) in enumerate(zip("ABCD", _CENTERS + offset), 1)
            )
            + "END\n"
        )
        path = tmp_path / f"ref{i}.{extension}"
        if extension == "pdb":
            path.write_text(pdb)
        else:
            gemmi.read_pdb_string(pdb).make_mmcif_document().write_file(str(path))
        refs[f"ref{i}"] = {
            f"ref_{extension}": path.name,
            "atom_selection_ref_fit": fit,
            "atom_selection_target_fit": fit,
            "pairing": "identity",
        }
        selections[f"atom_selection{i}"] = f"ref{i} and chain {chain}"
    entry = {**selections, "refs": refs, "harmonic": {"target_chiral": 0.8}}
    (tmp_path / "chiral.json").write_text(json.dumps([entry]))
    parsed = RestraintsConfig.from_dict(
        {"chiral_restraints_config": {"config_path": "chiral.json"}}, base_dir=tmp_path
    )
    data = parsed.custom_data[0]
    assert data.move_free == (False, False, False, True)
    data.resolve_sites(adapter)
    spec = build_spec(custom_restraints=[data])
    custom = _custom_spec(
        adapter,
        selections={
            c: selections[f"atom_selection{i}"] for i, c in enumerate("ABCD", 1)
        },
        refs={
            name: {
                **value,
                f"ref_{extension}": str(tmp_path / value[f"ref_{extension}"]),
            }
            for name, value in refs.items()
        },
        energy="harmonic(chiral(A,B,C,D), 0.8)",
    )
    expected = (_volume(coords, groups) - 0.8) ** 2
    for result in (spec, custom):
        closure = build_terms(result.custom, "numpy")[0][-1]
        assert closure(coords[result.active_sites]) == pytest.approx(expected, abs=1e-8)
    torch = pytest.importorskip("torch")
    gradients = []
    for result in (spec, custom):
        tensor = torch.tensor(
            coords[result.active_sites], dtype=torch.float64, requires_grad=True
        )
        closure = build_terms(result.custom, "torch")[0][-1]
        gradient = torch.autograd.grad(closure(tensor), tensor)[0].numpy()
        for indices in groups[:3]:
            assert not np.any(gradient[np.searchsorted(result.active_sites, indices)])
        gradients.append(gradient)
    np.testing.assert_allclose(*gradients, atol=1e-8)


@pytest.mark.parametrize("route", ["registered", "add_custom"])
def test_python_registration_and_add_custom(route, monkeypatch):
    from rgi_toolkit.custom import registry

    adapter, coords, groups = _system()
    restraint = CombinedRestraints()
    if route == "registered":
        monkeypatch.setattr(registry, "_CUSTOM_FNS", {})
        registry.custom_restraint("chiral_test")(_custom_fn)
        config = {
            "custom_restraints_config": [
                {"use": "chiral_test", "selections": _CUSTOM_SELECTIONS}
            ]
        }
    else:
        restraint.add_custom(fn=_custom_fn, selections=_CUSTOM_SELECTIONS)
        config = {}
    restraint.setup(adapter, config=config)
    spec = restraint.spec
    closure = build_terms(spec.custom, "numpy")[0][-1]
    assert closure(coords[spec.active_sites]) == pytest.approx(
        (_volume(coords, groups) - 0.8) ** 2
    )
