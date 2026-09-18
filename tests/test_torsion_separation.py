"""Independent E/Z and general torsion configuration through public APIs."""

from types import SimpleNamespace

import numpy as np
import pytest

from rgi_toolkit import AtomRecord, CombinedRestraints
from rgi_toolkit.energy import numpy_energy
from rgi_toolkit.featurizer import build_spec
from tests.test_featurizer import _lig_heavy


def _fixture():
    ligand = _lig_heavy("CC(=O)NC/C=C/C")
    config = {
        "relax_force_field": {"ligand": "none"},
        **{key: {"weight": 0} for key in ("bond", "angle", "chiral", "plane", "vdw")},
        "cistrans": {"weight": 2},
        "torsion": {"weight": 3},
    }
    initial = ligand.conf_coords + np.random.default_rng(731).normal(
        scale=0.15, size=ligand.conf_coords.shape
    )
    return ligand, config, initial


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("use_esd", [False, True])
def test_cistrans_and_torsion_have_independent_weights_and_breakdowns(backend, use_esd):
    ligand, config, initial = _fixture()
    config["use_esd"] = use_esd
    both = build_spec([ligand], conformer_config=config)
    assert len(both.cistrans.idx) > 0 and len(both.torsion.idx) > 0
    np.testing.assert_array_equal(both.cistrans.period, 1)
    np.testing.assert_array_equal(both.torsion.period, 2)
    expected = numpy_energy.energy_breakdown(
        initial[both.active_sites], numpy_energy.prepare_spec(both)
    )
    assert expected["cistrans"] > 0 and expected["torsion"] > 0

    if backend == "torch":
        import torch

        from rgi_toolkit.energy import torch_energy as module

        def native(x):
            return torch.tensor(x, dtype=torch.float64)

        def prepare(spec):
            return module.prepare_spec(spec, dtype=torch.float64)
    elif backend == "jax":
        import jax

        from rgi_toolkit.energy import jax_energy as module

        jax.config.update("jax_enable_x64", True)
        native, prepare = jax.numpy.asarray, module.prepare_spec
    else:
        module = numpy_energy
        native, prepare = np.asarray, module.prepare_spec

    for use_cistrans, use_torsion in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        options = {
            **config,
            "cistrans": {"weight": 2 if use_cistrans else 0},
            "torsion": {"weight": 3 if use_torsion else 0},
        }
        spec = build_spec([ligand], conformer_config=options)
        assert (spec.cistrans is not None) == use_cistrans
        assert (spec.torsion is not None) == use_torsion
        coords, prepared = native(initial[spec.active_sites]), prepare(spec)
        breakdown = module.energy_breakdown(coords, prepared)
        assert breakdown["cistrans"] == pytest.approx(
            expected["cistrans"] if use_cistrans else 0, abs=1e-10
        )
        assert breakdown["torsion"] == pytest.approx(
            expected["torsion"] if use_torsion else 0, abs=1e-10
        )
        assert float(module.total_energy(coords, prepared)) == pytest.approx(
            sum(breakdown.values()), abs=1e-10
        )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize(
    "options", [{"line_search": "armijo"}, {}, {"method": "l-bfgs"}]
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_public_solvers_reduce_both_torsions_and_respect_window(
    backend, options, device
):
    ligand, config, initial = _fixture()
    config["start_sigma"] = 1.0
    atoms = [AtomRecord("L", 1, i) for i in range(len(initial))]
    adapter = SimpleNamespace(
        iter_atoms=lambda: iter(atoms),
        iter_ligand_confs=lambda: iter([ligand]),
        get_elements=lambda: np.asarray(
            [a.GetAtomicNum() for a in ligand.mol.GetAtoms()]
        ),
    )
    restraints = CombinedRestraints()
    restraints.setup(
        adapter,
        config={
            **options,
            "gpu": device == "gpu",
            "conformer_restraints_config": config,
        },
    )
    assert restraints.spec.cistrans is not None
    assert restraints.spec.torsion is not None
    if backend == "torch":
        import torch

        target = "cuda" if device == "gpu" else "cpu"

        def run(sigma):
            return (
                restraints.minimize(torch.tensor(initial, device=target), sigma=sigma)
                .cpu()
                .numpy()
            )
    else:
        import jax

        jax.config.update("jax_enable_x64", True)
        target = jax.devices(device)[0]
        minimize = jax.jit(restraints.get_minimizer())

        def run(sigma):
            return np.asarray(minimize(jax.device_put(initial, target), sigma))

    np.testing.assert_array_equal(run(2.0), initial)
    output = run(0.5)
    assert np.isfinite(output).all()
    spec = restraints.spec
    prepared = numpy_energy.prepare_spec(spec)
    before = numpy_energy.energy_breakdown(initial[spec.active_sites], prepared)
    after = numpy_energy.energy_breakdown(output[spec.active_sites], prepared)
    for key in ("cistrans", "torsion"):
        assert before[key] > 1e-4
        assert after[key] < 1e-6, (backend, options, key, before, after)
