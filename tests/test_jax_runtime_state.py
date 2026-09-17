"""Runtime restraint values must change results without changing the JIT signature."""

from types import SimpleNamespace

import numpy as np
import pytest

from rgi_toolkit.atom_context import AtomRecord
from rgi_toolkit.combined import CombinedRestraints
from rgi_toolkit.optim.scan_runner import ScanMinimizer


def _restraints(
    target=4.0,
    weight=1.0,
    *,
    window=None,
    custom=False,
    method="CG",
    line_search="armijo",
):
    rows = [
        {
            "atom_selection1": "index 0",
            "atom_selection2": "index 1",
            "move": 2,
            "harmonic": {"target_distance": value},
            "weight": w,
            **(window or {}),
        }
        for value, w in ((target, weight), (10.0, 3.0))
    ]
    config = {"method": method, "distance_restraints_config": rows}
    if line_search is not None:
        config["line_search"] = line_search
    if custom:
        config["distance_restraints_config"] = rows[:1]
        config["custom_restraints_config"] = [
            {
                "energy": "(distance(A, B) - 10)**2",
                "selections": {"A": "index 0", "B": "index 1"},
                "move": "B",
                "weight": 3.0,
                **(window or {}),
            }
        ]
    rgi = CombinedRestraints()
    atoms = [AtomRecord("A", i + 1, i) for i in range(2)]
    rgi.setup(SimpleNamespace(iter_atoms=lambda: iter(atoms)), config=config)
    return rgi


@pytest.mark.parametrize(
    "method,line_search", [("CG", "armijo"), ("CG", "strong-wolfe"), ("l-bfgs", None)]
)
@pytest.mark.parametrize("custom", [False, True])
def test_values_and_windows_reuse_one_trace(method, line_search, custom):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    if method == "l-bfgs":
        pytest.importorskip("jaxopt")
    import jax.numpy as jnp

    traces = []

    @jax.jit
    def run(minimizer, coords, sigma, step):
        traces.append(1)
        return minimizer(coords, sigma, step)

    coords = jnp.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=jnp.float64)
    options = dict(custom=custom, method=method, line_search=line_search)
    first = _restraints(**options).get_minimizer()
    changed = _restraints(target=6.0, weight=3.0, **options).get_minimizer()
    sigma_off = _restraints(window={"start_sigma": 0.5}, **options).get_minimizer()
    step_on = _restraints(
        window={"start_step": 2, "stop_step": 3}, **options
    ).get_minimizer()
    for minimizer, sigma, step, expected in (
        (first, 1.0, 0, 8.5),
        (changed, 1.0, 0, 8.0),
        (sigma_off, 1.0, 0, 6.0),
        (step_on, 1.0, 1, 6.0),
        (step_on, 1.0, 2, 8.5),
        (first, 1.0, 0, 8.5),
    ):
        out = np.asarray(run(minimizer, coords, sigma, step))
        np.testing.assert_allclose(out[0], coords[0], atol=1e-6)
        assert out[1, 0] == pytest.approx(expected, abs=2e-4)
    if custom:
        reweighted = _restraints(**options)
        reweighted.spec.custom[0].weight = 1.0
        out = np.asarray(run(reweighted.get_minimizer(), coords, 1.0, 0))
        assert out[1, 0] == pytest.approx(7.0, abs=2e-4)
    assert len(traces) == 1


def test_scan_vmap_and_new_instances_share_runtime_state_signature():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    traces = []

    @jax.jit
    def predict(state, positions):
        traces.append(1)

        def sample(coords):
            def step(carry, index):
                return state.minimize_gpu(carry, 1.0, index), None

            return jax.lax.scan(step, coords, jnp.arange(2))[0]

        return jax.vmap(sample)(positions)

    positions = jnp.array([[[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]]], dtype=jnp.float32)
    positions = jnp.repeat(positions, 2, axis=0)
    for target, expected in ((4.0, 8.5), (6.0, 9.0)):
        rgi = _restraints(target=target)
        state = ScanMinimizer(rgi, rgi.get_minimizer()).as_pytree()
        assert state.is_active()
        out = np.asarray(predict(state, positions))
        np.testing.assert_allclose(out[:, 0, 1, 0], expected, atol=2e-4)
    assert len(traces) == 1


def test_mixed_conformer_coordinate_map_and_gates_are_dynamic():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from rgi_toolkit.optim.jax_optim import make_minimizer
    from rgi_toolkit.spec import BondArrays

    rgi = CombinedRestraints()
    atoms = [AtomRecord("A", i + 1, i) for i in range(6)]
    rgi.setup(
        SimpleNamespace(iter_atoms=lambda: iter(atoms)),
        config={
            "distance_restraints_config": [
                {
                    "atom_selection1": "index 0 1 2",
                    "atom_selection2": "index 3 4 5",
                    "harmonic": {"target_distance": 4.0},
                }
            ],
        },
    )
    spec = rgi.spec
    # Internal bonds and group translations exercise the mixed coordinate map.
    spec.bond = BondArrays(
        idx=np.array([[0, 1], [1, 2], [3, 4], [4, 5]]),
        r0=np.ones(4),
        slack=np.zeros(4),
        weight=np.ones(4),
        half=np.zeros(4),
        mask=np.ones(4),
    )
    spec.conf_start_sigma = 2.0
    traces = []

    @jax.jit
    def run(minimizer, coords):
        traces.append(1)
        return minimizer(coords, 1.0)

    coords = jnp.array(
        [[x, 0.0, 0.0] for x in (0, 1.5, 3, 6, 7.5, 9)], dtype=jnp.float32
    )
    first = make_minimizer(spec)
    out = np.asarray(run(first, coords))
    assert np.linalg.norm(out[3:].mean(0) - out[:3].mean(0)) == pytest.approx(
        4.0, abs=3e-4
    )
    assert np.linalg.norm(out[1] - out[0]) == pytest.approx(1.0, abs=3e-4)
    spec.conf_start_sigma = 0.5
    out = np.asarray(run(make_minimizer(spec), coords))
    assert np.linalg.norm(out[3:].mean(0) - out[:3].mean(0)) == pytest.approx(
        4.0, abs=3e-4
    )
    assert np.linalg.norm(out[1] - out[0]) == pytest.approx(1.5, abs=3e-4)
    assert len(traces) == 1


def test_reference_targets_and_coordinates_are_runtime_values():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from rgi_toolkit.custom.data import CustomSpec
    from rgi_toolkit.optim.jax_optim import make_minimizer
    from rgi_toolkit.spec import RestraintSpec

    term = CustomSpec(
        name="anchor",
        kind="ref_geom",
        selections={},
        ast=None,
        fn=None,
        weight=1.0,
        start_sigma=np.inf,
        stop_sigma=-1.0,
        start_step=-np.inf,
        stop_step=np.inf,
        geom="distance",
        target1=2.0,
        groups=[("pred", (np.array([0]), True)), ("ref", ("anchor", "ref1"))],
        ref_blocks={("anchor", "ref1"): np.zeros((1, 3))},
    )
    spec = RestraintSpec(n_active=1, active_sites=np.array([0]), custom=[term])
    traces = []

    @jax.jit
    def run(minimizer, coords):
        traces.append(1)
        return minimizer(coords, 1.0)

    coords = jnp.array([[5.0, 0.0, 0.0]])
    out = np.asarray(run(make_minimizer(spec), coords))
    assert np.linalg.norm(out[0]) == pytest.approx(2.0, abs=1e-4)
    term.target1 = 3.0
    term.ref_blocks[("anchor", "ref1")][0, 0] = 1.0
    out = np.asarray(run(make_minimizer(spec), coords))
    assert np.linalg.norm(out[0] - [1, 0, 0]) == pytest.approx(3.0, abs=1e-4)
    assert len(traces) == 1


@pytest.mark.parametrize("method", ["CG", "l-bfgs"])
def test_dynamic_vdw_weight_changes_the_compromise_without_retracing(method):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from rgi_toolkit.optim.jax_optim import make_minimizer
    from rgi_toolkit.spec import VdwConfig

    spec = _restraints().spec
    spec.distance.target1[:] = 1.0
    spec.distance.weight[:] = 0.5
    spec.conf_start_sigma = np.inf
    spec.vdw_config = VdwConfig(
        weight=0.04,
        ligand_local=np.array([1]),
        ligand_radii=np.ones(1),
        background_global=np.array([2]),
        background_radii=np.ones(1),
        scale=1.0,
    )
    traces = []

    @jax.jit
    def run(minimizer, coords):
        traces.append(1)
        return minimizer(coords, 1.0)

    coords = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    if method == "l-bfgs":
        pytest.importorskip("jaxopt")
    for weight, expected in ((0.04, 1.5), (0.12, 1.75)):
        spec.vdw_config.weight = weight
        out = np.asarray(run(make_minimizer(spec, method=method), coords))
        assert out[1, 0] == pytest.approx(expected, abs=1e-4)
        np.testing.assert_allclose(out[[0, 2]], np.asarray(coords)[[0, 2]], atol=1e-6)
    assert len(traces) == 1
