"""Default public workflows and independent complete-pair VdW regression oracles."""

import jax
import numpy as np
import pytest
import torch
from rdkit import Chem

from rgi_toolkit import AtomRecord, CGStatus, CombinedRestraints, LigandConf
from rgi_toolkit.optim._vdw_runtime import VdwRuntime


class Groups:
    def __init__(self, n1, n2):
        self.n1, self.n2 = n1, n2

    def num_atoms(self):
        return self.n1 + self.n2 + 1

    def iter_atoms(self):
        for i in range(self.num_atoms()):
            chain = "A" if i < self.n1 else "B" if i < self.n1 + self.n2 else "C"
            yield AtomRecord(chain, 1, i, name="C", mol_type="ligand")

    def get_elements(self):
        return np.full(self.num_atoms(), 6)

    def iter_ligand_confs(self):
        mol = Chem.MolFromSmiles("C")
        mol.AddConformer(Chem.Conformer(1))
        yield LigandConf(
            mol, np.zeros((1, 3)), np.array([0]), conformer_restraints=True
        )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "sizes,conformer",
    [
        ((1, 1), "absent"),
        ((1, 1), None),
        ((1, 1), {}),
        ((1, 1), {"bond": {}}),
        ((1, 1), {"vdw": {}}),
        ((7, 32), {}),
        ((624, 690), {}),
    ],
)
def test_default_distance_moves_with_conformer(backend, device, sizes, conformer):
    jax.config.update("jax_enable_x64", True)
    n1, n2 = sizes
    cr = CombinedRestraints()
    config = {
        "gpu": device == "cuda",
        "distance_restraints_config": [
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "harmonic": {"target_distance": 5},
            }
        ],
    }
    if conformer != "absent":
        config["conformer_restraints_config"] = conformer
    cr.setup(Groups(n1, n2), config=config)
    points = np.zeros((2, n1 + n2 + 1, 3))
    points[:, n1 : n1 + n2, 0] = np.array([6.0, 8.0])[:, None]
    points[:, -1, 0] = 100.0
    if backend == "torch":
        out, info = cr.minimize(
            torch.tensor(points, device=device), sigma=0.0, return_info=True
        )
        out = out.cpu().numpy()
    else:
        out, info = jax.jit(cr.get_minimizer(return_info=True))(
            jax.device_put(
                points, jax.devices("gpu" if device == "cuda" else "cpu")[0]
            ),
            0.0,
        )
        out = np.asarray(out)
    distance = np.linalg.norm(
        out[:, :n1].mean(1) - out[:, n1 : n1 + n2].mean(1), axis=-1
    )
    np.testing.assert_allclose(distance, 5.0, atol=1e-4)
    assert int(info.status) == CGStatus.CONVERGED
    assert int(info.nit) <= 100
    np.testing.assert_array_equal(out[:, -1], points[:, -1])
    # Ordinary equal per-atom mobility gives the minimum squared displacement split.
    np.testing.assert_allclose(
        out[:, :n1].mean(1) * n1 + out[:, n1 : n1 + n2].mean(1) * n2,
        points[:, :n1].mean(1) * n1 + points[:, n1 : n1 + n2].mean(1) * n2,
        atol=1e-7,
    )


def dense_oracle(active, background, fixed, moving):
    """Independent analytic sum over every eligible pair, without neighbour lists."""
    gradient = np.zeros_like(active)
    energy = 0.0

    def pair(i, j, contact, fixed_partner):
        nonlocal energy
        delta = active[i] - (background[j] if fixed_partner else active[j])
        distance = np.sqrt(delta @ delta + 1e-12)
        residual = min(distance - contact, 0.0)
        energy += 25 * residual**2
        force = 50 * residual / distance * delta
        gradient[i] += force
        if not fixed_partner:
            gradient[j] -= force

    if fixed:
        for i in (0, 2):
            for j in range(1, len(background)):
                pair(i, j, 3.4, True)
    if moving:
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if (i, j) != (0, 1):
                    pair(i, j, 3.4, False)
    return gradient, energy


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_jax_vmap_preserves_conditional_work_and_sample_results(monkeypatch, device):
    calls = []
    original = VdwRuntime._build

    def counted(self, a, moving):
        jax.debug.callback(lambda: calls.append(moving))
        return original(self, a, moving)

    monkeypatch.setattr(VdwRuntime, "_build", counted)
    cr = CombinedRestraints()
    cr.setup(
        Groups(7, 32),
        config={
            "conformer_restraints_config": {"start_step": 1, "stop_step": 2},
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "harmonic": {"target_distance": 5},
                    "start_step": 1,
                    "stop_step": 2,
                }
            ],
        },
    )
    points = np.zeros((2, 40, 3))
    points[:, 7:39, 0] = 6.0
    points[:, -1, 0] = 100.0
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    args = jax.device_put((points, np.zeros(2), np.array([0, 1])), target)
    fn = cr.get_minimizer(return_info=True)
    expected = jax.jit(lambda xs: jax.lax.map(lambda x: fn(*x), xs))(args)
    jax.block_until_ready(expected)
    jax.effects_barrier()
    expected_calls = len(calls)
    assert expected_calls > 0
    calls.clear()
    actual = jax.jit(jax.vmap(fn))(*args)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert len(calls) == expected_calls
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10)
    np.testing.assert_array_equal(actual[0][0], points[0])
    assert int(actual[1].status[0]) == CGStatus.INACTIVE
    assert int(actual[1].status[1]) == CGStatus.CONVERGED


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("capacity", [1, 128])
def test_every_trial_and_overflow_match_complete_pair_oracle(
    backend, device, mode, capacity
):
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(302)
    points = rng.normal(0, 1.2, (5, 3))
    background = rng.normal(0, 2, (79, 3))
    background[40:] += [12, 0, 0]

    def native(value):
        value = np.asarray(value)
        if backend == "torch":
            return torch.as_tensor(value, device=device)
        return jax.device_put(
            value, jax.devices("gpu" if device == "cuda" else "cpu")[0]
        )

    common = dict(
        weight=native(1.0),
        scale=native(1.0),
        dmax=native(1.0),
        contact=native(3.4),
        max_neighbors=capacity,
        chemistry=None,
    )
    fixed = (
        dict(
            common,
            lig_local=native([0, 2]),
            lig_r=native([1.7, 1.7]),
            bg_r=native([0.0] + [1.7] * 78),
        )
        if mode & 1
        else None
    )
    moving = (
        dict(
            common,
            radii=native([1.7] * 5),
            polymer_mask=native([True] * 5),
            excluded_codes=native([1]),
        )
        if mode & 2
        else None
    )
    runtime = VdwRuntime(
        backend,
        native(points),
        fixed=fixed,
        moving=moving,
        background=native(background),
        skin=0.5,
    )

    def evaluate(a, cache):
        cache = runtime.prepare(a, cache)
        g, f = runtime.grad_value(runtime.sparse_energy)(a, cache)
        dg, df = runtime.dense_value_grad(a, cache)
        return g + dg, f + df, cache

    if backend == "jax":
        evaluate = jax.jit(evaluate)
    cache = runtime.empty(native(points))
    # A rejected far trial followed by a return must not retain the wrong cell list.
    for shift in (0.0, 0.1, 12.0, 100.0, 0.2, -9.0, 0.0):
        current = points + [shift, 0, 0]
        g, f, cache = evaluate(native(current), cache)
        expected_g, expected_f = dense_oracle(current, background, mode & 1, mode & 2)
        if backend == "torch":
            g, f = g.cpu(), f.cpu()
        np.testing.assert_allclose(f, expected_f, atol=1e-9, rtol=1e-12)
        np.testing.assert_allclose(g, expected_g, atol=1e-9, rtol=1e-12)
    if capacity == 1:
        assert any(bool(c.overflow.any()) for c in cache if c is not None)


@pytest.mark.parametrize("key", ["max_atom_step", "neighbor_rebuild_interval"])
def test_retired_vdw_controls_explain_migration(key):
    cr = CombinedRestraints()
    with pytest.raises(ValueError, match="remove retired key"):
        cr.setup(
            Groups(1, 1), config={"conformer_restraints_config": {"vdw": {key: 1}}}
        )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("move", [0, 1, 2])
@pytest.mark.parametrize("overlap", [False, True])
def test_centroid_coordinates_are_an_exact_invertible_energy_map(
    backend, move, overlap
):
    from rgi_toolkit.optim._coordinates import CentroidCoordinates

    jax.config.update("jax_enable_x64", True)
    first, second = np.arange(7), np.arange(4 if overlap else 7, 39)
    cr = CombinedRestraints()
    entry = {
        "atom_selection1": "index 0 to 6",
        "atom_selection2": f"index {second[0]} to 38",
        "harmonic": {"target_distance": 5},
        "start_step": 1,
        "stop_step": 2,
    }
    if move:
        entry["move"] = move
    cr.setup(
        Groups(7, 32),
        config={
            "distance_restraints_config": [entry],
            "conformer_restraints_config": {},
        },
    )
    n = cr.spec.n_active
    row = np.zeros(n)
    if move in (0, 1):
        row[first] += 1 / len(first)
    if move in (0, 2):
        row[second] -= 1 / len(second)
    norm = np.linalg.norm(row)
    q = row / norm
    matrix = np.eye(n) + (max(1, 1 / norm) - 1) * np.outer(q, q)
    assert np.linalg.eigvalsh(matrix).min() >= 1 - 1e-12
    rng = np.random.default_rng(51)
    origin, trial = rng.normal(size=(2, n, 3))
    stiffness = np.linspace(1, 100, n)[:, None]
    if backend == "torch":
        native = torch.as_tensor
    else:
        import jax.numpy as jnp

        native = jnp.asarray
    x0, u = native(origin), native(trial)
    coordinates = CentroidCoordinates(cr.spec)
    transform = coordinates.bind(backend, u, step=1)
    mapped = transform(u, x0)
    expected = origin + matrix @ (trial - origin)
    np.testing.assert_allclose(np.asarray(mapped), expected, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.solve(matrix, expected - origin), trial - origin
    )

    def energy(v):
        x = transform(v, x0)
        return (native(stiffness) * x**2).sum() / 2

    gradient = (
        torch.func.grad(energy)(u)
        if backend == "torch"
        else jax.jit(jax.grad(energy))(u)
    )
    np.testing.assert_allclose(
        np.asarray(gradient), matrix.T @ (stiffness * expected), atol=1e-10
    )
    inactive = coordinates.bind(backend, u, step=0)
    np.testing.assert_array_equal(np.asarray(inactive(u, x0)), trial)
    if move == 0:
        np.testing.assert_allclose(np.asarray(mapped).sum(0), trial.sum(0), atol=1e-12)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_mixed_centroid_and_stiff_geometry_reach_target_with_default_budget(backend):
    from rgi_toolkit.optim.jax_optim import make_minimizer
    from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer
    from rgi_toolkit.spec import BondArrays, ChiralArrays

    jax.config.update("jax_enable_x64", True)
    n1, n2 = 624, 690
    cr = CombinedRestraints()
    cr.setup(
        Groups(n1, n2),
        config={
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "harmonic": {"target_distance": 5},
                }
            ]
        },
    )
    spec = cr.spec
    n = n1 + n2
    spec.n_active = n + 4
    spec.active_sites = np.arange(n + 4)
    indices = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) + n
    spec.bond = BondArrays(
        idx=indices,
        r0=np.sqrt([1, 1, 1, 2, 2, 2]),
        slack=np.zeros(6),
        weight=np.ones(6),
        half=np.zeros(6),
        mask=np.ones(6),
    )
    spec.chiral = ChiralArrays(
        idx=np.arange(n, n + 4).reshape(1, 4),
        vol0=np.ones(1),
        slack=np.zeros(1),
        weight=np.ones(1),
        mask=np.ones(1),
    )
    points = np.zeros((n + 4, 3))
    points[n1:n, 0] = 8.0
    points[n:] = 100 + np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]])
    if backend == "torch":
        out = TorchRestraintOptimizer(spec).minimize(torch.tensor(points)).numpy()
    else:
        out = np.asarray(jax.jit(make_minimizer(spec))(jax.numpy.asarray(points), 0.0))
    gap = np.linalg.norm(out[:n1].mean(0) - out[n1:n].mean(0))
    assert gap == pytest.approx(5.0, abs=1e-4)
    lengths = np.linalg.norm(out[indices[:, 0]] - out[indices[:, 1]], axis=-1)
    np.testing.assert_allclose(lengths, spec.bond.r0, atol=1e-4)
