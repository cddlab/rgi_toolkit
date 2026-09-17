"""Exact packed-overflow evaluation on each backend independently."""

import numpy as np
import pytest

from rgi_toolkit.optim._vdw_runtime import VdwRuntime


def _native(backend, device):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        return lambda value: torch.as_tensor(np.asarray(value), device=device)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    return lambda value: jax.device_put(np.asarray(value), target)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("typed", [False, True])
def test_packed_overflow_matches_dense_batched_energy_and_gradient(
    backend, device, typed
):
    """Unequal overflow counts and a partial second chunk preserve every pair."""
    rng = np.random.default_rng(604)
    points = rng.normal(0, 0.2, (2, 270, 3))
    points[1, 10:, 0] += np.arange(1, 261) * 10

    native = _native(backend, device)

    moving = dict(
        weight=native(1.0),
        scale=native(1.0),
        dmax=native(5.0),
        contact=native(3.4),
        max_neighbors=1,
        chemistry=None,
        radii=native(np.full(270, 1.7)),
        polymer_mask=native(np.ones(270, dtype=bool)),
        excluded_codes=native([1]),
    )
    if typed:
        from rgi_toolkit._array_ops import get_ops
        from rgi_toolkit.energy._nonbonded import prepare_chemistry

        host = dict(
            query_types=np.zeros(270, dtype=int),
            target_types=np.zeros(270, dtype=int),
            contacts=np.array([[3.4]]),
            inv_variances=np.array([[25.0]]),
            one_four_contacts=np.array([[3.4]]),
            one_four_inv_variances=np.array([[25.0]]),
            excluded=np.array([1, 270]),
            one_four=np.zeros(0, dtype=int),
            query_molecules=np.zeros(270, dtype=int),
            target_molecules=np.zeros(270, dtype=int),
            query_moving=np.ones(270, dtype=int),
            target_moving=np.ones(270, dtype=int),
            query_static=np.zeros(270, dtype=int),
            target_static=np.zeros(270, dtype=int),
            mode=np.array(0),
        )
        moving["chemistry"] = prepare_chemistry(
            get_ops(backend, device=device), host, native(points)
        )
    runtime = VdwRuntime(backend, native(points), moving=moving, skin=0.5)

    def evaluate(a):
        cache = runtime.prepare(a, runtime.empty(a))
        g, f = runtime.grad_value(runtime.sparse_energy)(a, cache)
        dg, df = runtime.dense_value_grad(a, cache)
        return g + dg, f + df

    if backend == "jax":
        evaluate = pytest.importorskip("jax").jit(evaluate)
    gradient, energy = evaluate(native(points))
    if backend == "torch" and device == "cuda":
        assert runtime._compiled_dense.get(True) is not None
    diff = points[:, :, None, :] - points[:, None, :, :]
    distance = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    allowed = ~np.eye(270, dtype=bool)
    allowed[0, 1] = allowed[1, 0] = False
    residual = np.minimum(distance - 3.4, 0) * allowed
    expected_g = 50 * np.sum((residual / distance)[..., None] * diff, axis=2)
    expected_f = 12.5 * np.sum(residual**2)
    if backend == "torch":
        gradient, energy = gradient.cpu(), energy.cpu()
    np.testing.assert_allclose(gradient, expected_g, rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(energy, expected_f, rtol=1e-11)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_harmless_candidates_do_not_overflow_and_cached_contact_is_retained(backend):
    """A pair may enter contact within the skin without a rebuild or lost energy."""
    points = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    background = np.array([[99.0, 0.0, 0.0], [3.5, 0.0, 0.0], [4.8, 0.0, 0.0]])
    native = _native(backend, "cpu")
    fixed = dict(
        weight=native(1.0),
        scale=native(1.0),
        dmax=native(5.0),
        contact=native(3.4),
        max_neighbors=1,
        chemistry=None,
        lig_local=native(np.array([0, 2])),
        lig_r=native(np.array([1.7, 1.7])),
        bg_r=native(np.array([0.0, 1.7, 1.7])),
    )
    runtime = VdwRuntime(
        backend, native(points), fixed=fixed, background=native(background), skin=0.5
    )
    cache = runtime.prepare(native(points), runtime.empty(native(points)))
    assert not bool(cache[0].overflow.any())
    moved = points.copy()
    moved[0, 0] += 0.2
    updated = runtime.prepare(native(moved), cache)
    np.testing.assert_array_equal(updated[0].reference, cache[0].reference)
    gradient, energy = runtime.grad_value(runtime.sparse_energy)(native(moved), updated)
    diff = moved[[0, 2], None, :] - background[None, 1:, :]
    distance = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    residual = np.minimum(distance - 3.4, 0)
    expected_f = 25 * np.sum(residual**2)
    expected_g = np.zeros_like(moved)
    expected_g[[0, 2]] = 50 * np.sum((residual / distance)[..., None] * diff, axis=1)
    assert expected_f > 0
    np.testing.assert_allclose(energy, expected_f, rtol=1e-12)
    np.testing.assert_allclose(gradient, expected_g, atol=1e-10)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_topology_rows_and_traced_flat_codes_resolve_the_same_contacts(backend, device):
    from rgi_toolkit._array_ops import get_ops
    from rgi_toolkit.energy._nonbonded import pair_parameters, prepare_chemistry

    if backend == "numpy" and device == "cuda":
        pytest.skip("NumPy has no CUDA backend")
    native = np.asarray if backend == "numpy" else _native(backend, device)
    host = dict(
        query_types=np.zeros(6, dtype=int),
        target_types=np.zeros(11, dtype=int),
        contacts=np.array([[3.4]]),
        inv_variances=np.array([[25.0]]),
        one_four_contacts=np.array([[2.8]]),
        one_four_inv_variances=np.array([[4.0]]),
        excluded=np.array([1, 3, 8, 12, 32, 55, 64]),
        one_four=np.array([2, 13, 26, 53]),
        query_molecules=np.zeros(6, dtype=int),
        target_molecules=np.zeros(11, dtype=int),
        query_moving=np.ones(6, dtype=int),
        target_moving=np.zeros(11, dtype=int),
        query_static=np.zeros(6, dtype=int),
        target_static=np.zeros(11, dtype=int),
        mode=np.array(0),
    )
    ops = get_ops(backend, device=device)
    like = native(np.zeros((6, 3)))
    prepared = prepare_chemistry(ops, host, like)
    sources = np.arange(6).reshape(1, 6, 1)
    targets = np.broadcast_to(np.arange(11), (2, 6, 11)).copy()
    codes = sources * 11 + targets
    expected = (
        np.where(np.isin(codes, host["one_four"]), 2.8, 3.4),
        np.where(np.isin(codes, host["one_four"]), 4.0, 25.0),
        ~np.isin(codes, host["excluded"]),
    )

    def evaluate(chemistry, source, target):
        # JAX builders recast prepared constants while tracing; Torch prepares
        # host chemistry once per device/dtype instead.
        if backend != "torch":
            chemistry = prepare_chemistry(ops, chemistry, like)
        return pair_parameters(ops, chemistry, source, target)

    if backend == "jax":
        evaluate = pytest.importorskip("jax").jit(evaluate)
    flat = {key: value for key, value in prepared.items() if not key.endswith("_rows")}
    for chemistry in (prepared, flat):
        actual = evaluate(chemistry, native(sources), native(targets))
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(
                value.cpu() if backend == "torch" else value, reference
            )
    scalar = pair_parameters(ops, prepared, 0, native(np.array([1, 2])))
    np.testing.assert_array_equal(
        scalar[2].cpu() if backend == "torch" else scalar[2], [False, True]
    )
