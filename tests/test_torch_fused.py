"""CUDA trial predicates, compiler fallback, and overflow cache invalidation."""

import numpy as np
import pytest
import torch

from rgi_toolkit.optim import _torch_fused as fused
from rgi_toolkit.optim._vdw_runtime import VdwRuntime


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("condition", ["finite", "stationary", "nan", "inf"])
def test_trial_statistics_preserve_values_and_invalid_point_checks(dtype, condition):
    generator = torch.Generator(device="cuda").manual_seed(72)
    g = torch.randn((2, 137, 3), generator=generator, device="cuda", dtype=dtype)
    d = torch.randn(g.shape, generator=generator, device="cuda", dtype=dtype)
    x = torch.randn(g.shape, generator=generator, device="cuda", dtype=dtype)
    base = x.clone() if condition == "stationary" else x + 0.1
    f = g.new_tensor(3.2)
    if condition == "nan":
        g[0, 0, 0] = float("nan")
    elif condition == "inf":
        x[0, 0, 0] = float("inf")
    expected = fused._trial_values(f, g, d, x, base)
    actual = fused.trial_values(f, g, d, x, base)
    torch.testing.assert_close(actual, expected, equal_nan=True)
    assert bool(actual[4]) == (condition in ("finite", "stationary"))
    assert bool(actual[5]) == (condition != "stationary")
    assert fused._COMPILED[fused._trial_values] is not None


@pytest.mark.gpu
def test_compiler_failure_preserves_eager_statistics(monkeypatch):
    calls = []

    def fail(*args):
        calls.append(True)
        raise RuntimeError("test compiler failure")

    monkeypatch.setitem(fused._COMPILED, fused._trial_values, fail)
    g = torch.ones((4, 3), device="cuda")
    args = (g.new_tensor(1.0), g, -g, g, g + 1)
    for _ in range(2):
        torch.testing.assert_close(
            fused.trial_values(*args), fused._trial_values(*args)
        )
    assert calls == [True]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_overflow_predicate_changes_after_rebuilding(device):
    def native(value):
        return torch.as_tensor(np.asarray(value), device=device)

    active = native([[20.0, 0.0, 0.0]])
    background = native([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    fixed = dict(
        weight=native(1.0),
        scale=native(1.0),
        dmax=native(5.0),
        contact=native(3.4),
        max_neighbors=1,
        chemistry=None,
        lig_local=native([0]),
        lig_r=native([1.7]),
        bg_r=native([1.7, 1.7]),
    )
    runtime = VdwRuntime("torch", active, fixed=fixed, background=background, skin=0.5)
    empty = runtime.empty(active)
    assert not runtime.has_overflow(empty)
    far = runtime.prepare(active, empty)
    assert not runtime.has_overflow(far)
    near = native([[0.8, 0.0, 0.0]])
    crowded = runtime.prepare(near, far)
    assert runtime.has_overflow(crowded)
    reused = runtime.prepare(near + 0.1, crowded)
    assert reused[0] is crowded[0]
    assert runtime.has_overflow(reused)
    gradient, energy = runtime.dense_value_grad(near, crowded)
    assert float(energy) > 0 and bool(torch.isfinite(gradient).all())
    clear = runtime.prepare(active, reused)
    assert not runtime.has_overflow(clear)
    # Independently computed pairs remain present whenever they enter contact.
    again = runtime.prepare(near, clear)
    assert runtime.has_overflow(again)
    g2, f2 = runtime.dense_value_grad(near, again)
    torch.testing.assert_close(g2, gradient)
    torch.testing.assert_close(f2, energy)
