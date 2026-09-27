"""CUDA trial predicates, compiler fallback, and overflow cache invalidation."""

import numpy as np
import pytest
import torch

from rgi_toolkit.optim import _torch_fused as fused
from rgi_toolkit.optim._vdw_runtime import VdwRuntime


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("mode", [1, 2, 3])
def test_combined_trial_checks_preserve_cache_boundaries_and_invalid_points(
    device, mode
):
    a = torch.zeros((2, 3, 3), device=device)
    indices = torch.tensor([0, 2], device=device)
    valid = torch.tensor(True, device=device)
    fixed = (a[..., indices, :].clone(), indices, valid, 1.0) if mode & 1 else None
    moving = (a.clone(), None, valid, 0.25) if mode & 2 else None
    previous = a.clone()
    for displacement in (0.0, 0.5, 0.75, 1.0, 1.25):
        a[1, 0, 0] = displacement
        actual = fused.trial_checks(a, previous, fixed, moving)
        assert actual == [
            displacement == 0.0,
            bool(mode & 1) and displacement > 1.0,
            bool(mode & 2) and displacement > 0.5,
        ]
    for invalid in (float("nan"), float("inf")):
        a[0, 1, 2] = invalid
        assert fused.trial_checks(a, previous, fixed, moving) == [False, False, False]
    if device == "cuda":
        assert fused._COMPILED[fused._trial_checks] is not None


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_combined_trial_preparation_preserves_duplicates_and_overflow(device):
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
    cache = runtime.prepare(active, runtime.empty(active))
    same, reused = runtime.prepare_trial(active.clone(), active, cache)
    assert same and reused is cache
    near = native([[0.8, 0.0, 0.0]])
    same, rebuilt = runtime.prepare_trial(near, active, cache)
    assert not same and runtime.has_overflow(rebuilt)
    expected = runtime.prepare(near, cache)
    for result, reference in zip(rebuilt[0], expected[0]):
        if reference is not None:
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    same, within_skin = runtime.prepare_trial(near + 0.1, near, rebuilt)
    assert not same and within_skin[0] is rebuilt[0]
    same, clear = runtime.prepare_trial(active, near, rebuilt)
    assert not same and not runtime.has_overflow(clear)


@pytest.mark.gpu
def test_combined_trial_checks_fall_back_after_compile_failure(monkeypatch):
    calls = []

    def fail(*args):
        calls.append(True)
        raise RuntimeError("test compiler failure")

    monkeypatch.setitem(fused._COMPILED, fused._trial_checks, fail)
    a = torch.ones((4, 3), device="cuda")
    for _ in range(2):
        assert fused.trial_checks(a, a.clone(), None, None) == [True, False, False]
        assert fused.trial_checks(a, a + 1, None, None) == [False, False, False]
    assert calls == [True]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("restart", [False, True])
def test_fused_direction_matches_host_pr_plus(dtype, restart):
    generator = torch.Generator(device="cuda").manual_seed(409)
    g = torch.randn((2, 137, 3), generator=generator, device="cuda", dtype=dtype)
    old_g = g * (2 if restart else 0.5)
    old_d = torch.randn(g.shape, generator=generator, device="cuda", dtype=dtype)
    denominator = float((old_g * old_g).sum())
    numerator = float((g * (g - old_g)).sum())
    expected = -g + max(0.0, numerator / denominator) * old_d
    actual, slope = fused.direction(g, old_g, old_d, denominator)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(slope, (expected * g).sum())
    if restart:
        torch.testing.assert_close(actual, -g, rtol=0, atol=0)
    assert fused._COMPILED[fused._direction] is not None


@pytest.mark.gpu
@pytest.mark.parametrize("moving", [False, True])
def test_fused_cache_predicate_preserves_skin_boundary_and_invalid_trials(moving):
    a = torch.zeros((2, 3, 3), device="cuda")
    indices = None if moving else torch.tensor([0, 2], device="cuda")
    reference = a.clone() if moving else a[..., indices, :].clone()
    valid = torch.tensor(True, device="cuda")
    threshold = 0.25
    assert not bool(fused.cache_needed(a, reference, indices, valid, threshold, True))
    # At the skin boundary a contact cannot have crossed the protected shell.
    a[1, 0, 0] = 0.5
    assert not bool(fused.cache_needed(a, reference, indices, valid, threshold, True))
    a[1, 0, 0] = torch.nextafter(a[1, 0, 0], a.new_tensor(1.0))
    assert bool(fused.cache_needed(a, reference, indices, valid, threshold, True))
    assert not bool(fused.cache_needed(a, reference, indices, valid, threshold, False))
    a[0, 1, 1] = float("nan")
    assert not bool(fused.cache_needed(a, reference, indices, valid, threshold, True))
    assert fused._COMPILED[fused._cache_needed] is not None


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
