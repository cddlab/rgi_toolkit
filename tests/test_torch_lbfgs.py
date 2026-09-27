"""CUDA L-BFGS agrees with the upstream optimizer and has an exact fallback."""

import numpy as np
import pytest
import torch

from rgi_toolkit.optim import _torch_lbfgs as impl


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_device_axpy_preserves_native_rounding_and_storage(dtype):
    generator = torch.Generator(device="cuda").manual_seed(822)
    x, y = [
        torch.randn(5179, generator=generator, device="cuda", dtype=dtype)
        for _ in range(2)
    ]
    alpha = x.new_tensor(0.133849394184)
    expected = x.clone().add_(y, alpha=float(alpha))
    pointer = x.data_ptr()
    result = impl.add_device_scalar(x, y, alpha)
    assert result.data_ptr() == pointer
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert impl._ADD is not None and not impl._ADD_FAILED


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [93, 5286])
def test_fused_two_loop_matches_independent_dense_recursion(dtype, size):
    from rgi_toolkit.optim._torch_lbfgs_cuda import direction

    rng = np.random.default_rng(55)
    g = rng.normal(size=size)
    s = rng.normal(size=(7, size))
    y = s * np.linspace(0.5, 2.0, size)
    rho = 1 / np.sum(s * y, axis=1)
    h_diag = np.sum(s[-1] * y[-1]) / np.sum(y[-1] ** 2)
    q, a = -g.copy(), np.empty(7)
    for i in range(6, -1, -1):
        a[i] = np.dot(s[i], q) * rho[i]
        q -= a[i] * y[i]
    expected = q * h_diag
    for i in range(7):
        beta = np.dot(y[i], expected) * rho[i]
        expected += (a[i] - beta) * s[i]

    def native(v):
        return torch.tensor(v, device="cuda", dtype=dtype)

    actual, coefficients = direction(
        native(g), list(native(y)), list(native(s)), list(native(rho)), native(h_diag)
    )
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        torch.stack(coefficients).cpu(), a, atol=tolerance, rtol=tolerance
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("history_size", [1, 5, 100])
def test_cuda_lbfgs_matches_upstream_minimum_with_history_and_resumption(
    dtype, history_size
):
    initial = torch.linspace(-2.0, 2.0, 93, dtype=dtype, device="cuda")
    target = initial.sin()
    diagonal = torch.linspace(0.5, 2.0, 93, dtype=dtype, device="cuda")
    outputs = []
    for optimizer in (torch.optim.LBFGS, impl.CudaLBFGS):
        x = initial.clone().requires_grad_()
        opt = optimizer(
            [x],
            max_iter=7,
            history_size=history_size,
            tolerance_grad=1e-5,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss = ((x - target).square() * diagonal).sum()
            loss.backward()
            return loss

        for _ in range(3):
            opt.step(closure)
        assert opt.state[x]["n_iter"] > 7
        outputs.append(x.detach())
    for result in outputs:
        torch.testing.assert_close(result, target, atol=3e-5, rtol=0)
    torch.testing.assert_close(*outputs, atol=3e-5, rtol=0)


@pytest.mark.gpu
def test_failed_cuda_kernels_use_original_updates_and_do_not_retry(monkeypatch):
    calls = []

    def fail(*args):
        calls.append(True)
        raise RuntimeError("test compile failure")

    monkeypatch.setattr(impl, "_ADD", fail)
    monkeypatch.setattr(impl, "_ADD_FAILED", False)
    x = torch.ones(7, device="cuda")
    y, alpha = x * 2, x.new_tensor(0.25)
    for _ in range(2):
        reference = x.clone().add_(y, alpha=float(alpha))
        torch.testing.assert_close(
            impl.add_device_scalar(x, y, alpha), reference, rtol=0, atol=0
        )
    assert calls == [True]


@pytest.mark.gpu
def test_device_updates_follow_the_callers_cuda_stream():
    from rgi_toolkit.optim._torch_lbfgs_cuda import add, direction

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(93, device="cuda", dtype=torch.float32)
        other = x.square()
        result = add(x.clone(), other, x.new_tensor(0.25))
        updated, _ = direction(
            result, [other], [x], [x.new_tensor(0.01)], x.new_tensor(1)
        )
        expected = result.clone()
    stream.synchronize()
    torch.testing.assert_close(expected, x + other * 0.25, rtol=0, atol=0)
    assert torch.isfinite(updated).all()
