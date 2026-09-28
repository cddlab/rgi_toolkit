"""Native CUDA control flow agrees with independent solver contracts."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def test_strong_wolfe_contracts(tmp_path):
    import json
    import warnings

    import numpy as np
    import scipy
    import torch
    from scipy.optimize import minimize
    from scipy.optimize._dcsrch import dcstep as scipy_dcstep
    from scipy.optimize._optimize import _line_search_wolfe12
    from torch.utils import _pytree

    from rgi_toolkit.optim._cg import run_cg, torch_cg
    from rgi_toolkit.optim._cg_linesearch import dcstep, strong_wolfe
    from rgi_toolkit.optim._cuda_graph import (
        FusedConditionalGraph as LeanConditionalGraph,
    )
    from rgi_toolkit.optim._cuda_graph_ops import DeviceCG, DeviceScalars, copy_tree

    output = tmp_path
    records = []
    _context_seed = torch.zeros(1, device="cuda")

    def record(name, **values):
        row = dict(case=name, **values)
        records.append(row)
        print(json.dumps(row), flush=True)
        (output / "wolfe-contracts.json").write_text(
            json.dumps(dict(scipy=scipy.__version__, tests=records), indent=2) + "\n"
        )

    for fp, dp in ((2.0, 0.5), (0.5, 0.5), (0.5, -0.5), (0.5, -1.5)):
        for bracket in (False, True):
            args = (0.0, 1.0, -1.0, 2.0, 0.7, 1.0, 0.5, fp, dp, bracket, 0.0, 4.0)
            expected = scipy_dcstep(*args)
            graph = LeanConditionalGraph()
            s = DeviceScalars(graph, torch.zeros(1, device="cuda", dtype=torch.float64))
            graph.begin()
            converted = [
                s.boolean(x) if isinstance(x, bool) else s.scalar(x) for x in args
            ]
            result = dcstep(s, *converted)
            graph.finish(result)
            actual = [x.item() for x in graph.replay()]
            np.testing.assert_allclose(actual[:7], expected[:7], rtol=1e-12, atol=1e-12)
            assert actual[-1] == bool(expected[-1])
            record(
                "dcstep",
                fp=fp,
                dp=dp,
                bracket=bracket,
                max_difference=float(
                    np.max(np.abs(np.array(actual[:7]) - expected[:7]))
                ),
            )
            graph.close()

    for reject in (False, True):
        graph = LeanConditionalGraph()
        x = torch.zeros(1, device="cuda", dtype=torch.float64)
        b = DeviceCG(graph, x)
        s = b.s
        graph.begin()
        d = torch.ones_like(x)
        vg = torch.func.grad_and_value(lambda x: (x[0] - 3.0) ** 2)
        origin = b.evaluate(vg, x, x, d, s.scalar(0), s.integer(-1))._replace(
            alpha=s.scalar(float("nan"))
        )

        def evaluate(alpha, old):
            return s.cond(
                alpha == old.alpha,
                lambda _: old,
                lambda _: b.evaluate(
                    vg, x + b.cast(alpha, x) * d, x, d, alpha, old.nfev
                ),
                None,
            )

        result = strong_wolfe(
            s,
            evaluate,
            lambda t: t.alpha < 2.8 if reject else s.boolean(True),
            origin,
            origin.f,
            origin.slope,
            s.scalar(12),
            s.scalar(1e-100),
            s.scalar(1e100),
        )
        graph.finish(result)
        actual, success, phase = graph.replay()
        expected = _line_search_wolfe12(
            lambda x: (x[0] - 3.0) ** 2,
            lambda x: 2 * (x - 3),
            np.zeros(1),
            np.ones(1),
            np.array([-6.0]),
            9.0,
            12.0,
            c1=1e-4,
            c2=0.4,
            amin=1e-100,
            amax=1e100,
            extra_condition=(lambda alpha, *_: alpha < 2.8)
            if reject
            else (lambda *_: True),
        )
        assert bool(success) and int(phase) == (2 if reject else 1)
        assert abs(float(actual.alpha) - expected[0]) <= 1e-12
        record(
            "search_order",
            reject_primary=reject,
            alpha=float(actual.alpha),
            phase=int(phase),
        )
        graph.close()

    cases = [
        (
            "exhausted_zoom",
            torch.zeros(3, dtype=torch.float64),
            lambda x: ((x - 100) ** 2).sum(),
            dict(more_maxiter=0, wolfe_maxiter=1),
        ),
        (
            "nonfinite_initial",
            torch.zeros(3, dtype=torch.float64),
            lambda x: torch.sqrt(x[0]),
            {},
        ),
        (
            "rounded_no_progress",
            torch.full((3,), 1e16, dtype=torch.float64),
            lambda x: x.sum(),
            {},
        ),
        (
            "nonfinite_trial",
            torch.ones(3, dtype=torch.float64),
            lambda x: (torch.sqrt(x[0]) - 0.5) ** 2,
            {},
        ),
    ]
    for name, initial, energy, kwargs in cases:
        vg = torch.func.grad_and_value(energy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected, host = torch_cg(vg, initial, 100, **kwargs)
        x = initial.cuda()
        graph = LeanConditionalGraph()
        graph.begin()
        result = run_cg(DeviceCG(graph, x), vg, x, 100, **kwargs)
        graph.finish(result)
        actual, state = graph.replay()
        np.testing.assert_allclose(
            actual.cpu().numpy(), expected.numpy(), rtol=1e-9, atol=1e-10
        )
        assert int(state.info.status) == int(host.info.status)
        assert int(state.info.nit) == int(host.info.nit)
        record(
            name,
            status=int(state.info.status),
            iterations=int(state.info.nit),
            evaluations=int(state.info.nfev),
        )
        graph.close()

    for diagonal in ((1.0, 1.0, 1.0), (16.0, 1.0, 1.0)):
        target, diagonal = np.array([0.5, -0.25, 0.75]), np.array(diagonal)
        initial = target + [1.0, -1.0, 0.5]
        target_cpu, diagonal_cpu = torch.tensor(target), torch.tensor(diagonal)
        target_gpu, diagonal_gpu = target_cpu.cuda(), diagonal_cpu.cuda()

        def energy(x):
            selected_target, selected_diagonal = (
                (target_gpu, diagonal_gpu) if x.is_cuda else (target_cpu, diagonal_cpu)
            )
            return 0.5 * ((x - selected_target) ** 2 * selected_diagonal).sum() + 0.25

        expected_trace = []
        minimize(
            lambda x: 0.5 * np.sum((x - target) ** 2 * diagonal) + 0.25,
            initial,
            jac=lambda x: (x - target) * diagonal,
            method="CG",
            callback=lambda x: expected_trace.append(x.copy()),
            options=dict(gtol=1e-5, maxiter=100),
        )
        _, initial_state = torch_cg(
            torch.func.grad_and_value(energy), torch.tensor(initial), 0
        )
        static = torch.tensor(initial, device="cuda")
        state = _pytree.tree_map(
            lambda x: (
                None
                if x is None
                else torch.as_tensor(
                    bool(x) if isinstance(x, np.bool_) else x, device="cuda"
                )
            ),
            initial_state,
        )
        graph = LeanConditionalGraph()
        graph.begin()
        result = run_cg(
            DeviceCG(graph, static),
            torch.func.grad_and_value(energy),
            static,
            1,
            state=state,
        )
        graph.finish(result)
        actual_trace = []
        for _ in range(100):
            value, updated = graph.replay()
            torch.cuda.synchronize()
            actual_trace.append(value.cpu().numpy().copy())
            if not bool(updated.valid):
                break
            static.copy_(value)
            copy_tree(state, updated)
        np.testing.assert_allclose(actual_trace, expected_trace, rtol=1e-9, atol=1e-10)
        record(
            "accepted_trajectory",
            diagonal=diagonal.tolist(),
            iterations=len(actual_trace),
            max_difference=float(
                np.max(np.abs(np.array(actual_trace) - expected_trace))
            ),
        )
        graph.close()
    print("DEVICE_WOLFE_CONTRACTS_PASSED", len(records), flush=True)


@pytest.mark.parametrize("iterations", [0, 1, 2, 100])
@pytest.mark.parametrize("objective", ["quadratic", "rosenbrock"])
def test_lbfgs_matches_native_limits_and_solution(iterations, objective):
    from rgi_toolkit.optim._cuda_graph import FusedConditionalGraph
    from rgi_toolkit.optim._cuda_graph_ops import DeviceScalars
    from rgi_toolkit.optim._cuda_lbfgs import run_lbfgs

    initial = torch.tensor([-1.2, 1.0], dtype=torch.float64, device="cuda")

    def energy(x):
        if objective == "quadratic":
            return ((x - 0.3) ** 2).sum()
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    reference = initial.clone().requires_grad_()
    native = torch.optim.LBFGS(
        [reference],
        max_iter=iterations,
        tolerance_grad=1e-5,
        line_search_fn="strong_wolfe",
    )

    def closure():
        native.zero_grad()
        loss = energy(reference)
        loss.backward()
        return loss

    native.step(closure)
    graph = FusedConditionalGraph()
    try:
        graph.begin()
        result = run_lbfgs(
            DeviceScalars(graph, initial),
            torch.func.grad_and_value(energy),
            initial,
            iterations,
        )
        graph.finish(result)
        state = graph.replay()
        torch.testing.assert_close(state.x, reference.detach(), atol=1e-5, rtol=1e-5)
        assert int(state.iteration) == native.state[reference]["n_iter"]
        assert int(state.evaluations) == native.state[reference]["func_evals"]
    finally:
        graph.close()


def test_library_scalar_capture_keeps_float32_rounding():
    from rgi_toolkit.optim._cuda_graph import FusedConditionalGraph
    from rgi_toolkit.optim._cuda_lbfgs_scalars import LibraryScalars

    x = torch.ones((), device="cuda", dtype=torch.float32)
    graph = FusedConditionalGraph()
    try:
        graph.begin()
        s = LibraryScalars(graph, x)
        tensor_value = (s.scalar(x) + 2**-24) - s.scalar(x)
        python_value = (s.host_value(x) + 2**-24) - s.host_value(x)
        graph.finish((tensor_value, python_value))
        tensor_result, python_result = graph.replay()
        assert float(tensor_result.value) == 0.0
        assert bool(tensor_result.tensor)
        assert float(python_result.value) == 2**-24
        assert not bool(python_result.tensor)
    finally:
        graph.close()


def test_library_cubic_retains_native_reverse_division():
    from torch.optim.lbfgs import _cubic_interpolate

    from rgi_toolkit.optim._cuda_lbfgs import Trial, cubic
    from rgi_toolkit.optim._cuda_lbfgs_scalars import LibraryScalars

    # Near-flat intervals amplify a one-ulp change in reciprocal arithmetic.
    t = torch.tensor(0.8939066529273987)
    f1, f2 = 4.556903839111328, 4.556352138519287
    g1, g2 = torch.tensor(-0.0006160561461001635), torch.tensor(-0.0006160188931971788)
    expected = _cubic_interpolate(0, f1, g1, t, f2, g2, (t + 0.01 * t, t * 10))
    s = LibraryScalars(None, t.cuda())
    first = Trial(s.scalar(0), s.host_value(f1), None, s.scalar(g1.cuda()))
    second = Trial(s.scalar(t.cuda()), s.host_value(f2), None, s.scalar(g2.cuda()))
    actual = cubic(first, second, (second.t + 0.01 * second.t, second.t * 10))
    assert float(actual.value) == float(expected)


def _bond_spec():
    from rgi_toolkit.spec import BondArrays, RestraintSpec

    return RestraintSpec(
        n_active=2,
        active_sites=np.arange(2),
        bond=BondArrays(
            np.array([[0, 1]]),
            np.array([1.5]),
            np.zeros(1),
            np.ones(1),
            np.zeros(1),
            np.ones(1),
        ),
        conf_start_sigma=float("inf"),
    )


@pytest.mark.parametrize(
    "method,search", [("CG", "strong-wolfe"), ("CG", "armijo"), ("l-bfgs", None)]
)
def test_public_replay_refreshes_inputs_batch_and_options(method, search):
    from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

    optimizer = TorchRestraintOptimizer(_bond_spec(), method=method, line_search=search)
    start = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 8.0, 7.0]]], device="cuda"
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = optimizer.minimize(start.clone())
        assert optimizer._device_graph_cache is not None
        graph = optimizer._device_graph_cache[1]
        # Reuse the graph with a different bond direction and absolute position.
        second = start.flip(-1).clone() + 5
        expected = second.clone()
        optimizer.minimize(second)
        assert optimizer._device_graph_cache[1] is graph
        torch.testing.assert_close(second[..., 2, :], expected[..., 2, :])
        torch.testing.assert_close(
            (second[..., 0, :] - second[..., 1, :]).norm(dim=-1),
            torch.tensor([1.5], device="cuda"),
            atol=1e-5,
            rtol=0,
        )
        # Changed shape and iteration limit must rebuild instead of reusing constants.
        batch = optimizer.minimize(start.repeat(2, 1, 1), max_iter=1)
        assert optimizer._device_graph_cache[1] is not graph
        assert batch.shape == (2, 3, 3)
    torch.testing.assert_close(
        (first[0, 0] - first[0, 1]).norm(),
        torch.tensor(1.5, device="cuda"),
        atol=1e-5,
        rtol=0,
    )
    if search is not None:
        _, info = optimizer.minimize(start.clone(), return_info=True)
        assert isinstance(info.nit, int)
        assert info.grad_norm < 1e-5
    optimizer._ensure(start.device, torch.float64)
    assert optimizer._device_graph_cache is None


def test_compile_failure_falls_back_once_without_dispatch_leak(monkeypatch):
    from rgi_toolkit.optim import _cuda_graph, _cuda_minimize
    from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

    calls = []

    def fail(self, *args):
        calls.append(True)
        raise RuntimeError("deliberate capture failure")

    monkeypatch.setattr(_cuda_graph.FusedConditionalGraph, "finish", fail)
    optimizer = TorchRestraintOptimizer(_bond_spec())
    start = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda")
    for _ in range(2):
        result = optimizer.minimize(start.clone())
        assert optimizer._device_graph_cache is None
        torch.testing.assert_close(
            (result[0] - result[1]).norm(),
            torch.tensor(1.5, device="cuda"),
            atol=1e-5,
            rtol=0,
        )
    assert calls == [True]
    # Ordinary tensor operations remain outside the recorder after failure.
    assert torch.ones(1).item() == 1
    monkeypatch.setenv("RGI_DISABLE_COMPILE", "1")
    assert not _cuda_minimize.eligible(optimizer, start, 100, True)


def test_replay_does_not_read_host_scalars():
    from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

    start = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda")
    optimizer = TorchRestraintOptimizer(_bond_spec())
    optimizer.minimize(start.clone())
    assert optimizer._device_graph_cache is not None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        optimizer.minimize(start.clone())
    assert not any(
        event.key == "aten::_local_scalar_dense" for event in profile.key_averages()
    )


def test_constant_gradient_exports_and_nondefault_stream():
    from rgi_toolkit.optim._cg import run_cg
    from rgi_toolkit.optim._cuda_graph import FusedConditionalGraph
    from rgi_toolkit.optim._cuda_graph_ops import DeviceCG
    from rgi_toolkit.optim.info import CGStatus

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.full((3,), 1e16, dtype=torch.float64, device="cuda")
        graph = FusedConditionalGraph()
        try:
            graph.begin()
            result = run_cg(
                DeviceCG(graph, x), torch.func.grad_and_value(lambda y: y.sum()), x, 100
            )
            graph.finish(result)
            actual, state = graph.replay()
            torch.testing.assert_close(actual, x, rtol=0, atol=0)
            assert int(state.info.status) == CGStatus.NO_PROGRESS
            assert graph.materialized_exports > 0
        finally:
            graph.close()


def test_lbfgs_nonfinite_trials_keep_native_evaluation_budget():
    from rgi_toolkit.optim._cuda_graph import FusedConditionalGraph
    from rgi_toolkit.optim._cuda_graph_ops import DeviceScalars
    from rgi_toolkit.optim._cuda_lbfgs import run_lbfgs

    x = torch.zeros(2, dtype=torch.float64, device="cuda")

    def energy(y):
        return torch.sqrt(y[0])

    expected = x.clone().requires_grad_()
    optimizer = torch.optim.LBFGS([expected], max_iter=1, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        value = energy(expected)
        value.backward()
        return value

    optimizer.step(closure)
    graph = FusedConditionalGraph()
    try:
        graph.begin()
        result = run_lbfgs(
            DeviceScalars(graph, x), torch.func.grad_and_value(energy), x, 1
        )
        graph.finish(result)
        state = graph.replay()
        assert int(state.iteration) == optimizer.state[expected]["n_iter"]
        assert int(state.evaluations) == optimizer.state[expected]["func_evals"]
    finally:
        graph.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3, 3), (1, 3, 3), (20, 3, 3)])
@pytest.mark.parametrize("kind", ["svd", "eigh"])
def test_captured_geometry_decompositions(dtype, shape, kind):
    from rgi_toolkit.optim._cuda_linalg import eigh3, release_stream, svd3

    generator = torch.Generator(device="cuda").manual_seed(42)
    initial = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    a = initial if kind == "svd" else initial.mT @ initial
    if shape[0] == 20:
        a[0] = 0
        a[1] = torch.eye(3, device="cuda", dtype=dtype)
        a[2] = 1
    flag = torch.zeros((), device="cuda", dtype=torch.bool)
    function = svd3 if kind == "svd" else eigh3
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function(a, flag)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            flag.zero_()
            result = function(a, flag)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    assert not bool(flag)
    tolerance = 3e-6 if dtype == torch.float32 else 2e-13
    if kind == "svd":
        u, values, v = result
        reference = torch.linalg.svd(a)
        torch.testing.assert_close(values, reference[1], atol=tolerance, rtol=tolerance)
        reconstruction = (u * values[..., None, :]) @ v
    else:
        values, vectors = result
        reference = torch.linalg.eigh(a)
        torch.testing.assert_close(values, reference[0], atol=tolerance, rtol=tolerance)
        reconstruction = (vectors * values[..., None, :]) @ vectors.mT
    torch.testing.assert_close(reconstruction, a, atol=tolerance, rtol=tolerance)
    release_stream(a.device.index, stream.cuda_stream)


def test_geometry_failure_retries_original_solver(monkeypatch):
    from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

    optimizer = TorchRestraintOptimizer(_bond_spec())
    initial = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda")
    optimizer.minimize(initial.clone())
    graph = optimizer._device_graph_cache[1].graph
    graph.uses_linalg = True
    replay = graph.replay

    def failure():
        result = replay()
        graph.library_failure.fill_(True)
        return result

    monkeypatch.setattr(graph, "replay", failure)
    result = optimizer.minimize(initial.clone())
    assert optimizer._device_graph_cache is None
    assert len(optimizer._device_graph_failures) == 1
    torch.testing.assert_close(
        (result[1] - result[0]).norm(), result.new_tensor(1.5), atol=1e-5, rtol=0
    )
