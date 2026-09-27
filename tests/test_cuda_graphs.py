"""Graph replay must preserve optimizer history and compiled fallbacks."""

import pytest
import torch

from rgi_toolkit.optim import _torch_cg_gpu as impl


def _energy(coordinates, prepared):
    return 2 * (coordinates - prepared["reference"]).square().sum()


@pytest.mark.parametrize("failure", ["construction", "evaluation"])
def test_graph_failure_keeps_default_compilation(monkeypatch, failure):
    calls = {"graph": 0, "default": 0}

    def compile_mock(function, **options):
        if options.get("mode") == "reduce-overhead":
            calls["graph"] += 1
            if failure == "construction":
                raise RuntimeError("Graph compiler unavailable")

            def unavailable(*args):
                raise RuntimeError("Graph replay unavailable")

            return unavailable

        def default(*args):
            calls["default"] += 1
            return function(*args)

        return default

    monkeypatch.setattr(torch, "compile", compile_mock)
    objective = impl._compile_value_gradient(_energy, cuda_graphs=True)
    for offset in (1.0, 2.0):
        coordinates = torch.full((2, 3), offset)
        reference = torch.zeros_like(coordinates)
        gradient, value = objective(coordinates, {"reference": reference})
        torch.testing.assert_close(gradient, coordinates * 4)
        torch.testing.assert_close(value, coordinates.square().sum() * 2)
    assert calls == {"graph": 1, "default": 2}


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replay_preserves_retained_gradients_across_shapes(monkeypatch, dtype):
    replay = torch.cuda.CUDAGraph.replay
    replay_count = 0

    def counted_replay(self):
        nonlocal replay_count
        replay_count += 1
        return replay(self)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)
    monkeypatch.setattr(impl, "_COMPILE_DISABLED", False)
    monkeypatch.setattr(impl, "_CVG_BY_MODE", {})
    monkeypatch.setattr(impl, "_compile_failed", {i: False for i in range(4)})
    monkeypatch.setitem(impl._ENERGY_BY_MODE, 1, _energy)
    objective = impl._get_cvg(1)
    retained = []
    for count in (6, 9, 6):
        reference = torch.zeros((count, 3), device="cuda", dtype=dtype)
        for offset in (1.0, 2.0, 3.0, 4.0):
            coordinates = torch.full_like(reference, offset)
            gradient, value = objective(coordinates, {"reference": reference})
            retained.append((gradient, value, count, offset))
    torch.cuda.synchronize()
    assert replay_count > 0, (
        "The retention check must exercise actual CUDA Graph replay"
    )
    for gradient, value, count, offset in retained:
        torch.testing.assert_close(gradient, torch.full_like(gradient, 4 * offset))
        torch.testing.assert_close(value, value.new_tensor(2 * count * 3 * offset**2))
