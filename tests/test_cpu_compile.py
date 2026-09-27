"""Default CPU compilation, explicit opt-out, and isolated fallback state."""

from types import SimpleNamespace

import pytest
import torch

from rgi_toolkit import AtomRecord, CombinedRestraints
from rgi_toolkit.config import RestraintsConfig
from rgi_toolkit.optim import _torch_cg_gpu as compiled
from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

pytestmark = pytest.mark.cpu_compile


def _distance_restraint(compile_cpu=None, custom=False, method="CG"):
    atoms = [AtomRecord("A", i + 1, i) for i in range(2)]
    config = {"gpu": False, "method": method}
    if compile_cpu is not None:
        config["compile_cpu"] = compile_cpu
    if custom:
        config["custom_restraints_config"] = [
            {
                "energy": "(distance(A, B) - 2)**2",
                "selections": {"A": "index 0", "B": "index 1"},
                "start_sigma": 1.0,
            }
        ]
    else:
        config["distance_restraints_config"] = [
            {
                "atom_selection1": "index 0",
                "atom_selection2": "index 1",
                "harmonic": {"target_distance": 2.0},
            }
        ]
    restraint = CombinedRestraints()
    restraint.setup(SimpleNamespace(iter_atoms=lambda: iter(atoms)), config=config)
    return restraint


def test_cpu_compile_defaults_on_and_coerces_false_strings():
    assert RestraintsConfig().compile_cpu
    assert RestraintsConfig.from_dict(None).compile_cpu
    assert RestraintsConfig.from_dict({}).compile_cpu
    assert not RestraintsConfig.from_dict({"compile_cpu": False}).compile_cpu
    assert not RestraintsConfig.from_dict({"compile_cpu": "false"}).compile_cpu
    assert RestraintsConfig.from_dict({"compile_cpu": "true"}).compile_cpu


@pytest.mark.parametrize("setting", [False, "false"])
def test_cpu_compile_can_be_disabled(monkeypatch, setting):
    def unexpected(*args, **kwargs):
        pytest.fail("An explicit opt-out must bypass CPU compilation")

    monkeypatch.setattr(compiled, "_get_cvg", unexpected)
    coords = torch.tensor([[0.0, 0, 0], [7.0, 0, 0]])
    restraint = _distance_restraint(compile_cpu=setting)
    restraint.minimize(coords, 0, 0.0)
    assert not restraint._optimizer.compile_cpu
    assert float(torch.linalg.norm(coords[0] - coords[1])) == pytest.approx(2, abs=1e-5)


def test_cpu_compile_failure_falls_back_without_disabling_cuda(monkeypatch):
    calls = []

    def fail(*args):
        calls.append(True)
        raise RuntimeError("CPU compiler unavailable")

    # Exercise dispatch failure without invoking a compiler, even in eager CI.
    monkeypatch.setattr(compiled, "_COMPILE_DISABLED", False)
    monkeypatch.setattr(compiled, "_CPU_CVG_BY_MODE", {0: fail})
    monkeypatch.setattr(compiled, "_cpu_compile_failed", dict.fromkeys(range(4), False))
    cuda_flags = compiled._compile_failed.copy()
    for _ in range(2):
        coords = torch.tensor([[0.0, 0, 0], [7.0, 0, 0]])
        _distance_restraint().minimize(coords, 0, 0.0)
        assert float(torch.linalg.norm(coords[0] - coords[1])) == pytest.approx(
            2, abs=1e-5
        )
    assert calls == [True]
    assert compiled._cpu_compile_failed[0]
    assert compiled._compile_failed == cuda_flags


@pytest.mark.parametrize("case", ["geometry", "rmsd"])
def test_cpu_compiled_objective_matches_eager_gradient(case):
    from tests.test_backend_parity import _make_spec, _positions
    from tests.test_optim import _require_python_dev_headers, _rmsd_spec

    _require_python_dev_headers()
    spec, positions = (
        (_make_spec(), _positions()) if case == "geometry" else _rmsd_spec()
    )
    x = torch.tensor(positions, dtype=torch.float64)
    optimizer = TorchRestraintOptimizer(spec)
    assert optimizer.compile_cpu
    optimizer._ensure(x.device, x.dtype)
    prepared = optimizer._gated_prepared(0.0)
    eager = torch.func.grad_and_value(compiled._energy)
    cvg = compiled._get_cvg(0, device_type="cpu")
    assert cvg is not None
    expected_g, expected_f = eager(x, prepared)
    actual_g, actual_f = cvg(x, prepared)
    torch.testing.assert_close(actual_f, expected_f, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual_g, expected_g, rtol=1e-9, atol=1e-9)
    initial = optimizer.energy(x)
    optimizer.minimize(x, sigma=0.0)
    assert optimizer.energy(x) < initial
    assert not compiled._cpu_compile_failed[0]


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("method", ["CG", "l-bfgs"])
def test_public_cpu_compile_preserves_target_dtype_and_custom_gate(custom, method):
    from tests.test_optim import _require_python_dev_headers

    _require_python_dev_headers()
    restraint = _distance_restraint(custom=custom, method=method)
    for dtype in (torch.float32, torch.float64):
        coords = torch.tensor([[0.0, 0, 0], [7.0, 0, 0]], dtype=dtype)
        if custom:
            restraint.minimize(coords, 0, 2.0)
            assert float(torch.linalg.norm(coords[0] - coords[1])) == 7.0
        restraint.minimize(coords, 1, 0.0)
        assert float(torch.linalg.norm(coords[0] - coords[1])) == pytest.approx(
            2, abs=1e-5
        )
        assert restraint._optimizer.compile_cpu
        assert restraint._optimizer._dtype == dtype
        if custom:
            assert all(
                v is not False for v in restraint._optimizer._custom_cvg.values()
            )
        else:
            assert not compiled._cpu_compile_failed[0]
