"""Capture 3x3 cuSOLVER decompositions without per-fit host status reads.

The SVD uses the same gesvdjBatched tolerance and 400-sweep limit as PyTorch 2.8.
Eigenproblems use the small-matrix Jacobi driver syevjBatched, also for one matrix.
Status is accumulated on the GPU. The adapter checks it once after the solve and
retries the ordinary optimizer on failure, retaining PyTorch's recovery path.
No energy derivative is implemented here: FX replaces already differentiated,
stop-gradient geometry calls, and all restraint gradients remain autodiff.
"""

import ctypes as ct
from importlib.util import find_spec
from pathlib import Path

import torch


class Solver:
    def __init__(self, device, stream, dtype):
        # Match PyTorch's wheel, not an unrelated system CUDA toolkit. SONAME-only
        # loading before Torch's first SVD can otherwise mix cuSOLVER/cuBLAS ABIs.
        library = "libcusolver.so.11"
        try:
            package = find_spec("nvidia.cusolver")
        except ModuleNotFoundError:
            package = None
        if package is not None:
            for directory in package.submodule_search_locations or ():
                candidate = Path(directory) / "lib" / library
                if candidate.is_file():
                    library = str(candidate)
                    break
        self.lib = ct.CDLL(library)
        self.handle = ct.c_void_p()
        self.svd = ct.c_void_p()
        self.eigh = ct.c_void_p()
        self.call("cusolverDnCreate", ct.byref(self.handle))
        self.call("cusolverDnSetStream", self.handle, ct.c_void_p(stream))
        self.call("cusolverDnCreateGesvdjInfo", ct.byref(self.svd))
        self.call(
            "cusolverDnXgesvdjSetTolerance",
            self.svd,
            ct.c_double(torch.finfo(dtype).eps),
        )
        self.call("cusolverDnXgesvdjSetMaxSweeps", self.svd, 400)
        self.call("cusolverDnXgesvdjSetSortEig", self.svd, 1)
        self.call("cusolverDnCreateSyevjInfo", ct.byref(self.eigh))
        self.call("cusolverDnXsyevjSetSortEig", self.eigh, 1)
        self.prefix = "S" if dtype == torch.float32 else "D"

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with cuSOLVER status {status}")

    def close(self):
        for name, value in (
            ("cusolverDnDestroyGesvdjInfo", self.svd),
            ("cusolverDnDestroySyevjInfo", self.eigh),
            ("cusolverDnDestroy", self.handle),
        ):
            if value.value:
                self.call(name, value)
                value.value = None

    def __del__(self):
        if hasattr(self, "handle"):
            try:
                self.close()
            except Exception:
                pass


_SOLVERS = {}


def _solver(device, stream, dtype):
    key = (device, stream, dtype)
    if key not in _SOLVERS:
        _SOLVERS[key] = Solver(device, stream, dtype)
    return _SOLVERS[key]


def release_stream(device, stream):
    for key in list(_SOLVERS):
        if key[:2] == (device, stream):
            _SOLVERS.pop(key).close()


def _prepare(a):
    if a.shape[-2:] != (3, 3) or a.dtype not in (torch.float32, torch.float64):
        raise ValueError("Captured geometry requires real 3x3 matrices")
    solver = _solver(
        a.device.index, torch.cuda.current_stream(a.device).cuda_stream, a.dtype
    )
    matrix = a.mT.contiguous().clone()
    values = torch.empty(a.shape[:-1], device=a.device, dtype=a.dtype)
    info = torch.empty(a.shape[:-2], device=a.device, dtype=torch.int32)
    return solver, matrix, values, info


def _pointer(tensor):
    return ct.c_void_p(tensor.data_ptr())


def _flag(info, failure):
    bad = info != 0
    failure.logical_or_(bad.any())
    return bad


@torch.library.custom_op("rgi_toolkit::cuda_svd3", mutates_args={"failure"})
def svd3(
    a: torch.Tensor, failure: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    solver, matrix, values, info = _prepare(a)
    left, right = torch.empty_like(matrix), torch.empty_like(matrix)
    length = ct.c_int()
    args = (
        solver.handle,
        1,
        3,
        3,
        _pointer(matrix),
        3,
        _pointer(values),
        _pointer(left),
        3,
        _pointer(right),
        3,
    )
    name = f"cusolverDn{solver.prefix}gesvdjBatched"
    solver.call(name + "_bufferSize", *args, ct.byref(length), solver.svd, info.numel())
    work = a.new_empty(length.value)
    solver.call(
        name, *args, _pointer(work), length, _pointer(info), solver.svd, info.numel()
    )
    bad = _flag(info, failure)
    nan = float("nan")
    return (
        torch.where(bad[..., None, None], nan, left.mT).contiguous(),
        torch.where(bad[..., None], nan, values),
        torch.where(bad[..., None, None], nan, right).contiguous(),
    )


@svd3.register_fake
def _fake_svd3(a, failure):
    return (
        torch.empty(a.shape, device=a.device, dtype=a.dtype),
        torch.empty(a.shape[:-1], device=a.device, dtype=a.dtype),
        torch.empty(a.shape, device=a.device, dtype=a.dtype),
    )


@torch.library.custom_op("rgi_toolkit::cuda_eigh3", mutates_args={"failure"})
def eigh3(
    a: torch.Tensor, failure: torch.Tensor, upper: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    solver, matrix, values, info = _prepare(a)
    length = ct.c_int()
    args = (solver.handle, 1, int(upper), 3, _pointer(matrix), 3, _pointer(values))
    name = f"cusolverDn{solver.prefix}syevjBatched"
    extra = (solver.eigh, info.numel())
    solver.call(name + "_bufferSize", *args, ct.byref(length), *extra)
    work = a.new_empty(length.value)
    solver.call(name, *args, _pointer(work), length, _pointer(info), *extra)
    bad = _flag(info, failure)
    return (
        torch.where(bad[..., None], float("nan"), values),
        torch.where(bad[..., None, None], float("nan"), matrix.mT).contiguous(),
    )


@eigh3.register_fake
def _fake_eigh3(a, failure, upper=False):
    return (
        torch.empty(a.shape[:-1], device=a.device, dtype=a.dtype),
        torch.empty(a.shape, device=a.device, dtype=a.dtype),
    )
