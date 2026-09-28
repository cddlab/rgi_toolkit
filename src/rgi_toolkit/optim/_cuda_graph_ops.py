"""CUDA graph primitives and scalar operations for shared solver control flow."""

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc
from torch.utils import _pytree


def checked(result):
    if int(result[0]):
        raise RuntimeError(str(result))
    return result[1] if len(result) == 2 else result[1:]


_MODULES = {}


def condition_kernel():
    device = torch.cuda.current_device()
    if device not in _MODULES:
        major, minor = torch.cuda.get_device_capability()
        code = b"""typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned int);
extern "C" __global__ void set_condition(cudaGraphConditionalHandle h, const bool *p) {
  cudaGraphSetConditional(h, *p ? 1 : 0);
}
"""
        program = checked(nvrtc.nvrtcCreateProgram(code, b"condition.cu", 0, [], []))
        options = [
            f"--gpu-architecture=compute_{major}{minor}".encode(),
            b"--std=c++17",
        ]
        try:
            status = nvrtc.nvrtcCompileProgram(program, len(options), options)
            if int(status[0]):
                log = b" " * checked(nvrtc.nvrtcGetProgramLogSize(program))
                checked(nvrtc.nvrtcGetProgramLog(program, log))
                raise RuntimeError(log.decode())
            ptx = b" " * checked(nvrtc.nvrtcGetPTXSize(program))
            checked(nvrtc.nvrtcGetPTX(program, ptx))
            module = checked(cu.cuModuleLoadData(ptx))
            kernel = checked(cu.cuModuleGetFunction(module, b"set_condition"))
        finally:
            checked(nvrtc.nvrtcDestroyProgram(program))
        _MODULES[device] = module, kernel
    return _MODULES[device][1]


def tree_map(function, *trees):
    return _pytree.tree_map(function, *trees)


def empty_tree(tree):
    return tree_map(
        lambda x: torch.empty_like(x) if isinstance(x, torch.Tensor) else x, tree
    )


def copy_tree(destination, source):
    # All reads must precede writes: loop-carried fields can swap or alias.
    staged = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, source)

    def copy(left, right):
        if isinstance(left, torch.Tensor):
            left.copy_(right)
        else:
            assert left is None and right is None
        return left

    return tree_map(copy, destination, staged)


class ArrayScalars:
    inf = float("inf")

    def __init__(self, owner):
        self.owner = owner

    def tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, bool):
            return self.owner.boolean(value)
        if isinstance(value, int):
            return self.owner.integer(value)
        return self.owner.scalar(value)

    def where(self, condition, yes, no):
        return torch.where(condition, self.tensor(yes), self.tensor(no))

    def maximum(self, a, b):
        return torch.maximum(self.tensor(a), self.tensor(b))

    def minimum(self, a, b):
        return torch.minimum(self.tensor(a), self.tensor(b))

    def clip(self, a, lo, hi):
        return self.minimum(self.maximum(a, lo), hi)

    sqrt = staticmethod(torch.sqrt)
    sign = staticmethod(torch.sign)
    isfinite = staticmethod(torch.isfinite)


class DeviceScalars:
    def __init__(self, builder, like):
        self.builder = builder
        self.device = like.device
        self.xp = ArrayScalars(self)

    def convert(self, value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype, device=self.device)
        return torch.full((), value, dtype=dtype, device=self.device)

    def scalar(self, value):
        return self.convert(value, torch.float64)

    def integer(self, value):
        return self.convert(value, torch.int64)

    def boolean(self, value):
        return self.convert(value, torch.bool)

    def cond(self, predicate, yes, no, operand):
        return self.builder.cond(predicate, yes, no, operand)

    def select(self, predicate, yes, no, operand):
        return tree_map(
            lambda a, b: None if a is None else self.xp.where(predicate, a, b),
            yes(operand),
            no(operand),
        )

    def loop(self, condition, body, state):
        return self.builder.loop(condition, body, state)


class DeviceCG:
    prepare = None

    def __init__(self, builder, like):
        self.s = DeviceScalars(builder, like)
        self.finfo = torch.finfo(like.dtype)

    def point(self, x, alpha, direction):
        # Native CG rounds the multiplication before adding it to coordinates.
        return self.s.builder.call(
            lambda a, t, d: a + t.to(a.dtype) * d,
            x,
            alpha,
            direction,
            native=True,
        )

    def fused_direction(self, g, old_g, old_d, denominator):
        from rgi_toolkit.optim._torch_fused import _direction

        return self.s.builder.call(_direction, g, old_g, old_d, denominator)

    def cast(self, value, like):
        return value.to(dtype=like.dtype)

    def dot(self, a, b):
        return self.s.scalar((a * b).sum())

    def same_point(self, a, b):
        return (a == b).all()

    def evaluate(self, vg, x, xbase, d, alpha, count, cache=None):
        from rgi_toolkit.optim._cg import Trial

        assert self.prepare is None
        gradient, value = vg(x)
        gradient, value = gradient.detach(), value.detach()
        from rgi_toolkit.optim._torch_fused import _trial_values

        statistics = self.s.builder.call(_trial_values, value, gradient, d, x, xbase)
        return Trial(
            alpha,
            x,
            self.s.scalar(statistics[0]),
            gradient,
            self.s.scalar(statistics[1]),
            self.s.scalar(statistics[2]),
            self.s.scalar(statistics[3]),
            statistics[4].to(torch.bool),
            statistics[5].to(torch.bool),
            count + 1,
            cache,
        )
