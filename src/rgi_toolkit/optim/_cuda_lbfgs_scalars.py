"""Emulate Python-number and Tensor arithmetic without host reads."""

from typing import NamedTuple

import torch

from rgi_toolkit.optim._cuda_graph_ops import DeviceScalars


class Number(NamedTuple):
    value: torch.Tensor
    tensor: torch.Tensor

    def tensor_dtype(self):
        return torch.float32

    def coerce(self, other):
        if isinstance(other, Number):
            return other
        if isinstance(other, torch.Tensor):
            return type(self)(other.to(torch.float64), torch.ones_like(self.tensor))
        return type(self)(
            torch.full_like(self.value, other), torch.zeros_like(self.tensor)
        )

    def binary(self, other, operation):
        other = self.coerce(other)
        tensor = self.tensor | other.tensor
        host_value = operation(self.value, other.value)
        # Round each Tensor operation as on the CPU, including division and sqrt.
        a = self.value.to(self.tensor_dtype()).to(torch.float64)
        b = other.value.to(self.tensor_dtype()).to(torch.float64)
        tensor_value = operation(a, b).to(self.tensor_dtype()).to(torch.float64)
        if operation is torch.div:
            # Tensor.__rtruediv__ uses a reciprocal followed by multiplication.
            reciprocal = (1 / b).to(self.tensor_dtype()).to(torch.float64)
            reverse_value = (a * reciprocal).to(self.tensor_dtype()).to(torch.float64)
            tensor_value = torch.where(
                ~self.tensor & other.tensor, reverse_value, tensor_value
            )
        return type(self)(torch.where(tensor, tensor_value, host_value), tensor)

    def compare(self, other, operation):
        other = self.coerce(other)
        return torch.where(
            self.tensor | other.tensor,
            operation(
                self.value.to(self.tensor_dtype()), other.value.to(self.tensor_dtype())
            ),
            operation(self.value, other.value),
        )

    def __add__(self, other):
        return self.binary(other, torch.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self.binary(other, torch.sub)

    def __rsub__(self, other):
        return self.coerce(other).__sub__(self)

    def __mul__(self, other):
        return self.binary(other, torch.mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.binary(other, torch.div)

    def __rtruediv__(self, other):
        return self.coerce(other).__truediv__(self)

    def __neg__(self):
        return type(self)(-self.value, self.tensor)

    def abs(self):
        return type(self)(self.value.abs(), self.tensor)

    def square(self):
        return self * self

    def sqrt(self):
        value = self.value.sqrt()
        return type(self)(
            torch.where(
                self.tensor, value.to(self.tensor_dtype()).to(torch.float64), value
            ),
            self.tensor,
        )

    def to(self, dtype):
        return self.value.to(dtype)

    def __lt__(self, other):
        return self.compare(other, torch.lt)

    def __le__(self, other):
        return self.compare(other, torch.le)

    def __gt__(self, other):
        return self.compare(other, torch.gt)

    def __ge__(self, other):
        return self.compare(other, torch.ge)

    def __bool__(self):
        raise RuntimeError("Library scalars cannot be read on the host")

    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if function is torch.sqrt:
            return args[0].sqrt()
        if function is torch.where:
            predicate, a, b = args
            if not isinstance(a, Number):
                a = b.coerce(a)
            b = a.coerce(b)
            return type(a)(
                torch.where(predicate, a.value, b.value),
                torch.where(predicate, a.tensor, b.tensor),
            )
        if function in (torch.minimum, torch.maximum):
            a, b = args
            if not isinstance(a, Number):
                a = b.coerce(a)
            b = a.coerce(b)
            take_b = b < a if function is torch.minimum else b > a
            return torch.where(take_b, b, a)
        if function is torch.clamp:
            result = args[0]
            if kwargs.get("min") is not None:
                result = torch.maximum(result, result.coerce(kwargs["min"]))
            if kwargs.get("max") is not None:
                result = torch.minimum(result, result.coerce(kwargs["max"]))
            return result
        raise TypeError(f"Unsupported scalar operation: {function}")


class DoubleNumber(Number):
    def tensor_dtype(self):
        return torch.float64


class LibraryScalars(DeviceScalars):
    def __init__(self, builder, like):
        super().__init__(builder, like)
        self.number = DoubleNumber if like.dtype == torch.float64 else Number

    def scalar(self, value):
        return self.number(
            self.convert(value, torch.float64),
            self.boolean(isinstance(value, torch.Tensor)),
        )

    def host_value(self, value):
        return self.number(self.convert(value, torch.float64), self.boolean(False))
