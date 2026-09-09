"""Scalar control flow for the shared Torch/JAX CG state machines."""

from __future__ import annotations

import numpy as np


class HostScalars:
    """Host-controlled Torch optimization, with NumPy scalar arithmetic."""

    xp = np

    def scalar(self, value):
        return np.float64(float(value))

    def integer(self, value):
        return int(value)

    def boolean(self, value):
        return np.bool_(value)

    def cond(self, predicate, yes, no, operand):
        return yes(operand) if predicate else no(operand)

    def loop(self, condition, body, state):
        with np.errstate(all="ignore"):
            while condition(state):
                state = body(state)
        return state


class JaxScalars:
    """The same transitions, with device-resident scalars and traced branches."""

    def __init__(self, like):
        import jax
        import jax.numpy as jnp

        self.xp = jnp
        self.lax = jax.lax
        self.dtype = like.dtype

    def scalar(self, value):
        return self.xp.asarray(value, dtype=self.dtype)

    def integer(self, value):
        return self.xp.asarray(value, dtype=self.xp.int32)

    def boolean(self, value):
        return self.xp.asarray(value, dtype=bool)

    def cond(self, predicate, yes, no, operand):
        return self.lax.cond(predicate, yes, no, operand)

    def loop(self, condition, body, state):
        return self.lax.while_loop(condition, body, state)
