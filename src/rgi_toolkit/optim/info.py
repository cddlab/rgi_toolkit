"""Framework-independent CG diagnostics; named tuples are also native JAX pytrees."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, NamedTuple


class CGStatus(IntEnum):
    """Termination of one minimization, not necessarily a stationary point."""

    INACTIVE = 0
    CONVERGED = 1
    MAX_ITER = 2
    LINE_SEARCH_FAILED = 3
    NONFINITE = 4
    NO_PROGRESS = 5


class CGInfo(NamedTuple):
    """Diagnostics for the entire batch and all neighbor-list blocks.

    ``nit`` counts accepted CG iterations. ``nfev`` and ``njev`` count actual
    objective/gradient evaluations, including rejected trials. ``grad_norm`` is
    the infinity norm of the gradient used by CG, including RGI modifications.
    Fields are ordinary scalars on Torch and scalar arrays under JAX tracing.
    """

    status: Any
    nit: Any
    nfev: Any
    njev: Any
    fun: Any
    grad_norm: Any


def inactive_info():
    """No objective was evaluated; the inactive gated objective is zero."""
    return CGInfo(CGStatus.INACTIVE, 0, 0, 0, 0.0, 0.0)
