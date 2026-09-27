"""Small CUDA reductions used by the host-controlled CG solver."""

from __future__ import annotations

import logging

import torch

from rgi_toolkit.optim._torch_cg_gpu import _COMPILE_DISABLED

logger = logging.getLogger(__name__)
_COMPILED = {}


def _trial_values(f, g, d, x, xbase):
    return torch.stack(
        (
            f,
            torch.sum(g * d),
            g.abs().max(),
            torch.sum(g * g),
            torch.isfinite(f) & torch.isfinite(g).all() & torch.isfinite(x).all(),
            torch.any(x != xbase),
        )
    )


def trial_values(f, g, d, x, xbase):
    """Fuse trial statistics on CUDA, retaining eager execution on failure."""
    if not g.is_cuda or _COMPILE_DISABLED:
        return _trial_values(f, g, d, x, xbase)
    try:
        if _trial_values not in _COMPILED:
            _COMPILED[_trial_values] = torch.compile(
                _trial_values, fullgraph=True, dynamic=False
            )
        compiled = _COMPILED[_trial_values]
        if compiled is not None:
            return compiled(f, g, d, x, xbase)
    except Exception as exc:
        logger.warning("compiled CG trial statistics failed (%s); eager", exc)
        _COMPILED[_trial_values] = None
    return _trial_values(f, g, d, x, xbase)
