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


def _run(function, like, *args):
    if not like.is_cuda or _COMPILE_DISABLED:
        return function(*args)
    try:
        if function not in _COMPILED:
            _COMPILED[function] = torch.compile(function, fullgraph=True, dynamic=False)
        compiled = _COMPILED[function]
        if compiled is not None:
            return compiled(*args)
    except Exception as exc:
        logger.warning("compiled %s failed (%s); eager", function.__name__, exc)
        _COMPILED[function] = None
    return function(*args)


def trial_values(f, g, d, x, xbase):
    """Fuse trial statistics on CUDA, retaining eager execution on failure."""
    return _run(_trial_values, g, f, g, d, x, xbase)


def _direction(g, old_g, old_d, denominator):
    numerator = torch.sum(g * (g - old_g)).to(torch.float64)
    beta = torch.clamp(numerator / denominator, min=0).to(g.dtype)
    direction = -g + beta * old_d
    return direction, torch.sum(direction * g)


def direction(g, old_g, old_d, denominator):
    """Use host-equivalent float64 scalar division for the PR+ coefficient."""
    denominator = g.new_tensor(float(denominator), dtype=torch.float64)
    return _run(_direction, g, g, old_g, old_d, denominator)


def _cache_needed(a, reference, lig_local, valid, threshold_squared, enabled):
    query = a if lig_local is None else a[..., lig_local, :]
    delta = query - reference
    stale = (~valid) | torch.any(torch.sum(delta * delta, dim=-1) > threshold_squared)
    return enabled & stale & torch.isfinite(a).all()


def cache_needed(a, reference, lig_local, valid, threshold_squared, enabled):
    """Check every trial's displacement and finiteness in one compiled region."""
    return _run(
        _cache_needed, a, a, reference, lig_local, valid, threshold_squared, enabled
    )
