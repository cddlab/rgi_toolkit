"""Optional loss-based termination around the native Torch L-BFGS algorithm."""

import torch


class _LossTargetReached(Exception):
    """Leave a native line search once the requested objective has been reached."""


def minimize(active, closure, max_iter, gtol, loss_tol):
    """Preserve native defaults unless an explicit objective target is requested."""
    options = {} if loss_tol is None else {"tolerance_change": 0.0}
    optimizer = torch.optim.LBFGS(
        [active],
        max_iter=max_iter,
        tolerance_grad=gtol if loss_tol is None else 0.0,
        line_search_fn="strong_wolfe",
        **options,
    )
    if loss_tol is None:
        optimizer.step(closure)
        return
    if max_iter == 0:
        return
    target = None

    def checked_closure():
        nonlocal target
        value = closure()
        if bool(
            torch.isfinite(value)
            & (value.abs() <= loss_tol)
            & torch.isfinite(active).all()
            & torch.isfinite(active.grad).all()
        ):
            target = active.detach().clone()
            raise _LossTargetReached
        return value

    try:
        optimizer.step(checked_closure)
    except _LossTargetReached:
        # A line search may restore its base point while unwinding. Keep the
        # actual evaluated target point, without depending on library internals.
        with torch.no_grad():
            active.copy_(target)
