"""Historical backtracking Armijo search, shared by Torch and JAX."""

from rgi_toolkit.optim._cg_config import ARMIJO_BACKTRACK, ARMIJO_C1


def armijo(s, evaluate, trial, value, slope, step, maxiter):
    """Return the first finite, moving trial satisfying sufficient decrease."""

    def condition(state):
        _step, _trial, accepted, iteration = state
        return ~accepted & (iteration < maxiter)

    def body(state):
        step, previous, _accepted, iteration = state
        current = evaluate(step, previous)
        accepted = (
            current.finite
            & current.moved
            & (current.f <= value + ARMIJO_C1 * step * slope)
        )
        return (
            s.xp.where(accepted, step, step * ARMIJO_BACKTRACK),
            current,
            accepted,
            iteration + 1,
        )

    _, result, accepted, _ = s.loop(
        condition, body, (step, trial, s.boolean(False), s.integer(0))
    )
    return result, accepted
