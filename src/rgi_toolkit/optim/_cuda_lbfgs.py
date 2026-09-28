"""CUDA control flow following PyTorch 2.8 L-BFGS and Wolfe rules.

Algorithm derived from torch.optim.lbfgs (BSD-3-Clause, PyTorch contributors).
The host adapter retains the native optimizer for unsupported CUDA environments.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch.utils import _pytree


@triton.jit
def _direction(
    g, ys, ss, rho, hdiag, count, alpha, out, N: tl.constexpr, B: tl.constexpr
):
    j = tl.arange(0, B)
    mask = j < N
    q = -tl.load(g + j, mask=mask, other=0)
    history = tl.load(count)
    for k in range(history - 1, -1, -1):
        s = tl.load(ss + k * N + j, mask=mask, other=0)
        y = tl.load(ys + k * N + j, mask=mask, other=0)
        a = tl.sum(s * q, 0) * tl.load(rho + k)
        tl.store(alpha + k, a)
        q = tl.fma(-a, y, q)
    r = q * tl.load(hdiag)
    for k in range(history):
        y = tl.load(ys + k * N + j, mask=mask, other=0)
        s = tl.load(ss + k * N + j, mask=mask, other=0)
        b = tl.sum(y * r, 0) * tl.load(rho + k)
        r = tl.fma(tl.load(alpha + k) - b, s, r)
    tl.store(out + j, r, mask=mask)


@torch.library.custom_op("rgi_toolkit::device_lbfgs_direction", mutates_args=())
def history_direction(
    g: torch.Tensor,
    ys: torch.Tensor,
    ss: torch.Tensor,
    rho: torch.Tensor,
    hdiag: torch.Tensor,
    count: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty_like(g)
    alpha = torch.empty_like(rho)
    _direction[(1,)](
        g,
        ys,
        ss,
        rho,
        hdiag,
        count,
        alpha,
        result,
        g.numel(),
        triton.next_power_of_2(g.numel()),
    )
    return result


@history_direction.register_fake
def _fake_direction(g, ys, ss, rho, hdiag, count):
    return torch.empty_like(g)


class Trial(NamedTuple):
    t: object
    f: object
    g: object
    slope: object


def choose(predicate, yes, no):
    return _pytree.tree_map(lambda a, b: torch.where(predicate, a, b), yes, no)


def cubic(first, second, bounds=None):
    x1, f1, _, g1 = first
    x2, f2, _, g2 = second
    lo, hi = (
        (torch.minimum(x1, x2), torch.maximum(x1, x2)) if bounds is None else bounds
    )
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    square = d1.square() - g1 * g2
    d2 = torch.sqrt(torch.clamp(square, min=0))
    position = torch.where(
        x1 <= x2,
        x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)),
        x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)),
    )
    return torch.where(
        square >= 0, torch.minimum(torch.maximum(position, lo), hi), (lo + hi) / 2
    )


def wolfe(s, vg, x, direction, value, gradient, slope, step):
    def evaluate(t):
        g, f = vg(x + t.to(x.dtype) * direction)
        return Trial(t, s.scalar(f), g, s.scalar((g * direction).sum()))

    initial = Trial(s.scalar(0), value, gradient, slope)
    trial = evaluate(step)

    # Expansion stops as soon as a bracket or a Wolfe point is found.
    def bracket_condition(state):
        previous, current, iteration, done, bracketed, evaluations = state
        return ~done & ~bracketed & (iteration < 25)

    def bracket_body(state):
        previous, current, iteration, done, bracketed, evaluations = state
        bad = (current.f > value + 1e-4 * current.t * slope) | (
            (iteration > 1) & (current.f >= previous.f)
        )
        done = ~bad & (current.slope.abs() <= -0.9 * slope)
        bracketed = bad | (~done & (current.slope >= 0))

        def expand(_):
            t = cubic(
                previous,
                current,
                (current.t + 0.01 * (current.t - previous.t), current.t * 10),
            )
            return current, evaluate(t), iteration + 1, done, bracketed, evaluations + 1

        return s.cond(
            ~done & ~bracketed,
            expand,
            lambda _: (previous, current, iteration, done, bracketed, evaluations),
            None,
        )

    previous, trial, iteration, done, _, evaluations = s.loop(
        bracket_condition,
        bracket_body,
        (
            initial,
            trial,
            s.integer(0),
            s.boolean(False),
            s.boolean(False),
            s.integer(1),
        ),
    )
    previous = choose(iteration == 25, initial, previous)
    previous = choose(done, trial, previous)
    low = choose(previous.f <= trial.f, previous, trial)
    high = choose(previous.f <= trial.f, trial, previous)
    norm = s.scalar(direction.abs().max())

    def zoom_condition(state):
        low, high, iteration, done, insufficient, evaluations = state
        return ~done & (iteration < 25) & ~((high.t - low.t).abs() * norm < 1e-9)

    def zoom_body(state):
        low, high, iteration, done, insufficient, evaluations = state
        t = cubic(low, high)
        lower, upper = torch.minimum(low.t, high.t), torch.maximum(low.t, high.t)
        epsilon = 0.1 * (upper - lower)
        near = torch.minimum(upper - t, t - lower) < epsilon
        replace = near & (insufficient | (t >= upper) | (t <= lower))
        adjusted = torch.where(
            (t - upper).abs() < (t - lower).abs(), upper - epsilon, lower + epsilon
        )
        t = torch.where(replace, adjusted, t)
        insufficient = near & ~replace
        current = evaluate(t)
        bad = (current.f > value + 1e-4 * t * slope) | (current.f >= low.f)
        satisfied = current.slope.abs() <= -0.9 * slope
        swap = ~satisfied & (current.slope * (high.t - low.t) >= 0)
        candidate_high = choose(swap, low, high)
        good_low, good_high = current, candidate_high
        bad_low = choose(low.f <= current.f, low, current)
        bad_high = choose(low.f <= current.f, current, low)
        return (
            choose(bad, bad_low, good_low),
            choose(bad, bad_high, good_high),
            iteration + 1,
            ~bad & satisfied,
            insufficient,
            evaluations + 1,
        )

    low, _, _, _, _, evaluations = s.loop(
        zoom_condition,
        zoom_body,
        (low, high, iteration, done, s.boolean(False), evaluations),
    )
    return low, evaluations


class State(NamedTuple):
    x: object
    f: object
    g: object
    d: object
    t: object
    previous_g: object
    previous_f: object
    ys: object
    ss: object
    rho: object
    hdiag: object
    count: object
    iteration: object
    evaluations: object
    running: object


def run_lbfgs(s, vg, x, max_iter=100, gtol=1e-5):
    history_size = 100
    g, f = vg(x)
    f = s.scalar(f)
    empty = torch.zeros((history_size, x.numel()), dtype=x.dtype, device=x.device)
    state = State(
        x,
        f,
        g,
        -g,
        s.scalar(0),
        g,
        f,
        empty,
        empty.clone(),
        torch.zeros(history_size, dtype=x.dtype, device=x.device),
        x.new_ones(()),
        s.integer(0),
        s.integer(0),
        s.integer(1),
        ~(g.abs().max() <= gtol),
    )
    if max_iter == 0:
        return state

    def condition(state):
        return (
            state.running
            & (state.iteration < max_iter)
            & ((state.iteration == 0) | (state.evaluations < max_iter * 5 // 4))
        )

    def body(state):
        def update(_):
            y = (state.g - state.previous_g).reshape(-1)
            step = (state.d * state.t.to(x.dtype)).reshape(-1)
            ys = (y * step).sum()
            accepted = ys > 1e-10
            shift = state.count == history_size
            indices = torch.arange(history_size, device=x.device)
            source = torch.where(
                shift, torch.clamp(indices + 1, max=history_size - 1), indices
            )
            position = torch.minimum(
                state.count, state.count.new_full((), history_size - 1)
            )
            replace = indices == position
            candidate_y = torch.where(replace[:, None], y, state.ys[source])
            candidate_s = torch.where(replace[:, None], step, state.ss[source])
            candidate_rho = torch.where(replace, 1.0 / ys, state.rho[source])
            new_y = torch.where(accepted, candidate_y, state.ys)
            new_s = torch.where(accepted, candidate_s, state.ss)
            new_rho = torch.where(accepted, candidate_rho, state.rho)
            new_h = torch.where(accepted, ys / (y * y).sum(), state.hdiag)
            new_count = torch.where(accepted, position + 1, state.count)
            d = history_direction(
                state.g.reshape(-1), new_y, new_s, new_rho, new_h, new_count
            ).reshape_as(x)
            return d, new_y, new_s, new_rho, new_h, new_count

        d, ys, ss, rho, hdiag, count = s.cond(
            state.iteration > 0,
            update,
            lambda _: (
                -state.g,
                state.ys,
                state.ss,
                state.rho,
                state.hdiag,
                state.count,
            ),
            None,
        )
        t = torch.where(
            state.iteration == 0,
            torch.minimum(s.scalar(1), 1 / s.scalar(state.g.abs().sum())),
            s.scalar(1),
        )
        slope = s.scalar((state.g * d).sum())

        def search(_):
            result, evaluations = wolfe(s, vg, state.x, d, state.f, state.g, slope, t)
            updated = state.x + result.t.to(x.dtype) * d
            running = (
                ~(result.g.abs().max() <= gtol)
                & ~((d * result.t.to(x.dtype)).abs().max() <= 1e-9)
                & ~((result.f - state.f).abs() < 1e-9)
            )
            return State(
                updated,
                result.f,
                result.g,
                d,
                result.t,
                state.g,
                state.f,
                ys,
                ss,
                rho,
                hdiag,
                count,
                state.iteration + 1,
                state.evaluations + evaluations,
                running,
            )

        def stop(_):
            return State(
                state.x,
                state.f,
                state.g,
                d,
                t,
                state.g,
                state.f,
                ys,
                ss,
                rho,
                hdiag,
                count,
                state.iteration + 1,
                state.evaluations,
                s.boolean(False),
            )

        return s.cond(~(slope > -1e-9), search, stop, None)

    return s.loop(condition, body, state)
