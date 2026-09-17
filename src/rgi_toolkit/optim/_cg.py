"""Shared PR+ solvers for Torch eager/CUDA and JAX JIT/scan.

Direction updates and step initialization follow scipy.optimize._minimize_cg
1.17.1, distributed under BSD-3-Clause (LICENSES/scipy.txt). Array evaluation is
backend-specific; the optimization and line-search transitions are shared.
The optional Armijo mode retains the historical PR+ update and stopping rules,
with expanding initial steps for ordinary mean derivatives.
"""

from __future__ import annotations

from typing import NamedTuple

from rgi_toolkit.optim._cg_armijo import armijo
from rgi_toolkit.optim._cg_config import (
    ARMIJO_BETA_EPS,
    ARMIJO_C1,
    ARMIJO_FTOL,
    ARMIJO_GG_FLOOR,
    ARMIJO_INITIAL_STEP,
    ARMIJO_MAX_ITER,
    ARMIJO_STEP_GROW,
    ARMIJO_STEP_MIN,
    DESCENT_C,
    GTOL,
    STEP_MAX,
    STEP_MIN,
    WOLFE1_MAX_ITER,
    WOLFE2_MAX_ITER,
    WOLFE_C2,
    ZOOM_MAX_ITER,
)
from rgi_toolkit.optim._cg_linesearch import strong_wolfe
from rgi_toolkit.optim._cg_scalar import HostScalars, JaxScalars
from rgi_toolkit.optim._options import resolve_line_search
from rgi_toolkit.optim.info import CGInfo, CGStatus


class CGState(NamedTuple):
    f: object
    g: object
    d: object
    gg: object
    previous_f: object
    valid: object
    info: CGInfo
    cache: object = None
    step: object = 1.0


class Trial(NamedTuple):
    alpha: object
    x: object
    f: object
    g: object
    slope: object
    grad_norm: object
    gg: object
    finite: object
    moved: object
    nfev: object
    cache: object = None


class TorchCG:
    prepare = None

    def __init__(self, like):
        import torch

        self.t = torch
        self.s = HostScalars()
        self.finfo = torch.finfo(like.dtype)

    def cast(self, scalar, like):
        return float(scalar)

    def dot(self, a, b):
        return self.s.scalar(self.t.sum(a * b))

    def same_point(self, a, b):
        return self.t.equal(a, b)

    def evaluate(self, vg, x, xbase, d, alpha, count, cache=None):
        if self.prepare is None:
            g, f = vg(x)
        else:
            cache = self.prepare(x, cache)
            g, f = vg(x, cache)
        g, f = g.detach(), f.detach()
        values = self.t.stack(
            (
                f,
                self.t.sum(g * d),
                g.abs().max(),
                self.t.sum(g * g),
                self.t.isfinite(f)
                & self.t.isfinite(g).all()
                & self.t.isfinite(x).all(),
                self.t.any(x != xbase),
            )
        ).tolist()
        f, slope, gnorm, gg, finite, moved = values
        return Trial(
            alpha,
            x,
            self.s.scalar(f),
            g,
            self.s.scalar(slope),
            self.s.scalar(gnorm),
            self.s.scalar(gg),
            self.s.boolean(finite),
            self.s.boolean(moved),
            count + 1,
            cache,
        )


class JaxCG:
    prepare = None

    def __init__(self, like, energy_fn):
        import jax
        import jax.numpy as jnp

        self.j = jnp
        self.s = JaxScalars(jnp.zeros((), dtype=jax.eval_shape(energy_fn, like).dtype))
        self.finfo = jnp.finfo(like.dtype)

    def cast(self, scalar, like):
        return self.j.asarray(scalar, dtype=like.dtype)

    def dot(self, a, b):
        return self.s.scalar(self.j.sum(a * b))

    def same_point(self, a, b):
        return self.j.all(a == b)

    def evaluate(self, vg, x, xbase, d, alpha, count, cache=None):
        if self.prepare is None:
            g, f = vg(x)
        else:
            cache = self.prepare(x, cache)
            g, f = vg(x, cache)
        return Trial(
            alpha,
            x,
            self.s.scalar(f),
            g,
            self.dot(g, d),
            self.s.scalar(self.j.max(self.j.abs(g))),
            self.dot(g, g),
            self.j.isfinite(f)
            & self.j.all(self.j.isfinite(g))
            & self.j.all(self.j.isfinite(x)),
            self.j.any(x != xbase),
            count + 1,
            cache,
        )


def run_cg(
    backend,
    vg,
    x0,
    max_iter,
    *,
    line_search=None,
    gtol=GTOL,
    ftol=ARMIJO_FTOL,
    max_ls=ARMIJO_MAX_ITER,
    state=None,
    cache=None,
    prepare=None,
    more_maxiter=WOLFE1_MAX_ITER,
    wolfe_maxiter=WOLFE2_MAX_ITER,
    zoom_maxiter=ZOOM_MAX_ITER,
):
    """Return coordinates and resumable state, including cumulative diagnostics."""
    is_armijo = resolve_line_search("CG", line_search) == "armijo"
    s, xp = backend.s, backend.s.xp
    backend.prepare = prepare
    zero, izero = s.scalar(0), s.integer(0)
    empty = CGInfo(s.integer(CGStatus.INACTIVE), izero, izero, izero, zero, zero)
    old_info = empty if state is None else state.info

    def fresh(_):
        t = backend.evaluate(vg, x0, x0, x0 * 0, zero, izero, cache)
        converged = t.finite & (t.grad_norm <= gtol)
        status = xp.where(
            t.finite,
            xp.where(converged, CGStatus.CONVERGED, CGStatus.MAX_ITER),
            CGStatus.NONFINITE,
        )
        info = CGInfo(
            s.integer(status),
            old_info.nit,
            old_info.nfev + 1,
            old_info.njev + 1,
            t.f,
            t.grad_norm,
        )
        return CGState(
            t.f,
            t.g,
            -t.g,
            t.gg,
            t.f + xp.sqrt(t.gg) / 2.0,
            t.finite & ~converged,
            info,
            t.cache,
            s.scalar(ARMIJO_INITIAL_STEP / ARMIJO_STEP_GROW),
        )

    current = (
        fresh(None)
        if state is None
        else s.cond(state.valid, lambda _: state, fresh, None)
    )
    # Bound extrapolation arithmetic as well as the coordinate step on float32.
    amin = s.scalar(max(STEP_MIN, float(backend.finfo.tiny)))
    global_amax = s.scalar(min(STEP_MAX, float(backend.finfo.max) / 8.0))

    def body(loop):
        x, st, iteration = loop
        slope = backend.dot(st.g, st.d)
        if is_armijo:
            st = s.cond(
                slope >= 0,
                lambda st: st._replace(d=-st.g),
                lambda st: st,
                st,
            )
            slope = backend.dot(st.g, st.d)
        amax = global_amax
        prototype = Trial(
            s.scalar(float("nan")),
            x,
            st.f,
            st.g,
            slope,
            st.info.grad_norm,
            st.gg,
            s.boolean(True),
            s.boolean(False),
            izero,
            st.cache,
        )

        def evaluate(alpha, cached):
            xt = x + backend.cast(alpha, x) * st.d
            # Different step lengths can round to identical coordinates.
            # Match ScalarFunction's coordinate-based value/gradient cache.
            return s.cond(
                (alpha == cached.alpha) | backend.same_point(xt, cached.x),
                lambda _: cached._replace(alpha=alpha),
                lambda _: backend.evaluate(
                    vg, xt, x, st.d, alpha, cached.nfev, cached.cache
                ),
                None,
            )

        def next_direction(t):
            numerator = backend.dot(t.g, t.g - st.g)
            denominator = st.gg + ARMIJO_BETA_EPS if is_armijo else st.gg
            beta = xp.maximum(0.0, numerator / denominator)
            d = -t.g + backend.cast(beta, t.g) * st.d
            return d, backend.dot(d, t.g)

        def extra(t):
            def descent(t):
                _d, dg = next_direction(t)
                return xp.isfinite(dg) & (dg <= -DESCENT_C * t.gg)

            return s.cond(t.grad_norm <= gtol, lambda _: s.boolean(True), descent, t)

        usable = (
            xp.isfinite(slope)
            & (slope < 0)
            & xp.isfinite(st.gg)
            & (st.gg > (ARMIJO_GG_FLOOR if is_armijo else 0))
            & (amax >= amin)
        )

        def search(t):
            if is_armijo:
                # Mean gradients in large selections need steps above one.
                # Expand the initial trial without rescaling the energy/gradient.
                step = xp.minimum(
                    amax,
                    xp.maximum(st.step, ARMIJO_STEP_MIN) * ARMIJO_STEP_GROW,
                )
                return armijo(s, evaluate, t, st.f, slope, step, max_ls)
            result, ok, _phase = strong_wolfe(
                s,
                evaluate,
                extra,
                t,
                st.f,
                slope,
                st.previous_f,
                amin,
                amax,
                more_maxiter=more_maxiter,
                wolfe_maxiter=wolfe_maxiter,
                zoom_maxiter=zoom_maxiter,
            )
            return result, ok

        t, ok = s.cond(
            usable,
            search,
            lambda t: (t, s.boolean(False)),
            prototype,
        )
        # Never accept Wolfe2's unverified last trial or a nonfinite/motionless point.
        ok = ok & t.finite & t.moved & (t.alpha > 0)
        ok = ok & (t.f <= st.f + ARMIJO_C1 * t.alpha * slope)
        if not is_armijo:
            ok = ok & (abs(t.slope) <= -WOLFE_C2 * slope)
            ok = s.cond(ok, extra, lambda _: s.boolean(False), t)

        def accepted(_):
            d, _dg = next_direction(t)
            converged = t.grad_norm <= gtol
            small_change = (
                abs(t.f - st.f) < ftol * (1.0 + abs(st.f))
                if is_armijo
                else s.boolean(False)
            )
            status = xp.where(
                converged,
                CGStatus.CONVERGED,
                xp.where(small_change, CGStatus.FUNCTION_TOLERANCE, CGStatus.MAX_ITER),
            )
            info = CGInfo(
                s.integer(status),
                st.info.nit + 1,
                st.info.nfev + t.nfev,
                st.info.njev + t.nfev,
                t.f,
                t.grad_norm,
            )
            return (
                t.x,
                CGState(
                    t.f,
                    t.g,
                    d,
                    t.gg,
                    st.f,
                    ~(converged | small_change),
                    info,
                    t.cache,
                    t.alpha,
                ),
                iteration + 1,
            )

        def failed(_):
            reason = xp.where(
                ~t.finite | ~xp.isfinite(slope),
                CGStatus.NONFINITE,
                xp.where(
                    usable & ~t.moved & xp.isfinite(t.alpha),
                    CGStatus.NO_PROGRESS,
                    CGStatus.LINE_SEARCH_FAILED,
                ),
            )
            info = CGInfo(
                s.integer(reason),
                st.info.nit,
                st.info.nfev + t.nfev,
                st.info.njev + t.nfev,
                st.f,
                st.info.grad_norm,
            )
            return (
                x,
                st._replace(valid=s.boolean(False), info=info, cache=t.cache),
                iteration + 1,
            )

        return s.cond(ok, accepted, failed, None)

    xf, result, _ = s.loop(
        lambda v: (v[2] < max_iter) & v[1].valid, body, (x0, current, izero)
    )
    return xf, result


def torch_cg(vg, x0, max_iter, **kwargs):
    """Use the shared solver with a Torch ``(gradient, value)`` callback."""
    x0 = x0.detach().clone()
    return run_cg(TorchCG(x0), vg, x0, max_iter, **kwargs)


def jax_cg(energy_fn, x0, max_iter, **kwargs):
    """Use the same solver with pure JAX autodiff and scalar control flow."""
    import jax

    value_grad = jax.value_and_grad(energy_fn)

    def vg(x):
        f, g = value_grad(x)
        return g, f

    return run_cg(JaxCG(x0, energy_fn), vg, x0, max_iter, **kwargs)
