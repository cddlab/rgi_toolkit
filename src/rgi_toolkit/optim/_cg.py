"""One SciPy-style PR+ algorithm for Torch eager/CUDA and JAX JIT/scan.

Direction updates and step initialization follow scipy.optimize._minimize_cg
1.17.1, distributed under BSD-3-Clause (LICENSES/scipy.txt). Array evaluation is
backend-specific; the optimization and line-search transitions are shared.
"""

from __future__ import annotations

from typing import NamedTuple

from rgi_toolkit.optim._cg_config import (
    ARMIJO_C1,
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
from rgi_toolkit.optim.info import CGInfo, CGStatus


class CGState(NamedTuple):
    f: object
    g: object
    d: object
    gg: object
    previous_f: object
    valid: object
    info: CGInfo


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


class TorchCG:
    def __init__(self, like):
        import torch

        self.t = torch
        self.s = HostScalars()
        self.finfo = torch.finfo(like.dtype)

    def cast(self, scalar, like):
        return float(scalar)

    def dot(self, a, b):
        return self.s.scalar(self.t.sum(a * b))

    def max_atom_norm(self, d):
        return self.s.scalar(self.t.linalg.vector_norm(d, dim=-1).max())

    def evaluate(self, vg, x, xbase, d, alpha, count):
        g, f = vg(x)
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
        )


class JaxCG:
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

    def max_atom_norm(self, d):
        return self.s.scalar(self.j.max(self.j.linalg.norm(d, axis=-1)))

    def evaluate(self, vg, x, xbase, d, alpha, count):
        g, f = vg(x)
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
        )


def run_cg(
    backend,
    vg,
    x0,
    max_iter,
    *,
    gtol=GTOL,
    max_atom_step=None,
    state=None,
    more_maxiter=WOLFE1_MAX_ITER,
    wolfe_maxiter=WOLFE2_MAX_ITER,
    zoom_maxiter=ZOOM_MAX_ITER,
):
    """Return coordinates and resumable state, including cumulative diagnostics."""
    s, xp = backend.s, backend.s.xp
    zero, izero = s.scalar(0), s.integer(0)
    empty = CGInfo(s.integer(CGStatus.INACTIVE), izero, izero, izero, zero, zero)
    old_info = empty if state is None else state.info

    def fresh(_):
        t = backend.evaluate(vg, x0, x0, x0 * 0, zero, izero)
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
            t.f, t.g, -t.g, t.gg, t.f + xp.sqrt(t.gg) / 2.0, t.finite & ~converged, info
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
        amax = global_amax
        if max_atom_step is not None:
            amax = xp.minimum(
                amax, s.scalar(max_atom_step) / backend.max_atom_norm(st.d)
            )
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
        )

        def evaluate(alpha, cached):
            def calculate(_):
                xt = x + backend.cast(alpha, x) * st.d
                return backend.evaluate(vg, xt, x, st.d, alpha, cached.nfev)

            return s.cond(alpha == cached.alpha, lambda _: cached, calculate, None)

        def next_direction(t):
            numerator = backend.dot(t.g, t.g - st.g)
            beta = xp.maximum(0.0, numerator / st.gg)
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
            & (st.gg > 0)
            & (amax >= amin)
        )
        t, ok, _phase = s.cond(
            usable,
            lambda t: strong_wolfe(
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
            ),
            lambda t: (t, s.boolean(False), izero),
            prototype,
        )
        # Never accept Wolfe2's unverified last trial or a nonfinite/motionless point.
        ok = ok & t.finite & t.moved & (t.alpha > 0)
        ok = ok & (t.f <= st.f + ARMIJO_C1 * t.alpha * slope)
        ok = ok & (abs(t.slope) <= -WOLFE_C2 * slope)
        ok = s.cond(ok, extra, lambda _: s.boolean(False), t)

        def accepted(_):
            d, _dg = next_direction(t)
            converged = t.grad_norm <= gtol
            info = CGInfo(
                s.integer(xp.where(converged, CGStatus.CONVERGED, CGStatus.MAX_ITER)),
                st.info.nit + 1,
                st.info.nfev + t.nfev,
                st.info.njev + t.nfev,
                t.f,
                t.grad_norm,
            )
            return (
                t.x,
                CGState(t.f, t.g, d, t.gg, st.f, ~converged, info),
                iteration + 1,
            )

        def failed(_):
            reason = xp.where(
                ~t.finite | ~xp.isfinite(slope),
                CGStatus.NONFINITE,
                xp.where(
                    usable & ~t.moved & (t.nfev > 0),
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
            return x, st._replace(valid=s.boolean(False), info=info), iteration + 1

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
