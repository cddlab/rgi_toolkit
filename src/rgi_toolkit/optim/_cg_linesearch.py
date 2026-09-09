"""SciPy 1.17.1 More--Thuente and bracket/zoom strong-Wolfe searches.

Adapted from scipy.optimize._dcsrch and _linesearch (BSD-3-Clause); see the
distributed LICENSES/scipy.txt. DCSRCH/dcstep originate in MINPACK-1 (1983) and
MINPACK-2 (1993), Argonne National Laboratory / University of Minnesota, by
Jorge J. More, David J. Thuente, Brett M. Averick, and Richard G. Carter.

The scalar state machines are shared by host-controlled Torch and pure JAX.
``evaluate(alpha, previous_trial)`` caches the current trial's value/gradient.
Unlike SciPy's unverified Wolfe2 result on iteration exhaustion, only explicit
successful termination is accepted. Nonfinite trials never establish Wolfe.
"""

from __future__ import annotations

from typing import NamedTuple

from rgi_toolkit.optim._cg_config import (
    ARMIJO_C1,
    WOLFE1_MAX_ITER,
    WOLFE2_MAX_ITER,
    WOLFE_C2,
    XTOL,
    ZOOM_MAX_ITER,
)


def dcstep(s, stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """The four safeguarded interpolation cases of MINPACK/SciPy dcstep."""
    xp = s.xp
    sgnd = xp.sign(dp) * xp.sign(dx)

    def cubic(left_f, right_f, left_d, right_d, width):
        theta = 3.0 * (left_f - right_f) / width + left_d + right_d
        scale = xp.maximum(xp.maximum(abs(theta), abs(left_d)), abs(right_d))
        rad = (theta / scale) ** 2 - (left_d / scale) * (right_d / scale)
        return theta, scale, rad

    def higher(_):
        theta, scale, rad = cubic(fx, fp, dx, dp, stp - stx)
        gamma = scale * xp.sqrt(rad)
        gamma = xp.where(stp < stx, -gamma, gamma)
        p, q = (gamma - dx) + theta, ((gamma - dx) + gamma) + dp
        stpc = stx + (p / q) * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        out = xp.where(
            abs(stpc - stx) <= abs(stpq - stx), stpc, stpc + (stpq - stpc) / 2.0
        )
        return out, s.boolean(True)

    def opposite(_):
        theta, scale, rad = cubic(fx, fp, dx, dp, stp - stx)
        gamma = scale * xp.sqrt(rad)
        gamma = xp.where(stp > stx, -gamma, gamma)
        p, q = (gamma - dp) + theta, ((gamma - dp) + gamma) + dx
        stpc = stp + (p / q) * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        return xp.where(abs(stpc - stp) > abs(stpq - stp), stpc, stpq), s.boolean(True)

    def smaller(_):
        theta, scale, rad = cubic(fx, fp, dx, dp, stp - stx)
        gamma = scale * xp.sqrt(xp.maximum(0.0, rad))
        gamma = xp.where(stp > stx, -gamma, gamma)
        p, q = (gamma - dp) + theta, (gamma + (dx - dp)) + gamma
        r = p / q
        stpc = xp.where(
            (r < 0) & (gamma != 0),
            stp + r * (stx - stp),
            xp.where(stp > stx, stpmax, stpmin),
        )
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        near = xp.where(abs(stpc - stp) < abs(stpq - stp), stpc, stpq)
        near = xp.where(
            stp > stx,
            xp.minimum(stp + 0.66 * (sty - stp), near),
            xp.maximum(stp + 0.66 * (sty - stp), near),
        )
        far = xp.where(abs(stpc - stp) > abs(stpq - stp), stpc, stpq)
        return xp.where(brackt, near, xp.clip(far, stpmin, stpmax)), brackt

    def other(_):
        def bracketed(_):
            theta, scale, rad = cubic(fp, fy, dp, dy, sty - stp)
            gamma = scale * xp.sqrt(rad)
            gamma = xp.where(stp > sty, -gamma, gamma)
            p, q = (gamma - dp) + theta, ((gamma - dp) + gamma) + dy
            return stp + (p / q) * (sty - stp)

        return s.cond(
            brackt, bracketed, lambda _: xp.where(stp > stx, stpmax, stpmin), None
        ), brackt

    stpf, bracket = s.cond(
        fp > fx,
        higher,
        lambda _: s.cond(
            sgnd < 0,
            opposite,
            lambda _: s.cond(abs(dp) < abs(dx), smaller, other, None),
            None,
        ),
        None,
    )
    lower = fp <= fx
    ny = xp.where(~lower, stp, xp.where(sgnd < 0, stx, sty))
    nfy = xp.where(~lower, fp, xp.where(sgnd < 0, fx, fy))
    ndy = xp.where(~lower, dp, xp.where(sgnd < 0, dx, dy))
    return (
        xp.where(lower, stp, stx),
        xp.where(lower, fp, fx),
        xp.where(lower, dp, dx),
        ny,
        nfy,
        ndy,
        stpf,
        bracket,
    )


class _MoreState(NamedTuple):
    x: object
    fx: object
    gx: object
    y: object
    fy: object
    gy: object
    step: object
    bracket: object
    stage: object
    width: object
    old_width: object
    lo: object
    hi: object


def _more_update(s, z, t, f0, slope0, amin, amax, c1, c2):
    """One DCSRCH FG transition: code 0 requests FG, 1 converges, 2 fails."""
    xp = s.xp
    gtest = c1 * slope0
    ftest = f0 + z.step * gtest
    stage = xp.where((z.stage == 1) & (t.f <= ftest) & (t.slope >= 0), 2, z.stage)
    warn = (
        (z.bracket & ((z.step <= z.lo) | (z.step >= z.hi)))
        | (z.bracket & (z.hi - z.lo <= XTOL * z.hi))
        | ((z.step == amax) & (t.f <= ftest) & (t.slope <= gtest))
        | ((z.step == amin) & ((t.f > ftest) | (t.slope >= gtest)))
    )
    ok = t.finite & (t.f <= ftest) & (abs(t.slope) <= c2 * -slope0)
    done = ok | warn | ~t.finite | ~t.moved

    def advance(_):
        modified = (stage == 1) & (t.f <= z.fx) & (t.f > ftest)
        shift = xp.where(modified, gtest, 0.0)
        x, fx, gx, y, fy, gy, step, bracket = dcstep(
            s,
            z.x,
            z.fx - z.x * shift,
            z.gx - shift,
            z.y,
            z.fy - z.y * shift,
            z.gy - shift,
            z.step,
            t.f - z.step * shift,
            t.slope - shift,
            z.bracket,
            z.lo,
            z.hi,
        )
        fx, fy, gx, gy = fx + x * shift, fy + y * shift, gx + shift, gy + shift
        step = xp.where(
            bracket & (abs(y - x) >= 0.66 * z.old_width), x + 0.5 * (y - x), step
        )
        old_width = xp.where(bracket, z.width, z.old_width)
        width = xp.where(bracket, abs(y - x), z.width)
        lo = xp.where(bracket, xp.minimum(x, y), step + 1.1 * (step - x))
        hi = xp.where(bracket, xp.maximum(x, y), step + 4.0 * (step - x))
        step = xp.clip(step, amin, amax)
        stuck = bracket & ((step <= lo) | (step >= hi) | (hi - lo <= XTOL * hi))
        step = xp.where(stuck, x, step)
        nz = _MoreState(
            x, fx, gx, y, fy, gy, step, bracket, stage, width, old_width, lo, hi
        )
        return nz, s.integer(xp.where(xp.isfinite(step), 0, 2))

    return s.cond(
        done, lambda _: (z, s.integer(xp.where(ok & t.moved, 1, 2))), advance, None
    )


def _more(s, evaluate, trial, step, f0, slope0, amin, amax, c1, c2, maxiter):
    zero = s.scalar(0.0)
    z = _MoreState(
        zero,
        f0,
        slope0,
        zero,
        f0,
        slope0,
        step,
        s.boolean(False),
        s.integer(1),
        amax - amin,
        2.0 * (amax - amin),
        zero,
        5.0 * step,
    )
    valid = (step >= amin) & (step <= amax) & (slope0 < 0) & (amax >= amin)

    def run(t):
        t = evaluate(step, t)

        def body(state):
            z, t, i, _code = state
            z, code = _more_update(s, z, t, f0, slope0, amin, amax, c1, c2)
            t = s.cond(code == 0, lambda t: evaluate(z.step, t), lambda t: t, t)
            return z, t, i + 1, code

        _, t, _, code = s.loop(
            lambda v: (v[2] < maxiter) & (v[3] == 0),
            body,
            (z, t, s.integer(1), s.integer(0)),
        )
        return t, code == 1

    return s.cond(valid & (maxiter > 0), run, lambda t: (t, s.boolean(False)), trial)


def _cubicmin(s, a, fa, da, b, fb, c, fc):
    db, dc = b - a, c - a
    denom = (db * dc) ** 2 * (db - dc)
    vb, vc = fb - fa - da * db, fc - fa - da * dc
    aa = (dc**2 * vb - db**2 * vc) / denom
    bb = (-(dc**3) * vb + db**3 * vc) / denom
    return a + (-bb + s.xp.sqrt(bb * bb - 3.0 * aa * da)) / (3.0 * aa)


def _quadmin(a, fa, da, b, fb):
    db = b - a * 1.0
    bb = (fb - fa - da * db) / (db * db)
    return a - da / (2.0 * bb)


def _zoom(
    s, evaluate, extra, trial, lo, hi, flo, fhi, dlo, f0, slope0, c1, c2, maxiter
):
    xp = s.xp

    def body(state):
        lo, hi, flo, fhi, dlo, recent, frecent, i, t, _ok, _done = state
        width = hi - lo
        a, b = xp.minimum(lo, hi), xp.maximum(lo, hi)
        cubic = s.cond(
            i > 0,
            lambda _: _cubicmin(s, lo, flo, dlo, hi, fhi, recent, frecent),
            lambda _: s.scalar(float("nan")),
            None,
        )
        bad_cubic = (
            ~xp.isfinite(cubic) | (cubic > b - 0.2 * width) | (cubic < a + 0.2 * width)
        )

        def quadratic(_):
            q = _quadmin(lo, flo, dlo, hi, fhi)
            bad = ~xp.isfinite(q) | (q > b - 0.1 * width) | (q < a + 0.1 * width)
            return xp.where(bad, lo + 0.5 * width, q)

        step = s.cond(bad_cubic, quadratic, lambda _: cubic, None)
        t = evaluate(step, t)
        # Invalid points are upper brackets, never inputs to gradient interpolation.
        high = ~t.finite | (t.f > f0 + c1 * step * slope0) | (t.f >= flo)
        wolfe = t.finite & t.moved & ~high & (abs(t.slope) <= -c2 * slope0)
        ok = s.cond(wolfe, extra, lambda _: s.boolean(False), t)
        reverse = t.slope * (hi - lo) >= 0
        nrecent = xp.where(high | reverse, hi, lo)
        nfrecent = xp.where(high | reverse, fhi, flo)
        nlo = xp.where(high, lo, step)
        nflo = xp.where(high, flo, t.f)
        ndlo = xp.where(high, dlo, t.slope)
        nhi = xp.where(high, step, xp.where(reverse, lo, hi))
        nfhi = xp.where(
            high, xp.where(t.finite, t.f, xp.inf), xp.where(reverse, flo, fhi)
        )
        done = ok | (step == lo) | (step == hi) | ~t.moved
        return nlo, nhi, nflo, nfhi, ndlo, nrecent, nfrecent, i + 1, t, ok, done

    # SciPy's i > maxiter termination permits the final i == maxiter trial.
    out = s.loop(
        lambda v: (v[7] <= maxiter) & ~v[10],
        body,
        (
            lo,
            hi,
            flo,
            fhi,
            dlo,
            s.scalar(0),
            f0,
            s.integer(0),
            trial,
            s.boolean(False),
            s.boolean(False),
        ),
    )
    return out[8], out[9]


def _wolfe2(
    s, evaluate, extra, trial, step, f0, slope0, amax, c1, c2, maxiter, zoom_maxiter
):
    xp = s.xp
    trial = evaluate(step, trial)

    def body(state):
        prev, fprev, dprev, step, t, i, _ok, _done = state
        high = ~t.finite | (t.f > f0 + c1 * step * slope0) | ((i > 0) & (t.f >= fprev))

        def zoom_forward(t):
            t, ok = _zoom(
                s,
                evaluate,
                extra,
                t,
                prev,
                step,
                fprev,
                t.f,
                dprev,
                f0,
                slope0,
                c1,
                c2,
                zoom_maxiter,
            )
            return prev, fprev, dprev, step, t, i + 1, ok, s.boolean(True)

        def low(t):
            wolfe = t.finite & t.moved & (abs(t.slope) <= -c2 * slope0)
            ok = s.cond(wolfe, extra, lambda _: s.boolean(False), t)

            def continue_search(t):
                def reverse_zoom(t):
                    t, ok = _zoom(
                        s,
                        evaluate,
                        extra,
                        t,
                        step,
                        prev,
                        t.f,
                        fprev,
                        t.slope,
                        f0,
                        slope0,
                        c1,
                        c2,
                        zoom_maxiter,
                    )
                    return prev, fprev, dprev, step, t, i + 1, ok, s.boolean(True)

                def grow(t):
                    nxt = xp.minimum(2.0 * step, amax)
                    stopped = (nxt == step) | ~t.moved
                    nt = s.cond(stopped, lambda t: t, lambda t: evaluate(nxt, t), t)
                    return step, t.f, t.slope, nxt, nt, i + 1, s.boolean(False), stopped

                return s.cond(t.slope >= 0, reverse_zoom, grow, t)

            return s.cond(
                ok,
                lambda t: (
                    prev,
                    fprev,
                    dprev,
                    step,
                    t,
                    i + 1,
                    s.boolean(True),
                    s.boolean(True),
                ),
                continue_search,
                t,
            )

        return s.cond(high, zoom_forward, low, t)

    out = s.loop(
        lambda v: (v[5] < maxiter) & ~v[7],
        body,
        (
            s.scalar(0),
            f0,
            slope0,
            step,
            trial,
            s.integer(0),
            s.boolean(False),
            s.boolean(False),
        ),
    )
    return out[4], out[6]


def strong_wolfe(
    s,
    evaluate,
    extra,
    trial,
    f0,
    slope0,
    previous_f,
    amin,
    amax,
    c1=ARMIJO_C1,
    c2=WOLFE_C2,
    more_maxiter=WOLFE1_MAX_ITER,
    wolfe_maxiter=WOLFE2_MAX_ITER,
    zoom_maxiter=ZOOM_MAX_ITER,
):
    """SciPy's Wolfe1 -> Wolfe2 ordering, returning (trial, success, phase)."""
    xp = s.xp
    guess = xp.minimum(1.0, 1.01 * 2.0 * (f0 - previous_f) / slope0)
    guess = xp.where(guess < 0, 1.0, guess)
    # Preserve the straight ray when VdW supplies a finite movement bound.
    guess = xp.minimum(guess, amax)
    t, ok = _more(
        s, evaluate, trial, guess, f0, slope0, amin, amax, c1, c2, more_maxiter
    )
    ok = s.cond(ok, extra, lambda _: s.boolean(False), t)

    def fallback(t):
        t, ok = _wolfe2(
            s,
            evaluate,
            extra,
            t,
            guess,
            f0,
            slope0,
            amax,
            c1,
            c2,
            wolfe_maxiter,
            zoom_maxiter,
        )
        return t, ok, s.integer(xp.where(ok, 2, 0))

    return s.cond(ok, lambda t: (t, s.boolean(True), s.integer(1)), fallback, t)
