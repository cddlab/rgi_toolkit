"""SciPy 1.17.1 PR+ and strong-Wolfe settings shared by all CG execution paths.

CG uses More--Thuente (DCSRCH), then the bracket/zoom Wolfe search if the first
search or its prospective-direction check fails. Every accepted point must pass
strong Wolfe and the next-direction sufficient-descent test. There is no Armijo
fallback or energy-change stopping rule. VdW supplies a scalar step bound rather
than clipping individual atoms off the search line.

The existing RGI gradient tolerance is retained; SciPy defaults to 1e-5 instead.
Resumable state carries the previous objective value used for SciPy's trial-step
guess. A neighbor rebuild invalidates the state but retains diagnostic counters.
Only exhaustion of a block budget allows resumption; convergence and failure end
the whole minimization invocation.
"""

GTOL = 1e-7
ARMIJO_C1 = 1e-4
WOLFE_C2 = 0.4
DESCENT_C = 0.01
WOLFE1_MAX_ITER = 100
WOLFE2_MAX_ITER = 10
ZOOM_MAX_ITER = 10
XTOL = 1e-14
STEP_MIN = 1e-100
STEP_MAX = 1e100

# This is a geometry regularizer, not a PR+ denominator adjustment.
EPS = 1e-12
