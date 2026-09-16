"""Shared PR+ settings for historical Armijo and SciPy 1.17.1 strong Wolfe.

Armijo restores the historical warm start and relative-function stopping rule.
Its trial step can grow above one for ordinary centroid/RMSD mean derivatives.
Strong Wolfe retains DCSRCH, Wolfe2 and the prospective sufficient-descent test.
Neither mode clips atom displacements. Exact neighbour-cache rebuilds preserve
the objective and the conjugate direction. The gradient tolerance remains the
historical RGI value; SciPy's default is 1e-5.
"""

GTOL = 1e-7
ARMIJO_C1 = 1e-4
ARMIJO_MAX_ITER = 20
ARMIJO_BACKTRACK = 0.5
ARMIJO_FTOL = 1e-9
ARMIJO_GG_FLOOR = 1e-20
ARMIJO_BETA_EPS = 1e-12
ARMIJO_INITIAL_STEP = 1.0
ARMIJO_STEP_GROW = 1.0 / ARMIJO_BACKTRACK
ARMIJO_STEP_MIN = ARMIJO_BACKTRACK**ARMIJO_MAX_ITER
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
