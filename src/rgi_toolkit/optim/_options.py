"""Backend-independent optimizer selection and validation."""

CG_METHODS = frozenset({"cg", "ncg", "nonlinear-cg", "nonlinearcg"})
LBFGS_METHODS = frozenset({"l-bfgs", "lbfgs"})
LINE_SEARCHES = frozenset({"armijo", "strong-wolfe"})


def resolve_line_search(method, line_search=None):
    """Resolve the CG default, or return None for L-BFGS's own line search."""
    name = str(method).lower()
    if name not in CG_METHODS | LBFGS_METHODS:
        raise ValueError(
            f"unknown method {method!r}: expected a CG alias "
            "(cg/ncg/nonlinear-cg/nonlinearcg) or l-bfgs (l-bfgs/lbfgs)"
        )
    if name in LBFGS_METHODS:
        if line_search is not None:
            raise ValueError("line_search is supported only for method='CG'")
        return None
    if line_search is None:
        return "strong-wolfe"
    if not isinstance(line_search, str) or line_search not in LINE_SEARCHES:
        raise ValueError("line_search must be 'armijo' or 'strong-wolfe'")
    return line_search
