"""GPU restraint optimizer for JAX tools (alphafold3).

Builds a pure JIT/scan/vmap-compatible minimizer over an autodiff energy.
CG selects historical Armijo (default) or SciPy 1.17.1 PR+ with DCSRCH/Wolfe2
strong Wolfe through ``optim/_cg.py``, shared with Torch. Failed searches
retain the last accepted point. ``return_info=True`` exposes traced CG diagnostics.
``method='l-bfgs'`` uses ``jaxopt.LBFGS`` (lazily imported). No callback or runtime
SciPy is used; optimization remains inside XLA on the selected device. Backend
floating-point evaluation orders can produce different search decisions near a
condition boundary; ordinary scalar objectives are checked against SciPy.

Dynamic VdW caches are validated before every trial evaluation by the shared
``_vdw_runtime``. Fixed partners receive the full Verlet skin displacement budget;
two moving partners each receive half. Overflow rows use complete pair sums in
bounded chunks. Rebuilding preserves the objective and CG history.

"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from jax.custom_batching import sequential_vmap

from rgi_toolkit._array_ops import VDW_OVERLAP_EPS, get_ops
from rgi_toolkit._config_util import (
    VDW_SCALE_DEFAULT,
)
from rgi_toolkit.energy import jax_energy
from rgi_toolkit.energy._nonbonded import pair_parameters, prepare_chemistry
from rgi_toolkit.optim._cell_list import (
    CELL_CHUNK_SIZE,
    CELL_HASH_PRIMES,
    CELL_OFFSETS,
)
from rgi_toolkit.optim._cg_config import EPS, GTOL
from rgi_toolkit.optim._options import resolve_line_search
from rgi_toolkit.spec import check_active_vdw_int32_safe

logger = logging.getLogger(__name__)


def _max_disp(current, reference):
    """Largest per-atom Euclidean displacement between two coordinate sets.

    The Euclidean norm, NOT a component-wise max: the latter underestimates the true
    displacement by up to sqrt(3) and would let a neighbour list go stale unnoticed. The
    torch optimizer computes the same quantity — keep the two in step.
    """
    return jnp.max(jnp.linalg.norm(current - reference, axis=-1))


def _cg_minimize(
    energy_fn,
    x0,
    max_iter,
    gtol=GTOL,
    state=None,
    return_state=False,
    return_info=False,
    **search_options,
):
    """Pure JAX entry to the selected shared CG solver."""
    from rgi_toolkit.optim._cg import jax_cg

    out, result = jax_cg(
        energy_fn,
        x0,
        max_iter,
        gtol=gtol,
        state=state,
        **search_options,
    )
    if return_state:
        return out, result
    return (out, result.info) if return_info else out


def _cell_hash_jax(cells):
    """Return an int32 spatial hash; callers verify cells after hash lookup."""
    return (
        cells[..., 0] * jnp.int32(CELL_HASH_PRIMES[0])
        ^ cells[..., 1] * jnp.int32(CELL_HASH_PRIMES[1])
        ^ cells[..., 2] * jnp.int32(CELL_HASH_PRIMES[2])
    )


def _batched_searchsorted_jax(sorted_values, queries, side):
    return jax.vmap(lambda values, query: jnp.searchsorted(values, query, side=side))(
        sorted_values, queries
    ).astype(jnp.int32)


def _merge_cell_candidates_jax(best_dist2, best_idx, dist2, candidate, k):
    """Keep the lexicographically smallest ``(distance squared, atom index)`` rows."""
    candidate = candidate.astype(best_idx.dtype)
    merged_dist2 = jnp.concatenate((best_dist2, dist2), axis=-1)
    merged_idx = jnp.concatenate((best_idx, candidate), axis=-1)
    merged_dist2, merged_idx = jax.lax.sort(
        (merged_dist2, merged_idx),
        dimension=-1,
        is_stable=True,
        num_keys=2,
    )
    return merged_dist2[..., :k], merged_idx[..., :k].astype(best_idx.dtype)


def _build_cell_pairs_jax(
    query,
    target,
    dmax,
    max_neighbors,
    exclude_self=False,
    query_radii=None,
    target_radii=None,
    pair_scale=None,
    query_polymer=None,
    target_polymer=None,
    excluded_codes=None,
    pair_code_size=None,
    chemistry=None,
):
    """Return target indices and ranking scores from a sorted cell list."""

    n_query, n_target = query.shape[-2], target.shape[-2]
    chemistry = prepare_chemistry(get_ops("jax"), chemistry, query)
    query_batch = jax.lax.stop_gradient(query.reshape((-1, n_query, 3)))
    target_batch = jax.lax.stop_gradient(target.reshape((-1, n_target, 3)))
    if query_batch.shape[0] != target_batch.shape[0]:
        raise ValueError("query and target VdW batch dimensions must match")
    if query_radii is not None:
        query_radii = jnp.asarray(query_radii, dtype=query_batch.dtype)
        target_radii = jnp.asarray(target_radii, dtype=query_batch.dtype)
        pair_scale = jnp.asarray(pair_scale, dtype=query_batch.dtype)
    max_candidates = n_target - int(exclude_self)
    if n_query == 0 or max_candidates <= 0:
        empty_idx = jnp.zeros((query_batch.shape[0], n_query, 0), dtype=jnp.int32)
        return empty_idx, empty_idx.astype(query.dtype)
    if n_target == 1:
        dist2 = jnp.sum(
            (query_batch - target_batch[:, :1, :]) ** 2, axis=-1, keepdims=True
        )
        dmax_value = jnp.asarray(dmax, dtype=query_batch.dtype)
        valid = (
            jnp.isfinite(dist2) & (dmax_value > 0) & (dist2 <= dmax_value * dmax_value)
        )
        score = dist2
        source = jnp.arange(n_query, dtype=jnp.int32).reshape((1, n_query, 1))
        neighbours = jnp.zeros(dist2.shape, dtype=jnp.int32)
        if chemistry is not None:
            contact, _, allowed = pair_parameters(
                get_ops("jax"), chemistry, source, neighbours
            )
            score = jnp.sqrt(dist2 + EPS) - pair_scale * contact
            valid = valid & allowed
        elif query_radii is not None:
            source_r = query_radii[source]
            target_r = target_radii[neighbours]
            valid = valid & (source_r > 0) & (target_r > 0)
            score = jnp.sqrt(dist2 + EPS) - pair_scale * (source_r + target_r)
        if chemistry is None and query_polymer is not None:
            valid = valid & (query_polymer[source] | target_polymer[neighbours])
        if (
            chemistry is None
            and excluded_codes is not None
            and excluded_codes.shape[0] > 0
        ):
            lo = jnp.minimum(source, neighbours)
            hi = jnp.maximum(source, neighbours)
            codes = lo * pair_code_size + hi
            positions = jnp.searchsorted(excluded_codes, codes)
            positions = jnp.minimum(positions, excluded_codes.shape[0] - 1)
            valid = valid & (excluded_codes[positions] != codes)
        return neighbours, jnp.where(valid, score, jnp.inf)
    k = min(int(max_neighbors), max_candidates)
    n_batch = query_batch.shape[0]
    dmax_value = jnp.asarray(dmax, dtype=query_batch.dtype)
    cell_width = jnp.where(
        dmax_value > 0, dmax_value, jnp.asarray(1.0, query_batch.dtype)
    )
    safe_query = jnp.nan_to_num(query_batch, nan=0.0, posinf=0.0, neginf=0.0)
    safe_target = jnp.nan_to_num(target_batch, nan=0.0, posinf=0.0, neginf=0.0)
    query_cells = jnp.floor(safe_query / cell_width).astype(jnp.int32)
    target_cells = jnp.floor(safe_target / cell_width).astype(jnp.int32)
    target_hashes = _cell_hash_jax(target_cells)
    order = jnp.argsort(target_hashes, axis=-1, stable=True).astype(jnp.int32)
    sorted_hashes = jnp.take_along_axis(target_hashes, order, axis=-1)
    own_start = _batched_searchsorted_jax(sorted_hashes, target_hashes, "left")
    own_end = _batched_searchsorted_jax(sorted_hashes, target_hashes, "right")
    max_bucket = jnp.max(own_end - own_start)

    best_dist2 = jnp.full((n_batch, n_query, k), jnp.inf, dtype=query_batch.dtype)
    best_idx = jnp.zeros((n_batch, n_query, k), dtype=jnp.int32)
    batch_idx = jnp.arange(n_batch, dtype=jnp.int32).reshape((-1, 1, 1))
    source = jnp.arange(n_query, dtype=jnp.int32).reshape((1, n_query, 1))
    offsets = jnp.asarray(CELL_OFFSETS, dtype=jnp.int32)
    chunk_offsets = jnp.arange(CELL_CHUNK_SIZE, dtype=jnp.int32).reshape((1, 1, -1))
    cutoff2 = dmax_value * dmax_value

    def offset_body(offset_index, state):
        best_dist2, best_idx = state
        adjacent_cells = query_cells + offsets[offset_index]
        query_hashes = _cell_hash_jax(adjacent_cells)
        starts = _batched_searchsorted_jax(sorted_hashes, query_hashes, "left")
        ends = _batched_searchsorted_jax(sorted_hashes, query_hashes, "right")

        def chunk_cond(chunk_state):
            base, _best_dist2, _best_idx = chunk_state
            return base < max_bucket

        def chunk_body(chunk_state):
            base, best_dist2, best_idx = chunk_state
            positions = starts[..., None] + base + chunk_offsets
            position_valid = positions < ends[..., None]
            safe_positions = jnp.minimum(positions, n_target - 1)
            candidate = order[batch_idx, safe_positions]
            candidate_cells = target_cells[batch_idx, candidate]
            same_cell = jnp.all(
                candidate_cells == adjacent_cells[..., None, :], axis=-1
            )
            candidate_coords = target_batch[batch_idx, candidate]
            delta = query_batch[:, :, None, :] - candidate_coords
            dist2 = jnp.sum(delta * delta, axis=-1)
            valid_candidate = (
                position_valid
                & same_cell
                & jnp.isfinite(dist2)
                & (dmax_value > 0)
                & (dist2 <= cutoff2)
            )
            if exclude_self:
                valid_candidate = valid_candidate & (candidate != source)
            score = dist2
            if chemistry is not None:
                contact, _, allowed = pair_parameters(
                    get_ops("jax"), chemistry, source, candidate
                )
                score = jnp.sqrt(dist2 + EPS) - pair_scale * contact
                valid_candidate = valid_candidate & allowed
            elif query_radii is not None:
                source_r = query_radii[source]
                target_r = target_radii[candidate]
                valid_candidate = valid_candidate & (source_r > 0) & (target_r > 0)
                score = jnp.sqrt(dist2 + EPS) - pair_scale * (source_r + target_r)
            if chemistry is None and query_polymer is not None:
                valid_candidate = valid_candidate & (
                    query_polymer[source] | target_polymer[candidate]
                )
            if (
                chemistry is None
                and excluded_codes is not None
                and excluded_codes.shape[0] > 0
            ):
                lo = jnp.minimum(source, candidate)
                hi = jnp.maximum(source, candidate)
                codes = lo * pair_code_size + hi
                positions = jnp.searchsorted(excluded_codes, codes)
                positions = jnp.minimum(positions, excluded_codes.shape[0] - 1)
                valid_candidate = valid_candidate & (excluded_codes[positions] != codes)
            score = jnp.where(valid_candidate, score, jnp.inf)
            candidate = jnp.where(valid_candidate, candidate, 0)
            best_dist2, best_idx = _merge_cell_candidates_jax(
                best_dist2, best_idx, score, candidate, k
            )
            return base + CELL_CHUNK_SIZE, best_dist2, best_idx

        _base, best_dist2, best_idx = jax.lax.while_loop(
            chunk_cond,
            chunk_body,
            (jnp.int32(0), best_dist2, best_idx),
        )
        return best_dist2, best_idx

    best_dist2, best_idx = jax.lax.fori_loop(
        0, len(CELL_OFFSETS), offset_body, (best_dist2, best_idx)
    )
    return best_idx, best_dist2


def _build_active_vdw_pairs(
    active,
    radii,
    polymer_mask,
    excluded_codes,
    dmax,
    max_neighbors,
    scale=VDW_SCALE_DEFAULT,
    chemistry=None,
    *,
    pair_factors=True,
):
    """Pure-jax sorted-cell neighbour builder matching the torch implementation.

    It is static-shape and JIT/scan compatible; exclusions are applied before K
    candidates are selected by smallest VdW clearance. Hash buckets are traversed completely in
    fixed-width chunks, so no dense or hash-colliding cell silently loses candidates.
    """

    n_atom = active.shape[-2]
    batch = jax.lax.stop_gradient(active.reshape((-1, n_atom, 3)))
    neighbours, best_dist2 = _build_cell_pairs_jax(
        batch,
        batch,
        dmax,
        max_neighbors,
        exclude_self=True,
        query_radii=radii,
        target_radii=radii,
        pair_scale=scale,
        query_polymer=polymer_mask,
        target_polymer=polymer_mask,
        excluded_codes=excluded_codes,
        pair_code_size=n_atom,
        chemistry=chemistry,
    )
    if neighbours.shape[-1] == 0:
        return neighbours, neighbours.astype(active.dtype)
    source = jnp.arange(n_atom, dtype=jnp.int32).reshape((1, n_atom, 1))
    valid = jnp.isfinite(best_dist2)
    if not pair_factors:
        return neighbours, valid.astype(active.dtype)

    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    reverse_neighbours = neighbours[batch_idx, neighbours]
    reverse_valid = valid[batch_idx, neighbours]
    reverse = jnp.any(
        (reverse_neighbours == source[..., None]) & reverse_valid, axis=-1
    )
    pair_factor = valid.astype(active.dtype) / (1.0 + reverse.astype(active.dtype))
    return neighbours, pair_factor


def _build_fixed_vdw_pairs(
    active,
    bg_pos,
    lig_local,
    dmax,
    max_neighbors,
    lig_r=None,
    bg_r=None,
    scale=None,
    chemistry=None,
):
    """Build moving-ligand to fixed-background neighbours at a trial point."""

    n_active = active.shape[-2]
    batch = jax.lax.stop_gradient(active.reshape((-1, n_active, 3)))
    lig = batch[:, lig_local, :]
    neighbours, best_dist2 = _build_cell_pairs_jax(
        lig,
        bg_pos,
        dmax,
        max_neighbors,
        query_radii=lig_r,
        target_radii=bg_r,
        pair_scale=scale,
        chemistry=chemistry,
    )
    return neighbours, jnp.isfinite(best_dist2).astype(active.dtype)


def _safe_vdw_diff_jax(diff, source, target, canonical):
    if canonical:
        lo = jnp.minimum(source, target)
        hi = jnp.maximum(source, target)
        code = lo * 31 + hi
        orientation = jnp.where(source <= target, 1.0, -1.0)
    else:
        code = source * 31 + target
        orientation = 1.0
    axis = code % 3
    base_sign = jnp.where((code // 3) % 2 == 0, 1.0, -1.0)
    unit = jnp.stack((axis == 0, axis == 1, axis == 2), axis=-1).astype(diff.dtype)
    fallback = VDW_OVERLAP_EPS * (base_sign * orientation)[..., None] * unit
    effective = diff + jax.lax.stop_gradient(fallback - diff)
    norm2 = jnp.sum(diff * diff, axis=-1)
    return jnp.where((norm2 < VDW_OVERLAP_EPS**2)[..., None], effective, diff)


def _vdw_pair_energy(
    active,
    bg_pos,
    lig_local,
    neighbours,
    pair_mask,
    lig_r,
    bg_r,
    scale,
    weight,
    chemistry=None,
):
    """Fixed-background VdW energy over the per-step neighbour list."""

    n_active, n_bg = active.shape[-2], bg_pos.shape[-2]
    batch = active.reshape((-1, n_active, 3))
    background = bg_pos.reshape((-1, n_bg, 3))
    lig = batch[:, lig_local, :]
    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    other = background[batch_idx, neighbours]
    diff = lig[:, :, None, :] - other
    source = lig_local.reshape((1, -1, 1))
    diff = _safe_vdw_diff_jax(diff, source, neighbours, canonical=False)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS)
    if chemistry is None:
        contact = lig_r[None, :, None] + bg_r[neighbours]
        inverse = 1 / 0.2**2
    else:
        query = jnp.arange(lig_local.shape[0], dtype=jnp.int32).reshape((1, -1, 1))
        contact, inverse, valid = pair_parameters(
            get_ops("jax"), chemistry, query, neighbours
        )
        pair_mask = pair_mask * valid
    r_min = scale * contact
    delta = jnp.minimum(dist - r_min, 0.0)
    return weight * jnp.sum(pair_mask * inverse * delta**2)


def _active_vdw_pair_energy(
    active,
    neighbours,
    pair_factor,
    radii,
    scale,
    weight,
    chemistry=None,
    *,
    source=None,
):
    """VdW energy over the per-step active-active neighbour list."""

    n_atom = active.shape[-2]
    batch = active.reshape((-1, n_atom, 3))
    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    other = batch[batch_idx, neighbours]
    if source is None:
        source = jnp.arange(n_atom, dtype=jnp.int32).reshape((1, n_atom, 1))
    query = batch[batch_idx, source]
    diff = query - other
    diff = _safe_vdw_diff_jax(diff, source, neighbours, canonical=True)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS)
    if chemistry is None:
        contact = radii[source] + radii[neighbours]
        inverse = 1 / 0.2**2
    else:
        contact, inverse, valid = pair_parameters(
            get_ops("jax"), chemistry, source, neighbours
        )
        pair_factor = pair_factor * valid
    r_min = scale * contact
    delta = jnp.minimum(dist - r_min, 0.0)
    return weight * jnp.sum(pair_factor * inverse * delta**2)


def make_minimizer(
    spec,
    max_iter: int = 100,
    method: str = "cg",
    *,
    line_search=None,
    return_info=False,
):
    """Return a callable pytree ``minimize(coords, sigma, step) -> coords``.

    ``coords`` has shape (..., n_atom, 3); ``step`` is the diffusion step index (for the
    step-window gate, alongside ``sigma`` for the sigma-window gate). The returned function
    is pure and JIT/vmap-able, so it runs inside the diffusion loop's ``hk.scan``/``hk.vmap``
    (``step`` is a traced scalar there). ``method='cg'`` (the default) runs the pure-jax
    ``_cg_minimize``; ``method='l-bfgs'`` uses ``jaxopt.LBFGS`` (lazily imported). Per-restraint
    gating uses the prepared window table and per-term masks. Pass the minimizer
    as an argument to the outermost JIT to keep restraint values dynamic. There is no
    ``start_sigma`` arg. ``return_info=True`` fixes the output as ``(coords, CGInfo)``
    with scalar JAX-array diagnostics; this is supported only for CG.
    """
    line_search = resolve_line_search(method, line_search)
    if return_info and line_search is None:
        raise ValueError("return_info is supported only for method='cg'")
    from rgi_toolkit.optim._jax_state import prepare_minimizer

    return prepare_minimizer(spec, max_iter, line_search, return_info)


@sequential_vmap
def _minimize(minimizer, coords, sigma, step):
    """Run one solve with runtime restraint arrays and static algorithm choices."""
    from rgi_toolkit.optim._gates import window_on
    from rgi_toolkit.optim.info import CGInfo, inactive_info

    parameters, options = minimizer.parameters, minimizer.options
    active_idx = parameters["active_idx"]
    prepared = parameters["prepared"]
    windows = parameters["windows"]
    is_cg = options.line_search is not None

    def result(coords, info):
        if not options.return_info:
            return coords
        dtype = jnp.result_type(coords.dtype, jnp.asarray(0.0).dtype)
        return coords, CGInfo(
            *(jnp.asarray(v, dtype=jnp.int32) for v in info[:4]),
            *(jnp.asarray(v, dtype=dtype) for v in info[4:]),
        )

    custom_terms = [
        (term.parameters, term.closure()) for term in minimizer.custom_terms
    ]
    # Read fixed-background positions at minimize time; they change each diffusion
    # step. Only indices and chemistry belong to the prepared spec.
    from rgi_toolkit.optim._cg import JaxCG, run_cg
    from rgi_toolkit.optim._coordinates import bind_coordinates
    from rgi_toolkit.optim._vdw_runtime import VdwRuntime

    def _descend(coords, sigma, step):
        active = coords[..., active_idx, :]
        prepared_step = jax_energy.bind_peptide_states(active, prepared)
        in_win = (
            (sigma <= prepared["conf_start_sigma"])
            & (sigma >= prepared["conf_stop_sigma"])
            & (step >= prepared["conf_start_step"])
            & (step <= prepared["conf_stop_step"])
        )
        fixed = moving = background = None
        if parameters["fixed"] is not None:
            fixed = dict(parameters["fixed"])
            background = coords[..., fixed.pop("bg_global"), :]
            fixed["weight"] = jnp.where(in_win, fixed["weight"], 0.0)
            fixed["max_neighbors"] = options.fixed_neighbors
        if parameters["moving"] is not None:
            moving = dict(parameters["moving"])
            moving["weight"] = jnp.where(in_win, moving["weight"], 0.0)
            moving["max_neighbors"] = options.moving_neighbors
        runtime = VdwRuntime(
            "jax",
            active,
            fixed=fixed,
            moving=moving,
            background=background,
            skin=parameters["skin"],
        )

        def sparse_energy(a, cache):
            e = jax_energy.total_energy(a, prepared_step, sigma, step)
            e = e + runtime.sparse_energy(a, cache)
            for term, closure in custom_terms:
                start, stop, start_step, stop_step = term["window"]
                gate = (
                    (sigma <= start)
                    & (sigma >= stop)
                    & (step >= start_step)
                    & (step <= stop_step)
                    & (term["weight"] != 0)
                )
                dtype = jax.eval_shape(closure, a).dtype
                e = e + jax.lax.cond(
                    gate, closure, lambda _: jnp.zeros((), dtype=dtype), a
                )
            return e

        sparse_vg = jax.value_and_grad(sparse_energy)

        def value_grad(a, cache):
            f, g = sparse_vg(a, cache)
            dg, df = runtime.dense_value_grad(a, cache)
            return g + dg, f + df

        def prepare(a, cache):
            return runtime.prepare(a, cache, in_win)

        cache = runtime.empty(active)
        if is_cg:
            mapping = bind_coordinates(
                "jax", active, parameters["coordinates"], sigma, step, enabled=in_win
            )

            def physical(u):
                return u if mapping is None else mapping(u, active)

            def mapped_value_grad(u, cache):
                g, f = value_grad(physical(u), cache)
                return (g if mapping is None else mapping(g)), f

            backend = JaxCG(active, lambda u: sparse_energy(physical(u), cache))
            opt, state = run_cg(
                backend,
                mapped_value_grad,
                active,
                options.max_iter,
                line_search=options.line_search,
                cache=cache,
                prepare=lambda u, c: prepare(physical(u), c),
            )
            opt = physical(opt)
            info = state.info
        else:
            from rgi_toolkit.optim._jax_lbfgs import CachedLBFGS

            def lbfgs_value_grad(a, previous_cache):
                current = prepare(a, previous_cache)
                g, f = value_grad(a, current)
                return (f, current), g

            opt = (
                CachedLBFGS(
                    fun=lbfgs_value_grad,
                    value_and_grad=True,
                    has_aux=True,
                    maxiter=options.max_iter,
                    tol=GTOL,
                    linesearch="zoom",
                    implicit_diff=False,
                )
                .run(active, cache)
                .params
            )
            info = inactive_info()
        active = jnp.where(jnp.all(jnp.isfinite(opt)), opt, active)
        return result(coords.at[..., active_idx, :].set(active), info)

    def minimize(coords, sigma, step=0):
        if not options.active:
            return result(coords, inactive_info())
        # Use the same host-spec window table as Torch, with traced scalar gates.
        return jax.lax.cond(
            window_on(windows, sigma, step, jnp),
            lambda c: _descend(c, sigma, step),
            lambda c: result(c, inactive_info()),
            coords,
        )

    # The outer sequential_vmap keeps neighbour rebuilds and overflow sums conditional.
    return minimize(coords, sigma, step)


def energy_of(spec, coords) -> float:
    """Restraint energy at ``coords`` (for stats); host-side, not for the loop."""
    if not spec.is_active():
        return 0.0
    from rgi_toolkit.custom.closure import build_terms

    coords = jnp.asarray(coords)
    active_idx = jnp.asarray(spec.active_sites, dtype=jnp.int32)
    prepared = jax_energy.prepare_spec(spec)
    active = coords[..., active_idx, :]
    energy = jax_energy.total_energy(active, prepared)
    for *_meta, closure in build_terms(spec.custom, "jax"):
        energy = energy + closure(active)
    return float(energy) + dynamic_vdw_energy(spec, coords)


def dynamic_vdw_energy(spec, coords) -> float:
    """Return both optimizer-only VdW residuals for host-side diagnostics."""
    if not spec.is_active():
        return 0.0
    vc = getattr(spec, "vdw_config", None)
    ac = getattr(spec, "active_vdw_config", None)
    has_vdw = vc is not None and vc.weight > 0
    has_active_vdw = ac is not None and ac.weight > 0
    if not (has_vdw or has_active_vdw):
        return 0.0

    coords = jnp.asarray(coords)
    active = coords[..., jnp.asarray(spec.active_sites, dtype=jnp.int32), :]
    from rgi_toolkit.optim._vdw_runtime import VdwRuntime

    def array(value):
        return jnp.asarray(value, dtype=active.dtype)

    fixed = moving = background = None
    if has_vdw:
        background = coords[..., jnp.asarray(vc.background_global, dtype=jnp.int32), :]
        fixed = dict(
            lig_local=jnp.asarray(vc.ligand_local, dtype=jnp.int32),
            lig_r=array(vc.ligand_radii),
            bg_r=array(vc.background_radii),
            scale=array(vc.scale),
            weight=array(vc.weight),
            dmax=array(vc.search_radius),
            max_neighbors=vc.max_neighbors,
            contact=array(vc.max_contact),
            chemistry=prepare_chemistry(get_ops("jax"), vc.chemistry, active),
        )
    if has_active_vdw:
        check_active_vdw_int32_safe(int(ac.radii.shape[0]))
        moving = dict(
            radii=array(ac.radii),
            polymer_mask=jnp.asarray(ac.polymer_mask, dtype=bool),
            excluded_codes=jnp.asarray(ac.excluded_codes, dtype=jnp.int32),
            scale=array(ac.scale),
            weight=array(ac.weight),
            dmax=array(ac.search_radius),
            max_neighbors=ac.max_neighbors,
            contact=array(ac.max_contact),
            chemistry=prepare_chemistry(get_ops("jax"), ac.chemistry, active),
        )
    runtime = VdwRuntime(
        "jax",
        active,
        fixed=fixed,
        moving=moving,
        background=background,
        skin=spec.vdw_neighbor_skin,
    )
    cache = runtime.prepare(active, runtime.empty(active))
    return float(
        runtime.sparse_energy(active, cache)
        + runtime.dense_value_grad(active, cache, gradient=False)[1]
    )
