"""GPU restraint optimizer for JAX tools (alphafold3).

Builds a pure JIT/scan/vmap-compatible minimizer over an autodiff energy.
The default CG follows SciPy 1.17.1 PR+ with DCSRCH/Wolfe2 strong Wolfe through
``optim/_cg.py`` and ``_cg_linesearch.py``, shared with Torch. Failed searches
retain the last accepted point. ``return_info=True`` exposes traced CG diagnostics.
``method='l-bfgs'`` uses ``jaxopt.LBFGS`` (lazily imported). No callback or runtime
SciPy is used; optimization remains inside XLA on the selected device. Backend
floating-point evaluation orders can produce different search decisions near a
condition boundary; ordinary scalar objectives are checked against SciPy.

The dynamic VdW neighbour lists are rebuilt on measured displacement against a Verlet
skin, not on a fixed cadence. ``lax.fori_loop`` needs a static trip count, so the number
of blocks stays fixed and it is the REBUILD that is ``lax.cond``-gated; the neighbour
arrays and the CG state ride in the loop carry, so a block that does not rebuild neither
re-evaluates the energy nor loses the conjugate direction. Keep this in step with
``torch_optim`` — the displacement metric, the 1x/2x skin asymmetry and the constant
movement bound are deliberately identical in both files.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from rgi_toolkit._array_ops import VDW_OVERLAP_EPS, get_ops
from rgi_toolkit._config_util import (
    VDW_MAX_ATOM_STEP_DEFAULT,
    VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT,
    VDW_NEIGHBOR_SKIN_DEFAULT,
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
    max_atom_step=None,
    state=None,
    return_state=False,
    return_info=False,
    **search_options,
):
    """Pure JAX entry to the shared SciPy-style strict-Wolfe CG solver."""
    from rgi_toolkit.optim._cg import jax_cg

    out, result = jax_cg(
        energy_fn,
        x0,
        max_iter,
        gtol=gtol,
        max_atom_step=max_atom_step,
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
    """Build moving-ligand to fixed-background neighbours for the current CG block."""

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
    active, neighbours, pair_factor, radii, scale, weight, chemistry=None
):
    """VdW energy over the per-step active-active neighbour list."""

    n_atom = active.shape[-2]
    batch = active.reshape((-1, n_atom, 3))
    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    other = batch[batch_idx, neighbours]
    diff = batch[:, :, None, :] - other
    source = jnp.arange(n_atom, dtype=jnp.int32).reshape((1, n_atom, 1))
    diff = _safe_vdw_diff_jax(diff, source, neighbours, canonical=True)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS)
    if chemistry is None:
        contact = radii[None, :, None] + radii[neighbours]
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
    return_info=False,
):
    """Return ``minimize(coords, sigma, step) -> coords``.

    ``coords`` has shape (..., n_atom, 3); ``step`` is the diffusion step index (for the
    step-window gate, alongside ``sigma`` for the sigma-window gate). The returned function
    is pure and JIT/vmap-able, so it runs inside the diffusion loop's ``hk.scan``/``hk.vmap``
    (``step`` is a traced scalar there). ``method='cg'`` (the default) runs the pure-jax
    ``_cg_minimize``; any other value uses ``jaxopt.LBFGS`` (lazily imported). Per-restraint
    gating uses the host-spec window table and per-term masks. There is no
    ``start_sigma`` arg. ``return_info=True`` fixes the output as ``(coords, CGInfo)``
    with scalar JAX-array diagnostics; this is supported only for CG.
    """
    active_idx = jnp.asarray(spec.active_sites, dtype=jnp.int32)
    prepared = jax_energy.prepare_spec(spec)
    is_cg = (method or "cg").lower() in ("cg", "ncg", "nonlinear-cg", "nonlinearcg")
    if return_info and not is_cg:
        raise ValueError("return_info is supported only for method='cg'")
    from rgi_toolkit.optim._gates import active_windows, window_on
    from rgi_toolkit.optim.info import CGInfo, inactive_info

    windows = jnp.asarray(active_windows(spec))

    def result(coords, info):
        if not return_info:
            return coords
        dtype = jnp.result_type(coords.dtype, jnp.asarray(0.0).dtype)
        return coords, CGInfo(
            *(jnp.asarray(v, dtype=jnp.int32) for v in info[:4]),
            *(jnp.asarray(v, dtype=dtype) for v in info[4:]),
        )

    has_builtin = spec.has_conformer() or spec.has_per_entry()
    # Custom closures use static selection indices so they trace inside lax.scan.
    has_custom = spec.has_custom()
    from rgi_toolkit.custom.closure import build_terms

    custom_terms = build_terms(spec.custom, "jax") if has_custom else []
    # Read fixed-background positions at minimize time; they change each diffusion
    # step. Only indices and chemistry belong to the prepared spec.
    _vc = getattr(spec, "vdw_config", None)
    has_vdw = _vc is not None and _vc.weight > 0
    if has_vdw:
        vdw_lig_local = jnp.asarray(_vc.ligand_local, dtype=jnp.int32)
        vdw_lig_r = jnp.asarray(_vc.ligand_radii)
        vdw_bg_global = jnp.asarray(_vc.background_global, dtype=jnp.int32)
        vdw_bg_r = jnp.asarray(_vc.background_radii)
        vdw_scale = jnp.asarray(float(_vc.scale))
        vdw_weight = jnp.asarray(float(_vc.weight))
        vdw_dmax = jnp.asarray(_vc.search_radius)
        vdw_max_neighbors = int(_vc.max_neighbors)
        vdw_chemistry = prepare_chemistry(get_ops("jax"), _vc.chemistry, vdw_lig_r)
    _ac = getattr(spec, "active_vdw_config", None)
    has_active_vdw = _ac is not None and _ac.weight > 0
    if has_active_vdw:
        # Pair-code encoding must fit in int32 (see spec.py).
        check_active_vdw_int32_safe(int(_ac.radii.shape[0]))
        active_vdw_radii = jnp.asarray(_ac.radii)
        active_vdw_polymer = jnp.asarray(_ac.polymer_mask, dtype=bool)
        active_vdw_excluded = jnp.asarray(_ac.excluded_codes, dtype=jnp.int32)
        active_vdw_scale = jnp.asarray(float(_ac.scale))
        active_vdw_weight = jnp.asarray(float(_ac.weight))
        active_vdw_dmax = jnp.asarray(_ac.search_radius)
        active_vdw_max_neighbors = int(_ac.max_neighbors)
        active_vdw_chemistry = prepare_chemistry(
            get_ops("jax"), _ac.chemistry, active_vdw_radii
        )
    has_static_vdw = spec.has_array_term("vdw")
    has_any_vdw = has_static_vdw or has_vdw or has_active_vdw
    vdw_step_limit = float(
        getattr(spec, "vdw_max_atom_step", VDW_MAX_ATOM_STEP_DEFAULT)
    )
    # Check staleness at this interval; measured displacement triggers rebuilds.
    vdw_rebuild_interval = int(
        getattr(
            spec,
            "vdw_neighbor_rebuild_interval",
            VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT,
        )
    )
    vdw_skin = float(getattr(spec, "vdw_neighbor_skin", VDW_NEIGHBOR_SKIN_DEFAULT))
    if has_any_vdw:
        conf_ss = jnp.asarray(float(spec.conf_start_sigma))
        conf_stop = jnp.asarray(float(getattr(spec, "conf_stop_sigma", -1.0)))
        conf_sstep = jnp.asarray(float(getattr(spec, "conf_start_step", float("-inf"))))
        conf_estep = jnp.asarray(float(getattr(spec, "conf_stop_step", float("inf"))))

    def _descend(coords, sigma, step):
        active = coords[..., active_idx, :]
        info = inactive_info()
        prepared_step = jax_energy.bind_peptide_states(active, prepared)
        if has_builtin or has_vdw or has_active_vdw or has_custom:
            if has_any_vdw:
                _s = jnp.asarray(sigma)
                _st = jnp.asarray(step)
                in_win = (
                    (_s <= conf_ss)
                    & (_s >= conf_stop)
                    & (_st >= conf_sstep)
                    & (_st <= conf_estep)
                )
                step_cap = jnp.where(in_win, vdw_step_limit, jnp.inf)
            else:
                step_cap = None
            if has_vdw:
                bg_pos = coords[..., vdw_bg_global, :]
                vdw_w = jnp.where(in_win, vdw_weight, 0.0)
            if has_active_vdw:
                active_vdw_w = jnp.where(in_win, active_vdw_weight, 0.0)

            def build_dynamic_pairs(a, fixed_cutoff, active_cutoff):
                fixed_neighbours = fixed_mask = None
                active_neighbours = active_factor = None
                if has_vdw:
                    fixed_neighbours, fixed_mask = _build_fixed_vdw_pairs(
                        a,
                        bg_pos,
                        vdw_lig_local,
                        fixed_cutoff,
                        vdw_max_neighbors,
                        vdw_lig_r,
                        vdw_bg_r,
                        vdw_scale,
                        vdw_chemistry,
                    )
                if has_active_vdw:
                    active_neighbours, active_factor = _build_active_vdw_pairs(
                        a,
                        active_vdw_radii,
                        active_vdw_polymer,
                        active_vdw_excluded,
                        active_cutoff,
                        active_vdw_max_neighbors,
                        active_vdw_scale,
                        active_vdw_chemistry,
                    )
                return (
                    fixed_neighbours,
                    fixed_mask,
                    active_neighbours,
                    active_factor,
                )

            def empty_dynamic_pairs(a):
                """Static-shape zero neighbour lists for an inactive VdW window."""
                n_batch = a.reshape((-1, a.shape[-2], 3)).shape[0]
                fixed_neighbours = fixed_mask = None
                active_neighbours = active_factor = None
                if has_vdw:
                    n_neighbour = min(vdw_max_neighbors, bg_pos.shape[-2])
                    shape = (
                        n_batch,
                        vdw_lig_local.shape[0],
                        n_neighbour,
                    )
                    fixed_neighbours = jnp.zeros(shape, dtype=jnp.int32)
                    fixed_mask = jnp.zeros(shape, dtype=a.dtype)
                if has_active_vdw:
                    n_neighbour = min(active_vdw_max_neighbors, max(0, a.shape[-2] - 1))
                    shape = (n_batch, a.shape[-2], n_neighbour)
                    active_neighbours = jnp.zeros(shape, dtype=jnp.int32)
                    active_factor = jnp.zeros(shape, dtype=a.dtype)
                return (
                    fixed_neighbours,
                    fixed_mask,
                    active_neighbours,
                    active_factor,
                )

            def initial_dynamic_pairs(a, fixed_cutoff, active_cutoff):
                # Another restraint can keep _descend active after the conformer window
                # closes. Avoid the O(N log N) cell-list build when both dynamic VdW
                # weights are zero while preserving the exact static carry shapes.
                if not (has_vdw or has_active_vdw):
                    return None, None, None, None
                return jax.lax.cond(
                    in_win,
                    lambda x: build_dynamic_pairs(x, fixed_cutoff, active_cutoff),
                    empty_dynamic_pairs,
                    a,
                )

            def energy_fn(
                a,
                fixed_neighbours,
                fixed_mask,
                active_neighbours,
                active_factor,
            ):
                e = jax_energy.total_energy(a, prepared_step, sigma, step)
                if has_vdw:
                    e = e + _vdw_pair_energy(
                        a,
                        bg_pos,
                        vdw_lig_local,
                        fixed_neighbours,
                        fixed_mask,
                        vdw_lig_r,
                        vdw_bg_r,
                        vdw_scale,
                        vdw_w,
                        vdw_chemistry,
                    )
                if has_active_vdw:
                    e = e + _active_vdw_pair_energy(
                        a,
                        active_neighbours,
                        active_factor,
                        active_vdw_radii,
                        active_vdw_scale,
                        active_vdw_w,
                        active_vdw_chemistry,
                    )
                for _name, start, stop, start_step, stop_step, closure in custom_terms:
                    _sc = jnp.asarray(sigma)
                    _stc = jnp.asarray(step)
                    gate = (
                        (_sc <= start)
                        & (_sc >= stop)
                        & (_stc >= start_step)
                        & (_stc <= stop_step)
                    )
                    # Multiplying a disabled expression by zero still propagates NaNs.
                    # Match its abstract scalar dtype without evaluating its value.
                    dtype = jax.eval_shape(closure, a).dtype
                    e = e + jax.lax.cond(
                        gate, closure, lambda _: jnp.zeros((), dtype=dtype), a
                    )
                return e

            if is_cg and (has_vdw or has_active_vdw):
                n_blocks = max(
                    1, (max_iter + vdw_rebuild_interval - 1) // vdw_rebuild_interval
                )
                # Bound unchecked travel using the constant check interval, not a traced
                # block length. Active-active pairs need twice the fixed-background allowance.
                movement = vdw_step_limit * vdw_rebuild_interval
                fixed_cutoff = None
                active_cutoff = None
                if has_vdw:
                    _mr = jnp.asarray(_vc.max_contact)
                    fixed_cutoff = jnp.maximum(vdw_dmax, _mr + movement + vdw_skin)
                if has_active_vdw:
                    _mr = jnp.asarray(_ac.max_contact)
                    active_cutoff = jnp.maximum(
                        active_vdw_dmax, _mr + 2.0 * movement + vdw_skin
                    )

                # Initialize concrete carry shapes before the loop. Infinite reference
                # coordinates cannot force a rebuild: inf - inf is NaN.
                _n0, _m0, _a0, _f0 = initial_dynamic_pairs(
                    active, fixed_cutoff, active_cutoff
                )
                # The CG state's dtypes must match the loop carry EXACTLY (fori_loop demands
                # an invariant carry), and the energy dtype is not simply the coord dtype --
                # it depends on x64 and on the prepared arrays. Take the exact structure from
                # an abstract trace rather than guessing: eval_shape runs no computation.
                _probe = jax.eval_shape(
                    lambda a: _cg_minimize(
                        lambda x: energy_fn(x, _n0, _m0, _a0, _f0),
                        a,
                        0,
                        max_atom_step=step_cap,
                        return_state=True,
                    ),
                    active,
                )
                carry = {
                    "x": active,
                    "stopped": jnp.asarray(False),
                    # Preserve the named-tuple pytree, including aggregate diagnostics.
                    "cg": jax.tree.map(
                        lambda s: jnp.zeros(s.shape, s.dtype), _probe[1]
                    ),
                }
                if has_vdw:
                    carry["fn"], carry["fm"] = _n0, _m0
                    carry["fref"] = active[..., vdw_lig_local, :]
                if has_active_vdw:
                    carry["an"], carry["af"] = _a0, _f0
                    carry["aref"] = active

                def block_body(block_index, c):
                    def run_block(c):
                        a = c["x"]
                        new = dict(c)
                        rebuilt = jnp.asarray(False)
                        if has_vdw:
                            lig = a[..., vdw_lig_local, :]
                            # Skip rebuilds outside the conformer window, where VdW has zero weight
                            # and the infinite step cap would otherwise force every block to rebuild.
                            need = jnp.logical_and(
                                in_win, _max_disp(lig, c["fref"]) > vdw_skin
                            )
                            nb, mask = jax.lax.cond(
                                need,
                                lambda: _build_fixed_vdw_pairs(
                                    a,
                                    bg_pos,
                                    vdw_lig_local,
                                    fixed_cutoff,
                                    vdw_max_neighbors,
                                    vdw_lig_r,
                                    vdw_bg_r,
                                    vdw_scale,
                                    vdw_chemistry,
                                ),
                                lambda: (c["fn"], c["fm"]),
                            )
                            new["fn"], new["fm"] = nb, mask
                            new["fref"] = jnp.where(need, lig, c["fref"])
                            rebuilt = jnp.logical_or(rebuilt, need)
                        if has_active_vdw:
                            # both endpoints move, so half the budget each
                            need = jnp.logical_and(
                                in_win, _max_disp(a, c["aref"]) > 0.5 * vdw_skin
                            )
                            nb, factor = jax.lax.cond(
                                need,
                                lambda: _build_active_vdw_pairs(
                                    a,
                                    active_vdw_radii,
                                    active_vdw_polymer,
                                    active_vdw_excluded,
                                    active_cutoff,
                                    active_vdw_max_neighbors,
                                    active_vdw_scale,
                                    active_vdw_chemistry,
                                ),
                                lambda: (c["an"], c["af"]),
                            )
                            new["an"], new["af"] = nb, factor
                            new["aref"] = jnp.where(need, a, c["aref"])
                            rebuilt = jnp.logical_or(rebuilt, need)

                        pairs = (
                            new.get("fn"),
                            new.get("fm"),
                            new.get("an"),
                            new.get("af"),
                        )
                        block_iters = jnp.minimum(
                            vdw_rebuild_interval,
                            max_iter - block_index * vdw_rebuild_interval,
                        )
                        # A rebuilt pair list invalidates the carried objective history.
                        cg_in = c["cg"]._replace(
                            valid=jnp.logical_and(
                                c["cg"].valid, jnp.logical_not(rebuilt)
                            )
                        )
                        updated, cg_out = _cg_minimize(
                            lambda x: energy_fn(x, *pairs),
                            a,
                            block_iters,
                            max_atom_step=step_cap,
                            state=cg_in,
                            return_state=True,
                        )
                        new["x"] = updated
                        new["cg"] = cg_out
                        new["stopped"] = jnp.logical_not(cg_out.valid)
                        return new

                    return jax.lax.cond(c["stopped"], lambda c: c, run_block, c)

                final = jax.lax.fori_loop(0, n_blocks, block_body, carry)
                opt, info = final["x"], final["cg"].info
            elif is_cg:
                opt, info = _cg_minimize(
                    lambda a: energy_fn(a, None, None, None, None),
                    active,
                    max_iter,
                    max_atom_step=step_cap,
                    return_info=True,
                )
            else:
                import jaxopt  # only the non-default l-bfgs method needs jaxopt

                def lbfgs_energy(a):
                    # Every line-search trial needs neighbours at its own coordinates.
                    pairs = initial_dynamic_pairs(
                        a,
                        vdw_dmax if has_vdw else None,
                        active_vdw_dmax if has_active_vdw else None,
                    )
                    return energy_fn(a, *pairs)

                opt = (
                    jaxopt.LBFGS(
                        fun=lbfgs_energy,
                        maxiter=max_iter,
                        linesearch="backtracking",
                        implicit_diff=False,
                    )
                    .run(active)
                    .params
                )
            active = jnp.where(jnp.all(jnp.isfinite(opt)), opt, active)
        return result(coords.at[..., active_idx, :].set(active), info)

    def minimize(coords, sigma, step=0):
        if not spec.is_active():
            return result(coords, inactive_info())
        # Use the same host-spec window table as Torch, with traced scalar gates.
        return jax.lax.cond(
            window_on(windows, sigma, step, jnp),
            lambda c: _descend(c, sigma, step),
            lambda c: result(c, inactive_info()),
            coords,
        )

    return minimize


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
    dtype = active.dtype
    total = 0.0
    if has_vdw:
        lig_local = jnp.asarray(vc.ligand_local, dtype=jnp.int32)
        lig_r = jnp.asarray(vc.ligand_radii, dtype=dtype)
        bg_r = jnp.asarray(vc.background_radii, dtype=dtype)
        scale = jnp.asarray(float(vc.scale), dtype=dtype)
        chemistry = prepare_chemistry(get_ops("jax"), vc.chemistry, active)
        bg_pos = coords[..., jnp.asarray(vc.background_global, dtype=jnp.int32), :]
        neighbours, pair_mask = _build_fixed_vdw_pairs(
            active,
            bg_pos,
            lig_local,
            jnp.asarray(vc.search_radius, dtype=dtype),
            int(vc.max_neighbors),
            lig_r,
            bg_r,
            scale,
            chemistry,
        )
        total += float(
            _vdw_pair_energy(
                active,
                bg_pos,
                lig_local,
                neighbours,
                pair_mask,
                lig_r,
                bg_r,
                scale,
                jnp.asarray(float(vc.weight), dtype=dtype),
                chemistry,
            )
        )
    if has_active_vdw:
        check_active_vdw_int32_safe(int(ac.radii.shape[0]))
        radii = jnp.asarray(ac.radii, dtype=dtype)
        scale = jnp.asarray(float(ac.scale), dtype=dtype)
        chemistry = prepare_chemistry(get_ops("jax"), ac.chemistry, active)
        neighbours, pair_factor = _build_active_vdw_pairs(
            active,
            radii,
            jnp.asarray(ac.polymer_mask, dtype=bool),
            jnp.asarray(ac.excluded_codes, dtype=jnp.int32),
            jnp.asarray(ac.search_radius, dtype=dtype),
            int(ac.max_neighbors),
            scale,
            chemistry,
        )
        total += float(
            _active_vdw_pair_energy(
                active,
                neighbours,
                pair_factor,
                radii,
                scale,
                jnp.asarray(float(ac.weight), dtype=dtype),
                chemistry,
            )
        )
    return total
