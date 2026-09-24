"""JAX pytrees separating restraint arrays from static solver/program choices."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from rgi_toolkit._array_ops import get_ops
from rgi_toolkit.custom.closure import build_closure
from rgi_toolkit.custom.dsl import parse_formula
from rgi_toolkit.energy import jax_energy
from rgi_toolkit.energy._nonbonded import prepare_chemistry
from rgi_toolkit.optim._coordinates import CentroidCoordinates
from rgi_toolkit.optim._gates import active_windows
from rgi_toolkit.spec import check_active_vdw_int32_safe


def _array(value):
    value = jnp.asarray(value)
    return value.astype(jnp.int32) if value.dtype.kind in "iu" else value


@dataclass(frozen=True)
class _CustomProgram:
    """Only code and selection labels belong to a custom term's static key."""

    kind: str
    formula: str | None
    fn: object
    selection_refs: tuple
    move_free: tuple
    groups: tuple
    geom: str

    @property
    def ast(self):
        return parse_formula(self.formula) if self.formula is not None else None


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class _CustomTerm:
    parameters: dict
    program: _CustomProgram

    def tree_flatten(self):
        return (self.parameters,), self.program

    @classmethod
    def tree_unflatten(cls, program, children):
        return cls(children[0], program)

    def closure(self):
        from rgi_toolkit.custom.backends import get_ops

        return build_closure(self.program, get_ops("jax"), parameters=self.parameters)


def _custom_term(spec):
    parameters = dict(
        selections=spec.selections,
        refs=spec.refs,
        ref_fits=spec.ref_fits,
        ref_blocks=spec.ref_blocks,
        group_indices=tuple(
            payload[0] if kind == "pred" else None for kind, payload in spec.groups
        ),
        weight=spec.weight,
        target1=spec.target1,
        target2=spec.target2,
        geom_type_code=spec.geom_type_code,
        window=[spec.start_sigma, spec.stop_sigma, spec.start_step, spec.stop_step],
    )
    parameters["window"] = jnp.asarray(parameters["window"])
    program = _CustomProgram(
        kind=spec.kind,
        formula=ast.unparse(spec.ast) if spec.ast is not None else None,
        fn=spec.fn,
        selection_refs=tuple(sorted(spec.selection_refs.items())),
        move_free=tuple(sorted(spec.move_free.items())),
        groups=tuple(
            (kind, (None, payload[1]) if kind == "pred" else tuple(payload))
            for kind, payload in spec.groups
        ),
        geom=spec.geom,
    )
    return _CustomTerm(jax.tree.map(_array, parameters), program)


@dataclass(frozen=True)
class _SolverOptions:
    max_iter: int
    gtol: float
    line_search: str | None
    return_info: bool
    active: bool
    fixed_neighbors: int | None
    moving_neighbors: int | None


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class JaxMinimizer:
    """Callable minimizer whose numeric restraint data are dynamic JIT arguments.

    Pass this object as an argument to the outermost ``jax.jit`` function to reuse
    compilation when values change without changing array shapes or program options.
    Closing over it remains supported, but embeds its values in the compiled program.
    """

    parameters: dict
    custom_terms: tuple
    options: _SolverOptions

    def tree_flatten(self):
        return (self.parameters, self.custom_terms), self.options

    @classmethod
    def tree_unflatten(cls, options, children):
        return cls(*children, options)

    def __call__(self, coords, sigma, step=0):
        from rgi_toolkit.optim.jax_optim import _minimize

        return _minimize(self, coords, sigma, step)

    def is_active(self):
        return self.options.active

    def minimize_gpu(self, positions, sigma, step=0):
        """Apply the scan hook while preserving the predictor's coordinate shape."""
        return self(positions.reshape(-1, 3), sigma, step).reshape(positions.shape)


def prepare_minimizer(spec, max_iter, line_search, return_info, gtol):
    """Prepare runtime arrays once on the host; retain only graph choices as metadata."""
    fixed = moving = None
    vc, ac = spec.vdw_config, spec.active_vdw_config
    has_vdw = vc is not None and vc.weight > 0
    has_active_vdw = ac is not None and ac.weight > 0
    if has_vdw:
        fixed = dict(
            lig_local=vc.ligand_local,
            lig_r=vc.ligand_radii,
            bg_global=vc.background_global,
            bg_r=vc.background_radii,
            scale=float(vc.scale),
            weight=float(vc.weight),
            dmax=vc.search_radius,
            contact=vc.max_contact,
            chemistry=prepare_chemistry(
                get_ops("jax"), vc.chemistry, jnp.asarray(vc.ligand_radii)
            ),
        )
    if has_active_vdw:
        check_active_vdw_int32_safe(int(ac.radii.shape[0]))
        moving = dict(
            radii=ac.radii,
            polymer_mask=ac.polymer_mask,
            excluded_codes=ac.excluded_codes,
            scale=float(ac.scale),
            weight=float(ac.weight),
            dmax=ac.search_radius,
            contact=ac.max_contact,
            chemistry=prepare_chemistry(
                get_ops("jax"), ac.chemistry, jnp.asarray(ac.radii)
            ),
        )
    parameters = dict(
        active_idx=spec.active_sites,
        prepared=jax_energy.prepare_spec(spec),
        windows=active_windows(spec),
        fixed=fixed,
        moving=moving,
        skin=float(spec.vdw_neighbor_skin),
        coordinates=CentroidCoordinates(spec).parameters(),
    )
    return JaxMinimizer(
        jax.tree.map(_array, parameters),
        tuple(_custom_term(term) for term in spec.custom if term.weight != 0),
        _SolverOptions(
            max_iter,
            gtol,
            line_search,
            return_info,
            spec.is_active(),
            int(vc.max_neighbors) if has_vdw else None,
            int(ac.max_neighbors) if has_active_vdw else None,
        ),
    )
