"""Parse the ``restraints_config`` dict shared by all tools.

One source of truth for defaults and the distance-restraint encoding, so boltz
(YAML), protenix (JSON) and AF3 only need to extract the dict from their input
format and hand it here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

from rgi_toolkit import monlib_geom
from rgi_toolkit._config_paths import resolve_restraints_config
from rgi_toolkit._config_util import (
    check_window_exclusive,
    coerce_bool,
    finite_float,
    validate_vdw_config,
)
from rgi_toolkit._mol_build import parse_relax_force_field
from rgi_toolkit.base_pair_restr_data import BasePairData
from rgi_toolkit.custom.data import CustomData
from rgi_toolkit.distance_restr_data import DistanceData
from rgi_toolkit.group_geom_restr_data import (
    AngleRestraintData,
    DihedralRestraintData,
    ImproperRestraintData,
)
from rgi_toolkit.plane_restr_data import PlaneRestraintData, count_plane_groups
from rgi_toolkit.ref_geom_restr_data import RefGeomData, is_ref_anchored
from rgi_toolkit.rmsd_restr_data import RmsdData


@dataclass(frozen=True)
class _EntryRoute:
    section: str
    destination: str
    data_type: type
    ref_geom: str | None = None
    ref_group_counter: object | None = None


_ENTRY_ROUTES = (
    _EntryRoute(
        "distance_restraints_config", "distance_data", DistanceData, "distance"
    ),
    _EntryRoute("rmsd_restraints_config", "rmsd_data", RmsdData),
    _EntryRoute("angle_restraints_config", "angle_data", AngleRestraintData, "angle"),
    _EntryRoute(
        "dihedral_restraints_config",
        "dihedral_data",
        DihedralRestraintData,
        "dihedral",
    ),
    _EntryRoute(
        "improper_restraints_config",
        "improper_data",
        ImproperRestraintData,
        "improper",
    ),
    _EntryRoute(
        "plane_restraints_config",
        "plane_data",
        PlaneRestraintData,
        "plane",
        count_plane_groups,
    ),
)

# Shared with config validation tools so every entry type is discovered.
RESTRAINT_SECTIONS = tuple(route.section for route in _ENTRY_ROUTES) + (
    "conformer_restraints_config",
    "custom_restraints_config",
    "base_pair_restraints_config",
)


def _entries(config: dict, section: str):
    entries = config.get(section)
    if entries is None:
        return ()
    if not isinstance(entries, (list, tuple)) or any(
        not isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(f"{section} must be a list of mappings")
    return entries


@dataclass
class RestraintsConfig:
    verbose: bool = False
    gpu: bool = True
    method: str = "CG"
    max_iter: int = 100
    # Shared conformer window; +inf starts at the first diffusion step.
    conf_start_sigma: float = float("inf")
    conf_stop_sigma: float = -1.0  # shared conformer lower bound; -1 = never released
    # Inclusive step window, mutually exclusive with an explicit sigma window.
    conf_start_step: float = float("-inf")
    conf_stop_step: float = float("inf")
    conformer_config: dict | None = None
    distance_data: list = field(default_factory=list)
    rmsd_data: list = field(default_factory=list)
    angle_data: list = field(default_factory=list)  # group-centroid angle restraints
    dihedral_data: list = field(
        default_factory=list
    )  # group-centroid dihedral restraints
    improper_data: list = field(
        default_factory=list
    )  # group-centroid improper restraints
    plane_data: list = field(
        default_factory=list
    )  # standalone best-fit-plane restraints
    custom_data: list = field(
        default_factory=list
    )  # custom restraints (rgi_toolkit.custom)
    base_pair_data: list = field(
        default_factory=list
    )  # nucleic-acid base-pair restraints (expand to distance + plane)

    def iter_resolvable_data(self):
        """Yield every ordinary built-in entry that resolves against an adapter."""
        for route in _ENTRY_ROUTES:
            yield from getattr(self, route.destination)

    @classmethod
    def from_dict(cls, config: dict | None, *, base_dir=None) -> "RestraintsConfig":
        config = resolve_restraints_config(config, base_dir=base_dir)
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise ValueError("restraints_config must be a mapping")
        _KNOWN_TOP_LEVEL = {
            "verbose",
            "gpu",
            "method",
            "max_iter",
            "conformer_restraints_config",
            "custom_restraints_config",
            "base_pair_restraints_config",
        } | {route.section for route in _ENTRY_ROUTES}
        _unknown_top = set(config) - _KNOWN_TOP_LEVEL - {"start_sigma"}
        if _unknown_top:
            hint = ""
            if "backend" in _unknown_top:
                hint = (
                    " Note: 'backend' is no longer configurable — it is inferred from "
                    "how the engine is invoked (get_minimizer() => jax; minimize() "
                    "with a torch/numpy array => torch)."
                )
            raise ValueError(
                f"restraints_config: unknown top-level key(s) {sorted(_unknown_top)}. "
                f"Known keys: {sorted(_KNOWN_TOP_LEVEL)}. A misspelled section name "
                f"(e.g. 'distance_restraint_config') would silently drop the whole "
                f"restraint block, so it is rejected here.{hint}"
            )
        if "start_sigma" in config:
            raise ValueError(
                "restraints_config: top-level 'start_sigma' is not supported — set it on "
                "each distance_restraints_config entry and inside conformer_restraints_config "
                "(or omit it: a restraint with no start_sigma is active at every step)."
            )
        _ALWAYS_ON = float("inf")  # omitted start_sigma -> active at every step
        conformer_config = config.get("conformer_restraints_config")
        conformer_config = {} if conformer_config is None else conformer_config
        if not isinstance(conformer_config, dict):
            raise ValueError("conformer_restraints_config must be a mapping")
        if "dihedral" in conformer_config:
            raise ValueError(
                "conformer_restraints_config: 'dihedral' was renamed to 'cistrans' "
                "(it restrains acyclic double bonds' cis/trans (E/Z) geometry)."
            )
        if "improper" in conformer_config:
            raise ValueError(
                "conformer_restraints_config: 'improper' was renamed to 'plane' "
                "(a best-fit-plane restraint over aromatic rings + sp2 groups)."
            )
        if "planarity" in conformer_config:
            raise ValueError(
                "conformer_restraints_config: 'planarity' was renamed to 'plane' "
                "(it is now a servalcat-style best-fit-plane restraint over whole planar "
                "atom groups — aromatic/conjugated rings + non-ring sp2 groups — not just "
                "per-centre sp2 signed volume)."
            )
        known_conformer_keys = {
            "start_sigma",
            "stop_sigma",
            "start_step",
            "stop_step",
            "bond",
            "angle",
            "chiral",
            "plane",
            "cistrans",
            "vdw",
            "monomer_library",
            "relax_force_field",
        }
        unknown_conformer = {
            key
            for key in conformer_config
            if not str(key).startswith("_") and key not in known_conformer_keys
        }
        if unknown_conformer:
            raise ValueError(
                "conformer_restraints_config: unknown key(s) "
                f"{sorted(unknown_conformer)}. Known keys: "
                f"{sorted(known_conformer_keys)}"
            )
        validate_vdw_config(conformer_config)
        for term in ("bond", "angle", "chiral", "cistrans", "plane"):
            block = conformer_config.get(term)
            if block is None:
                continue
            if not isinstance(block, dict):
                raise ValueError(
                    f"conformer_restraints_config.{term} must be a mapping"
                )
            unknown_term = {
                key
                for key in block
                if not str(key).startswith("_") and key not in {"weight", "slack"}
            }
            if unknown_term:
                raise ValueError(
                    f"conformer_restraints_config.{term}: unknown key(s) "
                    f"{sorted(unknown_term)}. Known keys: ['slack', 'weight']"
                )
            for key in ("weight", "slack"):
                value = block.get(key)
                if value is not None:
                    parsed = finite_float(value, f"conformer {term} {key}")
                    if key == "slack" and parsed < 0:
                        raise ValueError(f"conformer {term} slack must be >= 0")
        # Validate nested options even when no entity opts into conformer restraints.
        monlib_geom.parse_config(conformer_config)
        parse_relax_force_field(conformer_config)
        check_window_exclusive(conformer_config, "conformer_restraints_config")
        _css = conformer_config.get("start_sigma")
        conf_start_sigma = float(_css) if _css is not None else _ALWAYS_ON
        _csstop = conformer_config.get("stop_sigma")
        conf_stop_sigma = float(_csstop) if _csstop is not None else -1.0
        _csa = conformer_config.get("start_step")
        conf_start_step = float(_csa) if _csa is not None else float("-inf")
        _cso = conformer_config.get("stop_step")
        conf_stop_step = float(_cso) if _cso is not None else float("inf")
        # Quoted booleans such as "false" must not use Python's string truthiness.
        gpu = coerce_bool(config.get("gpu", True))
        method = config.get("method", "CG")
        _valid_methods = {"cg", "ncg", "nonlinear-cg", "nonlinearcg", "l-bfgs", "lbfgs"}
        if str(method).lower() not in _valid_methods:
            raise ValueError(
                f"unknown method {method!r}: expected a CG alias "
                "(cg/ncg/nonlinear-cg/nonlinearcg) or l-bfgs (l-bfgs/lbfgs)"
            )
        max_iter = config.get("max_iter", 100)
        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, Integral)
            or max_iter < 0
        ):
            raise ValueError("max_iter must be an integer >= 0")
        cfg = cls(
            verbose=coerce_bool(config.get("verbose", False)),
            gpu=gpu,
            method=method,
            max_iter=int(max_iter),
            conf_start_sigma=conf_start_sigma,
            conf_stop_sigma=conf_stop_sigma,
            conf_start_step=conf_start_step,
            conf_stop_step=conf_stop_step,
            conformer_config=(
                conformer_config
                if config.get("conformer_restraints_config") is not None
                else None
            ),
        )
        for route in _ENTRY_ROUTES:
            destination = getattr(cfg, route.destination)
            for entry in _entries(config, route.section):
                if route.ref_geom is not None and is_ref_anchored(entry):
                    n_groups = (
                        route.ref_group_counter(entry)
                        if route.ref_group_counter is not None
                        else None
                    )
                    reference = RefGeomData(route.ref_geom, n_groups=n_groups)
                    reference.set_config(entry)
                    cfg.custom_data.append(reference)
                    continue
                restraint = route.data_type()
                restraint.set_config(entry)
                if restraint.start_sigma is None:
                    restraint.start_sigma = _ALWAYS_ON
                destination.append(restraint)
        # custom restraints (expression DSL / code fn). start_sigma None -> +inf (active
        # every step) is applied when the CustomSpec is built (CustomData.build_spec).
        for entry in _entries(config, "custom_restraints_config"):
            cd = CustomData()
            cd.set_config(entry)
            cfg.custom_data.append(cd)
        for entry in _entries(config, "base_pair_restraints_config"):
            bp = BasePairData()
            bp.set_config(entry)
            if bp.start_sigma is None:
                bp.start_sigma = _ALWAYS_ON
            cfg.base_pair_data.append(bp)
        return cfg
