from __future__ import annotations

import logging
from dataclasses import dataclass

from rgi_toolkit._config_util import (
    apply_window_params,
    parse_geom_type,
    parse_move_indices,
    warn_unknown_keys,
)
from rgi_toolkit.atom_context import FrameworkAdapter, candidate_dict
from rgi_toolkit.selection import AtomSelector

logger = logging.getLogger(__name__)

_KNOWN_DISTANCE_KEYS = {
    "atom_selection1",
    "atom_selection2",
    "calc_method",
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
    "move",
    "harmonic",
    "flat-bottomed",
    "flat-bottomed1",
    "flat-bottomed2",
}


@dataclass
class DistanceData:
    atom_selection1: str
    atom_selection2: str
    target_distance: float
    target_distance1: float  # used in flat-bottomed, flat-bottomed1
    target_distance2: float  # used in flat-bottomed, flat-bottomed2
    distance_restraint_type: str | None  # harmonic / flat-bottomed /
    # flat-bottomed1 / flat-bottomed2 (assigned by set_config)
    target_sites1: list
    target_sites2: list
    calc_method: str  # ["unfixed-absolute"]
    run_restr: bool
    start_sigma: float  # apply this restraint only when noise level <= start_sigma
    stop_sigma: float  # RELEASE this restraint when noise level < stop_sigma (-1=never)
    start_step: float  # step-window lower bound (-inf = always); XOR the sigma window
    stop_step: float  # step-window upper bound (+inf = always)
    move_mode: int  # 0=both / 1=grp1 only / 2=grp2 only (the 'move' config key)
    # Per-restraint least-squares weight, balancing competing distance targets.
    weight: float

    def __init__(self):
        self.atom_selection1 = None
        self.atom_selection2 = None
        self.target_distance = None
        self.target_distance1 = None
        self.target_distance2 = None
        self.distance_restraint_type = None
        self.target_sites1 = None
        self.target_sites2 = None
        self.calc_method = None
        self.run_restr = None
        self.start_sigma = None
        self.stop_sigma = -1.0
        self.start_step = float("-inf")
        self.stop_step = float("inf")
        self.move_mode = 0
        self.weight = 1.0

    def set_config(self, config: dict):
        warn_unknown_keys(
            config, _KNOWN_DISTANCE_KEYS, "distance_restraints_config entry", logger
        )
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.calc_method = config.get("calc_method", "unfixed-absolute")
        apply_window_params(self, config, "distance_restraints_config entry")
        # Map the shared group-selection vocabulary onto the distance move-mode enum.
        idx = parse_move_indices(config.get("move"), 2)  # {1}, {2}, or {1, 2}
        if idx is not None:
            self.move_mode = {
                frozenset({1, 2}): 0,
                frozenset({1}): 1,
                frozenset({2}): 2,
            }[frozenset(idx)]
        gtype, t1, t2 = parse_geom_type(config, "target_distance", float)
        self.distance_restraint_type = gtype
        if gtype == "harmonic":
            self.target_distance = t1
        elif gtype == "flat-bottomed":
            self.target_distance1, self.target_distance2 = t1, t2
        elif gtype == "flat-bottomed1":
            self.target_distance1 = t1
        elif gtype == "flat-bottomed2":
            self.target_distance2 = t2
        self.run_restr = (
            (self.atom_selection1 is not None)
            and (self.atom_selection2 is not None)
            and (self.distance_restraint_type is not None)
        )

        if self.calc_method not in ["unfixed-absolute"]:
            raise ValueError("calc_method must be unfixed-absolute")

        if not self.run_restr:
            raise ValueError("distance restraints not run")

        logger.info(f"{self.distance_restraint_type=}")

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        """Resolve atom indices for distance restraint sites using a framework
        adapter."""
        if not self.run_restr:
            return

        self.target_sites1 = []
        self.target_sites2 = []

        atom_selector1 = AtomSelector(self.atom_selection1)
        atom_selector2 = AtomSelector(self.atom_selection2)

        for atom in adapter.iter_atoms():
            candidate = candidate_dict(atom)
            if atom_selector1.eval(candidate):
                self.target_sites1.append(atom.index)
            if atom_selector2.eval(candidate):
                self.target_sites2.append(atom.index)

        if len(self.target_sites1) == 0:
            raise ValueError(
                f"distance restraint atom_selection1 matched no atoms: "
                f"{self.atom_selection1!r}"
            )
        if len(self.target_sites2) == 0:
            raise ValueError(
                f"distance restraint atom_selection2 matched no atoms: "
                f"{self.atom_selection2!r}"
            )

        logger.info(
            "distance restraint resolved: group1=%d atoms, group2=%d atoms",
            len(self.target_sites1),
            len(self.target_sites2),
        )

    def iter_global_sites(self):
        """Yield every resolved global coordinate index used by this restraint."""
        yield from self.target_sites1 or ()
        yield from self.target_sites2 or ()

    def is_valid(self) -> bool:
        return self.run_restr
