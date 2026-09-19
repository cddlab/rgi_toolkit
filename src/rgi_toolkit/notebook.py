"""Small, framework-independent helpers for interactive RGI front ends.

Forms produce the ordinary shared configuration. The engine remains responsible
for parsing, atom selection, geometry and optimization.
"""

from __future__ import annotations

import math
import re

import numpy as np

from rgi_toolkit.config import RestraintsConfig, resolve_restraints_config
from rgi_toolkit.selection import AtomSelector


def residue_selection(chain: str, residues: str, atoms: str = "all") -> str:
    """Translate a chain and one-based ranges such as ``5-20,31`` to the DSL."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", chain.strip()):
        raise ValueError("Choose one chain ID from the input chain table.")
    groups = []
    for value in residues.split(","):
        match = re.fullmatch(r"\s*([1-9]\d*)\s*(?:-\s*([1-9]\d*))?\s*", value)
        if not match:
            raise ValueError("Residues must be one-based numbers or ranges: 5-20,31.")
        start, end = int(match[1]), int(match[2] or match[1])
        if end < start:
            raise ValueError("The last residue must be at least the first residue.")
        groups.append(f"resid {start} to {end}")
    suffixes = {"all": "", "CA": " and name CA", "backbone": " and backbone"}
    if atoms not in suffixes:
        raise ValueError("Atoms must be all, CA, or backbone.")
    selection = f"chain {chain.strip()} and ({' or '.join(groups)}){suffixes[atoms]}"
    AtomSelector(selection)
    return selection


def make_config(
    preset: str,
    *,
    selection1: str = "",
    selection2: str = "",
    distance: float = 25.0,
    tolerance: float = 0.0,
    custom: str = "",
    config_path: str = "",
    base_dir=None,
) -> dict:
    """Build a distance/ligand preset, or load a full shared YAML/JSON config.

    A nonzero tolerance creates a free interval around the requested distance.
    Ligand presets still require the host to opt the intended entities in.
    """
    presets = {"distance", "ligand_geometry", "distance+ligand_geometry", "custom"}
    if preset not in presets:
        raise ValueError(
            f"Unknown RGI preset {preset!r}; choose one of {sorted(presets)}."
        )
    if preset == "custom":
        if bool(custom.strip()) == bool(config_path.strip()):
            raise ValueError(
                "Provide either YAML/JSON text or a config file, not both."
            )
        if config_path.strip():
            config = {"config_path": config_path.strip()}
        else:
            import yaml

            config = yaml.safe_load(custom)
            if not isinstance(config, dict):
                raise ValueError("RGI YAML/JSON must contain a configuration mapping.")
            if set(config) == {"restraints_config"}:
                config = config["restraints_config"]
        config = resolve_restraints_config(config, base_dir=base_dir)
    else:
        config = {"verbose": True}
        if "distance" in preset:
            distance, tolerance = float(distance), float(tolerance)
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError(
                    "Distance must be a positive finite number in Angstrom."
                )
            if not math.isfinite(tolerance) or not 0 <= tolerance < distance:
                raise ValueError(
                    "Tolerance must be nonnegative and smaller than distance."
                )
            AtomSelector(selection1)
            AtomSelector(selection2)
            penalty = (
                {"harmonic": {"target_distance": distance}}
                if tolerance == 0
                else {
                    "flat-bottomed": {
                        "target_distance1": distance - tolerance,
                        "target_distance2": distance + tolerance,
                    }
                }
            )
            config["distance_restraints_config"] = [
                {
                    "atom_selection1": selection1,
                    "atom_selection2": selection2,
                    **penalty,
                }
            ]
        if "ligand_geometry" in preset:
            config["conformer_restraints_config"] = {"plane": {"weight": 1.0}}
    if not isinstance(config, dict):
        raise ValueError("RGI config must be a mapping.")
    RestraintsConfig.from_dict(config)
    return config


def restraint_inventory(restraints) -> dict[str, int]:
    """Report actual built rows, including separately represented dynamic VdW."""
    from rgi_toolkit.energy._terms import TERM_DEFS

    spec = restraints.spec
    counts = {"active_atoms": len(spec.active_sites)}
    for term in TERM_DEFS:
        arrays = getattr(spec, term.spec_attr)
        counts[term.key] = 0 if arrays is None else int(np.count_nonzero(arrays.mask))
    counts["custom"] = len(spec.custom)
    counts["vdw_background_atoms"] = (
        0 if spec.vdw_config is None else len(spec.vdw_config.background_global)
    )
    counts["vdw_active_atoms"] = (
        0 if spec.active_vdw_config is None else len(spec.active_vdw_config.radii)
    )
    return counts


def distance_report(restraints, coords) -> list[dict]:
    """Measure final centroid distances without treating energy as success."""
    positions = np.asarray(coords)
    if positions.ndim == 2:
        positions = positions[None]
    report = []
    for entry in restraints.config.distance_data:
        first = positions[:, entry.target_sites1].mean(axis=1)
        second = positions[:, entry.target_sites2].mean(axis=1)
        values = np.linalg.norm(first - second, axis=-1)
        kind = entry.distance_restraint_type
        lower = entry.target_distance if kind == "harmonic" else entry.target_distance1
        upper = entry.target_distance if kind == "harmonic" else entry.target_distance2
        report.append(
            {
                "selection1": entry.atom_selection1,
                "selection2": entry.atom_selection2,
                "atoms1": len(entry.target_sites1),
                "atoms2": len(entry.target_sites2),
                "type": kind,
                "lower": lower if kind != "flat-bottomed2" else None,
                "upper": upper if kind != "flat-bottomed1" else None,
                "distances_angstrom": values.tolist(),
            }
        )
    return report
