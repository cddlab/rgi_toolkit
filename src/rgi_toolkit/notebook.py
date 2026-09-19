"""Small, framework-independent helpers for interactive RGI front ends.

Forms produce the ordinary shared configuration. The engine remains responsible
for parsing, atom selection, geometry and optimization.
"""

from __future__ import annotations

import copy
import re

import numpy as np

from rgi_toolkit.config import (
    RESTRAINT_SECTIONS,
    RestraintsConfig,
    resolve_restraints_config,
)
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


FORM_SECTIONS = {
    "distance": "distance_restraints_config",
    "conformer": "conformer_restraints_config",
    "angle": "angle_restraints_config",
    "custom": "custom_restraints_config",
    "RMSD": "rmsd_restraints_config",
}


def make_config(config=None, *, config_text="", config_path="", base_dir=None):
    """Validate one ordinary RGI mapping, YAML/JSON text, or external file.

    ``custom`` keeps its toolkit meaning: a custom energy entry. Whole-config
    text and files are separate input methods, not restraint types.
    """
    sources = (config is not None, bool(config_text.strip()), bool(config_path.strip()))
    if sum(sources) != 1:
        raise ValueError("Provide exactly one config mapping, YAML/JSON text, or file.")
    if config_path.strip():
        config = {"config_path": config_path.strip()}
    elif config_text.strip():
        import yaml

        config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise ValueError("RGI config must be a mapping.")
    if set(config) == {"restraints_config"}:
        config = config["restraints_config"]
    config = resolve_restraints_config(copy.deepcopy(config), base_dir=base_dir)
    if not isinstance(config, dict):
        raise ValueError("RGI config must be a mapping.")
    RestraintsConfig.from_dict(config)
    if not any(
        config.get(key) is not None
        and (key == "conformer_restraints_config" or bool(config[key]))
        for key in RESTRAINT_SECTIONS
    ):
        raise ValueError("Add at least one enabled restraint.")
    return config


def compose_config(items, *, settings=None, base_dir=None):
    """Append native entries without losing repeated types or mixed restraints.

    Each item is a ``(type, native_entry)`` pair. Conformer is one shared mapping
    in the toolkit; its entity opt-ins belong to the predictor input.
    """
    config = copy.deepcopy(settings) if settings is not None else {"verbose": True}
    if not isinstance(config, dict):
        raise ValueError("Global settings must be a mapping.")
    for kind, entry in items:
        if kind not in FORM_SECTIONS:
            raise ValueError(
                f"Choose an RGI restraint type: {', '.join(FORM_SECTIONS)}."
            )
        section = FORM_SECTIONS[kind]
        if not isinstance(entry, dict):
            raise ValueError(f"{section} entries must be mappings.")
        if kind == "conformer":
            if config.get(section) is not None:
                raise ValueError(
                    "conformer uses one shared configuration; select multiple chains."
                )
            config[section] = copy.deepcopy(entry)
        else:
            config.setdefault(section, []).append(copy.deepcopy(entry))
    return make_config(config, base_dir=base_dir)


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
