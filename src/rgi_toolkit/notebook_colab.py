"""Native Colab fields, without a running kernel or a widget frontend.

The template is copied into notebook source so Colab can render its ordinary
forms before execution. Field values become the same native config as any other
RGI client; this module adds no selection or energy implementation.
"""

from __future__ import annotations

import copy
import json

from rgi_toolkit.config import resolve_restraints_config
from rgi_toolkit.notebook import compose_config

COLAB_FORM = """#@title RGI (optional)
#@markdown Fill in this form before **Runtime → Run all**. Leave `use_rgi` off for vanilla. Leave unused restraint
#@markdown fields empty.
use_rgi = False #@param {type:"boolean"}
#@markdown **distance** — enter two atom selections and their target distance in Å. Example: `chain A and resid
#@markdown 1 to 10` and `chain A and resid 40 to 50`, with `target_distance = 25`.
distance_atom_selection1 = "" #@param {type:"string"}
distance_atom_selection2 = "" #@param {type:"string"}
target_distance = 25 #@param {type:"raw"}
#@markdown **conformer** — enter chain IDs such as `B,C`, or `ligands` for all ligand chains. Leave empty to
#@markdown disable.
conformer_chains = "" #@param {type:"string"}
#@markdown **angle** — enter three atom selections and the target in degrees. Selection 2 is the vertex. Leave
#@markdown all three empty to disable.
angle_atom_selection1 = "" #@param {type:"string"}
angle_atom_selection2 = "" #@param {type:"string"}
angle_atom_selection3 = "" #@param {type:"string"}
target_angle = 90 #@param {type:"raw"}
#@markdown **custom** — set named selections and an energy expression, e.g. `harmonic(distance(A, B), 25.0)`.
#@markdown Leave the energy empty to disable.
custom_selections = {"A": "chain A and resid 1 to 10", "B": "chain A and resid 40 to 50"} #@param {type:"raw"}
custom_energy = "" #@param {type:"string"}
#@markdown **RMSD** — upload a reference using Colab's **Files → Upload** and enter its filename in either
#@markdown `ref_pdb` or `ref_cif`. Leave both filenames empty to disable. Empty target/ref selections mean the
#@markdown whole structure; e.g. `chain A and name CA` selects C-alpha atoms.
ref_pdb = "" #@param {type:"string"}
ref_cif = "" #@param {type:"string"}
atom_selection_target = "" #@param {type:"string"}
atom_selection_ref = "" #@param {type:"string"}
target_rmsd = 0 #@param {type:"raw"}
#@markdown **Multiple restraints:** enter lists in the same fields, e.g. `distance_atom_selection2 = ["chain A
#@markdown and resid 20", "chain A and resid 40"]` and `target_distance = [15, 25]`. A single selection/value is
#@markdown shared across the entries. All non-single lists must have the same length.
#@markdown **Selections:** all fields accept the full RGI-toolkit selection language. Residues start at 1 within
#@markdown each chain. The input cell prints the chain IDs. Distance and angle use atom-group centroids.
#@markdown **Advanced (optional)** — an ordinary RGI-toolkit dict for additional restraints, flat-bottomed
#@markdown penalties, weights, activation windows or conformer settings. Lists append to the fields above. Use
#@markdown `{"config_path": "restraints.yaml"}` to read a file. [Examples and native config
#@markdown reference](https://github.com/cddlab/rgi_toolkit/blob/main/docs/colabfold.md).
restraints_config = {} #@param {type:"raw"}
#@markdown After changing this form, rerun this cell and the prediction cell, or use **Run all**. Settings are
#@markdown saved in the notebook itself.
"""


def _values(value, field):
    """Accept a native scalar or a JSON list in a Colab string field."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'{field}: use a JSON list, e.g. ["chain A", "chain B"].'
                ) from error
    result = value if isinstance(value, list) else [value]
    if not result:
        raise ValueError(f"{field}: the list is empty; clear the field to disable.")
    return result


def _rows(values):
    columns = {key: _values(value, key) for key, value in values.items()}
    size = max(map(len, columns.values()))
    for key, column in columns.items():
        if len(column) not in (1, size):
            raise ValueError(
                f"{key}: got {len(column)} entries; expected 1 or {size}. "
                "Corresponding lists must have the same length."
            )
    return [
        {
            key: copy.deepcopy(col[0 if len(col) == 1 else i])
            for key, col in columns.items()
        }
        for i in range(size)
    ]


def _filled(value):
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def config_from_fields(fields, *, base_dir=None):
    """Read a native notebook form afresh, ignoring all values in vanilla mode.

    Basic fields create harmonic restraints. Advanced entries retain the entire
    native schema and append rather than replace basic entries. Conformer remains
    one shared config whose selected chains are returned separately.
    """
    if not fields.get("use_rgi", False):
        return None, ""
    settings = fields.get("restraints_config", {})
    if not isinstance(settings, dict):
        raise ValueError("restraints_config must be an RGI-toolkit dict, or {}.")
    if set(settings) == {"restraints_config"}:
        settings = settings["restraints_config"]
    settings = resolve_restraints_config(copy.deepcopy(settings), base_dir=base_dir)
    if not isinstance(settings, dict):
        raise ValueError("restraints_config must be an RGI-toolkit dict.")
    settings.setdefault("verbose", True)
    items = []
    for kind, count, default in (("distance", 2, 25), ("angle", 3, 90)):
        names = [f"{kind}_atom_selection{i}" for i in range(1, count + 1)]
        if not any(_filled(fields.get(name, "")) for name in names):
            continue
        for name in names:
            if not _filled(fields.get(name, "")):
                raise ValueError(f"{name} is required; enter an atom selection.")
        target = f"target_{kind}"
        columns = {name: fields[name] for name in names}
        columns[target] = fields.get(target, default)
        for row in _rows(columns):
            entry = {f"atom_selection{i}": row[name] for i, name in enumerate(names, 1)}
            entry["harmonic"] = {target: row[target]}
            items.append((kind, entry))

    chains = fields.get("conformer_chains", "").strip()
    if chains and settings.get("conformer_restraints_config") is None:
        items.append(("conformer", {}))
    if settings.get("conformer_restraints_config") is not None and not chains:
        raise ValueError(
            "Set conformer_chains to the chain IDs to restrain, or ligands."
        )

    if _filled(fields.get("custom_energy", "")):
        for row in _rows(
            {
                "custom_energy": fields["custom_energy"],
                "custom_selections": fields.get("custom_selections", {}),
            }
        ):
            items.append(
                (
                    "custom",
                    {
                        "energy": row["custom_energy"],
                        "selections": row["custom_selections"],
                    },
                )
            )

    references = [key for key in ("ref_pdb", "ref_cif") if _filled(fields.get(key, ""))]
    if len(references) > 1:
        raise ValueError(
            "Fill either ref_pdb or ref_cif, not both; use restraints_config for mixed formats."
        )
    if references:
        key = references[0]
        columns = {
            key: fields[key],
            "atom_selection_target": fields.get("atom_selection_target", ""),
            "atom_selection_ref": fields.get("atom_selection_ref", ""),
            "target_rmsd": fields.get("target_rmsd", 0),
        }
        for row in _rows(columns):
            target = row.pop("target_rmsd")
            items.append(
                (
                    "RMSD",
                    {k: v for k, v in row.items() if v}
                    | {"harmonic": {"target_rmsd": target}},
                )
            )
    if not items and not any(key.endswith("_restraints_config") for key in settings):
        raise ValueError(
            "RGI is on but no restraints are set. In RGI (optional), fill the two "
            "distance_atom_selection fields and target_distance, select conformer_chains, "
            "or configure another type. Leave use_rgi off for vanilla."
        )
    return compose_config(items, settings=settings, base_dir=base_dir), chains
