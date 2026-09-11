"""Resolve local JSON/YAML configuration references before schema validation."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _path(value, base, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a nonempty path string")
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return (base / path).resolve()


def _resources(value, base):
    """Copy a file's value, anchoring only known structure/dictionary paths."""
    if isinstance(value, (list, tuple)):
        return [_resources(item, base) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _resources(item, base) for key, item in value.items()}
    for key in ("ref_pdb", "ref_cif"):
        if isinstance(result.get(key), str):
            result[key] = str(_path(result[key], base, key))
    library = result.get("monomer_library")
    if isinstance(library, str):
        result["monomer_library"] = str(_path(library, base, "monomer_library"))
    elif isinstance(library, dict) and isinstance(library.get("path"), str):
        library["path"] = str(_path(library["path"], base, "monomer_library.path"))
    return result


def resolve_restraints_config(config, *, base_dir=None):
    """Return a new configuration with every section-level ``config_path`` expanded.

    Paths are relative to their containing file, or ``base_dir`` (the working
    directory by default) for a Python dictionary. Existing inline resource paths
    are unchanged. Files contain the replacement value, without a host-job wrapper.
    No file is read for an entirely inline configuration.
    """
    from rgi_toolkit.config import RESTRAINT_SECTIONS

    sections = set(RESTRAINT_SECTIONS)

    def resolve(value, base, stack, label, root=False):
        if isinstance(value, dict) and "config_path" in value:
            if set(value) != {"config_path"}:
                raise ValueError(
                    f"{label}: config_path cannot accompany inline settings"
                )
            path = _path(value["config_path"], base, f"{label}.config_path")
            if path in stack:
                chain = " -> ".join(str(p) for p in (*stack, path))
                raise ValueError(f"{label}: circular config_path reference: {chain}")
            try:
                text = path.read_text(encoding="utf-8")
                if path.suffix.lower() in (".yaml", ".yml"):
                    import yaml

                    loaded = yaml.safe_load(text)
                elif path.suffix.lower() == ".json":
                    loaded = json.loads(text)
                else:
                    raise ValueError("config_path requires a .json, .yaml or .yml file")
                if loaded is None:
                    raise ValueError(
                        "referenced configuration must not be null or empty"
                    )
                return _resources(
                    resolve(loaded, path.parent, (*stack, path), label, root),
                    path.parent,
                )
            except (OSError, ValueError) as exc:
                raise ValueError(f"{label}: config_path {path}: {exc}") from exc
            except Exception as exc:
                # YAML syntax errors do not inherit ValueError.
                if type(exc).__module__.startswith("yaml"):
                    raise ValueError(f"{label}: config_path {path}: {exc}") from exc
                raise
        if root:
            if value is None:
                return None
            if not isinstance(value, dict):
                raise ValueError(f"{label} must be a mapping")
            return {
                key: resolve(item, base, stack, f"{label}.{key}")
                if key in sections
                else _copy(item)
                for key, item in value.items()
            }
        if value is None:
            return None
        conformer = label.endswith(".conformer_restraints_config")
        if conformer:
            if not isinstance(value, dict):
                raise ValueError(f"{label} must be a mapping")
        elif not isinstance(value, (list, tuple)) or any(
            not isinstance(item, dict) for item in value
        ):
            raise ValueError(f"{label} must be a list of mappings")
        elif any("config_path" in item for item in value):
            raise ValueError(
                f"{label}: config_path replaces the whole section, not an entry"
            )
        return _copy(value)

    return resolve(
        config, Path(base_dir or Path.cwd()).resolve(), (), "restraints_config", True
    )


def _copy(value):
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    return value
