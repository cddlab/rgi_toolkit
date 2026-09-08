"""Acquire an immutable local snapshot of the public CCP4 monomer library."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

MONOMER_REPOSITORY = "https://github.com/MonomerLibrary/monomers.git"
_TIMEOUT = 900


def cache_directory() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME")
    return (
        (Path(root).expanduser() if root else Path.home() / ".config")
        / "rgi_toolkit"
        / "monomers"
    )


def validate_library_directory(path: Path) -> None:
    """Reject partial downloads before making them available to another process."""
    required = ("list/mon_lib_list.cif", "ener_lib.cif")
    if not path.is_dir():
        raise ValueError(f"monomer library path {str(path)!r} is not a directory")
    if any(not (path / name).is_file() for name in required):
        raise ValueError(
            "conformer_restraints_config.monomer_library: expected a directory "
            f"containing list/mon_lib_list.cif and ener_lib.cif, got {str(path)!r}"
        )
    import gemmi

    # Parse both shared dictionaries, rather than accepting merely nonempty files.
    for name in required:
        if not len(gemmi.cif.read_file(str(path / name))):
            raise ValueError(f"monomer library: empty dictionary {name}")


def revision(path: Path) -> str | None:
    """Read provenance without fetching or changing an existing checkout."""
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def ensure_cached_library() -> str:
    """Clone on first use; reuse a completed snapshot even when offline.

    The lock lives beside the destination, and the staging directory is on the same
    filesystem so publication is atomic. No reader can observe a half-cloned tree.
    """
    from filelock import FileLock

    destination = cache_directory()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(destination.with_name("monomers.lock")), timeout=_TIMEOUT):
            if destination.exists():
                validate_library_directory(destination)
                return str(destination)
            stage = Path(tempfile.mkdtemp(prefix=".monomers-", dir=destination.parent))
            try:
                logger.info(
                    "[rgi_toolkit] downloading monomer library from %s",
                    MONOMER_REPOSITORY,
                )
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--",
                        MONOMER_REPOSITORY,
                        str(stage),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
                validate_library_directory(stage)
                if revision(stage) is None:
                    raise ValueError("monomer library clone has no Git revision")
                stage.rename(destination)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not acquire the monomer library. Retry with network access and Git "
            "available, or supply monomer_library.path to an existing local library."
        ) from exc
    return str(destination)
