"""Fetch the receptor and matching ligand from the pinned PLINDER release."""

import hashlib
import time
from pathlib import Path
from zipfile import ZipFile

from common import protocol, read_json, slug, write_json


def archive_members(url, members):
    import aiohttp
    import fsspec

    for attempt in range(3):
        try:
            with (
                fsspec.open(
                    url,
                    "rb",
                    block_size=65536,
                    headers={"Accept-Encoding": "identity"},
                    client_kwargs={"timeout": aiohttp.ClientTimeout(total=60)},
                ) as stream,
                ZipFile(stream) as archive,
            ):
                return {key: archive.read(member) for key, member in members.items()}
        except (aiohttp.ClientError, OSError, ValueError) as error:
            if isinstance(error, ValueError) and "range requests" not in str(error):
                raise
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def pose_inputs(root, system, source_root=None):
    system = slug(system)
    parts = system.split("__")
    if len(parts) != 4 or len(parts[0]) != 4:
        raise ValueError(
            "Supply receptor and reference_ligand columns for custom systems"
        )
    base = protocol("conformer")["pose_reference"]["url"].rstrip("/")
    members = {
        "receptor": (f"{system}/receptor.pdb", f"receptor/{system}.pdb"),
        "reference_ligand": (
            f"{system}/ligand_files/{parts[-1]}.sdf",
            f"reference_ligand/{system}.sdf",
        ),
    }
    root = Path(root)
    metadata = root / "pose_sources" / f"{system}.json"
    if metadata.is_file() and source_root is None:
        cached = read_json(metadata)
        if cached.get("url") == base and all(
            (root / relative).is_file()
            and hashlib.sha256((root / relative).read_bytes()).hexdigest()
            == cached.get("sha256", {}).get(key)
            for key, (_, relative) in members.items()
        ):
            return {key: relative for key, (_, relative) in members.items()}
    if source_root is not None:
        payloads = {
            key: (Path(source_root) / member).read_bytes()
            for key, (member, _) in members.items()
        }
    else:
        url = f"{base}/systems/{parts[0][1:3]}.zip"
        payloads = archive_members(
            url, {key: member for key, (member, _) in members.items()}
        )
    for key, (_, relative) in members.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payloads[key])
        temporary.replace(path)
    write_json(
        metadata,
        {
            "url": base,
            "system": system,
            "sha256": {
                key: hashlib.sha256(value).hexdigest()
                for key, value in payloads.items()
            },
        },
    )
    return {key: relative for key, (_, relative) in members.items()}
