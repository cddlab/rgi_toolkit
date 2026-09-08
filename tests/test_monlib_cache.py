"""First-use acquisition, process exclusion, and offline cache reuse."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rgi_toolkit import _monlib_cache as cache
from rgi_toolkit import monlib_geom
from rgi_toolkit.config import RestraintsConfig


def _git(*args):
    return subprocess.run(
        ["git", *map(str, args)], check=True, text=True, capture_output=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "list").mkdir()
    (source / "list" / "mon_lib_list.cif").write_text(
        "data_link_list\n_chem_link.id .\n"
    )
    (source / "ener_lib.cif").write_text("data_energy\n_lib_atom.type .\n")
    _git("init", source)
    _git("-C", source, "add", ".")
    _git(
        "-C",
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Fixture dictionary\n\nCo-authored-by: Codex <noreply@openai.com>",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(cache, "MONOMER_REPOSITORY", source.as_uri())
    return source


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (False, None),
        (True, (None, "fallback")),
        ({}, (None, "fallback")),
        ({"on_missing": "error"}, (None, "error")),
        ("monomers", ("monomers", "fallback")),
        ({"path": "monomers", "on_missing": "error"}, ("monomers", "error")),
    ],
)
def test_configuration_parsing_never_acquires_a_library(monkeypatch, value, expected):
    def forbidden():
        pytest.fail("parsing performed cache or network I/O")

    monkeypatch.setattr(cache, "ensure_cached_library", forbidden)
    config = {"bond": {}, "monomer_library": value}
    assert monlib_geom.parse_config(config) == expected
    RestraintsConfig.from_dict({"conformer_restraints_config": config})


def test_cache_location_follows_xdg_or_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert cache.cache_directory() == tmp_path / ".config/rgi_toolkit/monomers"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert cache.cache_directory() == tmp_path / "xdg/rgi_toolkit/monomers"


def test_concurrent_first_use_clones_once_then_reuses_offline(repository, monkeypatch):
    run = cache.subprocess.run
    clones = []

    def counted(args, **kwargs):
        if args[:2] == ["git", "clone"]:
            clones.append(args)
        return run(args, **kwargs)

    monkeypatch.setattr(cache.subprocess, "run", counted)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: cache.ensure_cached_library(), range(4)))
    assert len(set(paths)) == len(clones) == 1
    destination = Path(paths[0])
    assert (destination / ".git/shallow").is_file()
    assert cache.revision(destination) == _git("-C", repository, "rev-parse", "HEAD")

    def offline(*args, **kwargs):
        pytest.fail("completed cache attempted an external command")

    monkeypatch.setattr(cache.subprocess, "run", offline)
    assert cache.ensure_cached_library() == paths[0]
    assert not list(destination.parent.glob(".monomers-*"))


def test_first_use_is_also_safe_across_processes(repository):
    script = (
        "from rgi_toolkit import _monlib_cache as c; "
        f"c.MONOMER_REPOSITORY = {repository.as_uri()!r}; "
        "print(c.ensure_cached_library())"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    outputs = [p.communicate(timeout=60) for p in processes]
    assert all(p.returncode == 0 for p in processes), outputs
    assert {out.strip() for out, _ in outputs} == {str(cache.cache_directory())}


def test_clone_failure_leaves_no_published_or_partial_cache(repository, monkeypatch):
    monkeypatch.setattr(cache, "MONOMER_REPOSITORY", (repository / "absent").as_uri())
    with pytest.raises(RuntimeError, match="Could not acquire"):
        cache.ensure_cached_library()
    assert not cache.cache_directory().exists()
    assert not list(cache.cache_directory().parent.glob(".monomers-*"))
    monkeypatch.setattr(cache, "MONOMER_REPOSITORY", repository.as_uri())
    assert Path(cache.ensure_cached_library()).is_dir()


def test_invalid_checkout_is_never_published(repository):
    (repository / "ener_lib.cif").write_text("")
    _git("-C", repository, "add", ".")
    _git(
        "-C",
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Broken dictionary\n\nCo-authored-by: Codex <noreply@openai.com>",
    )
    with pytest.raises(ValueError, match="empty dictionary"):
        cache.ensure_cached_library()
    assert not cache.cache_directory().exists()
    assert not list(cache.cache_directory().parent.glob(".monomers-*"))


def test_existing_invalid_cache_and_explicit_path_do_not_trigger_network(
    repository, monkeypatch
):
    def forbidden():
        pytest.fail("explicit path requested automatic acquisition")

    monkeypatch.setattr(cache, "ensure_cached_library", forbidden)
    monlib_geom.MonomerLibrary.load(str(repository), [])
    with pytest.raises(ValueError, match="not a directory"):
        monlib_geom.MonomerLibrary.load(str(repository / "absent"), [])


def test_unused_dictionary_setting_does_not_download(monkeypatch):
    from rgi_toolkit.polymer import _load_library

    def forbidden(*args, **kwargs):
        pytest.fail("inactive geometry attempted to load a dictionary")

    monkeypatch.setattr(monlib_geom.MonomerLibrary, "load", forbidden)
    for config in (
        {"monomer_library": True, "vdw": {}},
        {"monomer_library": True, "bond": {"weight": 0}},
    ):
        assert not _load_library(config, [], []).atoms
