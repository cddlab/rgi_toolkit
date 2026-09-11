"""File references exercise the same schema and objective as inline input."""

import copy
import json

import numpy as np
import pytest
import yaml

from rgi_toolkit import AtomRecord, CombinedRestraints, resolve_restraints_config
from rgi_toolkit.config import RestraintsConfig


def _distance():
    return {
        "atom_selection1": "index 0",
        "atom_selection2": "index 1",
        "harmonic": {"target_distance": 2.0},
    }


@pytest.mark.parametrize("extension", ["json", "yaml", "yml"])
def test_every_section_and_nested_paths(tmp_path, monkeypatch, extension):
    entries = {
        "conformer": {},
        "distance": [_distance()],
        "angle": [
            {
                "atom_selection1": "index 0",
                "atom_selection2": "index 1",
                "atom_selection3": "index 2",
                "harmonic": {"target_angle": 90},
            }
        ],
        "plane": [{"atom_selection1": "index 0 to 3"}],
        "rmsd": [{"ref_pdb": "ref.pdb", "harmonic": {"target_rmsd": 0}}],
        "base_pair": [{"residue1": "resid 1", "residue2": "resid 2"}],
        "custom": [
            {
                "energy": "(distance(A, B) - 2)**2",
                "selections": {"A": "index 0", "B": "index 1"},
            }
        ],
    }
    for kind in ("dihedral", "improper", "chiral"):
        entries[kind] = [
            {
                **{f"atom_selection{i + 1}": f"index {i}" for i in range(4)},
                "harmonic": {f"target_{kind}": 180},
            }
        ]
    folder = tmp_path / "configs"
    folder.mkdir()
    raw = {}
    for kind, value in entries.items():
        filename = f"{kind}.{extension}"
        (folder / filename).write_text(
            json.dumps(value) if extension == "json" else yaml.safe_dump(value)
        )
        raw[f"{kind}_restraints_config"] = {"config_path": filename}
    (folder / "rgi.json").write_text(json.dumps(raw))
    reference = {"config_path": "configs/rgi.json"}
    original = copy.deepcopy(reference)
    monkeypatch.chdir(tmp_path.parent)
    resolved = resolve_restraints_config(reference, base_dir=tmp_path)
    assert reference == original
    assert resolved["rmsd_restraints_config"][0]["ref_pdb"] == str(folder / "ref.pdb")
    parsed = RestraintsConfig.from_dict(reference, base_dir=tmp_path)
    assert parsed.conformer_config == {}
    assert len(list(parsed.iter_resolvable_data())) == 7
    assert len(parsed.custom_data) == len(parsed.base_pair_data) == 1


def test_external_resources_and_inline_compatibility(tmp_path):
    raw = {
        "conformer_restraints_config": {"monomer_library": {"path": "monomers"}},
        "custom_restraints_config": [{"refs": {"ref1": {"ref_cif": "target.cif"}}}],
    }
    (tmp_path / "rgi.json").write_text(json.dumps(raw))
    resolved = resolve_restraints_config({"config_path": "rgi.json"}, base_dir=tmp_path)
    assert resolved["conformer_restraints_config"]["monomer_library"]["path"] == str(
        tmp_path / "monomers"
    )
    assert resolved["custom_restraints_config"][0]["refs"]["ref1"]["ref_cif"] == str(
        tmp_path / "target.cif"
    )
    assert resolve_restraints_config(raw, base_dir=tmp_path) == raw


@pytest.mark.parametrize("value", [None, "", 2, []])
def test_invalid_path(tmp_path, value):
    with pytest.raises(ValueError, match="path string"):
        resolve_restraints_config({"config_path": value}, base_dir=tmp_path)


@pytest.mark.parametrize("content", ["{", "null", "[]", '{"config_path":"a.json"}'])
def test_bad_files(tmp_path, content):
    (tmp_path / "a.json").write_text(content)
    with pytest.raises(ValueError, match="a.json"):
        resolve_restraints_config({"config_path": "a.json"}, base_dir=tmp_path)


def test_missing_file_and_mixed_settings(tmp_path):
    with pytest.raises(ValueError, match="missing.json"):
        resolve_restraints_config({"config_path": "missing.json"}, base_dir=tmp_path)
    with pytest.raises(ValueError, match="inline settings"):
        resolve_restraints_config({"config_path": "a.json", "verbose": True})
    with pytest.raises(ValueError, match="whole section"):
        resolve_restraints_config(
            {"distance_restraints_config": [{"config_path": "a.json"}]}
        )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_external_public_setup_objective_matches_inline(tmp_path, backend):
    class Adapter:
        def iter_atoms(self):
            return iter([AtomRecord("A", 1, 0), AtomRecord("B", 1, 1)])

    config = {"distance_restraints_config": [_distance()]}
    (tmp_path / "rgi.json").write_text(json.dumps(config))
    inline, external = CombinedRestraints(), CombinedRestraints()
    inline.setup(Adapter(), config=config)
    external.setup(Adapter(), config={"config_path": str(tmp_path / "rgi.json")})
    np.testing.assert_array_equal(inline.spec.active_sites, external.spec.active_sites)
    from tests.test_monlib_esd import _energy_grad

    coords = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
    a, ga = _energy_grad(inline.spec, coords, backend)
    b, gb = _energy_grad(external.spec, coords, backend)
    assert a == pytest.approx(b)
    if ga is not None:
        np.testing.assert_allclose(ga, gb)
