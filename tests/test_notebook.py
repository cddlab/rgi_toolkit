"""Validate notebook presets against real configuration and selection behavior."""

import json

import numpy as np
import pytest

from rgi_toolkit.atom_context import AtomRecord
from rgi_toolkit.combined import CombinedRestraints
from rgi_toolkit.notebook import (
    distance_report,
    make_config,
    residue_selection,
    restraint_inventory,
)
from rgi_toolkit.selection import AtomSelector


class Adapter:
    def iter_atoms(self):
        yield AtomRecord("A", 1, 0, name="CA", mol_type="protein")
        yield AtomRecord("A", 2, 1, name="CA", mol_type="protein")
        yield AtomRecord("B", 1, 2, name="C1")


def test_chain_qualified_range_excludes_other_chains():
    selector = AtomSelector(residue_selection("A", "1-10,31", "CA"))
    assert selector.matches({"chain": "A", "resid": 31, "name": "CA"})
    assert not selector.matches({"chain": "B", "resid": 1, "name": "CA"})
    assert not selector.matches({"chain": "A", "resid": 11, "name": "CA"})


@pytest.mark.parametrize("residues", ["", "0", "-1", "9-2", "1,,3", "1 to 2"])
def test_invalid_ranges_fail_before_prediction(residues):
    with pytest.raises(ValueError):
        residue_selection("A", residues)


@pytest.mark.parametrize(
    "distance,tolerance", [(0, 0), (float("nan"), 0), (4, 4), (4, -1)]
)
def test_invalid_distances_fail_before_prediction(distance, tolerance):
    with pytest.raises(ValueError):
        make_config(
            "distance",
            selection1="chain A",
            selection2="chain B",
            distance=distance,
            tolerance=tolerance,
        )


def test_preset_builds_real_spec_and_measures_each_sample():
    config = make_config(
        "distance", selection1="chain A", selection2="chain B", distance=25, tolerance=2
    )
    restraints = CombinedRestraints()
    restraints.setup(Adapter(), config=config)
    assert restraint_inventory(restraints)["distance"] == 1
    coords = np.array(
        [[[0, 0, 0], [2, 0, 0], [25, 0, 0]], [[0, 0, 0], [2, 0, 0], [28, 0, 0]]]
    )
    report = distance_report(restraints, coords)[0]
    assert report["distances_angstrom"] == [24, 27]
    assert (report["lower"], report["upper"]) == (23, 27)
    assert (report["atoms1"], report["atoms2"]) == (2, 1)


def test_external_config_resolves_relative_references(tmp_path):
    config = {
        "rmsd_restraints_config": [
            {"ref_pdb": "target.pdb", "harmonic": {"target_rmsd": 1}}
        ]
    }
    (tmp_path / "rgi.json").write_text(json.dumps(config))
    actual = make_config("custom", config_path="rgi.json", base_dir=tmp_path)
    assert actual["rmsd_restraints_config"][0]["ref_pdb"] == str(
        tmp_path / "target.pdb"
    )


def test_custom_accepts_yaml_and_rejects_ambiguous_sources():
    assert make_config("custom", custom="conformer_restraints_config: {}")
    with pytest.raises(ValueError, match="either"):
        make_config("custom", custom="{}", config_path="rgi.yaml")
    with pytest.raises(ValueError):
        make_config("custom", custom="[1, 2]")
