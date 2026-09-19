"""Validate notebook configs against real configuration and selection behavior."""

import json

import numpy as np
import pytest

from rgi_toolkit.atom_context import AtomRecord
from rgi_toolkit.combined import CombinedRestraints
from rgi_toolkit.notebook import (
    compose_config,
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


@pytest.mark.parametrize("distance", [float("nan"), float("inf")])
def test_invalid_distances_fail_before_prediction(distance):
    with pytest.raises(ValueError):
        make_config(
            {
                "distance_restraints_config": [
                    {
                        "atom_selection1": "chain A",
                        "atom_selection2": "chain B",
                        "harmonic": {"target_distance": distance},
                    }
                ]
            }
        )


def test_native_config_builds_real_spec_and_measures_each_sample():
    config = make_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "flat-bottomed": {"target_distance1": 23, "target_distance2": 27},
                }
            ]
        }
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
    actual = make_config(config_path="rgi.json", base_dir=tmp_path)
    assert actual["rmsd_restraints_config"][0]["ref_pdb"] == str(
        tmp_path / "target.pdb"
    )


def test_native_config_text_is_separate_from_custom_energy():
    assert make_config(config_text="conformer_restraints_config: {}")
    with pytest.raises(ValueError, match="exactly one"):
        make_config(config_text="{}", config_path="rgi.yaml")
    with pytest.raises(ValueError):
        make_config(config_text="[1, 2]")


def test_multiple_types_and_repeated_entries_keep_native_fields():
    distance = {
        "atom_selection1": "chain A and name CA",
        "atom_selection2": "chain B",
        "harmonic": {"target_distance": 25},
        "move": 2,
        "start_step": 5,
    }
    angle = {
        "atom_selection1": "chain A",
        "atom_selection2": "chain B",
        "atom_selection3": "chain C",
        "harmonic": {"target_angle": 90},
    }
    custom = {"selections": {"P": "protein"}, "energy": "harmonic(rg(P), 12)"}
    rmsd = {
        "ref_cif": "reference.cif",
        "atom_selection_target_fit": "backbone",
        "atom_selection_ref_fit": "backbone",
        "harmonic": {"target_rmsd": 0},
    }
    config = compose_config(
        [
            ("distance", distance),
            ("distance", distance),
            ("conformer", {}),
            ("angle", angle),
            ("custom", custom),
            ("RMSD", rmsd),
            ("RMSD", rmsd),
        ]
    )
    assert len(config["distance_restraints_config"]) == 2
    assert len(config["rmsd_restraints_config"]) == 2
    assert config["angle_restraints_config"] == [angle]
    assert config["custom_restraints_config"] == [custom]
    assert config["conformer_restraints_config"] == {}
    config["distance_restraints_config"][0]["harmonic"]["target_distance"] = 35
    assert distance["harmonic"]["target_distance"] == 25
    assert config["distance_restraints_config"][1]["harmonic"]["target_distance"] == 25


def test_conformer_is_one_shared_config_and_empty_inputs_fail():
    with pytest.raises(ValueError, match="one shared"):
        compose_config([("conformer", {}), ("conformer", {})])
    with pytest.raises(ValueError, match="at least one"):
        compose_config([])
