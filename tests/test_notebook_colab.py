"""Exercise standard form values through the native configuration parser."""

import copy
import json

import pytest

from rgi_toolkit.notebook_colab import COLAB_FORM, config_from_fields


def form(**values):
    fields = {}
    exec(COLAB_FORM, fields)
    fields.update(use_rgi=True, **values)
    return fields


def test_form_is_visible_without_execution_and_defaults_to_vanilla():
    fields = {}
    exec(COLAB_FORM, fields)
    assert config_from_fields(fields) == (None, "")
    assert all(
        "#@param" in line
        for line in COLAB_FORM.splitlines()
        if line and not line.startswith("#")
    )
    with pytest.raises(ValueError, match="two distance_atom_selection"):
        config_from_fields(form())


def test_single_distance_uses_plain_text_and_numbers():
    config, chains = config_from_fields(
        form(distance_atom_selection1="chain A", distance_atom_selection2="chain B")
    )
    assert chains == ""
    assert config["distance_restraints_config"] == [
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "harmonic": {"target_distance": 25},
        }
    ]


def test_multiple_values_broadcast_and_all_five_types_combine():
    config, chains = config_from_fields(
        form(
            distance_atom_selection1="chain A",
            distance_atom_selection2='["chain B", "chain C"]',
            target_distance=[15, 25],
            conformer_chains="B,C",
            angle_atom_selection1="chain A",
            angle_atom_selection2="chain B",
            angle_atom_selection3="chain C",
            target_angle=[90, 120],
            custom_energy='["harmonic(distance(A, B), 25)", "harmonic(distance(A, B), 30)"]',
            ref_pdb='["first.pdb", "second.pdb"]',
            atom_selection_target="chain A and name CA",
            atom_selection_ref="chain B and name CA",
            target_rmsd=[0, 1],
        )
    )
    assert chains == "B,C"
    assert config["conformer_restraints_config"] == {}
    for kind in ("distance", "angle", "custom", "rmsd"):
        assert len(config[f"{kind}_restraints_config"]) == 2
    assert config["distance_restraints_config"][1]["harmonic"]["target_distance"] == 25
    assert config["rmsd_restraints_config"][1]["ref_pdb"].endswith("second.pdb")


@pytest.mark.parametrize(
    "values, message",
    [
        (
            {"distance_atom_selection1": "chain A"},
            "distance_atom_selection2 is required",
        ),
        (
            {
                "distance_atom_selection1": '["chain A", "chain B"]',
                "distance_atom_selection2": "chain C",
                "target_distance": [1, 2, 3],
            },
            "expected 1 or 3",
        ),
        (
            {
                "distance_atom_selection1": "[chain A]",
                "distance_atom_selection2": "chain B",
            },
            "use a JSON list",
        ),
        ({"ref_pdb": "one.pdb", "ref_cif": "two.cif"}, "either ref_pdb or ref_cif"),
    ],
)
def test_errors_point_to_the_actual_field(values, message):
    with pytest.raises(ValueError, match=message):
        config_from_fields(form(**values))


def test_native_advanced_entries_append_and_conformer_settings_survive():
    extra = {
        "distance_restraints_config": [
            {
                "atom_selection1": "chain B",
                "atom_selection2": "chain C",
                "flat-bottomed2": {"target_distance2": 8},
            }
        ],
        "conformer_restraints_config": {"use_esd": True, "plane": {"weight": 1}},
    }
    original = copy.deepcopy(extra)
    config, chains = config_from_fields(
        form(
            distance_atom_selection1="chain A",
            distance_atom_selection2="chain B",
            restraints_config=extra,
            conformer_chains="B",
        )
    )
    assert len(config["distance_restraints_config"]) == 2
    assert config["conformer_restraints_config"] == extra["conformer_restraints_config"]
    assert chains == "B"
    assert extra == original


def test_file_config_resolves_references_and_requires_conformer_opt_in(tmp_path):
    (tmp_path / "restraints.json").write_text(
        json.dumps(
            {
                "rmsd_restraints_config": [
                    {"ref_cif": "reference.cif", "harmonic": {"target_rmsd": 0}}
                ]
            }
        )
    )
    config, _ = config_from_fields(
        form(restraints_config={"config_path": "restraints.json"}), base_dir=tmp_path
    )
    assert config["rmsd_restraints_config"][0]["ref_cif"] == str(
        tmp_path / "reference.cif"
    )
    with pytest.raises(ValueError, match="Set conformer_chains"):
        config_from_fields(form(restraints_config={"conformer_restraints_config": {}}))


def test_vanilla_ignores_incomplete_settings_and_reruns_do_not_accumulate():
    fields = form(
        distance_atom_selection1="chain A", distance_atom_selection2="chain B"
    )
    first, _ = config_from_fields(fields)
    fields["target_distance"] = 30
    second, _ = config_from_fields(fields)
    assert len(second["distance_restraints_config"]) == 1
    assert first["distance_restraints_config"][0]["harmonic"]["target_distance"] == 25
    assert second["distance_restraints_config"][0]["harmonic"]["target_distance"] == 30
    fields.update(
        use_rgi=False, distance_atom_selection2="", restraints_config="invalid"
    )
    assert config_from_fields(fields) == (None, "")
