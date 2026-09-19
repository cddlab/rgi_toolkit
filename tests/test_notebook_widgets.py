"""Exercise live controls, mixed entries and native config round trips."""

import json

import pytest

pytest.importorskip("ipywidgets")

from rgi_toolkit.notebook_widgets import RestraintEditor  # noqa: E402


def test_add_duplicate_disable_and_remove_read_current_values():
    editor = RestraintEditor()
    first = editor.cards[0]
    first.fields["atom_selection1"].value = "chain A and (resid 1 to 4 or resid 8)"
    first.duplicate.click()
    second = editor.cards[1]
    second.fields["target_distance"].value = 35
    values = editor.get_config()["distance_restraints_config"]
    assert [item["harmonic"]["target_distance"] for item in values] == [25, 35]
    assert values[1]["atom_selection1"] == first.fields["atom_selection1"].value
    first.enabled.value = False
    assert len(editor.get_config()["distance_restraints_config"]) == 1
    first.remove.click()
    assert editor.cards == [second]
    second.fields["target_distance"].value = 40
    assert editor.get_config()["distance_restraints_config"][0]["harmonic"] == {
        "target_distance": 40
    }


def test_five_types_repeated_entries_and_conformer_defaults():
    editor = RestraintEditor()
    for kind in ("angle", "custom", "RMSD", "conformer"):
        editor.kind.value = kind
        editor.add_button.click()
    rmsd = editor.cards[3]
    rmsd.fields["reference_file"].value = "reference.cif"
    rmsd.fields["reference_format"].value = "ref_cif"
    rmsd.fields["atom_selection_target_fit"].value = "chain A and backbone"
    rmsd.duplicate.click()
    conformer = editor.cards[4]
    conformer.fields["conformer_chains"].value = "A,C"
    result = editor.get_config()
    assert len(result["rmsd_restraints_config"]) == 2
    assert len(result["angle_restraints_config"]) == 1
    assert len(result["custom_restraints_config"]) == 1
    assert result["conformer_restraints_config"]["plane"]["weight"] == 0
    assert result["conformer_restraints_config"]["torsion"]["weight"] == 0
    assert result["conformer_restraints_config"]["bond"]["weight"] == 1
    assert editor.get_conformer_chains() == "A,C"
    with pytest.raises(ValueError, match="existing conformer"):
        editor.add("conformer")


def test_custom_groups_formula_and_labels_are_independent():
    editor = RestraintEditor()
    card = editor.add("custom")
    card.selections.add("C", "chain C and name CA")
    card.fields["energy"].value = "(distance(A, B) - distance(A, C))**2"
    card.fields["move"].value = "[B, C]"
    card.duplicate.click()
    entries = editor.get_config()["custom_restraints_config"]
    assert entries[0]["move"] == ["B", "C"]
    assert set(entries[0]["selections"]) == {"A", "B", "C"}
    assert entries[0]["name"] != entries[1]["name"]
    card.selections.add("C", "chain A")
    with pytest.raises(ValueError, match="Duplicate custom selection"):
        editor.get_config()


def test_penalty_window_and_native_advanced_settings():
    editor = RestraintEditor()
    card = editor.cards[0]
    card.fields["penalty"].value = "flat-bottomed"
    card.fields["target_distance1"].value = 20
    card.fields["target_distance2"].value = 30
    card.fields["window"].value = "step"
    card.fields["start_step"].value = "5"
    card.fields["stop_step"].value = "10"
    card.fields["move"].value = "2"
    native = editor.get_config()["distance_restraints_config"][0]
    assert native["flat-bottomed"] == {"target_distance1": 20, "target_distance2": 30}
    assert (native["start_step"], native["stop_step"], native["move"]) == (5, 10, 2)
    assert "start_sigma" not in native
    card.extra.value = "weight: 2"
    with pytest.raises(ValueError, match="not twice"):
        editor.get_config()


def test_file_to_form_preserves_reference_paths_and_whole_structure_rmsd(tmp_path):
    native = {
        "rmsd_restraints_config": [
            {"ref_cif": "ref.cif", "harmonic": {"target_rmsd": 0}}
        ],
        "conformer_restraints_config": {"vdw": {"mode": "intramolecular"}},
    }
    path = tmp_path / "restraints.json"
    path.write_text(json.dumps(native))
    editor = RestraintEditor()
    editor.mode.value = "file"
    editor.config_path.value = str(path)
    editor.external_chains.value = "A,B"
    editor.import_button.click()
    assert editor.mode.value == "form"
    result = editor.get_config()
    rmsd = result["rmsd_restraints_config"][0]
    assert rmsd["ref_cif"] == str(tmp_path / "ref.cif")
    assert not any(key.startswith("atom_selection") for key in rmsd)
    assert result["conformer_restraints_config"]["vdw"]["mode"] == "intramolecular"
    assert editor.get_conformer_chains() == "A,B"


def test_yaml_supports_other_toolkit_sections_and_import_keeps_them():
    editor = RestraintEditor()
    editor.mode.value = "YAML/JSON"
    editor.config_text.value = "plane_restraints_config: [{atom_selection1: chain A, harmonic: {target_plane: 0}}]"
    before = editor.get_config()
    editor.import_button.click()
    assert editor.get_config() == before
    assert editor.cards == []


def test_disabled_invalid_entry_does_not_block_prediction():
    editor = RestraintEditor()
    card = editor.add("RMSD")
    card.enabled.value = False
    assert set(editor.get_config()) == {"verbose", "distance_restraints_config"}
