"""Self-contained CCP4 fixtures for modification and peptide-state resolution."""

from __future__ import annotations

import itertools
import math

import pytest

from rgi_toolkit._monlib_records import canonical, modified_restraints, read_restraints
from rgi_toolkit.monlib_geom import KINDS, MonomerLibrary, collect
from tests.test_monlib_geom import _ENER_LIB


def _loop(category, fields, rows):
    return (
        "loop_\n"
        + "".join(f"_{category}.{f}\n" for f in fields.split())
        + "".join(" ".join(map(str, row)) + "\n" for row in rows)
    )


_NAMES = ["N", "CA", "C", "O", "CB", "OXT"]


def _component(name, group):
    body = f"data_comp_list\n_chem_comp.id {name}\n_chem_comp.group {group}\n\ndata_comp_{name}\n"
    body += _loop(
        "chem_comp_atom",
        "comp_id atom_id type_symbol type_energy charge",
        [(name, a, a[0], a[0], 0) for a in _NAMES],
    )
    body += _loop(
        "chem_comp_bond",
        "comp_id atom_id_1 atom_id_2 type value_dist value_dist_esd",
        [
            (name, a, b, "single", v, s)
            for a, b, v, s in [
                ("N", "CA", 1.45, 0.01),
                ("CA", "C", 1.53, 0.02),
                ("CA", "CB", 1.54, 0.03),
                ("C", "O", 1.24, 0.02),
                ("C", "OXT", 1.30, 0.03),
            ]
        ],
    )
    body += _loop(
        "chem_comp_angle",
        "comp_id atom_id_1 atom_id_2 atom_id_3 value_angle value_angle_esd",
        [
            (name, a, b, c, v, 1.5)
            for a, b, c, v in [
                ("N", "CA", "C", 110),
                ("N", "CA", "CB", 109),
                ("C", "CA", "CB", 108),
                ("CA", "C", "O", 110),
                ("CA", "C", "OXT", 110),
                ("O", "C", "OXT", 110),
            ]
        ],
    )
    body += _loop(
        "chem_comp_tor",
        "comp_id id atom_id_1 atom_id_2 atom_id_3 atom_id_4 value_angle value_angle_esd period",
        [
            (name, label, *atoms, angle, 7, period)
            for label, atoms, angle, period in [
                ("sp2_sp2_test", ("N", "CA", "C", "O"), 37, 3),
                ("sp2_sp2_leaving", ("N", "CA", "C", "OXT"), 20, 2),
                ("chi1", ("CB", "CA", "C", "O"), 60, 3),
                ("phi", ("CB", "CA", "N", "C"), 70, 1),
            ]
        ],
    )
    body += _loop(
        "chem_comp_chir",
        "comp_id id atom_id_centre atom_id_1 atom_id_2 atom_id_3 volume_sign",
        [
            (name, "CA-chir", "CA", "N", "C", "CB", "positive"),
            (name, "leaving-chir", "C", "CA", "O", "OXT", "both"),
        ],
    )
    body += _loop(
        "chem_comp_plane_atom",
        "comp_id plane_id atom_id dist_esd",
        [(name, "free-plane", a, 0.02) for a in ("CA", "C", "O", "OXT")],
    )
    return body


def _link(name, cis, group, side2_target):
    body = f"data_link_{name}\n"
    body += _loop(
        "chem_link_bond",
        "link_id atom_1_comp_id atom_id_1 atom_2_comp_id atom_id_2 type value_dist value_dist_esd",
        [(name, 1, "C", 2, "N", "single", 1.40 if cis else 1.30, 0.01)],
    )
    body += _loop(
        "chem_link_tor",
        "link_id id atom_1_comp_id atom_id_1 atom_2_comp_id atom_id_2 "
        "atom_3_comp_id atom_id_3 atom_4_comp_id atom_id_4 "
        "value_angle value_angle_esd period",
        [
            (
                name,
                "omega",
                1,
                "CA",
                1,
                "C",
                2,
                "N",
                2,
                "CA",
                0 if cis else 180,
                10 if cis else 5,
                0,
            )
        ],
    )
    body += _loop(
        "chem_link_plane",
        "link_id plane_id atom_comp_id atom_id dist_esd",
        [
            (name, "local", side, a, 0.04 if cis else 0.02)
            for side, a in [(1, "CA"), (1, "C"), (1, "O"), (2, "N")]
        ],
    )
    body += f"data_mod_{name}-1\n"
    body += _loop(
        "chem_mod_atom",
        "mod_id function atom_id new_atom_id new_type_symbol new_type_energy new_charge",
        [(f"{name}-1", "delete", "OXT", ".", "O", "O", 0)],
    )
    body += _loop(
        "chem_mod_bond",
        "mod_id function atom_id_1 atom_id_2 new_type new_value_dist new_value_dist_esd",
        [
            (f"{name}-1", "change", "CA", "C", "single", 1.50 + 0.02 * cis, 0.04),
            (f"{name}-1", "change", "C", "O", "double", 1.22 + 0.05 * cis, 0.03),
        ],
    )
    body += f"data_mod_{name}-2\n"
    body += _loop(
        "chem_mod_bond",
        "mod_id function atom_id_1 atom_id_2 new_type new_value_dist new_value_dist_esd",
        [(f"{name}-2", "change", "N", "CA", "single", side2_target - 0.02 * cis, 0.02)],
    )
    return body


@pytest.fixture
def peptide_library(tmp_path):
    groups = {"AAA": "peptide", "PPP": "P-peptide", "MMM": "M-peptide"}
    for name, group in groups.items():
        directory = tmp_path / name[0].lower()
        directory.mkdir(exist_ok=True)
        (directory / f"{name}.cif").write_text(_component(name, group))
    links = [
        (prefix + state, state == "CIS", group, value)
        for prefix, group, value in [
            ("", "peptide", 1.44),
            ("P", "P-peptide", 1.46),
            ("NM", "M-peptide", 1.48),
        ]
        for state in ("TRANS", "CIS")
    ]
    body = "data_link_list\n" + _loop(
        "chem_link",
        "id comp_id_1 mod_id_1 group_comp_1 comp_id_2 mod_id_2 group_comp_2 name",
        [
            (name, ".", f"{name}-1", "peptide", ".", f"{name}-2", group, name)
            for name, _cis, group, _target in links
        ],
    )
    body += "".join(_link(*entry) for entry in links)
    body += "data_mod_EXTRA\n"
    body += _loop(
        "chem_mod_atom",
        "mod_id function atom_id new_atom_id new_type_symbol new_type_energy new_charge",
        [("EXTRA", "delete", "OXT", ".", "O", "O", 0)],
    )
    body += _loop(
        "chem_mod_bond",
        "mod_id function atom_id_1 atom_id_2 new_type new_value_dist new_value_dist_esd",
        [
            ("EXTRA", "add", "N", "C", "single", 1.7, 0.04),
            ("EXTRA", "delete", "CA", "CB", ".", ".", "."),
            ("EXTRA", "change", "N", "CA", "single", 1.6, 0.02),
        ],
    )
    body += _loop(
        "chem_mod_angle",
        "mod_id function atom_id_1 atom_id_2 atom_id_3 new_value_angle new_value_angle_esd",
        [
            ("EXTRA", "change", "N", "CA", "C", 100, 3),
            ("EXTRA", "delete", "N", "CA", "CB", ".", "."),
            ("EXTRA", "add", "N", "C", "O", 125, 2),
        ],
    )
    body += _loop(
        "chem_mod_tor",
        "mod_id function atom_id_1 atom_id_2 atom_id_3 atom_id_4 id new_value_angle new_value_angle_esd new_period",
        [
            ("EXTRA", "change", "N", "CA", "C", "O", "sp2_sp2_changed", 47, 4, 2),
            ("EXTRA", "delete", "CB", "CA", "N", "C", "phi", ".", ".", "."),
            ("EXTRA", "add", "O", "C", "N", "CB", "sp2_sp2_new", 60, 6, 3),
        ],
    )
    body += _loop(
        "chem_mod_chir",
        "mod_id function atom_id_centre atom_id_1 atom_id_2 atom_id_3 new_volume_sign",
        [("EXTRA", "change", "CA", "N", "C", "CB", "negative")],
    )
    body += _loop(
        "chem_mod_plane_atom",
        "mod_id function plane_id atom_id new_dist_esd",
        [
            ("EXTRA", "add", "free-plane", "CB", 0.03),
            ("EXTRA", "delete", "free-plane", "CA", "."),
        ],
    )
    (tmp_path / "list").mkdir()
    (tmp_path / "list/mon_lib_list.cif").write_text(body)
    (tmp_path / "ener_lib.cif").write_text(_ENER_LIB)
    return MonomerLibrary.load(str(tmp_path), groups)


def _residues(sequence=("AAA", "AAA", "PPP")):
    return [
        dict(
            uid=i,
            resname=name,
            mol_type="protein",
            names={a: 6 * i + j for j, a in enumerate(_NAMES)},
        )
        for i, name in enumerate(sequence)
    ]


def _targets(library, residues=None, enabled=KINDS, on_missing="error"):
    residues = residues or _residues()
    return collect(
        library, residues, on_missing, list(zip(residues, residues[1:])), enabled
    )


def _active(rows, states):
    return [r for r in rows if all(states[i] == cis for i, cis in r.conditions)]


def test_adjacent_link_modifications_are_local_and_never_score_alternatives_twice(
    peptide_library,
):
    targets = _targets(peptide_library)
    for states in itertools.product((0, 1), repeat=2):
        bonds = _active(targets.terms["bond"], states)
        by_atoms = {tuple(sorted(r.atoms)): r for r in bonds}
        assert len(by_atoms) == len(bonds)
        # The middle AAA's N-CA follows its incoming TRANS, even though its outgoing
        # edge is PTRANS. The last PPP follows PTRANS and its own cis/trans state.
        assert by_atoms[(6, 7)].value == pytest.approx(1.44 - 0.02 * states[0])
        assert by_atoms[(12, 13)].value == pytest.approx(1.46 - 0.02 * states[1])
        assert by_atoms[(7, 8)].value == pytest.approx(1.50 + 0.02 * states[1])
        assert by_atoms[(8, 9)].value == pytest.approx(1.22 + 0.05 * states[1])
        assert (2, 5) not in by_atoms and (
            8,
            11,
        ) not in by_atoms  # deleted leaving atoms
        assert by_atoms[(14, 17)].value == pytest.approx(1.30)  # free C terminus
        chirals = [
            r for r in _active(targets.terms["chiral"], states) if r.atoms[0] == 7
        ]
        assert len(chirals) == 1 and chirals[0].esd > 0
        assert len(chirals[0].conditions) == 2
        planes = _active(targets.terms["plane"], states)
        for edge, atoms in enumerate(((1, 2, 3, 6), (7, 8, 9, 12))):
            (plane,) = [r for r in planes if r.atoms == atoms]
            assert plane.esd == pytest.approx(0.04 if states[edge] else 0.02)
        assert not any({1, 7}.issubset(r.atoms) for r in planes)
    # An unmodified internal term is represented once, with no state gate.
    rows = [r for r in targets.terms["bond"] if r.atoms == (7, 10)]
    assert len(rows) == 1 and rows[0].conditions == ()


@pytest.mark.parametrize(
    "resname,expected", [("AAA", 1.44), ("PPP", 1.46), ("MMM", 1.48)]
)
def test_each_peptide_link_family_has_its_own_cis_and_trans_targets(
    peptide_library, resname, expected
):
    targets = _targets(peptide_library, _residues(("AAA", resname)))
    assert len(targets.peptides) == 1
    for cis in (0, 1):
        rows = _active(targets.terms["bond"], [cis])
        (row,) = [r for r in rows if r.atoms == (6, 7)]
        assert row.value == pytest.approx(expected - 0.02 * cis)
        (omega,) = [
            r
            for r in _active(targets.terms["cistrans"], [cis])
            if r.atoms == (1, 2, 6, 7)
        ]
        assert omega.value == pytest.approx(0 if cis else -math.pi)
        assert omega.esd == pytest.approx(math.radians(10 if cis else 5))
        assert omega.period == 1


def test_only_omega_and_sp2_torsions_are_selected_and_dictionary_sign_is_converted(
    peptide_library,
):
    targets = _targets(peptide_library, _residues(("AAA",)))
    rows = targets.terms["cistrans"]
    assert len(rows) == 2
    (sp2,) = [r for r in rows if r.atoms == (0, 1, 2, 3)]
    assert sp2.value == pytest.approx(-math.radians(37))
    assert sp2.esd == pytest.approx(math.radians(7))
    assert sp2.period == 3
    assert not any(r.atoms[0] == 4 for r in rows)


def test_gemmi_add_change_delete_operations_do_not_mutate_shared_components(
    peptide_library,
):
    comp = peptide_library._monlib.monomers["AAA"]
    original = read_restraints(comp.rt)
    rows = modified_restraints(
        comp.rt, [peptide_library._monlib.modifications["EXTRA"]]
    )
    assert read_restraints(comp.rt) == original
    assert all((1, "OXT") not in r.atoms for records in rows.values() for r in records)
    bonds = {canonical("bond", r.atoms): r for r in rows["bond"]}
    assert bonds[canonical("bond", ((1, "N"), (1, "C")))].value == pytest.approx(1.7)
    assert canonical("bond", ((1, "CA"), (1, "CB"))) not in bonds
    angles = {r.atoms: r for r in rows["angle"]}
    assert angles[((1, "N"), (1, "CA"), (1, "C"))].esd == pytest.approx(3)
    assert ((1, "N"), (1, "CA"), (1, "CB")) not in angles
    assert angles[((1, "N"), (1, "C"), (1, "O"))].value == pytest.approx(125)
    torsions = {r.label: r for r in rows["cistrans"]}
    assert "phi" not in torsions and "sp2_sp2_leaving" not in torsions
    assert (
        torsions["sp2_sp2_changed"].value,
        torsions["sp2_sp2_changed"].esd,
        torsions["sp2_sp2_changed"].period,
    ) == (47, 4, 2)
    assert torsions["sp2_sp2_new"].period == 3
    (chiral,) = rows["chiral"]
    assert chiral.sign == "Negative"
    (plane,) = rows["plane"]
    assert set(plane.atoms) == {(1, "C"), (1, "O"), (1, "CB")}
    assert canonical("chiral", (0, 1, 2, 3)) == canonical("chiral", (0, 2, 3, 1))
    assert canonical("chiral", (0, 1, 2, 3)) != canonical("chiral", (0, 1, 3, 2))


@pytest.mark.parametrize("esd", [0.0, -1.0])
def test_nonpositive_esd_disables_energy_but_retains_topological_exclusion(
    peptide_library, esd
):
    peptide_library._monlib.monomers["AAA"].rt.bonds[0].esd = esd
    rows = _targets(peptide_library, _residues(("AAA",)), enabled=["bond"])
    assert not any(r.atoms == (0, 1) for r in rows.terms["bond"])
    assert (0, 1) in rows.bond_pairs


@pytest.mark.parametrize(
    "field,value",
    [("esd", float("nan")), ("esd", float("inf")), ("value", float("nan"))],
)
def test_nonfinite_active_dictionary_measurement_is_an_error(
    peptide_library, field, value
):
    setattr(peptide_library._monlib.monomers["AAA"].rt.bonds[0], field, value)
    with pytest.raises(ValueError, match="nonfinite"):
        _targets(peptide_library, _residues(("AAA",)), enabled=["bond"])
    # A disabled term does not validate measurements unused by another active term.
    _targets(peptide_library, _residues(("AAA",)), enabled=["plane"])


def test_missing_chiral_dependency_obeys_policy_and_drops_partial_dictionary_replacement(
    peptide_library, caplog
):
    peptide_library._monlib.monomers["AAA"].rt.bonds[0].esd = 0
    with pytest.raises(ValueError, match="cannot derive chiral"):
        _targets(peptide_library, _residues(("AAA",)), enabled=["chiral"])
    targets = _targets(
        peptide_library, _residues(("AAA",)), enabled=["chiral"], on_missing="fallback"
    )
    assert 1 in targets.chiral_fallback and 1 not in targets.chiral_centers
    assert not any(r.atoms[0] == 1 for r in targets.terms["chiral"])
    assert "cannot derive chiral" in caplog.text


def test_missing_cis_dictionary_never_forces_trans_omega(peptide_library, caplog):
    del peptide_library._monlib.links["CIS"]
    with pytest.raises(ValueError, match="no cis counterpart"):
        _targets(peptide_library, _residues(("AAA", "AAA")))
    targets = _targets(
        peptide_library, _residues(("AAA", "AAA")), on_missing="fallback"
    )
    assert not targets.peptides
    assert not any(r.atoms == (1, 2, 6, 7) for r in targets.terms["cistrans"])
    assert "omega omitted" in caplog.text


@pytest.mark.parametrize("link,residue", [("TRANS", "AAA"), ("PTRANS", "PPP")])
def test_missing_link_obeys_policy(peptide_library, caplog, link, residue):
    del peptide_library._monlib.links[link]
    residues = _residues(("AAA", residue))
    with pytest.raises(ValueError, match="no library entry for link"):
        _targets(peptide_library, residues)
    targets = _targets(peptide_library, residues, on_missing="fallback")
    assert not targets.covered_links
    assert "using built-in link geometry" in caplog.text
