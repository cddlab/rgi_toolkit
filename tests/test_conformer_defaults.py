"""Default weights and torsion priority through the public spec-building path."""

from dataclasses import replace

import numpy as np
import pytest

from rgi_toolkit._config_util import conformer_weight
from rgi_toolkit._conformer_planes import prefer_cistrans
from rgi_toolkit._monlib_records import GeometryTarget
from rgi_toolkit.config import RestraintsConfig
from rgi_toolkit.energy import numpy_energy
from rgi_toolkit.featurizer import build_spec
from rgi_toolkit.monlib_geom import LibraryTargets, PeptideChoice
from rgi_toolkit.polymer import PolymerGeometry

TERMS = ("bond", "angle", "chiral", "cistrans", "vdw", "plane")


@pytest.mark.parametrize("config", [None, {}, {"conformer_restraints_config": None}])
def test_absent_conformer_remains_disabled(config):
    parsed = RestraintsConfig.from_dict(config)
    assert parsed.conformer_config is None
    assert all(conformer_weight(parsed.conformer_config, key) == 0 for key in TERMS)


@pytest.mark.parametrize("block", [{}, {"plane": {}}, {"start_sigma": 1.0}])
def test_empty_conformer_enables_five_terms(block):
    config = RestraintsConfig.from_dict({"conformer_restraints_config": block})
    assert [conformer_weight(config.conformer_config, key) for key in TERMS] == [
        1
    ] * 5 + [0]


@pytest.mark.parametrize("weight", [0, -1, None, 0.25])
@pytest.mark.parametrize("term", TERMS)
def test_explicit_weights(term, weight):
    assert conformer_weight({term: {"weight": weight}}, term) == (weight or 0)


def test_real_ligand_defaults_and_optin():
    from tests.test_featurizer import _lig_heavy

    ligand = _lig_heavy("C[C@H](O)/C=C/C")
    spec = build_spec([ligand], conformer_config={})
    assert all(getattr(spec, key) is not None for key in TERMS if key != "plane")
    assert spec.plane is None
    assert not build_spec([ligand], conformer_config=None).is_active()
    assert not build_spec(
        [replace(ligand, conformer_restraints=False)], conformer_config={}
    ).is_active()


def _geometry(torsions):
    library = LibraryTargets()
    library.terms["cistrans"] = torsions
    library.plane_groups = {(0, 1, 2, 3, 4)}
    return PolymerGeometry(
        [], np.arange(9), [], [], [(0, 1, 2, 3, 4), (5, 6, 7, 8)], library
    )


def test_only_conflicting_groups_removed_and_zero_weight_restores_them():
    row = GeometryTarget((0, 1, 2, 3), 0, 0.1)
    geometry = _geometry([row])
    config = {key: {"weight": int(key in ("plane", "cistrans"))} for key in TERMS}
    spec = build_spec([], conformer_config=config, polymer_geometry=geometry)
    assert len(spec.plane.idx) == 1
    assert set(spec.active_sites[spec.plane.idx[0]]) == {5, 6, 7, 8}
    assert geometry.library.plane_groups == {(0, 1, 2, 3, 4)}
    assert geometry.library.terms["cistrans"] == [row]
    config["cistrans"]["weight"] = 0
    restored = build_spec([], conformer_config=config, polymer_geometry=geometry)
    assert len(restored.plane.idx) == 2
    assert restored.cistrans is None


@pytest.mark.parametrize("source", ["reference", "dictionary"])
def test_plane_survives_only_in_nonconflicting_peptide_state(source):
    from tests.test_monlib_esd import _torsion_coords

    geometry = _geometry([GeometryTarget((0, 1, 2, 3), 0, 0.1, conditions=((0, 1),))])
    geometry.library.peptides = [PeptideChoice((0, 1, 2, 3), np.pi, 0)]
    geometry.link_planes = [(0, 1, 2, 3, 4)]
    if source == "dictionary":
        geometry.link_planes = []
        geometry.library.terms["plane"] = [GeometryTarget((0, 1, 2, 3, 4), 0, 0.1)]
    config = {key: {"weight": int(key in ("plane", "cistrans"))} for key in TERMS}
    spec = build_spec([], conformer_config=config, polymer_geometry=geometry)
    prepared = numpy_energy.prepare_spec(spec)
    for degrees, plane_mask in ((0, 0), (180, 1)):
        coords = np.vstack([_torsion_coords(degrees), [0, 1, 1]])
        bound = numpy_energy.bind_peptide_states(coords, prepared)
        assert bound["plane"]["mask"].tolist() == [plane_mask]
        assert bound["cistrans"]["mask"].tolist() == [1 - plane_mask]


def test_condition_subtraction_is_local_and_disjoint():
    library = LibraryTargets()
    library.terms["cistrans"] = [
        GeometryTarget((0, 1, 2, 3), 0, 0.1, conditions=((2, 0), (7, 1)))
    ]
    planes, conditions, _ = prefer_cistrans([(0, 1, 2, 3, 4)], [], library)
    assert len(planes) == 2
    for a in (0, 1):
        for b in (0, 1):
            state = {2: a, 7: b}
            active = sum(all(state[k] == v for k, v in row) for row in conditions)
            assert active == int(not (a == 0 and b == 1))


def test_disabled_dictionary_torsion_esd_does_not_suppress_plane():
    geometry = _geometry([GeometryTarget((0, 1, 2, 3), 0, 0)])
    config = {key: {"weight": int(key in ("plane", "cistrans"))} for key in TERMS}
    spec = build_spec([], conformer_config=config, polymer_geometry=geometry)
    assert spec.cistrans is None
    assert len(spec.plane.idx) == 2
