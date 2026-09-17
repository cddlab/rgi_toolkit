"""Reference-conformer uncertainties through the public spec-building path."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import pytest
from rdkit import Chem

from rgi_toolkit.atom_context import LigandConf
from rgi_toolkit.featurizer import build_spec
from rgi_toolkit.polymer import PolymerGeometry
from tests.test_featurizer import _lig_heavy
from tests.test_monlib_esd import _energy_grad, _torsion_coords

TERMS = ("bond", "angle", "chiral", "plane", "cistrans", "vdw")


def _config(term, *, weight=1, slack=0):
    return {
        **{key: {"weight": 0} for key in TERMS},
        term: {"weight": weight, "slack": slack},
        "relax_force_field": {"ligand": "none"},
    }


def _ligand(smiles, coords):
    return LigandConf(
        Chem.MolFromSmiles(smiles),
        np.asarray(coords, dtype=float),
        np.arange(len(coords)),
        conformer_restraints=True,
    )


def _chiral_sigma(coords):
    """Independent finite-difference propagation through a Gram determinant."""
    vectors = coords[1:] - coords[0]
    lengths = np.linalg.norm(vectors, axis=1)
    unit = vectors / lengths[:, None]
    angles = np.arccos([unit[i] @ unit[j] for i, j in ((0, 1), (1, 2), (2, 0))])
    values = np.r_[lengths, angles]
    sigmas = np.array([0.02] * 3 + [math.radians(3)] * 3)

    def volume(v):
        x, y, z = np.cos(v[3:])
        gram = np.array([[1, x, z], [x, 1, y], [z, y, 1]])
        return np.prod(v[:3]) * np.sqrt(np.linalg.det(gram))

    jacobian = np.array(
        [(volume(values + d) - volume(values - d)) / 2e-6 for d in np.eye(6) * 1e-6]
    )
    return np.linalg.norm(jacobian * sigmas)


def _case(term, use_esd=True):
    if term == "bond":
        ligand = _ligand("CC", [[0, 0, 0], [1.5, 0, 0]])
        query = ligand.conf_coords.copy()
        query[1, 0] += 0.04
        expected = (0.04 / (0.02 if use_esd else 1)) ** 2
    elif term == "angle":
        ligand = _ligand("CCC", [[1, 0, 0], [0, 0, 0], [0, 1, 0]])
        theta = math.radians(96)
        query = np.array([[1, 0, 0], [0, 0, 0], [math.cos(theta), math.sin(theta), 0]])
        expected = (math.radians(6) / (math.radians(3) if use_esd else 1)) ** 2
    elif term == "cistrans":
        ligand = _ligand("C/C=C/C", _torsion_coords(170))
        query = _torsion_coords(180)
        expected = (math.radians(10) / (math.radians(5) if use_esd else 1)) ** 2
    elif term == "plane":
        ligand = _lig_heavy("c1ccccc1")
        query = ligand.conf_coords.copy()
        query[0, 2] += 0.3
        centered = query - query.mean(axis=0)
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
        expected = np.square(centered @ normal / (0.02 if use_esd else 1)).sum()
    else:
        ligand = _ligand(
            "F[C@](Cl)(Br)I",
            [[1, 1, 1], [0, 0, 0], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        )
        query = ligand.conf_coords.copy()
        query[0, 0] += 0.1
        expected = 0.0
        # Four modeled neighbors give four signed volume restraints.
        for neighbors in ((0, 2, 3), (0, 2, 4), (0, 3, 4), (2, 3, 4)):
            indices = [1, *neighbors]
            reference = ligand.conf_coords[indices]
            original = np.linalg.det(reference[1:] - reference[0])
            moved = query[indices]
            actual = np.linalg.det(moved[1:] - moved[0])
            sigma = _chiral_sigma(reference) if use_esd else 1
            expected += ((actual - original) / sigma) ** 2
    config = dict(_config(term), use_esd=use_esd)
    spec = build_spec([ligand], conformer_config=config)
    return spec, query[spec.active_sites], expected


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("term", TERMS[:-1])
@pytest.mark.parametrize("use_esd", [True, False])
def test_reference_energy_and_gradient_match_residuals(term, backend, use_esd):
    spec, coords, expected = _case(term, use_esd)
    energy, grad = _energy_grad(spec, coords, backend)
    assert energy == pytest.approx(expected, rel=2e-7, abs=1e-8)
    if grad is not None:
        numeric = np.empty_like(coords)
        for index in np.ndindex(coords.shape):
            plus, minus = coords.copy(), coords.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numeric[index] = (
                _energy_grad(spec, plus, "numpy")[0]
                - _energy_grad(spec, minus, "numpy")[0]
            ) / 2e-6
        np.testing.assert_allclose(grad, numeric, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("slack", [0.0, 0.005])
@pytest.mark.parametrize("use_esd", [True, False])
def test_link_esd_is_weight_and_user_slack_is_separate(slack, use_esd):
    geometry = PolymerGeometry(
        [],
        np.arange(3),
        [(0, 1, 1.329, 0.011)],
        [(0, 1, 2, 2.0, math.radians(1.5))],
        [],
    )
    config = _config("bond", weight=2, slack=slack)
    config["angle"] = {"weight": 3, "slack": slack}
    config["use_esd"] = use_esd
    spec = build_spec([], conformer_config=config, polymer_geometry=geometry)
    np.testing.assert_allclose(spec.bond.weight, 2 / (0.011**2 if use_esd else 1))
    np.testing.assert_allclose(
        spec.angle.weight, 3 / (math.radians(1.5) ** 2 if use_esd else 1)
    )
    np.testing.assert_allclose(spec.bond.slack, slack)
    np.testing.assert_allclose(spec.angle.slack, slack)
    # A half-ESD deviation is penalized; ESD is never an implicit tolerance band.
    coords = np.array([[0, 0, 0], [1.3345, 0, 0], [2, 1, 0]])
    isolated = replace(spec, angle=None)
    assert _energy_grad(isolated, coords, "numpy")[0] == pytest.approx(
        2 * (max(0.0055 - slack, 0) / (0.011 if use_esd else 1)) ** 2
    )


def test_chiral_normalization_does_not_depend_on_bond_angle_activation():
    ligand = _lig_heavy("C[C@H](O)N")
    config = _config("chiral", slack=0.05)
    single = build_spec([ligand], conformer_config=config)
    config.update(bond={"weight": 9}, angle={"weight": 0.2})
    combined = build_spec([ligand], conformer_config=config)
    np.testing.assert_allclose(single.chiral.weight, combined.chiral.weight)
    np.testing.assert_allclose(single.chiral.slack, 0.05)
    inverted = build_spec(
        [replace(ligand, invert_chirality=True)], conformer_config=config
    )
    np.testing.assert_allclose(inverted.chiral.weight, combined.chiral.weight)
    np.testing.assert_allclose(inverted.chiral.vol0, -combined.chiral.vol0)


def test_default_esd_preserves_existing_arrays_and_toggle_changes_only_weights():
    from tests.test_conformer_chemistry import _ligand as make_ligand

    ligand = make_ligand("C[C@H](O)CC/C=C/c1ccccc1")
    config = {key: {"weight": 2, "slack": 0.01} for key in TERMS[:-1]}
    config["vdw"] = {"weight": 3}
    default = build_spec([ligand], conformer_config=config)
    enabled = build_spec([ligand], conformer_config=dict(config, use_esd=True))
    disabled = build_spec([ligand], conformer_config=dict(config, use_esd=False))
    np.testing.assert_array_equal(default.active_sites, disabled.active_sites)
    for kind in TERMS:
        a, b, c = (getattr(spec, kind) for spec in (default, enabled, disabled))
        assert a is not None, kind
        for field in fields(a):
            np.testing.assert_array_equal(
                getattr(a, field.name), getattr(b, field.name)
            )
            if field.name != "weight":
                np.testing.assert_array_equal(
                    getattr(a, field.name), getattr(c, field.name)
                )
        expected = 3 if kind == "vdw" else 2
        if kind == "plane":
            expected *= c.grp_mask.sum(axis=-1)
        np.testing.assert_array_equal(c.weight, expected)


@pytest.mark.parametrize("use_esd", [True, False])
def test_partial_dictionary_and_reference_follow_the_same_esd_option(use_esd):
    from rgi_toolkit._monlib_records import GeometryTarget
    from rgi_toolkit.monlib_geom import LibraryTargets

    ligand = _ligand("CC", [[0, 0, 0], [1.5, 0, 0]])
    library = LibraryTargets()
    library.terms["bond"] = [GeometryTarget((2, 3), 1.2, 0.01)]
    geometry = PolymerGeometry([], np.arange(2, 4), [], [], [], library)
    spec = build_spec(
        [ligand],
        conformer_config=dict(_config("bond"), use_esd=use_esd),
        polymer_geometry=geometry,
    )
    np.testing.assert_allclose(
        spec.bond.weight, [1 / 0.02**2, 1 / 0.01**2] if use_esd else [1, 1]
    )
    np.testing.assert_array_equal(spec.bond.slack, 0)


@pytest.mark.parametrize("term", ["bond", "angle"])
@pytest.mark.parametrize("esd", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("use_esd", [True, False])
def test_active_invalid_reference_link_esd_raises(term, esd, use_esd):
    geometry = PolymerGeometry(
        [], np.arange(3), [(0, 1, 1.3, esd)], [(0, 1, 2, 2.0, esd)], []
    )
    with pytest.raises(ValueError, match=f"reference conformer {term}: ESD"):
        build_spec(
            [],
            conformer_config=dict(_config(term), use_esd=use_esd),
            polymer_geometry=geometry,
        )
