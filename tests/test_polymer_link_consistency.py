"""Link targets must admit coordinates with the residue-local reference geometry."""

from dataclasses import replace

import numpy as np
import pytest

from rgi_toolkit._monlib_records import GeometryTarget
from rgi_toolkit._polymer_links import (
    cohere_dictionary_fallback_links,
    cohere_mixed_library_links,
    cohere_reference_links,
)
from rgi_toolkit.combined import CombinedRestraints
from rgi_toolkit.energy import numpy_energy
from rgi_toolkit.monlib_geom import LibraryTargets
from rgi_toolkit.polymer import build_polymer_geometry
from tests.test_polymer_conformer import _ALA_COORDS, _ALA_NAMES, _PolymerAdapter


def _angle(x, y, z):
    a, b = x - y, z - y
    return np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))


@pytest.mark.parametrize("use_esd", [False, True])
@pytest.mark.parametrize("omega", [0.0, np.pi])
@pytest.mark.parametrize("carbonyl_degrees", [121.1, 128.44209870142407])
def test_peptide_has_zero_energy_witness(use_esd, omega, carbonyl_degrees):
    # The larger carbonyl angle reproduces an AF3 free-CCD reference. Its sum
    # with the old 116.2/122.7-degree link targets exceeded 360 degrees.
    xyz = _ALA_COORDS.copy()
    u = xyz[1] - xyz[2]
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0], 0.0])
    theta = np.deg2rad(carbonyl_degrees)
    xyz[3] = xyz[2] + 1.23 * (np.cos(theta) * u + np.sin(theta) * v)
    adapter = _PolymerAdapter("protein", _ALA_NAMES, xyz)
    cfg = {
        "use_esd": use_esd,
        "plane": {"weight": 1, "slack": 0},
        "chiral": {"slack": 0},
        "vdw": {"weight": 0},
    }
    geometry = build_polymer_geometry(adapter, cfg)
    targets = {tuple(row[:3]): row[3] for row in geometry.link_angles}
    assert theta + targets[1, 2, 5] + targets[3, 2, 5] == pytest.approx(2 * np.pi)
    # Construct a complete peptide without altering either residue internally.
    direction = np.cos(targets[1, 2, 5]) * u - np.sin(targets[1, 2, 5]) * v
    nitrogen = xyz[2] + 1.329 * direction
    toward_carbon = -direction
    sideways = u - (u @ toward_carbon) * toward_carbon
    sideways /= np.linalg.norm(sideways)
    ca_direction = (
        np.cos(targets[2, 5, 6]) * toward_carbon
        + np.cos(omega) * np.sin(targets[2, 5, 6]) * sideways
    )
    old = xyz[1] - xyz[0]
    old /= np.linalg.norm(old)
    cross = np.cross(old, ca_direction)
    skew = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    rotation = np.eye(3) + skew + skew @ skew / (1 + old @ ca_direction)
    assert np.linalg.det(rotation) == pytest.approx(1)
    second = (xyz - xyz[0]) @ rotation.T + nitrogen
    positions = np.concatenate([xyz, second])
    restraint = CombinedRestraints()
    restraint.setup(adapter, config={"gpu": False, "conformer_restraints_config": cfg})
    # Select the same local peptide alternative as a real minimizer invocation.
    prepared = numpy_energy.prepare_spec(restraint.spec)
    energies = numpy_energy.energy_breakdown(positions, prepared)
    assert energies["bond"] < 1e-20
    assert energies["angle"] < 1e-20
    assert energies["chiral"] < 1e-20
    # The plane RMS kernel retains its 1e-6-A numerical floor at zero slack.
    assert energies["plane"] <= 1.01e-8
    np.testing.assert_allclose(positions[[1, 2, 3, 5], 2], 0, atol=1e-15)
    assert energies["cistrans"] < 1e-20


def _phosphate():
    xyz = np.array(
        [[0, 0, 0], [-0.5, 1.4, 0.3], [-0.5, -0.7, 1.2], [1.6, 0, 0], [8, 4, 3]], float
    )
    residues = [
        dict(uid=10, names={"P": 0, "OP1": 1, "OP2": 2, "O5'": 3}),
        dict(uid=11, names={"O3'": 4}),
    ]
    angles = [
        (i, 0, 4, np.deg2rad(a), np.deg2rad(1.5))
        for i, a in [(1, 109.493), (2, 109.493), (3, 100.661)]
    ]
    return xyz, residues, angles


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("planar_reference", [False, True])
def test_phosphate_angles_have_a_unit_vector_witness(count, planar_reference):
    xyz, residues, angles = _phosphate()
    if planar_reference:
        xyz[:, 2] = 0
    angles = angles[:count]
    result = cohere_reference_links(angles, [], residues, xyz)
    directions = xyz[[a[0] for a in result]] - xyz[0]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cosines = np.cos([a[3] for a in result])
    partner, _, rank, _ = np.linalg.lstsq(directions, cosines, rcond=None)
    np.testing.assert_allclose(directions @ partner, cosines, atol=1e-12)
    if rank == 3:
        assert np.linalg.norm(partner) == pytest.approx(1)
    else:
        assert np.linalg.norm(partner) <= 1 + 1e-12
    for old, new in zip(angles, result, strict=True):
        assert old[:3] == new[:3]
        assert old[4] == new[4]


def test_link_targets_ignore_independent_residue_frames():
    xyz, residues, angles = _phosphate()
    reference = cohere_reference_links(angles, [], residues, xyz)
    rotation, _ = np.linalg.qr(np.random.default_rng(13).normal(size=(3, 3)))
    transformed = xyz.copy()
    transformed[:4] = xyz[:4] @ rotation + [101, -25, 1]
    transformed[4] = [-200, 16, -105]
    np.testing.assert_allclose(
        cohere_reference_links(angles, [], residues, transformed), reference, atol=1e-12
    )


def test_mixed_library_reference_links_keep_states_and_dictionary_targets():
    xyz = np.array(
        [[0, 0, 0], [1, 0, 0], [-0.65, 0.76, 0], [9, 2, 0], [10, 2, 0]], float
    )
    residues = [
        dict(uid=0, names={"C": 0, "CA": 1, "O": 2}),
        dict(uid=1, names={"N": 3, "CA": 4}),
    ]
    library = LibraryTargets(atoms={3, 4}, plane_groups=[(0, 1, 2, 3)])
    for state, shift in [(0, 0), (1, 1)]:
        library.terms["angle"].extend(
            [
                GeometryTarget(
                    (1, 0, 3), np.deg2rad(116.2 + shift), 0.02, conditions=((0, state),)
                ),
                GeometryTarget(
                    (2, 0, 3), np.deg2rad(122.7 - shift), 0.03, conditions=((0, state),)
                ),
                GeometryTarget(
                    (0, 3, 4), np.deg2rad(121.7), 0.04, conditions=((0, state),)
                ),
            ]
        )
    original = list(library.terms["angle"])
    cohere_mixed_library_links(library, residues, xyz)
    for old, new in zip(original, library.terms["angle"], strict=True):
        assert replace(new, value=old.value) == old
        if old.atoms[1] in library.atoms:
            assert new == old
    for state in (0, 1):
        rows = [
            r
            for r in library.terms["angle"]
            if r.conditions == ((0, state),) and r.atoms[1] == 0
        ]
        assert sum(r.value for r in rows) + _angle(
            xyz[1], xyz[0], xyz[2]
        ) == pytest.approx(2 * np.pi)


def test_disabled_angles_do_not_require_a_nondegenerate_reference_plane():
    xyz = _ALA_COORDS.copy()
    xyz[3] = xyz[2] + (xyz[2] - xyz[1])
    geometry = build_polymer_geometry(
        _PolymerAdapter("protein", _ALA_NAMES, xyz), {"angle": {"weight": 0}}
    )
    assert len(geometry.link_angles) == 3


def test_missing_dictionary_link_uses_dictionary_local_geometry():
    # The reference angle is irrelevant at a dictionary-covered center.
    local = [
        GeometryTarget(
            (1, 0, 2), np.deg2rad(120 + state), 0.03, conditions=((0, state),)
        )
        for state in (0, 1)
    ]
    library = LibraryTargets(atoms={0, 1, 2}, terms={"angle": list(local)})
    angles = [(1, 0, 3, np.deg2rad(116.2), 0.02), (2, 0, 3, np.deg2rad(122.7), 0.04)]
    result = cohere_dictionary_fallback_links(angles, [(0, 1, 2, 3)], library)
    assert result == []
    assert library.terms["angle"][:2] == local
    assert library.angle_tuples == {(1, 0, 3), (2, 0, 3)}
    for state in (0, 1):
        active = [r for r in library.terms["angle"] if r.conditions == ((0, state),)]
        assert len(active) == 3
        assert sum(r.value for r in active) == pytest.approx(2 * np.pi)
        assert [r.esd for r in active] == [0.03, 0.02, 0.04]


def test_missing_dictionary_phosphate_link_preserves_local_gram_matrix():
    import itertools

    xyz, _residues, angles = _phosphate()
    neighbours = (1, 2, 3)
    local = [
        GeometryTarget((i, 0, j), _angle(xyz[i], xyz[0], xyz[j]), 0.03)
        for i, j in itertools.combinations(neighbours, 2)
    ]
    library = LibraryTargets(atoms={0, 1, 2, 3}, terms={"angle": list(local)})
    assert cohere_dictionary_fallback_links(angles, [], library) == []
    directions = xyz[list(neighbours)] - xyz[0]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    partner = np.linalg.solve(
        directions, np.cos([r.value for r in library.terms["angle"][3:]])
    )
    assert np.linalg.norm(partner) == pytest.approx(1)
    assert library.terms["angle"][:3] == local
