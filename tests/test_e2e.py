"""Public-API E2E checks against independent NumPy/SciPy geometry objectives.

All fixtures are small, offline, and generated in memory or pytest's temporary
directory. Neither the oracle nor its derivatives call RGI's energy kernels.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from rgi_toolkit import CombinedRestraints
from rgi_toolkit.atom_context import AtomRecord, LigandConf


@dataclass
class Adapter:
    atoms: list[AtomRecord]
    ligands: list[LigandConf] = field(default_factory=list)
    elements: np.ndarray | None = None

    def iter_atoms(self):
        return iter(self.atoms)

    def iter_ligand_confs(self):
        return iter(self.ligands)

    def num_atoms(self):
        return len(self.atoms)

    def get_elements(self):
        return self.elements


@pytest.fixture(autouse=True)
def isolate_verbose_logging(monkeypatch):
    """Do not leave a handler pointing at pytest's closed capture stream."""
    logger = logging.getLogger("rgi_toolkit")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)


@pytest.fixture(params=itertools.product(("torch", "jax"), ("CG", "l-bfgs")))
def solver(request):
    jax.config.update("jax_enable_x64", True)
    return request.param


def setup(adapter, config, solver, max_iter=1000):
    cr = CombinedRestraints()
    cr.setup(
        adapter,
        nbatch=2,
        config={
            "gpu": False,
            "method": solver[1],
            "max_iter": max_iter,
            "verbose": True,
            **config,
        },
    )
    assert cr.is_active(), "the public config must build an active restraint"
    if solver[0] == "jax":
        assert cr.get_minimizer() is not None
    return cr


def native(coords, solver):
    return (
        torch.tensor(coords, dtype=torch.float64)
        if solver[0] == "torch"
        else jnp.asarray(coords)
    )


def diagnostic(cr, coords, solver, capsys, expected):
    value = native(coords, solver)
    before = np.asarray(coords).copy()
    capsys.readouterr()
    cr.finalize(value)
    text = capsys.readouterr().out
    totals = re.findall(r"finalize .*total=([-+\deE.]+)", text)
    assert len(totals) == 1, text
    assert float(totals[0]) == pytest.approx(expected, rel=1e-6, abs=1e-5)
    np.testing.assert_array_equal(np.asarray(value), before)


def drive(cr, initial, solver, sigmas=(0.0,), steps=None, kicks=None):
    sigmas = np.asarray(sigmas, dtype=float)
    steps = np.arange(len(sigmas)) if steps is None else np.asarray(steps)
    kicks = (
        np.zeros((len(sigmas), *initial.shape)) if kicks is None else np.asarray(kicks)
    )
    value = native(initial, solver)
    if solver[0] == "torch":
        history = []
        for sigma, step, kick in zip(sigmas, steps, kicks, strict=True):
            value += torch.tensor(kick)
            result = cr.minimize(value, istep=int(step), sigma=float(sigma))
            assert result is value, "Torch's public API updates its input in place"
            history.append(value.numpy().copy())
        return np.asarray(history)
    minimizer = cr.get_minimizer()

    def scan(coords):
        def body(x, item):
            sigma, step, kick = item
            out = minimizer(x + kick, sigma, step)
            return out, out

        return jax.lax.scan(
            body, coords, (jnp.asarray(sigmas), jnp.asarray(steps), jnp.asarray(kicks))
        )

    _, history = jax.jit(scan)(value)
    np.testing.assert_array_equal(np.asarray(value), initial)
    return np.asarray(history)


def scipy_solution(fun, initial, method, jac="3-point"):
    options = {"maxiter": 10_000, "gtol": 1e-9}
    name = "CG" if method == "CG" else "L-BFGS-B"
    if name == "L-BFGS-B":
        options["ftol"] = 1e-14
    result = minimize(
        fun, np.asarray(initial).ravel(), jac=jac, method=name, options=options
    )
    assert np.isfinite(result.fun) and np.isfinite(result.x).all(), result
    return result


def finite_gradient(fun, x, step=1e-6):
    x = np.asarray(x, dtype=float).ravel()
    gradient = np.empty_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = step
        gradient[i] = (fun(x + delta) - fun(x - delta)) / (2 * step)
    return gradient


PENALTIES = {
    "harmonic": {"target_distance": 6.0},
    "flat-bottomed": {"target_distance1": 4.0, "target_distance2": 6.0},
    "flat-bottomed1": {"target_distance1": 6.0},
    "flat-bottomed2": {"target_distance2": 6.0},
}


def distance_residual(distance, kind):
    if kind == "harmonic":
        return distance - 6.0
    if kind == "flat-bottomed":
        return distance - np.clip(distance, 4.0, 6.0)
    return (
        min(distance - 6.0, 0.0)
        if kind == "flat-bottomed1"
        else max(distance - 6.0, 0.0)
    )


@pytest.mark.parametrize("kind", PENALTIES)
@pytest.mark.parametrize("move", ("both", 1, 2))
def test_distance_penalties_groups_and_batch(solver, kind, move, capsys):
    first, second = np.array([1, 4]), np.array([0, 5, 8])
    unused = np.array([2, 3, 6, 7])
    atoms = [
        AtomRecord("A" if i in first else "B" if i in second else "Z", 1, i)
        for i in range(9)
    ]
    coords = np.full((2, 9, 3), 20.0)
    coords[:, first] = [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]]
    coords[:, second] = [[0.0, 1.0, 0.0], [0.0, -0.5, 0.5], [0.0, -0.5, -0.5]]
    coords[:, second, 0] = np.array(
        [2.0, 3.0] if kind == "flat-bottomed1" else [12.0, 14.0]
    )[:, None]
    cr = setup(
        Adapter(atoms),
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "move": move,
                    kind: PENALTIES[kind],
                }
            ]
        },
        solver,
    )
    np.testing.assert_array_equal(cr.spec.active_sites, sorted([*first, *second]))

    def energy(batch):
        distance = np.linalg.norm(batch[first].mean(0) - batch[second].mean(0))
        return distance_residual(distance, kind) ** 2

    diagnostic(cr, coords, solver, capsys, sum(map(energy, coords)))
    out = drive(cr, coords, solver)[-1]
    np.testing.assert_array_equal(out[:, unused], coords[:, unused])
    for group in (first, second):
        # Gradient rescaling translates each group rigidly, even for unequal sizes.
        np.testing.assert_allclose(
            out[:, group] - out[:, group].mean(1, keepdims=True),
            coords[:, group] - coords[:, group].mean(1, keepdims=True),
            atol=1e-10,
        )
    fixed = second if move == 1 else first if move == 2 else unused
    np.testing.assert_array_equal(out[:, fixed], coords[:, fixed])
    if move == "both":
        np.testing.assert_allclose(
            out[:, np.r_[first, second]].mean(1),
            coords[:, np.r_[first, second]].mean(1),
            atol=1e-10,
        )
    moving = np.r_[first, second] if move == "both" else first if move == 1 else second
    for original, actual in zip(coords, out, strict=True):

        def objective(flat):
            trial = original.copy()
            trial[moving] = flat.reshape(-1, 3)
            return energy(trial)

        ref = scipy_solution(objective, original[moving], solver[1])
        assert ref.fun < 1e-8, ref
        assert energy(actual) <= ref.fun + 1e-6
    diagnostic(cr, out, solver, capsys, sum(map(energy, out)))


@pytest.mark.parametrize("window", ("sigma", "step"))
@pytest.mark.parametrize("custom", (False, True))
def test_schedule_selection_and_re_setup_match_scipy(solver, window, custom, capsys):
    anchors = np.array([0, 3, 5, 8])
    moving = 4
    target = np.array([1.2, 0.8, 1.1])
    coords = np.full((9, 3), 20.0)
    coords[anchors] = [[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]]
    coords[moving] = target + [0.3, -0.5, 0.4]
    atoms = [
        AtomRecord("A", i + 1, i, name="CA", mol_type="protein", resname="ALA")
        for i in range(9)
    ]
    gate = (
        {"start_sigma": 2.0, "stop_sigma": 1.0}
        if window == "sigma"
        else {"start_step": 1, "stop_step": 2}
    )
    distances = np.linalg.norm(coords[anchors] - target, axis=1)
    entries = []
    for i, distance in zip(anchors, distances, strict=True):
        if custom:
            entries.append(
                {
                    "energy": f"harmonic(distance(M, F), {float(distance)!r})",
                    "selections": {
                        "M": "chain A and resid 5 and name CA",
                        "F": f"index {i}",
                    },
                    "move": "M",
                    **gate,
                }
            )
        else:
            entries.append(
                {
                    "atom_selection1": "chain A and resid 5 and name CA",
                    "atom_selection2": f"index {i}",
                    "move": 1,
                    "harmonic": {"target_distance": float(distance)},
                    **gate,
                }
            )
    config = {
        "custom_restraints_config" if custom else "distance_restraints_config": entries
    }
    cr = setup(Adapter(atoms), config, solver)

    def energy(x):
        return np.sum((np.linalg.norm(x - coords[anchors], axis=1) - distances) ** 2)

    diagnostic(cr, coords, solver, capsys, energy(coords[moving]))
    kick = np.zeros_like(coords)
    kick[moving] = [0.05, 0.1, -0.08]
    history = drive(
        cr, coords, solver, sigmas=[3.0, 2.0, 1.0, 0.5], kicks=np.tile(kick, (4, 1, 1))
    )
    np.testing.assert_array_equal(history[0], coords + kick)
    for index in (1, 2):
        ref = scipy_solution(
            energy, history[index - 1, moving] + kick[moving], solver[1]
        )
        assert ref.fun < 1e-8, ref
        np.testing.assert_allclose(history[index, moving], ref.x, rtol=0, atol=1e-3)
        assert np.max(np.abs(finite_gradient(energy, history[index, moving]))) < 1e-3
    np.testing.assert_allclose(history[3], history[2] + kick, rtol=0, atol=1e-14)
    untouched = np.delete(np.arange(9), moving)
    np.testing.assert_array_equal(
        history[:, untouched],
        np.broadcast_to(coords[untouched], (4, len(untouched), 3)),
    )
    diagnostic(cr, history[-1], solver, capsys, energy(history[-1, moving]))
    # A reused instance must lose all old selections, gates, and optimizer state.
    cr.setup(Adapter(atoms), config={})
    assert not cr.is_active()
    value = native(coords, solver)
    np.testing.assert_array_equal(np.asarray(cr.minimize(value, sigma=0.0)), coords)
    fresh = setup(Adapter(atoms), config, solver)
    np.testing.assert_allclose(
        drive(fresh, coords, solver, sigmas=[2.0], steps=[1])[-1, moving],
        target,
        rtol=0,
        atol=1e-3,
    )


def angle(points):
    first, second = points[0] - points[1], points[2] - points[1]
    return np.arccos(
        np.clip(
            first @ second / (np.linalg.norm(first) * np.linalg.norm(second)), -1, 1
        )
    )


def torsion(points):
    axis = points[2] - points[1]
    axis /= np.linalg.norm(axis)
    first, second = points[0] - points[1], points[3] - points[2]
    first -= (first @ axis) * axis
    second -= (second @ axis) * axis
    return np.arctan2(-axis @ np.cross(first, second), first @ second)


def plane_rms(points):
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
    return singular[-1] / np.sqrt(len(points))


@pytest.mark.parametrize("kind", ("angle", "dihedral", "improper", "plane"))
def test_group_geometry_matches_independent_scipy_objective(solver, kind, capsys):
    ids = np.array([1, 3, 5, 7])
    coords = np.full((9, 3), 20.0)
    coords[ids] = [[0.3, 1.2, 0.1], [0, 0, 0], [1, 0, 0], [1, -1, -0.1]]
    target = np.pi / 3 if kind == "angle" else np.deg2rad(-179)
    entry = {
        f"atom_selection{i + 1}": f"index {atom}"
        for i, atom in enumerate(ids[:3] if kind == "angle" else ids)
    }
    free = ids[0] if kind in ("angle", "plane") else ids[3]
    if kind == "plane":
        coords[ids] = [[0.5, 0.3, 0.5], [0, 0, 0], [2, 0, 0], [0, 1, 0]]
        entry = {
            "atom_selection1": "index 1",
            "atom_selection2": "index 3 or index 5 or index 7",
            "move": 1,
        }
    else:
        entry.update(
            {
                "move": 1 if kind == "angle" else 4,
                "unit": "radians",
                "harmonic": {f"target_{kind}": target},
            }
        )
    cr = setup(
        Adapter([AtomRecord("A", i + 1, i) for i in range(9)]),
        {f"{kind}_restraints_config": [entry]},
        solver,
    )

    def energy(flat):
        points = coords.copy()
        points[free] = flat
        points = points[ids]
        if kind == "plane":
            return plane_rms(points) ** 2
        measured = angle(points[:3]) if kind == "angle" else torsion(points)
        residual = measured - target
        if kind != "angle":
            residual = np.angle(np.exp(1j * residual))
        return residual**2

    diagnostic(cr, coords, solver, capsys, energy(coords[free]))
    reference = scipy_solution(energy, coords[free], solver[1])
    assert reference.fun < 1e-8, reference
    out = drive(cr, coords, solver)[-1]
    assert energy(out[free]) <= reference.fun + 1e-6
    assert np.max(np.abs(finite_gradient(energy, out[free]))) < 1e-3
    fixed = np.delete(np.arange(9), free)
    np.testing.assert_array_equal(out[fixed], coords[fixed])
    diagnostic(cr, out, solver, capsys, energy(out[free]))


def reference_structure(tmp_path, coords, suffix):
    structure = gemmi.Structure()
    model, chain = gemmi.Model("1"), gemmi.Chain("A")
    for i, xyz in enumerate(coords):
        residue = gemmi.Residue()
        residue.name, residue.seqid = "ALA", gemmi.SeqId(i + 1, " ")
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = (
            "CA",
            gemmi.Element("C"),
            gemmi.Position(*xyz),
        )
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    path = tmp_path / f"reference.{suffix}"
    if suffix == "pdb":
        structure.write_pdb(str(path))
    else:
        structure.make_mmcif_document().write_file(str(path))
    return path


def fitted_mse(points, reference):
    moving = points - points.mean(0)
    target = reference - reference.mean(0)
    rotation, _ = Rotation.align_vectors(target, moving)
    return np.mean(np.sum((rotation.apply(moving) - target) ** 2, axis=1))


@pytest.mark.parametrize("suffix", ("pdb", "cif"))
def test_rmsd_reference_io_and_minimize_match_scipy(solver, suffix, tmp_path, capsys):
    reference = np.array(
        [[0, 0, 0], [2, 0, 0], [0.4, 1.5, 0], [0.2, 0.3, 1.8], [1.1, 1.2, 0.5]]
    )
    path = reference_structure(tmp_path, reference, suffix)
    coords = np.vstack(
        [
            Rotation.from_rotvec([0.2, -0.3, 0.1]).apply(reference) + [10, 2, -3],
            [20, 20, 20],
        ]
    )
    coords[:5] += np.random.default_rng(731).normal(0, 0.08, (5, 3))
    atoms = [
        AtomRecord(
            "A" if i < 5 else "Z",
            i + 1,
            i,
            name="CA",
            mol_type="protein",
            resname="ALA",
        )
        for i in range(6)
    ]
    cr = setup(
        Adapter(atoms),
        {
            "rmsd_restraints_config": [
                {
                    f"ref_{suffix}": str(path),
                    "atom_selection_target": "chain A and name CA",
                    "atom_selection_ref": "chain A and name CA",
                    "pairing": "identity",
                    "harmonic": {"target_rmsd": 0.0},
                }
            ]
        },
        solver,
    )
    np.testing.assert_array_equal(cr.spec.active_sites, np.arange(5))

    def energy(flat):
        return fitted_mse(flat.reshape(5, 3), reference)

    diagnostic(cr, coords, solver, capsys, energy(coords[:5].ravel()))
    result = scipy_solution(energy, coords[:5], solver[1])
    assert result.fun < 1e-8, result
    out = drive(cr, coords, solver)[-1]
    np.testing.assert_array_equal(out[5], coords[5])
    assert energy(out[:5].ravel()) <= result.fun + 1e-6
    assert np.max(np.abs(finite_gradient(energy, out[:5]))) < 1e-3
    diagnostic(cr, out, solver, capsys, energy(out[:5].ravel()))


def test_custom_callable_rosenbrock_through_public_api(solver, capsys):
    def energy(ctx):
        xyz = ctx.coords("index 1")
        x, y, z = xyz[..., 0, 0], xyz[..., 0, 1], xyz[..., 0, 2]
        return ctx.sum(100 * (y - x * x) ** 2 + (1 - x) ** 2 + z * z)

    coords = np.array([[20, 20, 20], [-1.2, 1.0, 0.5], [30, 30, 30]])
    cr = setup(
        Adapter([AtomRecord("A", i + 1, i) for i in range(3)]),
        {"custom_restraints_config": [{"fn": energy}]},
        solver,
        max_iter=10_000,
    )
    from scipy.optimize import rosen, rosen_der

    def objective(x):
        return rosen(x[:2]) + x[2] ** 2

    def gradient(x):
        return np.r_[rosen_der(x[:2]), 2 * x[2]]

    diagnostic(cr, coords, solver, capsys, objective(coords[1]))
    ref = scipy_solution(objective, coords[1], solver[1], jac=gradient)
    assert np.max(np.abs(gradient(ref.x))) < 1e-6, ref
    out = drive(cr, coords, solver)[-1]
    np.testing.assert_array_equal(out[[0, 2]], coords[[0, 2]])
    np.testing.assert_allclose(out[1], ref.x, rtol=0, atol=1e-3)
    assert np.max(np.abs(gradient(out[1]))) < (1e-6 if solver[1] == "CG" else 1e-3)
    diagnostic(cr, out, solver, capsys, objective(out[1]))


def ligand(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=731) == 0
    AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol, np.asarray(mol.GetConformer().GetPositions()).copy()


@pytest.mark.parametrize("kind", ("chiral", "cistrans", "plane"))
def test_conformer_targets_and_minima_match_scipy(solver, kind, capsys):
    smiles = {"chiral": "F[C@](Cl)(Br)I", "cistrans": "C/C=C/C", "plane": "c1ccccc1"}[
        kind
    ]
    mol, reference = ligand(smiles)
    n = len(reference)
    ids = 2 * np.arange(n) + 1
    coords = np.full((2 * n + 1, 3), 20.0)
    coords[ids] = reference + np.random.default_rng(732).normal(0, 0.03, (n, 3))
    elements = np.zeros(len(coords), dtype=int)
    elements[ids] = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    adapter = Adapter(
        [AtomRecord("L" if i in ids else "Z", 1, i) for i in range(len(coords))],
        [LigandConf(mol, reference, ids, conformer_restraints=True)],
        elements,
    )
    config = {
        "relax_force_field": {"ligand": "none"},
        **{key: {"weight": 0} for key in ("chiral", "cistrans", "vdw")},
        "bond": {},
        "angle": {},
        kind: {"weight": 1, "slack": 0.0},
    }
    cr = setup(adapter, {"conformer_restraints_config": config}, solver, max_iter=5000)
    # Explicit topology of the three fixtures, independent of the spec builder.
    if kind == "chiral":
        bonds = [(1, i) for i in (0, 2, 3, 4)]
        angles = [(i, 1, j) for i, j in itertools.combinations((0, 2, 3, 4), 2)]
        chirals = [(1, *triple) for triple in itertools.combinations((0, 2, 3, 4), 3)]
    elif kind == "cistrans":
        bonds, angles = [(0, 1), (1, 2), (2, 3)], [(0, 1, 2), (1, 2, 3)]
        chirals = []
    else:
        bonds = [(i, (i + 1) % 6) for i in range(6)]
        angles = [((i - 1) % 6, i, (i + 1) % 6) for i in range(6)]
        chirals = []
    assert len(cr.spec.bond.idx) == len(bonds)
    assert len(cr.spec.angle.idx) == len(angles)
    assert len(getattr(cr.spec, kind).idx) == (4 if kind == "chiral" else 1)
    np.testing.assert_array_equal(cr.spec.active_sites, ids)
    bond_targets = [np.linalg.norm(reference[i] - reference[j]) for i, j in bonds]
    angle_targets = [angle(reference[list(row)]) for row in angles]

    def volume(points):
        return np.linalg.det(points[1:] - points[0])

    chiral_targets = [volume(reference[list(row)]) for row in chirals]

    def objective(flat):
        points = flat.reshape(n, 3)
        total = sum(
            (np.linalg.norm(points[i] - points[j]) - value) ** 2
            for (i, j), value in zip(bonds, bond_targets, strict=True)
        )
        total += sum(
            (angle(points[list(row)]) - value) ** 2
            for row, value in zip(angles, angle_targets, strict=True)
        )
        total += sum(
            (volume(points[list(row)]) - value) ** 2
            for row, value in zip(chirals, chiral_targets, strict=True)
        )
        if kind == "cistrans":
            total += np.angle(np.exp(1j * (torsion(points) - torsion(reference)))) ** 2
        elif kind == "plane":
            total += plane_rms(points) ** 2
        return total

    diagnostic(cr, coords, solver, capsys, objective(coords[ids].ravel()))
    result = scipy_solution(objective, coords[ids], solver[1])
    assert result.fun < 1e-6, result
    out = drive(cr, coords, solver)[-1]
    np.testing.assert_array_equal(out[::2], coords[::2])
    actual = objective(out[ids].ravel())
    assert abs(actual - result.fun) <= 1e-6 * (1 + abs(result.fun))
    assert np.max(np.abs(finite_gradient(objective, out[ids]))) < 1e-3
    diagnostic(cr, out, solver, capsys, actual)


def test_intramolecular_vdw_matches_dense_scipy(solver, capsys):
    mol, reference = ligand("CCCC")
    coords = reference.copy()
    coords[0], coords[3] = [0, 0, 0], [2, 0, 0]
    adapter = Adapter(
        [AtomRecord("L", 1, i) for i in range(4)],
        [LigandConf(mol, reference, np.arange(4), conformer_restraints=True)],
        np.full(4, 6),
    )
    cr = setup(
        adapter,
        {
            "conformer_restraints_config": {
                "relax_force_field": {"ligand": "none"},
                "vdw": {"mode": "intramolecular", "weight": 0.04, "max_atom_step": 2.0},
                **{
                    key: {"weight": 0}
                    for key in ("bond", "angle", "chiral", "cistrans")
                },
            }
        },
        solver,
    )
    # Only terminal carbons are a nonexcluded 1-4 pair: (1.94 - .15) * 2 = 3.58 A.
    assert len(cr.spec.vdw.idx) == 1
    assert cr.spec.vdw_config is None

    def objective(flat):
        points = flat.reshape(4, 3)
        return min(np.linalg.norm(points[0] - points[3]) - 3.58, 0.0) ** 2

    diagnostic(cr, coords, solver, capsys, objective(coords.ravel()))
    result = scipy_solution(objective, coords, solver[1])
    assert result.fun < 1e-8, result
    out = drive(cr, coords, solver)[-1]
    assert objective(out.ravel()) <= result.fun + 1e-6
    np.testing.assert_array_equal(out[1:3], coords[1:3])
    np.testing.assert_allclose(out[[0, 3]].mean(0), coords[[0, 3]].mean(0), atol=1e-10)
    diagnostic(cr, out, solver, capsys, objective(out.ravel()))


def test_dynamic_vdw_new_contacts_match_dense_scipy(solver, capsys):
    mol = Chem.MolFromSmiles("C")
    mol.AddConformer(Chem.Conformer(1))
    ids = (1, 3, 5)
    coords = np.full((6, 3), 20.0)
    # Initial energy lies below the far-side stationary point. All monotone solvers
    # can therefore be compared in the same basin, with no initial clash.
    coords[list(ids)] = [[1, 0, 0], [8, 0, 0], [15, 0, 0]]
    elements = np.zeros(6, dtype=int)
    elements[list(ids)] = 6
    adapter = Adapter(
        [AtomRecord(str(i), 1, i) for i in range(6)],
        [
            LigandConf(
                Chem.Mol(mol),
                np.zeros((1, 3)),
                np.array([i]),
                conformer_restraints=i == 1,
            )
            for i in ids
        ],
        elements,
    )

    def pull(ctx):
        xyz = ctx.coords("index 1")
        return ctx.sum((xyz[..., 0] - 5) ** 2 + xyz[..., 1] ** 2 + xyz[..., 2] ** 2)

    cr = setup(
        adapter,
        {
            "conformer_restraints_config": {
                "relax_force_field": {"ligand": "none"},
                "vdw": {
                    "mode": "intermolecular",
                    "weight": 0.04,
                    "dmax": 0.5,
                    "neighbor_skin": 0.0,
                    "neighbor_rebuild_interval": 1,
                    "max_atom_step": 4.0,
                },
            },
            "custom_restraints_config": [{"fn": pull}],
        },
        solver,
    )
    np.testing.assert_array_equal(cr.spec.active_sites, [1])
    assert cr.spec.vdw_config is not None

    # Both atoms of an ordinary carbon contact have a 1.94 A radius. The chosen
    # weight cancels ESD**2 = .2**2; score EVERY background on every reference call.
    def objective(point):
        distances = np.linalg.norm(point - coords[[3, 5]], axis=1)
        return np.sum((point - [5, 0, 0]) ** 2) + np.sum(
            np.minimum(distances - 3.88, 0) ** 2
        )

    diagnostic(cr, coords, solver, capsys, objective(coords[1]))
    result = scipy_solution(objective, coords[1], solver[1])
    assert np.max(np.abs(finite_gradient(objective, result.x))) < 1e-6, result
    np.testing.assert_allclose(result.x, [4.56, 0, 0], atol=1e-6)
    out = drive(cr, coords, solver)[-1]
    np.testing.assert_allclose(out[1], result.x, rtol=0, atol=1e-3)
    assert abs(objective(out[1]) - result.fun) < 1e-6
    assert np.max(np.abs(finite_gradient(objective, out[1]))) < 1e-3
    np.testing.assert_array_equal(out[[0, 2, 3, 4, 5]], coords[[0, 2, 3, 4, 5]])
    diagnostic(cr, out, solver, capsys, objective(out[1]))
