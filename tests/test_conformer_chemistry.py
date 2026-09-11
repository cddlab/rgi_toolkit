"""Linear-angle stability, polymer torsions, and chemical VdW parity."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_toolkit._array_ops import get_ops
from rgi_toolkit._vdw_chemistry import AtomType, build_chemistry, pair_contact
from rgi_toolkit.atom_context import AtomRecord, LigandConf
from rgi_toolkit.energy import numpy_energy
from rgi_toolkit.energy._kernels import angle_energy
from rgi_toolkit.energy._nonbonded import prepare_chemistry
from rgi_toolkit.featurizer import build_spec


def _value_grad(function, coords, backend):
    if backend == "numpy":
        return float(function(np.asarray(coords))), None
    if backend == "torch":
        torch = pytest.importorskip("torch")
        x = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
        value = function(x)
        return float(value.detach()), torch.autograd.grad(value, x)[0].detach().numpy()
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    value, gradient = jax.jit(jax.value_and_grad(function))(jax.numpy.asarray(coords))
    return float(value), np.asarray(gradient)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("target", [179.49, 179.5, 179.51, 180.0])
@pytest.mark.parametrize("slack", [0.0, 5.0])
def test_linear_angle_threshold_and_angular_slack(backend, target, slack):
    ops = get_ops(backend)
    angle = math.radians(170)
    coords = np.array([[1.0, 0, 0], [0, 0, 0], [math.cos(angle), math.sin(angle), 0]])

    def energy(x):
        return angle_energy(
            ops,
            x,
            ops.asint([[0, 1, 2]]),
            ops.const_like([math.radians(target)], x),
            ops.const_like([math.radians(slack)], x),
            ops.const_like([1 / math.radians(2) ** 2], x),
            ops.const_like([1.0], x),
        )

    value, grad = _value_grad(energy, coords, backend)
    if target > 179.5:
        residual = 2 * (
            math.sin(math.radians(10) / 2) - math.sin(math.radians(slack) / 2)
        )
    else:
        residual = math.radians(target - 170 - slack)
    assert value == pytest.approx(residual**2 / math.radians(2) ** 2, rel=1e-8)
    if grad is not None:
        numeric = np.zeros_like(coords)
        for idx in np.ndindex(coords.shape):
            plus, minus = coords.copy(), coords.copy()
            plus[idx] += 1e-6
            minus[idx] -= 1e-6
            # Use the same backend's value; this checks the actual differentiable objective.
            if backend == "torch":
                import torch

                plus, minus = torch.as_tensor(plus), torch.as_tensor(minus)
            numeric[idx] = (float(energy(plus)) - float(energy(minus))) / 2e-6
        np.testing.assert_allclose(grad, numeric, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("length", [0.0, 0.001, 1.0])
def test_exact_linear_angle_has_finite_energy_and_gradient(backend, length):
    ops = get_ops(backend)

    def energy(x):
        constants = [ops.const_like([v], x) for v in (math.pi, 0.0, 1.0, 1.0)]
        return angle_energy(ops, x, ops.asint([[0, 1, 2]]), *constants)

    coords = np.array([[-length, 0, 0], [0, 0, 0], [length, 0, 0]])
    value, grad = _value_grad(energy, coords, backend)
    assert math.isfinite(value)
    if length == 1:
        assert value == 0
    if grad is not None:
        assert np.isfinite(grad).all()
        if length == 1:
            np.testing.assert_array_equal(grad, 0)


def _peptide(sequence):
    mol = Chem.AddHs(Chem.MolFromSequence(sequence))
    assert AllChem.EmbedMolecule(mol, randomSeed=319) == 0
    mol = Chem.RemoveHs(mol)
    coords = np.asarray(mol.GetConformer().GetPositions())
    records = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        records.append(
            AtomRecord(
                "A",
                info.GetResidueNumber(),
                atom.GetIdx(),
                info.GetName().strip(),
                "protein",
                info.GetResidueName().strip(),
                True,
            )
        )
    elements = np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()])
    return SimpleNamespace(
        iter_atoms=lambda: iter(records),
        get_elements=lambda: elements,
        get_reference_positions=lambda: coords,
    ), coords


def test_dictionary_free_chi_omega_and_sp2_keep_reference_periods(monkeypatch):
    from rgi_toolkit import monlib_geom
    from rgi_toolkit.polymer import build_polymer_geometry

    def no_download(*args, **kwargs):
        pytest.fail("dictionary-free chemistry must not load a monomer library")

    monkeypatch.setattr(monlib_geom.MonomerLibrary, "load", no_download)
    adapter, coords = _peptide("SFR")
    config = {key: {"weight": 0} for key in ("bond", "angle", "chiral", "vdw")}
    geometry = build_polymer_geometry(adapter, config)
    targets = geometry.library
    assert not targets.atoms
    assert len(targets.peptides) == 2
    lookup = {(r.resid, r.name): r.index for r in adapter.iter_atoms()}
    for resid, names, period, esd in (
        (1, ("N", "CA", "CB", "OG"), 3, 10),
        (2, ("CA", "CB", "CG", "CD1"), 6, 10),
        (3, ("CD", "NE", "CZ", "NH2"), 2, 5),
    ):
        quad = tuple(lookup[resid, name] for name in names)
        (row,) = [r for r in targets.terms["cistrans"] if r.atoms == quad]
        assert row.period == period
        assert row.esd == pytest.approx(math.radians(esd))
    assert all(r.period in (1, 2, 3, 6) for r in targets.terms["cistrans"])
    spec = build_spec(conformer_config=config, polymer_geometry=geometry)
    prepared = numpy_energy.prepare_spec(spec)
    # Only omega deviates at the embedded reference; every chi uses its own reference.
    free = replace(
        spec,
        cistrans=replace(
            spec.cistrans, mask=spec.cistrans.mask * (spec.cistrans.period != 1)
        ),
    )
    assert numpy_energy.total_energy(
        coords[spec.active_sites], numpy_energy.prepare_spec(free)
    ) == pytest.approx(0, abs=1e-15)
    assert numpy_energy.total_energy(coords[spec.active_sites], prepared) >= 0


def test_missing_chi_atoms_warn_instead_of_inventing_coordinates(caplog):
    from rgi_toolkit.polymer import build_polymer_geometry

    adapter, _ = _peptide("S")
    records = [r for r in adapter.iter_atoms() if r.name != "OG"]
    adapter.iter_atoms = lambda: iter(records)
    geometry = build_polymer_geometry(adapter, {"cistrans": {}})
    assert not geometry.library.terms["cistrans"]
    assert "chi1 (missing atoms)" in caplog.text


def _ligand(smiles, offset=0, enabled=True):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=631) == 0
    mol = Chem.RemoveHs(mol)
    return LigandConf(
        mol,
        np.asarray(mol.GetConformer().GetPositions()),
        np.arange(mol.GetNumAtoms()) + offset,
        conformer_restraints=enabled,
    )


def test_ligand_sp2_uses_relaxed_coords_and_preserves_double_bond_ez(monkeypatch):
    from rgi_toolkit import featurizer

    ligand = _ligand("CC(=O)NC/C=C/C")
    relaxed = ligand.conf_coords.copy()
    relaxed[4, 2] += 0.4
    monkeypatch.setattr(featurizer, "ff_relax", lambda *args, **kwargs: relaxed)
    config = {key: {"weight": 0} for key in ("bond", "angle", "chiral", "vdw")}
    spec = build_spec([ligand], conformer_config=config)
    assert 1 in spec.cistrans.period
    assert 2 in spec.cistrans.period
    assert numpy_energy.total_energy(
        relaxed[spec.active_sites], numpy_energy.prepare_spec(spec)
    ) == pytest.approx(0, abs=1e-15)
    assert np.all(
        spec.cistrans.weight[spec.cistrans.period == 2]
        == pytest.approx(1 / math.radians(5) ** 2)
    )


@pytest.mark.parametrize(
    "first,second,one_four,expected",
    [
        (AtomType(1.75, 0, "N", 6), AtomType(1.75, 0, "N", 6), False, (3.5, 0.2)),
        (AtomType(1.52, 1.28, "A", 8), AtomType(1.6, 1.32, "D", 7), False, (2.82, 0.2)),
        (AtomType(1.52, 1.28, "A", 8), AtomType(1.2, 0, "H", 1), False, (1.62, 0.2)),
        (
            AtomType(1.39, 0.74, "N", 30),
            AtomType(1.52, 1.28, "A", 8),
            False,
            (2.02, 0.2),
        ),
        (AtomType(1.75, 0, "N", 6, True), AtomType(1.75, 0, "N", 6), False, (2.8, 0.3)),
        (
            AtomType(1.75, 0, "N", 6, True),
            AtomType(1.75, 0, "N", 6, True),
            False,
            (3.5, 0.3),
        ),
        (AtomType(1.52, 1.28, "A", 8), AtomType(1.6, 1.32, "D", 7), True, (2.92, 0.2)),
    ],
)
def test_servalcat_contact_rules(first, second, one_four, expected):
    assert pair_contact(first, second, one_four) == pytest.approx(expected)
    assert pair_contact(second, first, one_four) == pytest.approx(expected)


def test_one_four_exclusions_depend_on_planes_independently_of_energy_blocks():
    chain = _ligand("CCCCC")
    plain = build_spec([chain], conformer_config={"vdw": {}})
    np.testing.assert_array_equal(plain.vdw.idx, [[0, 3], [0, 4], [1, 4]])
    assert np.all(plain.vdw.weight == pytest.approx(25))
    np.testing.assert_allclose(
        plain.vdw.r_min, [1.94 + 1.92 - 0.3, 3.88, 1.92 + 1.94 - 0.3]
    )
    aromatic = _ligand("c1ccccc1")
    assert build_spec([aromatic], conformer_config={"vdw": {}}).vdw is None
    explicit = build_spec([aromatic], conformer_config={"vdw": {}, "plane": {}})
    assert explicit.vdw is None


def test_background_residue_and_ligand_chemistry_are_typed_without_opt_in():
    adapter, _ = _peptide("SA")
    records = [replace(r, conformer_restraints=False) for r in adapter.iter_atoms()]
    elements = adapter.get_elements()
    ligand = _ligand("C[NH3+]", len(elements), enabled=False)
    full_elements = np.r_[elements, [a.GetAtomicNum() for a in ligand.mol.GetAtoms()]]
    chemistry = build_chemistry([ligand], full_elements, records)
    by_name = {
        (r.resid, r.name): chemistry.types[chemistry.type_ids[r.index]] for r in records
    }
    assert by_name[1, "OG"].hb == "B"
    assert by_name[2, "N"].hb == "D"
    assert by_name[1, "O"].hb == "A"
    assert chemistry.types[chemistry.type_ids[-1]].hb == "D"
    assert chemistry.molecules[0] != chemistry.molecules[-1]


def test_dictionary_vdw_types_cover_ligands_and_fixed_background(tmp_path, caplog):
    # Deliberately nonstandard radii prove that type_energy and ener_lib are used.
    from tests.test_monlib_dictionary import _loop

    (tmp_path / "list").mkdir()
    (tmp_path / "l").mkdir()
    (tmp_path / "list/mon_lib_list.cif").write_text("data_link_list\n")
    component = "data_comp_list\n_chem_comp.id LIG\n_chem_comp.group non-polymer\n\ndata_comp_LIG\n"
    component += _loop(
        "chem_comp_atom",
        "comp_id atom_id type_symbol type_energy charge",
        [
            ("LIG", "N1", "N", "SPECIAL_D", 1),
            ("LIG", "O1", "O", "SPECIAL_A", 0),
            ("LIG", "C1", "C", "MISSING", 0),
        ],
    )
    (tmp_path / "l/LIG.cif").write_text(component)
    energy = "data_energy\n" + _loop(
        "lib_atom",
        "type weight hb_type vdw_radius vdwh_radius ion_radius element valency sp",
        [
            ("SPECIAL_D", 14, "D", 1.4, 1.41, 1.32, "N", 4, 3),
            ("SPECIAL_A", 16, "A", 1.3, 1.31, 1.28, "O", 2, 2),
            ("C", 12, "N", 1.7, 1.71, 0, "C", 4, 3),
        ],
    )
    (tmp_path / "ener_lib.cif").write_text(energy)
    records = [
        AtomRecord("L", 1, i, name, "ligand", "LIG")
        for i, name in enumerate(("N1", "O1", "C1"))
    ]
    chemistry = build_chemistry(
        [], np.array([7, 8, 6]), records, {"monomer_library": str(tmp_path)}
    )
    assert chemistry.pair(0, 1) == pytest.approx((2.42, 25))
    assert chemistry.radii[2] == pytest.approx(1.71)
    assert "MISSING" in caplog.text
    assert "using elemental types" in caplog.text

    # Even a known source-graph donor must become elemental when its dictionary
    # energy type is unknown and ener_lib has no elemental N entry.
    (tmp_path / "l/LIG.cif").write_text(component.replace("SPECIAL_D", "UNKNOWN_N"))
    donor = LigandConf(Chem.MolFromSmiles("[NH4+]"), np.zeros((1, 3)), np.array([0]))
    fallback = build_chemistry(
        [donor], np.array([7, 8, 6]), records, {"monomer_library": str(tmp_path)}
    )
    nitrogen = fallback.types[fallback.type_ids[0]]
    assert nitrogen.radius == pytest.approx(1.6)
    assert nitrogen.hb == "N"


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_static_vdw_energy_and_gradient_use_contact_esd(backend):
    ligands = [
        LigandConf(
            Chem.MolFromSmiles(smiles),
            np.zeros((1, 3)),
            np.array([i]),
            conformer_restraints=True,
        )
        for i, smiles in enumerate(("[Zn+2]", "[O-]"))
    ]
    spec = build_spec(ligands, conformer_config={"vdw": {}})
    assert spec.vdw.r_min[0] == pytest.approx(2.02)
    assert spec.vdw.weight[0] == pytest.approx(25)
    ops = get_ops(backend)
    from rgi_toolkit.energy._kernels import vdw_energy

    def energy(x, esd_scale=1):
        return vdw_energy(
            ops,
            x,
            ops.asint(spec.vdw.idx),
            ops.const_like(spec.vdw.r_min, x),
            ops.const_like(spec.vdw.weight / esd_scale**2, x),
            ops.const_like(spec.vdw.mask, x),
        )

    coords = np.array([[0.0, 0, 0], [1.7, 0, 0]])
    value, grad = _value_grad(energy, coords, backend)
    assert value == pytest.approx((0.32 / 0.2) ** 2, abs=1e-9)
    quarter, quarter_grad = _value_grad(lambda x: energy(x, 2), coords, backend)
    assert quarter == pytest.approx(value / 4)
    if grad is not None:
        np.testing.assert_allclose(grad, [[16.0, 0, 0], [-16.0, 0, 0]], atol=1e-8)
        np.testing.assert_allclose(quarter_grad, grad / 4, atol=1e-10)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("active", [False, True])
def test_typed_cell_lists_match_dense_contacts_in_collapsed_batches(backend, active):
    # More than one cell chunk, multiple molecules, excluded and eligible 1-4 pairs,
    # exact overlaps and heterogeneous atom radii all share the same bucket.
    ligands = [_ligand("CCNCO", 5 * i) for i in range(12)]
    chemistry = build_chemistry(ligands, None)
    n = len(chemistry.type_ids)
    query = np.arange(n) if active else np.arange(0, n, 2)
    target = np.arange(n) if active else np.arange(1, n, 2)
    moving, static = set(query), set(range(5)) if active else set()
    coords = np.random.default_rng(519).normal(scale=0.03, size=(2, n, 3))
    coords[0] = 0
    host = chemistry.subset(query, target, moving, static, active=active)
    k = 7
    expected = np.zeros((2, len(query), k), dtype=int)
    valid = np.zeros_like(expected, dtype=bool)
    for batch in range(2):
        for i, first in enumerate(query):
            candidates = []
            for j, second in enumerate(target):
                if first == second or (first in static and second in static):
                    continue
                pair = chemistry.pair(first, second)
                if pair is None:
                    continue
                distance = math.sqrt(
                    np.square(coords[batch, first] - coords[batch, second]).sum()
                    + 1e-12
                )
                candidates.append((distance - pair[0], j))
            selected = sorted(candidates)[:k]
            expected[batch, i, : len(selected)] = [j for _, j in selected]
            valid[batch, i, : len(selected)] = True
    ops = get_ops(backend)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from rgi_toolkit.optim._torch_cg_gpu import _build_cell_pairs_torch as builder

        x = torch.as_tensor(coords)
    else:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        from rgi_toolkit.optim.jax_optim import _build_cell_pairs_jax as builder

        x = jax.numpy.asarray(coords)
    typed = prepare_chemistry(ops, host, x)

    def build(x):
        return builder(
            x[..., query, :],
            x[..., target, :],
            5.0,
            k,
            exclude_self=active,
            pair_scale=1.0,
            chemistry=typed,
        )

    neighbours, scores = build(x) if backend == "torch" else jax.jit(build)(x)
    np.testing.assert_array_equal(np.asarray(neighbours), expected)
    np.testing.assert_array_equal(np.isfinite(np.asarray(scores)), valid)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("active", [False, True])
def test_typed_dynamic_ranking_energy_gradient_and_esd_scaling(backend, active):
    # The nearer donor has a shorter hydrogen-bond contact than the farther carbon.
    # K=1 must retain the carbon, which has the larger true overlap.
    elements = np.array([8, 7, 6])
    chemical = build_chemistry([], elements)
    chemical.types = [
        AtomType(1.52, 1.28, "A", 8),
        AtomType(1.6, 1.32, "D", 7),
        AtomType(1.75, 0, "N", 6),
    ]
    chemical.type_ids = np.arange(3)
    for i in range(3):
        for j in range(3):
            contact, sigma = pair_contact(chemical.types[i], chemical.types[j])
            chemical.contact_table[i, j] = contact
            chemical.inv_variance_table[i, j] = 1 / sigma**2
    coords = np.array([[0.0, 0, 0], [2.7, 0, 0], [3.0, 0, 0]])
    query = np.arange(3) if active else np.array([0])
    target = np.arange(3) if active else np.array([1, 2])
    host = chemical.subset(query, target, {0}, set(), active=active)
    ops = get_ops(backend)

    def evaluate(x, wider=False):
        typed = prepare_chemistry(ops, host, x)
        if wider:
            typed = {
                **typed,
                "inv_variances": typed["inv_variances"] / 4,
                "one_four_inv_variances": typed["one_four_inv_variances"] / 4,
            }
        radii = ops.const_like([1.52, 1.6, 1.75], x)
        if backend == "torch":
            from rgi_toolkit.optim._torch_cg_gpu import _vdw_pair_energy as fixed_energy
            from rgi_toolkit.optim._torch_cg_gpu import (
                active_vdw_pair_energy as active_energy,
            )
            from rgi_toolkit.optim._torch_cg_gpu import (
                build_active_vdw_pairs as active_pairs,
            )
            from rgi_toolkit.optim._torch_cg_gpu import (
                build_fixed_vdw_pairs as fixed_pairs,
            )
        else:
            from rgi_toolkit.optim.jax_optim import (
                _active_vdw_pair_energy as active_energy,
            )
            from rgi_toolkit.optim.jax_optim import (
                _build_active_vdw_pairs as active_pairs,
            )
            from rgi_toolkit.optim.jax_optim import (
                _build_fixed_vdw_pairs as fixed_pairs,
            )
            from rgi_toolkit.optim.jax_optim import _vdw_pair_energy as fixed_energy
        if active:
            neighbours, factor = active_pairs(
                x, radii, ops.asint([1, 0, 0]) > 0, ops.asint([]), 5.0, 2, 1.0, typed
            )
            return active_energy(x, neighbours, factor, radii, 1.0, 1.0, typed)
        neighbours, mask = fixed_pairs(
            x[:1], x[1:], ops.asint([0]), 5.0, 1, radii[:1], radii[1:], 1.0, typed
        )
        return fixed_energy(
            x[:1],
            x[1:],
            ops.asint([0]),
            neighbours,
            mask,
            radii[:1],
            radii[1:],
            1.0,
            1.0,
            typed,
        )

    value, grad = _value_grad(evaluate, coords, backend)
    expected = (0.27**2 + (0.12**2 if active else 0)) / 0.2**2
    assert value == pytest.approx(expected, abs=1e-9)
    assert np.isfinite(grad).all()
    assert np.linalg.norm(grad[2]) > 0
    wider, wider_grad = _value_grad(lambda x: evaluate(x, True), coords, backend)
    assert wider == pytest.approx(value / 4)
    np.testing.assert_allclose(wider_grad, grad / 4, atol=1e-10)


def _typed_optimizer_spec(custom=False):
    from rgi_toolkit.combined import CombinedRestraints

    records = [
        AtomRecord(chain, 1, i, name, "ligand", resname)
        for i, (chain, name, resname) in enumerate(
            (("L", "ZN", "ZN"), ("B", "O", "HOH"), ("C", "C", "UNK"))
        )
    ]
    ligand = LigandConf(
        Chem.MolFromSmiles("[Zn+2]"),
        np.zeros((1, 3)),
        np.array([0]),
        conformer_restraints=True,
    )
    adapter = SimpleNamespace(
        iter_atoms=lambda: iter(records),
        iter_ligand_confs=lambda: iter([ligand]),
        get_elements=lambda: np.array([30, 8, 6]),
    )
    config = {
        # This fixture checks typed scoring and caches, independently of cap failure.
        "conformer_restraints_config": {"vdw": {"max_atom_step": 2.0}},
        "distance_restraints_config": [
            {
                "atom_selection1": "index 0",
                "atom_selection2": "index 2",
                "harmonic": {"target_distance": 3.5},
            }
        ],
    }
    if custom:
        config["custom_restraints_config"] = [
            {
                "energy": "distance(A,B)**2",
                "weight": 0.01,
                "selections": {"A": "index 0", "B": "index 2"},
            }
        ]
    restraint = CombinedRestraints()
    restraint.setup(adapter, config=config)
    assert restraint.spec.vdw_config is not None
    assert restraint.spec.active_vdw_config is not None
    return restraint.spec


@pytest.mark.gpu
@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("custom", [False, True])
def test_typed_vdw_cuda_compilation_and_dtype_cache(mode, custom):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from rgi_toolkit.optim import _torch_cg_gpu
    from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer

    spec = _typed_optimizer_spec(custom)
    if mode == 1:
        spec.active_vdw_config = None
    elif mode == 2:
        spec.vdw_config = None
    optimizer = TorchRestraintOptimizer(spec, max_iter=50)
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor(
            [[0.0, 0, 0], [1.5, 0, 0], [0, 2.0, 0]], device="cuda", dtype=dtype
        )
        before = optimizer.energy(x)
        out = optimizer.minimize(x.clone())
        assert optimizer.energy(out) < before
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out[1], x[1])
        for data in (optimizer._vdw, optimizer._active_vdw):
            if data is not None:
                assert data["chemistry"]["contacts"].device == x.device
                assert data["chemistry"]["contacts"].dtype == x.dtype
        assert not _torch_cg_gpu._compile_failed[mode]
        if custom:
            assert optimizer._custom_cvg
            assert all(value is not False for value in optimizer._custom_cvg.values())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_typed_jax_jit_matches_dense_fixed_and_active_contacts(dtype):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_toolkit.optim.jax_optim import dynamic_vdw_energy, make_minimizer

    spec = _typed_optimizer_spec()
    coords = jnp.array(
        [[0.0, 0, 0], [1.5, 0, 0], [0, 2.0, 0]], dtype=getattr(jnp, dtype)
    )
    result = jax.jit(make_minimizer(spec, max_iter=50))(coords, 0.0, 0)
    distances = np.linalg.norm(np.asarray(result[0] - result[1:]), axis=-1)
    expected = np.square(np.minimum(distances - [2.02, 3.14], 0) / 0.2).sum()
    assert dynamic_vdw_energy(spec, result) == pytest.approx(expected, abs=2e-6)
    np.testing.assert_array_equal(result[1], coords[1])
