"""Geometry controls for E/Z preservation, including inconsistent substituents."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from test_workflow import write_cif

from common import coordinates
from metrics import conformer_metrics, ligand_names
from pose import coordinate_molecule
from stereo import assess_ez


def reference():
    mol = Chem.MolFromSmiles("F/C(Cl)=C(/Br)I")
    coords = np.array(
        [(-1, 1, 0), (0, 0, 0), (-1, -1, 0), (1.3, 0, 0), (2.3, -1, 0), (2.3, 1, 0)],
        dtype=float,
    )
    return mol, coords


def reverse_bonds(mol):
    edit = Chem.RWMol()
    for atom in mol.GetAtoms():
        edit.AddAtom(Chem.Atom(atom))
    for bond in reversed(list(mol.GetBonds())):
        edit.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    result = edit.GetMol()
    for bond in mol.GetBonds():
        other = result.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if len(bond.GetStereoAtoms()) == 2:
            other.SetStereoAtoms(*bond.GetStereoAtoms())
        other.SetStereo(bond.GetStereo())
    Chem.SanitizeMol(result)
    return result


def test_all_substituents_distinguish_matching_inverted_and_undefined():
    mol, xyz = reference()
    assert assess_ez(mol, xyz, [(1, 3)]) == {(1, 3): True}
    xyz[4:, 1] *= -1
    assert assess_ez(mol, xyz, [(1, 3)]) == {(1, 3): False}
    xyz[2, 1] *= -1
    assert assess_ez(mol, xyz, [(1, 3)]) == {(1, 3): None}


@pytest.mark.parametrize("state", ["match", "opposite", "ambiguous"])
def test_atom_and_bond_order_and_rigid_transform_invariance(state):
    mol, xyz = reference()
    if state == "opposite":
        xyz[4:, 1] *= -1
    elif state == "ambiguous":
        xyz[2, 1] *= -1
    expected = {"match": True, "opposite": False, "ambiguous": None}[state]
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    xyz = xyz @ rotation + [11, -7, 3]
    rng = np.random.default_rng(71)
    for graph in (mol, reverse_bonds(mol)):
        for _ in range(8):
            order = rng.permutation(mol.GetNumAtoms()).tolist()
            reordered = Chem.RenumberAtoms(graph, order)
            pair = (order.index(1), order.index(3))
            assert assess_ez(reordered, xyz[order], [pair])[pair] is expected


@pytest.mark.parametrize(
    "failure", ["axis", "substituent", "linear", "right_angle", "nan"]
)
def test_degenerate_geometry_is_undefined(failure):
    mol, xyz = reference()
    if failure == "axis":
        xyz[3] = xyz[1]
    elif failure == "substituent":
        xyz[0] = xyz[1]
    elif failure == "linear":
        xyz[0] = [-1, 0, 0]
    elif failure == "right_angle":
        xyz[0] = [-1, 0, 1]
    else:
        xyz[0, 0] = np.nan
    assert assess_ez(mol, xyz, [(1, 3)]) == {(1, 3): None}


@pytest.mark.parametrize("smiles", ["F/C=C/F", r"F/C=C\F", "C/C=N/OC", "C/N=N/C"])
def test_embedded_alkene_oxime_ether_and_azo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    expanded = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(expanded, randomSeed=37) == 0
    xyz = Chem.RemoveHs(expanded).GetConformer().GetPositions()
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
    ]
    assert bonds
    assert all(value is True for value in assess_ez(mol, xyz, bonds).values())
    assert assess_ez(mol, xyz * [-1, 1, 1], bonds) == assess_ez(mol, xyz, bonds)


def test_wrong_coordinate_shape_rejected():
    mol, xyz = reference()
    with pytest.raises(ValueError, match="atom order"):
        assess_ez(mol, xyz[:-1], [(1, 3)])


def test_undefined_geometry_fails_the_public_metric_and_remains_scored(tmp_path):
    mol, xyz = reference()
    folder = tmp_path / "jobs/case"
    folder.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    with Chem.SDWriter(str(data / "toy.sdf")) as writer:
        writer.write(coordinate_molecule(mol, xyz))
    xyz[2, 1] *= -1
    names = ligand_names(mol, "boltz_restr")
    path = folder / "prediction.cif"
    write_cif(
        path,
        [
            (names[i], atom.GetSymbol(), "B", 1, "LIG", xyz[i])
            for i, atom in enumerate(mol.GetAtoms())
        ],
    )
    atoms, coords = coordinates(path)
    result = conformer_metrics(
        atoms,
        coords,
        {"smiles": Chem.MolToSmiles(mol), "model": "boltz_restr", "system": "toy"},
        folder,
    )
    assert result["ez_sites"] == 1
    assert result["ez_undefined_sites"] == 1
    assert result["ez_match"] is False
