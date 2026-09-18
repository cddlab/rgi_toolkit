"""RF3 atom order, token ordinals, reference geometry and chemical graph mapping."""

import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem

from rgi_toolkit.rf3.adapter import RF3Adapter
from tests.test_adapters_shared import _FakeAtomArray


def atom_array():
    return _FakeAtomArray(
        element=["N", "C", "C", "N", "C", "C", "O"],
        coord=np.zeros((7, 3)),
        bonds=[[5, 6, 1]],
        annots=["conformer_restraints"],
        chain_id=["A", "A", "A", "B", "B", "L", "L"],
        res_id=[100, 100, 900, 17, 17, 4, 4],
        atom_name=["N", "CA", "CA", "N", "CA", "C1", "O1"],
        res_name=["GLY", "GLY", "ALA", "MSE", "MSE", "LIG", "LIG"],
        hetero=[False, False, False, True, True, True, True],
        conformer_restraints=[False, False, False, True, True, True, True],
    )


def adapter(**overrides):
    values = dict(
        atom_to_token_map=[7, 7, 15, 31, 31, 50, 51],
        ref_pos=[
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 3],
            [1.4, 0, 3],
        ],
        ref_space_uid=[0, 0, 1, 2, 2, 3, 3],
        mol_types=["protein"] * 5 + ["ligand"] * 2,
    )
    values.update(overrides)
    return RF3Adapter(atom_array(), **values)


def test_processed_rows_and_per_chain_tokens_ignore_author_numbers():
    records = list(adapter().iter_atoms())
    assert [record.index for record in records] == list(range(7))
    assert [record.resid for record in records] == [1, 1, 2, 1, 1, 1, 2]
    assert records[3].resname == "MSE"
    assert records[3].mol_type == "protein"
    assert records[3].conformer_restraints
    assert not records[0].conformer_restraints


def test_ligand_uses_reference_not_zeroed_structure_and_not_hetero():
    ad = adapter()
    (ligand,) = ad.iter_ligand_confs()
    assert ligand.global_indices.tolist() == [5, 6]
    assert ligand.conformer_restraints
    np.testing.assert_allclose(
        ligand.conf_coords[1] - ligand.conf_coords[0], [1.4, 0, 0]
    )
    assert ad.get_elements().tolist() == [7, 6, 6, 7, 6, 6, 8]
    assert ad.get_reference_space_uid().tolist() == [0, 0, 1, 2, 2, 3, 3]


def test_default_conformer_opt_out():
    aa = atom_array()
    aa._annots = []
    ad = adapter()
    ad.atom_array = aa
    assert not any(record.conformer_restraints for record in ad.iter_atoms())
    assert not next(ad.iter_ligand_confs()).conformer_restraints


def test_source_atom_names_reorder_and_remove_leaving_atoms():
    source = Chem.MolFromSmiles("OCC")
    for atom, name in zip(source.GetAtoms(), ["O1", "C1", "C2"]):
        atom.SetProp("atom_name", name)
    (ligand,) = adapter(ligand_mols={"L": source}).iter_ligand_confs()
    assert ligand.stereo_mol.GetNumAtoms() == 2
    assert [a.GetSymbol() for a in ligand.stereo_mol.GetAtoms()] == ["C", "O"]
    assert source.GetNumAtoms() == 3


def test_incompatible_source_graph_fails_loudly():
    with pytest.raises(ValueError, match="cannot map source ligand chemistry"):
        list(adapter(ligand_mols={"L": Chem.MolFromSmiles("CC")}).iter_ligand_confs())


@pytest.mark.parametrize(
    "field,value",
    [
        ("atom_to_token_map", [0]),
        ("ref_space_uid", [0]),
        ("ref_pos", np.zeros((8, 3))),
        ("ref_pos", np.full((7, 3), np.nan)),
        ("mol_types", ["invalid"] * 7),
    ],
)
def test_misaligned_or_invalid_features_are_rejected(field, value):
    with pytest.raises(ValueError, match="RF3"):
        adapter(**{field: value})


def test_import_is_framework_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rgi_toolkit.rf3.adapter; "
            "assert not {'torch', 'jax', 'rf3', 'foundry', 'atomworks', 'biotite'} & sys.modules.keys()",
        ],
        check=True,
    )
