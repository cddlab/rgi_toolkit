"""BioEmu2 atom mapping, padding and source-chemistry contracts."""

import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem

from rgi_toolkit.bioemu2 import BioEmu2Adapter


def _codes(names):
    return [[ord(c) - 32 for c in name.ljust(4)] for name in names]


def adapter(**overrides):
    source = Chem.MolFromSmiles("C[N+](C)(C)C")
    for atom in source.GetAtoms():
        atom.SetProp("atom_name", f"{atom.GetSymbol()}{atom.GetIdx()}")
    values = dict(
        chain_ids=["A", "B", "L"],
        atom_chain_indices=[0, 0, 0, 1, 1, 2, 2, 2, 2, 2, -1],
        atom_to_token_map=[4, 4, 9, 20, 20, 35, 36, 37, 38, 39, -1],
        atom_mask=[True] * 10 + [False],
        atom_name_codes=_codes(
            ["N", "CA", "CA", "N", "CA", "C4", "C3", "C2", "N1", "C0", ""]
        ),
        residue_names=["GLY", "GLY", "ALA", "MSE", "MSE"] + ["LIG"] * 5 + [None],
        mol_types=["protein"] * 5 + ["ligand"] * 5 + [None],
        elements=[7, 6, 6, 7, 6, 6, 6, 6, 7, 6, 99],
        ref_pos=np.arange(33).reshape(11, 3) * 0.1,
        ref_space_uid=[1, 1, 2, 3, 3, 4, 4, 4, 4, 4, -1],
        conformer_chains=["B", "L"],
        ligand_mols={"L": source},
    )
    values.update(overrides)
    return BioEmu2Adapter(**values)


def test_padded_atom_indices_and_per_chain_token_ordinals():
    ad = adapter()
    records = list(ad.iter_atoms())
    assert [record.index for record in records] == list(range(10))
    assert [record.resid for record in records] == [1, 1, 2, 1, 1, 1, 2, 3, 4, 5]
    assert records[3].mol_type == "protein" and records[3].resname == "MSE"
    assert records[3].conformer_restraints and not records[0].conformer_restraints
    assert ad.num_atoms() == 11
    assert ad.get_elements()[-1] == 0
    assert not ad.get_reference_positions()[-1].any()
    assert ad.get_reference_space_uid()[-1] == -1


def test_name_mapping_preserves_charge_and_does_not_mutate_source():
    ad = adapter()
    source = ad._ligand_mols["L"]
    before = Chem.MolToSmiles(source)
    (ligand,) = ad.iter_ligand_confs()
    assert ligand.global_indices.tolist() == [5, 6, 7, 8, 9]
    assert ligand.mol.GetAtomWithIdx(3).GetFormalCharge() == 1
    assert ligand.stereo_mol.GetAtomWithIdx(3).GetFormalCharge() == 1
    assert ligand.conformer_restraints
    assert Chem.MolToSmiles(source) == before
    np.testing.assert_equal(ligand.conf_coords, ad.get_reference_positions()[5:10])


def test_named_leaving_atoms_are_removed_without_renumbering_coordinates():
    source = Chem.MolFromSmiles("CCO")
    for atom, name in zip(source.GetAtoms(), ["C0", "C2", "O1"]):
        atom.SetProp("atom_name", name)
    ad = adapter(
        atom_mask=[True] * 7 + [False] * 4,
        atom_name_codes=_codes(
            ["N", "CA", "CA", "N", "CA", "C2", "C0", "", "", "", ""]
        ),
        ligand_mols={"L": source},
    )
    (ligand,) = ad.iter_ligand_confs()
    assert ligand.global_indices.tolist() == [5, 6]
    assert ligand.mol.GetNumAtoms() == 2
    assert ligand.mol.GetNumBonds() == 1
    assert source.GetNumAtoms() == 3


def test_no_conformer_flags_means_no_opt_in():
    ad = adapter(conformer_chains=[])
    assert not any(record.conformer_restraints for record in ad.iter_atoms())
    assert not next(ad.iter_ligand_confs()).conformer_restraints


def test_invalid_source_mapping_fails_instead_of_restraining_wrong_atoms():
    ad = adapter()
    ad._ligand_mols["L"].GetAtomWithIdx(0).SetProp("atom_name", "missing")
    with pytest.raises(ValueError, match="ambiguous ligand atom names"):
        list(ad.iter_ligand_confs())
    with pytest.raises(ValueError, match="missing source ligand"):
        list(adapter(ligand_mols={}).iter_ligand_confs())
    with pytest.raises(ValueError, match="element order"):
        list(adapter(elements=[7, 6, 6, 7, 6, 8, 6, 6, 7, 6, 0]).iter_ligand_confs())


@pytest.mark.parametrize(
    "overrides",
    [
        {"atom_to_token_map": [0]},
        {"atom_name_codes": np.zeros((11, 3))},
        {"ref_pos": np.full((11, 3), np.nan)},
        {"atom_chain_indices": [3] * 11},
        {"conformer_chains": ["unknown"]},
    ],
)
def test_inconsistent_metadata_is_rejected(overrides):
    with pytest.raises(ValueError, match="BioEmu2"):
        adapter(**overrides)


def test_import_does_not_load_predictor_or_compute_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rgi_toolkit.bioemu2; "
            "assert not {'torch', 'jax', 'bioemu2', 'biotite'} & sys.modules.keys()",
        ],
        check=True,
    )
