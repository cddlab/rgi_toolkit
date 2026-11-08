"""New predictor adapters preserve atom layout, chemistry and chain ordinals."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_toolkit.intellifold.adapter import IntelliFoldAdapter
from rgi_toolkit.mol1.adapter import Mol1Adapter


class AtomArray:
    def __init__(self, **annotations):
        self.__dict__.update(annotations)
        self.bonds = None

    def __len__(self):
        return len(self.element)

    def get_annotation_categories(self):
        return list(self.__dict__)


def test_mol1_uses_model_tokens_and_preserves_padding():
    aa = AtomArray(
        label_asym_id=np.array(["Z", "Z", "Z", "B", "B", "B"]),
        res_id=np.array([42, 42, 42, 100, 100, 101]),
        hetero=np.zeros(6, dtype=bool),
        atom_name=np.array(["CA", "CB", "X", "P", "C1'", "P"]),
        res_name=np.array(["ALA", "ALA", "MOD", "DA", "DA", "DT"]),
        mol_type=np.array(["protein"] * 3 + ["dna"] * 3),
        conformer_restraints=np.array([True] * 3 + [False] * 3),
        element=np.array(["C", "C", "N", "P", "C", "P"]),
    )
    ref = np.arange(24).reshape(8, 3)
    adapter = Mol1Adapter(
        {
            "atom_array": aa,
            "atom_to_token_idx": np.array([4, 4, 5, 6, 6, 7, 0, 0]),
            "ref_pos": ref,
            "ref_space_uid": np.arange(8),
        }
    )
    atoms = list(adapter.iter_atoms())
    assert [a.resid for a in atoms] == [1, 1, 2, 1, 1, 2]
    assert [a.index for a in atoms] == list(range(6))
    assert [a.conformer_restraints for a in atoms] == [True] * 3 + [False] * 3
    assert atoms[4].name == "C1'" and atoms[4].mol_type == "dna"
    assert adapter.num_atoms() == 8
    np.testing.assert_array_equal(adapter.get_elements(), [6, 6, 7, 15, 6, 15, 0, 0])
    np.testing.assert_array_equal(adapter.get_reference_positions(), ref)
    np.testing.assert_array_equal(adapter.get_reference_space_uid(), np.arange(8))


def test_mol1_rejects_missing_atom_metadata():
    with pytest.raises(ValueError, match="atom_array"):
        Mol1Adapter({"atom_to_token_idx": np.arange(2)})


def test_intellifold_flat_indices_source_stereo_and_opt_in():
    smol = Chem.AddHs(Chem.MolFromSmiles("F/C=C/Cl"))
    assert AllChem.EmbedMolecule(smol, randomSeed=11) == 0
    smol = Chem.RemoveHs(smol)
    coords = smol.GetConformer().GetPositions()
    mask = np.zeros((6, 3), dtype=bool)
    mask[:5, 0] = True
    names = np.zeros((6, 3, 4), dtype=int)
    for token, name in enumerate(["CA", "F1", "C1", "C2", "CL1"]):
        names[token, 0, : len(name)] = [ord(c) - 32 for c in name]
    ref = np.zeros((6, 3, 3))
    ref[1:5, 0] = coords
    batch = {
        "asym_id": np.array([1, 2, 2, 2, 2, 0]),
        "ref_mask": mask,
        "ref_pos": ref,
        "ref_element": np.array(
            [[6, 0, 0], [9, 0, 0], [6, 0, 0], [6, 0, 0], [17, 0, 0], [0, 0, 0]]
        ),
        "ref_atom_name_chars": names,
        "is_protein": np.array([True, False, False, False, False, False]),
        "is_dna": np.zeros(6, dtype=bool),
        "is_rna": np.zeros(6, dtype=bool),
        "aatype": np.zeros(6, dtype=int),
    }
    adapter = IntelliFoldAdapter(
        batch,
        {"P": 1, "L": 2},
        ("ALA",),
        [("L", smol, True)],
        conformer_restraints_by_asym={1: False, 2: True},
    )
    atoms = list(adapter.iter_atoms())
    assert [(a.chain, a.resid, a.index) for a in atoms] == [
        ("P", 1, 0),
        ("L", 1, 3),
        ("L", 2, 6),
        ("L", 3, 9),
        ("L", 4, 12),
    ]
    (conf,) = adapter.iter_ligand_confs()
    np.testing.assert_array_equal(conf.global_indices, [3, 6, 9, 12])
    np.testing.assert_allclose(conf.conf_coords, coords)
    assert conf.conformer_restraints
    assert Chem.MolToSmiles(conf.stereo_mol) == "F/C=C/Cl"
    assert adapter.num_atoms() == 18
    assert not list(
        IntelliFoldAdapter(batch, {"P": 1, "L": 2}, ("ALA",), []).iter_ligand_confs()
    )


def test_new_adapters_import_without_predictors_or_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rgi_toolkit.mol1.adapter; "
            "import rgi_toolkit.intellifold.adapter; "
            "assert not {'mol', 'intellifold', 'alphafold3', 'torch', 'jax'} & sys.modules.keys()",
        ],
        check=True,
    )
