import numpy as np
import pytest
from rdkit import Chem
from test_workflow import write_cif

from common import coordinates
from metrics import conformer_metrics, ligand_names
from pose import coordinate_molecule


@pytest.mark.parametrize("same", [True, False])
def test_conflicting_substituents_use_rdkit_labels(tmp_path, same):
    smiles = "F/C(Cl)=C/Br" if same else r"F/C(Cl)=C\Br"
    mol = Chem.MolFromSmiles(smiles)
    xyz = np.array([[-1, 1, 0], [0, 0, 0], [-0.9, 0.8, 0.1], [1.3, 0, 0], [2.3, -1, 0]])
    folder = tmp_path / "jobs/case"
    folder.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    crystal_xyz = xyz.copy()
    crystal_xyz[2] = [-1, -1, 0]
    if not same:
        crystal_xyz[-1, 1:] *= -1
    with Chem.SDWriter(str(data / "toy.sdf")) as writer:
        writer.write(coordinate_molecule(mol, crystal_xyz))
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
        {"smiles": smiles, "model": "boltz_restr", "system": "toy"},
        folder,
    )
    assert result["ez_sites"] == 1
    assert result["ez_match"] is same
    assert "ez_undefined_sites" not in result
