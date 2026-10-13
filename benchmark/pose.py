"""Ligand pose RMSD after fitting sequence-paired pocket C-alpha atoms."""

import hashlib
from dataclasses import dataclass

import gemmi
import numpy as np
from Bio import Align
from Bio.PDB import Polypeptide, is_aa
from Bio.SVDSuperimposer import SVDSuperimposer
from rdkit import Chem
from rdkit.Chem import rdMolAlign

POSE_METHOD = "pocket-ca-10a-symmetry-v1"
POSE_SETTINGS = {
    "method": POSE_METHOD,
    "pocket_cutoff_A": 10.0,
    "sequence_alignment": {
        "mode": "global",
        "match": 1.0,
        "mismatch": 0.0,
        "open_gap": -10.0,
        "extend_gap": -0.5,
        "end_gap": 0.0,
    },
    "ligand_superposition": False,
    "symmetry_max_matches": 1000,
    "symmetrize_conjugated_terminal_groups": True,
}


@dataclass
class ProteinCA:
    sequence: str
    coordinates: np.ndarray


def protein_ca(path, chain=None):
    structure = gemmi.read_structure(str(path))
    if not len(structure) or not len(structure[0]):
        raise ValueError("No protein chains for the pocket fit")
    selected = structure[0][chain] if chain is not None else structure[0][0]
    sequence, coordinates = [], []
    for residue in selected:
        atoms = [atom for atom in residue if atom.name == "CA"]
        if not atoms:
            continue
        if not is_aa(residue.name):
            raise ValueError("Non-protein C-alpha atom in the receptor chain")
        atom = max(atoms, key=lambda item: item.occ)
        sequence.append(Polypeptide.protein_letters_3to1.get(residue.name, "X"))
        coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return ProteinCA("".join(sequence), np.asarray(coordinates, dtype=np.float32))


def protein_fit(reference, predicted, ligand):
    if not reference.sequence or not predicted.sequence:
        raise ValueError("No protein C-alpha atoms for the pocket fit")
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    indices = aligner.align(reference.sequence, predicted.sequence)[0].indices
    paired = indices[:, np.all(indices >= 0, axis=0)]
    fixed = np.asarray(reference.coordinates[paired[0]], dtype=float)
    mobile = np.asarray(predicted.coordinates[paired[1]], dtype=float)
    if not len(fixed):
        raise ValueError("No aligned protein C-alpha atoms")
    pocket = np.any(
        np.linalg.norm(fixed[None, :, :] - ligand[:, None, :], axis=2) < 10.0,
        axis=0,
    )
    if (
        pocket.sum() < 3
        or np.linalg.matrix_rank(fixed[pocket] - fixed[pocket].mean(0)) < 2
        or np.linalg.matrix_rank(mobile[pocket] - mobile[pocket].mean(0)) < 2
    ):
        raise ValueError("Fewer than three non-collinear paired pocket C-alpha atoms")
    fit = SVDSuperimposer()
    fit.set(fixed[pocket], mobile[pocket])
    fit.run()
    rotation, translation = fit.get_rotran()
    return rotation, translation, fit.get_rms(), len(fixed), int(pocket.sum())


def coordinate_molecule(graph, coordinates):
    mol = Chem.Mol(graph)
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(coordinates):
        conformer.SetAtomPosition(index, tuple(map(float, point)))
    mol.AddConformer(conformer)
    return mol


def pose_rmsd(graph, crystal, predicted, rotation, translation):
    fixed = coordinate_molecule(graph, crystal)
    mobile = coordinate_molecule(graph, predicted @ rotation + translation)
    return rdMolAlign.CalcRMS(
        fixed,
        mobile,
        maxMatches=1000,
        symmetrizeConjugatedTerminalGroups=True,
    )


def ligand_coordinates(path, graph):
    mol = next((m for m in Chem.SDMolSupplier(str(path), removeHs=True) if m), None)
    if mol is None:
        raise ValueError("Cannot read crystal ligand SDF")
    mol = Chem.RemoveAllHs(mol)
    match = mol.GetSubstructMatch(graph)
    if len(match) != graph.GetNumAtoms() or mol.GetNumAtoms() != graph.GetNumAtoms():
        raise ValueError("Crystal ligand graph differs from the input SMILES")
    return mol.GetConformer().GetPositions()[list(match)]


def check_reference_frame(graph, crystal_sdf, native_sdf):
    crystal = ligand_coordinates(crystal_sdf, graph)
    native = ligand_coordinates(native_sdf, graph)
    error = pose_rmsd(graph, crystal, native, np.eye(3), np.zeros(3))
    if not np.isfinite(error) or error > 0.002:
        raise ValueError(
            "Crystal ligand and reference receptor are in different frames"
        )
    return error


def ligand_pose_metrics(mol, job, folder):
    if not job.get("receptor"):
        raise ValueError("Missing receptor reference; prepare the run with pose inputs")
    receptor = folder / job["receptor"]
    for relative, expected in job.get("pose_reference_sha256", {}).items():
        if hashlib.sha256((folder / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Ligand pose reference changed after preparation")
    crystal_sdf = folder / "../../data" / f"{job['system']}.sdf"
    if job.get("reference_ligand"):
        check_reference_frame(mol, crystal_sdf, folder / job["reference_ligand"])
    crystal = ligand_coordinates(crystal_sdf, mol)
    fit = protein_fit(
        protein_ca(receptor), protein_ca(folder / "prediction.cif", "A"), crystal
    )
    value = pose_rmsd(mol, crystal, mol.GetConformer().GetPositions(), *fit[:2])
    return {
        "ligand_rmsd_A": value,
        "protein_fit_rmsd_A": fit[2],
        "paired_ca_atoms": fit[3],
        "pocket_ca_atoms": fit[4],
    }
