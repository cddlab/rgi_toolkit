"""Pocket fitting must retain ligand displacement and respect molecular symmetry."""

import hashlib
import io
from zipfile import ZipFile

import numpy as np
import pytest
from rdkit import Chem
from test_workflow import write_cif

from common import read_json
from pose import (
    ProteinCA,
    check_reference_frame,
    coordinate_molecule,
    ligand_pose_metrics,
    pose_rmsd,
    protein_ca,
    protein_fit,
)
from pose_inputs import archive_members, pose_inputs

PROTEIN = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [30, 30, 30.0]])
LIGAND = np.array([[1, 1, 1], [2, 1, 1], [2, 2, 1.0]])


def protein(coords=PROTEIN, sequence="ACDEF"):
    return ProteinCA(sequence, np.asarray(coords))


def test_joint_rotation_and_translation_removed_without_fitting_remote_domain():
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
    translation = np.array([12, -5, 4.0])
    moving = PROTEIN @ rotation + translation
    moving[-1] += [30, 10, -20]
    fit = protein_fit(protein(), protein(moving), LIGAND)
    assert fit[3:] == (5, 4)
    assert fit[2] < 1e-12
    assert (
        pose_rmsd(
            Chem.MolFromSmiles("CCO"), LIGAND, LIGAND @ rotation + translation, *fit[:2]
        )
        < 1e-12
    )


def test_ligand_translation_is_retained_in_pose_rmsd():
    fit = protein_fit(protein(), protein(), LIGAND)
    value = pose_rmsd(Chem.MolFromSmiles("CCO"), LIGAND, LIGAND + [2, -1, 0], *fit[:2])
    assert value == pytest.approx(np.sqrt(5), abs=1e-12)


def test_sequence_pairing_handles_missing_internal_residue():
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 2
    coords = np.random.default_rng(19).normal(size=(len(sequence), 3)) + 30
    coords[:4] = PROTEIN[:4]
    keep = [i for i in range(len(sequence)) if i != 20]
    fit = protein_fit(
        protein(coords, sequence),
        protein(coords[keep], "".join(sequence[i] for i in keep)),
        LIGAND,
    )
    assert fit[3:] == (39, 4)
    assert fit[2] < 1e-12


def test_equivalent_terminal_oxygen_atoms_are_permuted_without_ligand_fit():
    graph = Chem.MolFromSmiles("CC(=O)[O-]")
    crystal = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0.0]])
    swapped = crystal[[0, 1, 3, 2]]
    assert pose_rmsd(graph, crystal, swapped, np.eye(3), np.zeros(3)) < 1e-12
    assert pose_rmsd(graph, crystal, swapped + [0, 0, 2], np.eye(3), np.zeros(3)) == 2


def test_insufficient_or_collinear_pocket_is_rejected():
    with pytest.raises(ValueError, match="non-collinear"):
        protein_fit(protein(), protein(), LIGAND + 100)
    with pytest.raises(ValueError, match="non-collinear"):
        protein_fit(protein(), protein(np.zeros_like(PROTEIN)), LIGAND)


def write_sdf(path, graph, coords):
    with Chem.SDWriter(str(path)) as writer:
        writer.write(coordinate_molecule(graph, coords))


def test_mismatched_ligand_and_receptor_coordinate_frames_rejected(tmp_path):
    graph = Chem.MolFromSmiles("CCO")
    crystal, native = tmp_path / "crystal.sdf", tmp_path / "native.sdf"
    write_sdf(crystal, graph, LIGAND)
    write_sdf(native, graph, LIGAND)
    assert check_reference_frame(graph, crystal, native) < 1e-12
    write_sdf(native, graph, LIGAND + [1, 0, 0])
    with pytest.raises(ValueError, match="different frames"):
        check_reference_frame(graph, crystal, native)


def test_complete_pose_metric_and_reference_hash_guard(tmp_path):
    folder = tmp_path / "jobs/case"
    folder.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    graph = Chem.MolFromSmiles("CCO")
    write_sdf(data / "toy.sdf", graph, LIGAND)
    atoms = [
        ("CA", "C", "A", i + 1, name, xyz)
        for i, (name, xyz) in enumerate(
            zip(["ALA", "CYS", "ASP", "GLU", "PHE"], PROTEIN, strict=True)
        )
    ]
    write_cif(data / "receptor.cif", atoms)
    write_cif(folder / "prediction.cif", atoms)
    job = {
        "system": "toy",
        "receptor": "../../data/receptor.cif",
        "pose_reference_sha256": {
            "../../data/receptor.cif": hashlib.sha256(
                (data / "receptor.cif").read_bytes()
            ).hexdigest()
        },
    }
    assert protein_ca(data / "receptor.cif").sequence == "ACDEF"
    result = ligand_pose_metrics(
        coordinate_molecule(graph, LIGAND + [2, -1, 0]), job, folder
    )
    assert result["ligand_rmsd_A"] == pytest.approx(np.sqrt(5))
    assert result["pocket_ca_atoms"] == 4
    (data / "receptor.cif").write_text("changed")
    with pytest.raises(ValueError, match="changed after preparation"):
        ligand_pose_metrics(coordinate_molecule(graph, LIGAND), job, folder)


def test_archive_fetch_preserves_exact_receptor_frame_and_detects_cache_changes(
    tmp_path, monkeypatch
):
    import fsspec

    sid = "1abc__1__1.A__1.B"
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr(f"{sid}/receptor.pdb", b"receptor")
        archive.writestr(f"{sid}/ligand_files/1.B.sdf", b"native ligand")
    data = payload.getvalue()
    calls = []

    def remote(url, mode, **kwargs):
        calls.append(url)
        assert url.endswith("/2024-06/v2/systems/ab.zip")
        assert kwargs["block_size"] == 65536
        return io.BytesIO(data)

    monkeypatch.setattr(fsspec, "open", remote)
    paths = pose_inputs(tmp_path, sid)
    assert (tmp_path / paths["receptor"]).read_bytes() == b"receptor"
    assert (tmp_path / paths["reference_ligand"]).read_bytes() == b"native ligand"
    assert read_json(tmp_path / "pose_sources" / f"{sid}.json")["sha256"]
    assert pose_inputs(tmp_path, sid) == paths and len(calls) == 1
    (tmp_path / paths["receptor"]).write_bytes(b"modified")
    assert pose_inputs(tmp_path, sid) == paths and len(calls) == 2
    assert (tmp_path / paths["receptor"]).read_bytes() == b"receptor"


def test_archive_range_failure_is_retried_without_caching_partial_files(
    tmp_path, monkeypatch
):
    import fsspec

    calls, delays = [], []

    def unavailable(*args, **kwargs):
        calls.append(args[0])
        raise ValueError("The HTTP server doesn't appear to support range requests.")

    monkeypatch.setattr(fsspec, "open", unavailable)
    monkeypatch.setattr("pose_inputs.time.sleep", delays.append)
    with pytest.raises(ValueError, match="range requests"):
        pose_inputs(tmp_path, "1abc__1__1.A__1.B")
    assert len(calls) == 3 and delays == [1, 2]
    assert not list(tmp_path.iterdir())


def test_archive_recovers_after_a_transient_range_failure(monkeypatch):
    import fsspec

    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("member", b"recovered")
    attempts = []

    def remote(*args, **kwargs):
        attempts.append(None)
        if len(attempts) == 1:
            raise ValueError(
                "The HTTP server doesn't appear to support range requests."
            )
        return io.BytesIO(payload.getvalue())

    monkeypatch.setattr(fsspec, "open", remote)
    monkeypatch.setattr("pose_inputs.time.sleep", lambda _: None)
    assert archive_members("unused", {"data": "member"}) == {"data": b"recovered"}
    assert len(attempts) == 2
