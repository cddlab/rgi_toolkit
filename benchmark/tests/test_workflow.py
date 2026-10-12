import argparse
import json
import lzma
import shutil
import sys

import numpy as np
import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

from arms import uses_rgi
from common import MODELS, coordinates, read_json, verify_inputs, write_json
from evaluate import evaluate
from fetch import conformer, place
from metrics import angle, benchmark_ez, conformer_metrics, ligand_names, rmsd, selected
from prepare import prepare
from run import check_restraints, run, runtime_for


def arguments(tmp_path, benchmark="distance", **overrides):
    values = dict(
        benchmark=benchmark,
        output=tmp_path / "run",
        models=list(MODELS),
        seeds=[0],
        msa=None,
        single_sequence=True,
        index=None,
        dataset=tmp_path / "dataset",
        references=tmp_path / "refs",
        limit=1,
    )
    values.update(overrides)
    values["references"].mkdir(exist_ok=True)
    for name in ("1GGG", "1WDN", "4AKE", "1AKE", "6E9O", "6E9N"):
        path = values["references"] / f"{name}.cif"
        if not path.exists():
            path.write_text("data_ref\n")
    return argparse.Namespace(**values)


def write_cif(path, atoms):
    fields = [
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_comp_id",
        "label_asym_id",
        "label_seq_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "label_alt_id",
        "occupancy",
        "B_iso_or_equiv",
        "auth_asym_id",
        "auth_seq_id",
        "auth_atom_id",
        "auth_comp_id",
        "pdbx_PDB_model_num",
        "pdbx_PDB_ins_code",
    ]
    lines = ["data_test", "loop_", *["_atom_site." + f for f in fields]]
    for i, (name, element, chain, resid, resname, xyz) in enumerate(atoms, 1):
        lines.append(
            f"ATOM {i} {element} {name} {resname} {chain} {resid} "
            + " ".join(str(v) for v in xyz)
            + f" . 1.0 90.0 {chain} {resid} {name} {resname} 1 ?"
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("benchmark", ["distance", "angle", "custom"])
def test_prepared_runs_are_relocatable(tmp_path, benchmark):
    args = arguments(tmp_path, benchmark)
    prepare(args)
    moved = tmp_path / "with spaces" / "moved"
    moved.parent.mkdir()
    shutil.move(args.output, moved)
    manifest = verify_inputs(moved)
    assert len(manifest["jobs"]) == 12
    for relative in manifest["jobs"]:
        path = moved / relative
        job = read_json(path)
        assert (path.parent / job["msa"]).is_file()
        assert bool(job["restraints"]) == uses_rgi(job["arm"])
        for file in path.parent.iterdir():
            assert str(tmp_path) not in file.read_text()
        if job["model"] == "openfold-3_restr":
            assert read_json(path.with_name("input.json"))["seeds"] == [job["seed"]]


def test_missing_inputs_and_overwrite_are_rejected(tmp_path):
    args = arguments(tmp_path)
    prepare(args)
    with pytest.raises(ValueError, match="empty"):
        prepare(args)
    path = next((args.output / "data").glob("*.a3m"))
    path.unlink()
    with pytest.raises(ValueError, match="Missing prepared input"):
        verify_inputs(args.output)


def test_rmsd_preserves_targets_and_relative_references(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ["1GGG.cif", "1WDN.cif"]:
        (refs / name).write_text("data_ref\n")
    args = arguments(tmp_path, "rmsd", models=["alphafold3_restr"], limit=None)
    prepare(args)
    manifest = verify_inputs(args.output)
    jobs = [read_json(args.output / path) for path in manifest["jobs"]]
    guided = [j for j in jobs if j["arm"] == "rgi"]
    assert guided[1]["target"] == [1.325, 3.975]
    for job in guided:
        assert "conformer_restraints_config" not in job["restraints"]
        assert [
            e["harmonic"]["target_rmsd"]
            for e in job["restraints"]["rmsd_restraints_config"]
        ] == job["target"]
        for entry in job["restraints"]["rmsd_restraints_config"]:
            assert (args.output / "jobs" / job["id"] / entry["ref_cif"]).is_file()


def test_auto_fetch_and_conformer_opt_in(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(root, limit):
        calls.append((root, limit))
        root.mkdir()
        (root / "input.a3m").write_text(">query\nAG\n")
        (root / "ligand.sdf").write_text("placeholder")
        (root / "receptor.pdb").write_text("placeholder")
        (root / "index.csv").write_text(
            "id,smiles,msa,sdf,receptor\ntoy,C/C=C/C,input.a3m,ligand.sdf,receptor.pdb\n"
        )

    monkeypatch.setattr("fetch.conformer", fake_fetch)
    args = arguments(tmp_path, "conformer", single_sequence=False)
    prepare(args)
    assert calls == [(args.dataset, 1)]
    for relative in verify_inputs(args.output)["jobs"]:
        path = args.output / relative
        job = read_json(path)
        assert (path.parent / job["receptor"]).is_file()
        assert job["pose_reference_sha256"]
        text = "\n".join(p.read_text() for p in path.parent.iterdir())
        if job["arm"] == "rgi":
            assert '"conformer_restraints_config"' in text
            if job["model"] == "chai-lab_restr":
                sidecar = yaml.safe_load(path.with_name("restraints.yaml").read_text())
                assert sidecar["conformer_restraints"] == {"B": True}
        else:
            assert "conformer_restraints_config" not in text


def test_truncated_payload_is_not_published(tmp_path):
    path = tmp_path / "payload"
    with pytest.raises(ValueError, match="size"):
        place(path, b"short", 10)
    assert not path.exists()


def test_fetch_refreshes_cached_files_of_the_same_size(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    (remote / "msa").mkdir(parents=True)
    (remote / "sdf").mkdir()
    (remote / "index.csv").write_text("sys_id,smiles,a3m_bytes,sdf_bytes\ntoy,CC,6,6\n")
    (remote / "sdf/toy.sdf").write_bytes(b"ligand")

    def download(repo, name, **kwargs):
        assert kwargs["revision"] == "main"
        return remote / name

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(
        "fetch.pose_inputs",
        lambda *_args: {
            "receptor": "receptor/toy.pdb",
            "reference_ligand": "reference_ligand/toy.sdf",
        },
    )
    cache = tmp_path / "dataset"
    for sequence in (b">q\nAG\n", b">q\nAA\n"):
        (remote / "msa/toy.a3m.xz").write_bytes(lzma.compress(sequence))
        conformer(cache, 1)
        assert (cache / "msa/toy.a3m").read_bytes() == sequence


def test_run_resumes_and_retries_invalid_predictions(tmp_path, monkeypatch):
    root = tmp_path / "run"
    folder = root / "jobs/toy"
    job = {
        "id": "toy",
        "model": "boltz_restr",
        "benchmark": "distance",
        "arm": "off",
        "sequence": "A",
    }
    write_json(folder / "job.json", job)
    write_json(
        root / "manifest.json",
        {"jobs": ["jobs/toy/job.json"], "files": ["jobs/toy/job.json"]},
    )
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({"boltz_restr": {"python": sys.executable}}))
    software = {"python": "test"}
    calls = []

    def execute(command, **kwargs):
        if command[-1] == "--probe":
            return argparse.Namespace(stdout=json.dumps(software))
        calls.append(command)
        prediction = folder / "raw/predictions/toy/toy_model_0.cif"
        prediction.parent.mkdir(parents=True)
        write_cif(prediction, [("CA", "C", "A", 1, "ALA", (0, 0, 0))])
        write_json(folder / "software.json", software)

    monkeypatch.setattr("run.subprocess.run", execute)
    args = argparse.Namespace(
        run=root, runtime=runtime, model=None, task=None, dry_run=False, retry=False
    )
    run(args)
    run(args)
    assert len(calls) == 1
    write_cif(
        folder / "prediction.cif",
        [("CA", "C", "A", 1, "ALA", (float("nan"), 0, 0))],
    )
    with pytest.raises(ValueError, match="non-finite"):
        run(args)
    args.retry = True
    run(args)
    assert len(calls) == 2


def test_finite_coordinates_and_ordinal_selection(tmp_path):
    path = tmp_path / "model.cif"
    write_cif(
        path,
        [
            ("CA", "C", "A", 51, "ALA", (0, 0, 0)),
            ("CA", "C", "A", 55, "GLY", (3, 0, 0)),
        ],
    )
    atoms, xyz = coordinates(path)
    result, _ = selected(atoms, xyz, "chain A and resid 2 and name CA")
    np.testing.assert_equal(result, [[3, 0, 0]])
    write_cif(path, [("CA", "C", "A", 1, "ALA", (float("nan"), 0, 0))])
    with pytest.raises(ValueError):
        coordinates(path)


def test_geometry_and_inactive_restraints():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert angle(xyz[1], xyz[0], xyz[2]) == pytest.approx(90)
    assert rmsd(xyz, xyz + [4, 5, 6]) == pytest.approx(0, abs=1e-12)
    job = {"arm": "rgi", "benchmark": "distance"}
    with pytest.raises(ValueError, match="distances"):
        check_restraints("built spec: distances=0\nfinalize distance=0", job)
    check_restraints("built spec: distances=1", job)


def test_ez_excludes_rings_and_exchangeable_hydrogens():
    from rgi_toolkit._mol_build import _graph_stereo_labels

    for smiles in ("C/C=N/O", "C1CCC/C=C/CCC1"):
        mol = Chem.MolFromSmiles(smiles)
        assert not benchmark_ez(mol, _graph_stereo_labels(mol)[1])
    mol = Chem.MolFromSmiles("C/C=C/C")
    assert len(benchmark_ez(mol, _graph_stereo_labels(mol)[1])) == 1


def test_alphafold3_runtime_paths_are_relative_to_config(tmp_path):
    from models import command

    (tmp_path / "python").touch()
    (tmp_path / "run_alphafold.py").touch()
    (tmp_path / "model weights").mkdir()
    runtime = runtime_for(
        {
            "alphafold3_restr": {
                "python": "python",
                "script": "run_alphafold.py",
                "model_dir": "model weights",
            }
        },
        "alphafold3_restr",
        tmp_path,
    )
    cmd = command({"model": "alphafold3_restr", "seed": 0, "arm": "vanilla"}, runtime)
    assert f"--model_dir={tmp_path / 'model weights'}" in cmd
    assert runtime["script"] == str(tmp_path / "run_alphafold.py")


@pytest.mark.parametrize("model", MODELS)
def test_conformer_mapping_handles_reordered_atoms(tmp_path, model):
    smiles = "C/C=C/[C@H](O)Cl"
    mol = Chem.MolFromSmiles(smiles)
    expanded = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(expanded, randomSeed=42) == 0
    mol3d = Chem.RemoveHs(expanded)
    folder = tmp_path / "jobs" / "case"
    folder.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    with Chem.SDWriter(str(data / "toy.sdf")) as writer:
        writer.write(mol3d)
    xyz = mol3d.GetConformer().GetPositions()
    names = ligand_names(mol, model)
    atoms = [
        (names[i], atom.GetSymbol(), "B", 1, "LIG", xyz[i])
        for i, atom in enumerate(mol.GetAtoms())
    ]
    path = folder / "prediction.cif"
    write_cif(path, atoms[::-1])
    records, coords = coordinates(path)
    result = conformer_metrics(
        records, coords, {"smiles": smiles, "model": model, "system": "toy"}, folder
    )
    assert result["ligand_fit_rmsd_A"] < 1e-3
    assert result["chiral_match"] is True
    assert result["ez_match"] is True


def test_evaluation_rejects_missing_samples_and_preserves_results(tmp_path):
    args = arguments(tmp_path, models=["boltz_restr"])
    prepare(args)
    manifest = verify_inputs(args.output)
    job_path = args.output / manifest["jobs"][0]
    job = read_json(job_path)
    import gemmi

    names = {
        gemmi.find_tabulated_residue(name).one_letter_code: name
        for name in [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]
    }
    prediction = job_path.with_name("prediction.cif")
    write_cif(
        prediction,
        [
            ("CA", "C", "A", i, names[aa], (i, 0, 0))
            for i, aa in enumerate(job["sequence"], 1)
        ],
    )
    write_json(
        job_path.with_name("complete.json"),
        {
            "seconds": 1.0,
        },
    )
    assert (
        evaluate(
            args.output, skip=("usalign", "molprobity", "confidence"), make_report=False
        )
        == 1
    )
    saved = (args.output / "results.csv").read_bytes()
    job_path.with_name("complete.json").unlink()
    with pytest.raises(ValueError, match="preserved"):
        evaluate(args.output)
    assert (args.output / "results.csv").read_bytes() == saved
