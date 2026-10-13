import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from test_workflow import write_cif

from clashes import intermolecular, intramolecular
from common import write_json
from confidence import plddt, ptm
from evaluate import evaluate, summarize
from manuscript import export, source_tables
from quality import molprobity, parse_molprobity, parse_usalign, settings, tm_scores

MP_OUTPUT = """
Molprobity validation
  Ramachandran outliers = 1.38 %
                favored = 88.30 %
       Rotamer outliers = 18.78 %
      C-beta deviations = 5
             Clashscore = 18.83
             RMS(bonds) = 0.0161
            RMS(angles) = 1.86
        MolProbity score = 3.33
"""


def test_evaluation_paths_resolve_before_entering_temporary_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PHENIX_ENV", "tools/phenix env.sh")
    assert settings()["phenix_env"] == str(tmp_path / "tools/phenix env.sh")
    config = tmp_path / "local/evaluation.yaml"
    config.parent.mkdir()
    config.write_text(
        "usalign: ../tools/USalign\nphenix_env: ../phenix/env.sh\ntimeout: 15\n"
    )
    values = settings(config)
    assert values["usalign"] == str(tmp_path / "tools/USalign")
    assert values["phenix_env"] == str(tmp_path / "phenix/env.sh")
    config.write_text("timeout: .nan\n")
    with pytest.raises(ValueError, match="finite"):
        settings(config)


def test_external_output_parsers_preserve_normalization_and_require_all_fields():
    text = "TM-score= 0.8 (normalized by length of Structure_1: L=10)\nTM-score= 0.4 (normalized by length of Structure_2: L=20)"
    assert parse_usalign(text) == 0.8
    with pytest.raises(ValueError, match="reference-normalized"):
        parse_usalign("TM-score= 0.4 (normalized by length of Structure_2: L=20)")
    result = parse_molprobity(MP_OUTPUT)
    assert len(result) == 8
    assert result["rama_favored"] == 88.3
    assert result["rama_outliers"] == 1.38
    assert result["rms_bond"] == 0.0161
    with pytest.raises(ValueError, match="Missing MolProbity"):
        parse_molprobity("Clashscore = 0")


def test_external_commands_bound_time_use_reference_first_and_isolate_phenix(
    tmp_path, monkeypatch
):
    reference = tmp_path / "reference.cif"
    write_cif(reference, [("CA", "C", "A", 1, "ALA", (0, 0, 0))])
    prediction = tmp_path / "prediction.cif"
    prediction.write_bytes(reference.read_bytes())
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["timeout"] == 7
        if command[0] == "USalign":
            assert command[1:3] == [str(reference), str(prediction)]
            assert command[-4:] == ["-chain1", "A", "-chain2", "A"]
            return SimpleNamespace(
                stdout="TM-score= 1.0 (normalized by length of Structure_1: L=1)",
                returncode=0,
            )
        directory = Path(kwargs["cwd"])
        assert directory != tmp_path
        assert (directory / "prediction.pdb").is_file()
        assert command[:2] == ["bash", "-c"]
        assert command[4] == str(tmp_path / "phenix env.sh")
        if not command[-1].startswith("pdb_interpretation."):
            return SimpleNamespace(
                returncode=1,
                stderr="Bond distance > max_reasonable_bond_distance: 80 > 50",
            )
        assert command[-1] == "pdb_interpretation.max_reasonable_bond_distance=80.8"
        (directory / "molprobity.out").write_text(MP_OUTPUT)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("quality.subprocess.run", execute)
    config = settings()
    config.update(timeout=7, phenix_env=str(tmp_path / "phenix env.sh"))
    result, _ = tm_scores(
        prediction, [{"file": reference}, {"file": reference}], config
    )
    assert result == {"tmscore_1": 1, "tmscore_2": 1}
    result, _, retry = molprobity(prediction, config)
    assert result["molprobity_score"] == 3.33
    assert retry == 80.8
    assert not Path(calls[-1][1]["cwd"]).exists()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("quality.subprocess.run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        tm_scores(prediction, [{"file": reference}, {"file": reference}], config)


@pytest.mark.parametrize(
    "model,raw_path,sidecar",
    [
        (
            "boltz_restr",
            "predictions/toy/toy_model_0.cif",
            "confidence_toy_model_0.json",
        ),
        (
            "protenix_restr",
            "toy/seed_0/predictions/toy_sample_0.cif",
            "toy_summary_confidence_sample_0.json",
        ),
        (
            "alphafold3_restr",
            "toy/seed-0_sample-0/model.cif",
            "summary_confidences.json",
        ),
        ("chai-lab_restr", "pred.model_idx_0.cif", "scores.model_idx_0.npz"),
        ("openfold-3_restr", "seed_0/toy_model.cif", "toy_confidences_aggregated.json"),
        ("esm_restr", "model.cif", "model.confidence.json"),
    ],
)
def test_confidence_requires_the_exact_paired_prediction(
    tmp_path, model, raw_path, sidecar
):
    raw = tmp_path / "raw" / raw_path
    raw.parent.mkdir(parents=True)
    raw.write_text("same coordinates")
    prediction = tmp_path / "prediction.cif"
    prediction.write_bytes(raw.read_bytes())
    confidence = raw.parent / sidecar
    if model == "chai-lab_restr":
        np.savez(confidence, ptm=np.array([0.75]))
    else:
        confidence.write_text(json.dumps({"ptm": 0.75}))
    assert ptm(tmp_path, model) == {"ptm": 0.75}
    prediction.write_text("different coordinates")
    with pytest.raises(ValueError, match="differs"):
        ptm(tmp_path, model)
    prediction.write_bytes(raw.read_bytes())
    confidence.unlink()
    with pytest.raises(ValueError, match="Missing paired"):
        ptm(tmp_path, model)


def test_plddt_is_an_atom_mean_and_missing_values_are_errors(tmp_path):
    path = tmp_path / "confidence.cif"
    path.write_text("data_conf\nloop_\n_atom_site.B_iso_or_equiv\n80\n90\n100\n")
    assert plddt(path) == {"plddt": 90}
    path.write_text("data_conf\nloop_\n_atom_site.B_iso_or_equiv\n?\n")
    with pytest.raises(ValueError, match="missing"):
        plddt(path)


def ligand(terminal):
    mol = Chem.MolFromSmiles("CCCC")
    conformer = Chem.Conformer(4)
    for i, xyz in enumerate([(0, 0, 0), (1.5, 0, 0), (3, 0.7, 0), terminal]):
        conformer.SetAtomPosition(i, xyz)
    mol.AddConformer(conformer)
    return mol


def test_posebusters_real_internal_and_protein_clash_thresholds(tmp_path):
    extended = ligand((4.5, 0.8, 0))
    compact = ligand((0.5, 0, 0))
    good = intramolecular(extended)
    bad = intramolecular(compact)
    assert good["iclash_ok"] is True
    assert bad["iclash_ok"] is False
    assert bad["iclash_vdw_rel"] == pytest.approx(0.5 / 3.4)
    assert bad["iclash_rel"] < 0.7
    path = tmp_path / "complex.cif"
    write_cif(path, [("CA", "C", "A", 1, "ALA", (-1, 0, 0))])
    clash = intermolecular(extended, path)
    assert clash["clash_ok"] is False
    assert clash["clash_min"] == pytest.approx(1)
    assert clash["clash_rel"] == pytest.approx(1 / 3.4)
    write_cif(path, [("CA", "C", "A", 1, "ALA", (0, 0, 4))])
    assert intermolecular(extended, path)["clash_ok"] is True
    small = Chem.MolFromSmiles("CC")
    assert all(v is None for v in intramolecular(small).values())


def test_summary_keeps_independent_metrics_and_applicable_pass_denominators():
    base = {"model": "boltz_restr", "arm": "rgi", "target": "null"}
    rows = [
        {
            **base,
            "status": "ok",
            "prediction_status": "ok",
            "clash_ok": True,
            "ptm": 0.8,
        },
        {
            **base,
            "status": "partial",
            "prediction_status": "ok",
            "clash_ok": False,
            "ptm": None,
        },
        {**base, "status": "error", "clash_ok": None},
    ]
    summary = summarize(rows, ["clash_ok", "ptm"])[0]
    assert summary["expected"] == 3
    assert summary["prediction_success_rate"] == pytest.approx(2 / 3)
    assert summary["clash_ok_pass_rate"] == 0.5
    assert summary["clash_ok_n"] == 2
    assert summary["ptm_n"] == 1
    assert summary["ptm_median"] == 0.8


def test_partial_evaluation_generates_figures_and_portable_manuscript_tables(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    folder = root / "jobs/toy"
    job = {
        "id": "toy",
        "benchmark": "distance",
        "model": "openfold-3_restr",
        "system": "toy",
        "arm": "rgi",
        "seed": 0,
        "target": 2,
        "sequence": "A",
    }
    write_json(folder / "job.json", job)
    write_json(folder / "complete.json", {"seconds": 1})
    write_json(
        root / "manifest.json",
        {
            "benchmark": "distance",
            "jobs": ["jobs/toy/job.json"],
            "files": ["jobs/toy/job.json"],
        },
    )
    (root / "protocol.yaml").write_text(
        "kind: distance\ntm_references:\n- {file: first.cif, label: open}\n- {file: second.cif, label: closed}\n"
    )
    write_cif(folder / "prediction.cif", [("CA", "C", "A", 1, "ALA", (0, 0, 0))])
    monkeypatch.setattr(
        "evaluate.state_metrics", lambda *args: {"distance_A": 2, "absolute_error": 0}
    )
    monkeypatch.setattr("evaluate.plddt", lambda *args: {"plddt": 95})
    monkeypatch.setattr("evaluate.ptm", lambda *args: {"ptm": 0.8})
    monkeypatch.setattr(
        "evaluate.tm_scores",
        lambda *args: ({"tmscore_1": 0.8, "tmscore_2": 0.9}, "TM log"),
    )

    def fail(*args):
        raise RuntimeError("MolProbity unavailable")

    monkeypatch.setattr("evaluate.molprobity", fail)
    assert evaluate(root) == 1
    from report import read_rows

    row = read_rows(root / "results.csv")[0]
    assert row["status"] == "partial"
    assert row["ptm"] == "0.8" and row["molprobity_score"] == ""
    assert "MolProbity unavailable" in row["molprobity_error"]
    assert (root / "report/figures/openfold-3_restr-states.pdf").is_file()
    assert (root / "report/figures/target-vs-achieved.png").is_file()
    output = export(root, tmp_path / "manuscript")
    converted = read_rows(output / "source_data/distance/openfold-3_restr_rgi.csv")[0]
    assert converted["method"] == "openfold-3_restr"
    assert converted["target_distance"] == "2"
    assert converted["tmscore_2"] == "0.9"
    assert str(tmp_path) not in (output / "manifest.json").read_text()
    with pytest.raises(ValueError, match="empty"):
        export(root, output)


def test_conformer_manuscript_boolean_conversion_keeps_missing_values():
    rows = [
        {
            "model": "boltz_restr",
            "arm": "rgi+g",
            "system": "toy",
            "seed": "0",
            "status": "ok",
            "chiral_match": "True",
            "ez_match": "",
            "clash_ok": "False",
        }
    ]
    row = source_tables(rows, "conformer")["boltz_restr_rgi+g.csv"][0]
    assert row["match"] == 1
    assert row["ez_match"] is None
    assert row["clash_ok"] == 0
