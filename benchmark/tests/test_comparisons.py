import json
from collections import Counter

import pytest
import yaml
from test_workflow import arguments

from arms import uses_rgi
from common import read_json, verify_inputs
from inputs import write_input
from models import command
from prepare import prepare
from sampling import subsample_msa


def test_default_protein_campaign_has_three_matched_45_sample_arms(tmp_path):
    from common import protocol

    sequence = protocol("distance")["sequence"]
    msa = tmp_path / "query.a3m"
    msa.write_text(
        f">query\n{sequence}\n"
        + "".join(f">homolog-{i}\n{sequence}\n" for i in range(30))
    )
    args = arguments(
        tmp_path,
        seeds=None,
        limit=None,
        models=["boltz_restr", "openfold-3_restr"],
        msa=msa,
        single_sequence=False,
    )
    prepare(args)
    jobs = [read_json(args.output / p) for p in verify_inputs(args.output)["jobs"]]
    assert Counter((j["model"], j["arm"]) for j in jobs) == {
        (model, arm): 45 for model in args.models for arm in ("vanilla", "ms", "rgi")
    }
    for model in args.models:
        rgi = [j for j in jobs if j["model"] == model and j["arm"] == "rgi"]
        assert set(Counter(j["target"] for j in rgi).values()) == {9}
        baseline = [j for j in jobs if j["model"] == model and j["arm"] == "vanilla"]
        assert {j["seed"] for j in baseline} == set(range(45))
    ms = [j for j in jobs if j["arm"] == "ms"]
    assert {j["msa_depth"] for j in ms} == {16, 32, 64, 128, 256}
    assert {j["msa_rows"] for j in ms} == {17, 31}
    assert all(j["target"] is None and j["restraints"] is None for j in ms)
    assert len({j["msa"] for j in ms}) == 15


def test_subsampling_preserves_wrapped_query_and_is_reproducible(tmp_path):
    source = tmp_path / "input.a3m"
    original = "# metadata\n>query\nAC\nDE\n" + "".join(
        f">row{i}\nAcD-\nEF\n" for i in range(12)
    )
    source.write_text(original)
    first, second = tmp_path / "first.a3m", tmp_path / "second.a3m"
    assert subsample_msa(source, first, 4, 123) == 5
    assert subsample_msa(source, second, 4, 123) == 5
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text().startswith(">query\nACDE\n")
    assert first.read_text().count(">row") == 4
    assert source.read_text() == original
    subsample_msa(source, second, 100, 0)
    assert second.read_text().count(">") == 13
    subsample_msa(source, second, 4, 321)
    assert first.read_bytes() != second.read_bytes()


@pytest.mark.parametrize("model", ["boltz_restr", "protenix_restr"])
@pytest.mark.parametrize("arm", ["off", "g-only", "rgi", "rgi+g"])
def test_guidance_and_rgi_are_independent_controls(tmp_path, model, arm):
    restraints = (
        {"conformer_restraints_config": {"bond": {"weight": 1}}}
        if uses_rgi(arm)
        else None
    )
    job = {
        "model": model,
        "arm": arm,
        "seed": 1,
        "restraints": restraints,
        "sequence": "AG",
        "smiles": "CCCC",
        "msa": "input.a3m",
    }
    write_input(tmp_path, job)
    cmd = command(job, {"script": "inference.py"})
    guided = arm in {"g-only", "rgi+g"}
    if model == "boltz_restr":
        assert ("--use_potentials" in cmd) is guided
        parsed = yaml.safe_load((tmp_path / "input.yaml").read_text())
        opt_in = parsed["sequences"][1]["ligand"]["conformer_restraints"]
        assert cmd[cmd.index("--model") + 1] == "boltz2"
    else:
        assert (
            cmd[cmd.index("--sample_diffusion.guidance.enable") + 1]
            == str(guided).lower()
        )
        parsed = json.loads((tmp_path / "input.json").read_text())[0]
        opt_in = parsed["sequences"][1]["ligand"]["conformer_restraints"]
    assert opt_in is uses_rgi(arm)
    assert ("restraints_config" in parsed) is uses_rgi(arm)


def test_invalid_guidance_and_single_sequence_subsampling_are_rejected(tmp_path):
    args = arguments(tmp_path, arms=["ms"])
    with pytest.raises(ValueError, match="MSA subsampling"):
        prepare(args)
    with pytest.raises(ValueError, match="unsupported"):
        command({"model": "openfold-3_restr", "arm": "g-only", "seed": 0}, {})
    with pytest.raises(ValueError, match="--arms"):
        command(
            {"model": "boltz_restr", "arm": "vanilla", "seed": 0},
            {"args": ["--use_potentials"]},
        )
