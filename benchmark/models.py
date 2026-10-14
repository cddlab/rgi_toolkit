"""Run inside a predictor's existing Python environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from arms import GUIDANCE_MODELS, uses_guidance, uses_rgi


def _pop_option(args, names):
    remaining = []
    value = None
    iterator = iter(args)
    for arg in iterator:
        name, separator, inline = arg.partition("=")
        if name in names:
            value = inline if separator else next(iterator)
        else:
            remaining.append(arg)
    return remaining, value


def msa_recycle_command(cmd, model):
    if model == "openfold-3_restr":
        import yaml

        cmd, source = _pop_option(cmd, {"--runner-yaml", "--runner_yaml"})
        config = (yaml.safe_load(Path(source).read_text()) or {}) if source else {}
        config.setdefault("model_update", {}).setdefault("presets", ["predict"])
        shared = (
            config.setdefault("model_update", {})
            .setdefault("custom", {})
            .setdefault("architecture", {})
            .setdefault("shared", {})
        )
        shared["num_recycles"] = 1
        runner = Path("ms_runner.yaml")
        runner.write_text(yaml.safe_dump(config))
        return cmd + ["--runner-yaml", str(runner)]
    flag = {
        "boltz_restr": "--recycling_steps",
        "protenix_restr": "--model.N_cycle",
        "alphafold3_restr": "--num_recycles",
        "chai-lab_restr": "--num-trunk-recycles",
    }[model]
    cmd, _ = _pop_option(cmd, {flag})
    return cmd + [flag, "1"]


def provenance():
    versions = {"python": sys.version.split()[0]}
    for name in (
        "rgi-toolkit",
        "boltz",
        "protenix",
        "chai-lab",
        "openfold3",
        "esm",
        "transformers",
        "torch",
        "jax",
        "rdkit",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def command(job, runtime):
    seed = str(job["seed"])
    model = job["model"]
    python = sys.executable
    extra = runtime.get("args", [])
    if any(
        arg.split("=")[0] in {"--use_potentials", "--sample_diffusion.guidance.enable"}
        for arg in extra
    ):
        raise ValueError("Select native guidance with --arms, not runtime args")
    if uses_guidance(job["arm"]) and model not in GUIDANCE_MODELS:
        raise ValueError(f"Native guidance is unsupported by {model}")
    if model == "boltz_restr":
        cmd = [
            python,
            "-m",
            "boltz.main",
            "predict",
            "input.yaml",
            "--out_dir",
            "raw",
            "--model",
            "boltz2",
            "--seed",
            seed,
            "--diffusion_samples",
            "1",
        ]
    elif model == "protenix_restr":
        cmd = [
            python,
            runtime["script"],
            "--model_name",
            "protenix-v2",
            "--num_workers",
            "0",
            "--seeds",
            seed,
            "--dump_dir",
            "raw",
            "--input_json_path",
            "input.json",
            "--model.N_cycle",
            "10",
            "--sample_diffusion.N_sample",
            "1",
            "--sample_diffusion.N_step",
            "200",
        ]
    elif model == "alphafold3_restr":
        cmd = [
            python,
            runtime["script"],
            "--json_path=input.json",
            "--output_dir=raw",
            "--run_data_pipeline=False",
            "--num_diffusion_samples=1",
            f"--model_dir={runtime['model_dir']}",
        ]
    elif model == "openfold-3_restr":
        cmd = [
            python,
            "-m",
            "openfold3.run_openfold",
            "predict",
            "--query-json",
            "input.json",
            "--output-dir",
            "raw",
            "--num-diffusion-samples",
            "1",
            "--use-msa-server",
            "false",
            "--use-templates",
            "false",
        ]
    elif model == "chai-lab_restr":
        cmd = [
            python,
            "-m",
            "chai_lab.main",
            "fold",
            "input.fasta",
            "raw",
            "--msa-directory",
            "msa",
            "--num-trunk-samples",
            "1",
            "--num-diffn-samples",
            "1",
            "--seed",
            seed,
        ]
        if job["restraints"]:
            cmd.extend(["--restraints-config-path", "restraints.yaml"])
    else:
        raise ValueError(f"No CLI for {model}")
    if model == "boltz_restr" and uses_guidance(job["arm"]):
        cmd.append("--use_potentials")
    elif model == "protenix_restr":
        cmd.extend(
            [
                "--sample_diffusion.guidance.enable",
                str(uses_guidance(job["arm"])).lower(),
            ]
        )
    cmd += extra
    return msa_recycle_command(cmd, model) if job["arm"] == "ms" else cmd


def chai_msa(job):
    import pandas as pd
    from chai_lab.data.parsing.msas.aligned_pqt import (
        expected_basename,
        merge_a3m_in_directory,
    )

    output = Path("msa")
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        lines = Path(job["msa"]).read_text().splitlines()
        Path(temporary, "uniref90.a3m").write_text(
            "\n".join(line for line in lines if not line.startswith("#")) + "\n"
        )
        merge_a3m_in_directory(temporary, output_directory=temporary)
        fresh = Path(temporary) / expected_basename(job["sequence"])
        if not fresh.is_file():
            raise ValueError(
                "Chai MSA query mismatch: expected sequence hash not written"
            )
        table = pd.read_parquet(fresh)
        if table.empty or table.iloc[0]["sequence"].upper() != job["sequence"].upper():
            raise ValueError("Chai MSA query mismatch")
        fresh.replace(output / fresh.name)
    print(f"Chai MSA: rows={len(table)} query_ok=True", flush=True)


def esm_fold(job):
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        EsmFold2Model,
        LigandInput,
        ProteinInput,
        StructurePredictionInput,
    )
    from esm.utils.msa import MSA

    msa = MSA.from_a3m(path=job["msa"], remove_insertions=True)
    if msa.depth < 1 or msa.query != job["sequence"]:
        raise ValueError("ESMFold2 MSA query mismatch")
    entities = [ProteinInput(id="A", sequence=job["sequence"], msa=msa)]
    if job.get("smiles"):
        entities.append(
            LigandInput(
                id="B", smiles=job["smiles"], conformer_restraints=uses_rgi(job["arm"])
            )
        )
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device="cuda")
    model.train(False)
    result = ESMFold2InputBuilder().fold(
        model,
        StructurePredictionInput(sequences=entities),
        seed=job["seed"],
        num_loops=1 if job["arm"] == "ms" else 20,
        num_sampling_steps=200,
        msa_max_depth=1024,
        msa_column_mask_rate=0.1,
        restraints_config=job["restraints"],
    )
    Path("raw").mkdir(exist_ok=True)
    Path("raw/model.cif").write_text(result.complex.to_mmcif())
    Path("raw/model.confidence.json").write_text(
        json.dumps(
            {
                "ptm": None if result.ptm is None else float(result.ptm),
                "iptm": None if result.iptm is None else float(result.iptm),
            },
            allow_nan=False,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", nargs="?", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(provenance(), sort_keys=True))
        return
    runtime = json.loads(args.runtime.read_text())
    job = json.loads(args.job.read_text())
    os.chdir(args.job.resolve().parent)
    Path("software.json").write_text(json.dumps(provenance(), indent=2) + "\n")
    if job["model"] == "chai-lab_restr":
        chai_msa(job)
    if job["model"] == "esm_restr":
        esm_fold(job)
    else:
        subprocess.run(command(job, runtime), check=True)


if __name__ == "__main__":
    main()
