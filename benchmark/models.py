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
    if model == "boltz2":
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
    elif model == "protenix":
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
    elif model == "af3":
        cmd = [
            python,
            runtime["script"],
            "--json_path=input.json",
            "--output_dir=raw",
            "--run_data_pipeline=False",
            "--num_diffusion_samples=1",
            f"--model_dir={runtime['model_dir']}",
        ]
    elif model == "of3":
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
    elif model == "chai":
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
    return cmd + runtime.get("args", [])


def chai_msa(job):
    import pandas as pd
    from chai_lab.data.parsing.msas.aligned_pqt import (
        expected_basename,
        merge_a3m_in_directory,
    )

    output = Path("msa")
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        Path(temporary, "uniref90.a3m").write_text(Path(job["msa"]).read_text())
        merge_a3m_in_directory(temporary, output_directory=str(output))
    query = pd.read_parquet(output / expected_basename(job["sequence"])).iloc[0][
        "sequence"
    ]
    if query.upper().replace("-", "") != job["sequence"]:
        raise ValueError("Chai MSA query mismatch")


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
                id="B", smiles=job["smiles"], conformer_restraints=job["arm"] == "rgi"
            )
        )
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device="cuda")
    model.train(False)
    result = ESMFold2InputBuilder().fold(
        model,
        StructurePredictionInput(sequences=entities),
        seed=job["seed"],
        num_loops=20,
        num_sampling_steps=200,
        msa_max_depth=1024,
        msa_column_mask_rate=0.1,
        restraints_config=job["restraints"],
    )
    Path("raw").mkdir(exist_ok=True)
    Path("raw/model.cif").write_text(result.complex.to_mmcif())


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
    if job["model"] == "chai":
        chai_msa(job)
    if job["model"] == "esm":
        esm_fold(job)
    else:
        subprocess.run(command(job, runtime), check=True)


if __name__ == "__main__":
    main()
