"""Reproducible, isolated nine-model CG verification and paired timing campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = {
    "boltz1": "boltz",
    "boltz2": "boltz",
    "protenix1": "protenix",
    "protenix2": "protenix",
    "chai": "chai",
    "af3": "af3",
    "of3": "of3",
    "esm": "esm",
    "opendde": "opendde",
}
CASES = (
    "off",
    "distance",
    "distance_conformer",
    "distance_vdw",
    "angle",
    "rmsd",
    "custom",
)
BASELINE = "11de8b4aa49c9298a7de4fdc112afa322886aa40"
ATP_SMILES = (
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_cluster(path):
    data = yaml.safe_load(path.read_text())

    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if not isinstance(value, str):
            return value

        def substitute(match):
            key = match[1]
            if key == "self_dir":
                return str(path.parent)
            current = data
            for part in key.split("."):
                current = current[part]
            return str(resolve(current))

        return os.path.expandvars(re.sub(r"\$\{([^}]+)\}", substitute, value))

    return resolve(data)


def case_data(case, fixtures):
    from rgi_toolkit.config import resolve_restraints_config

    family, name = {
        "angle": ("angle", "adk_72.85"),
        "custom": ("custom/dist-diff", "dgot_0.00"),
        "rmsd": ("rmsd", "qbp_3.00"),
    }.get(case, ("distance", "qbp_25.00"))
    source = ROOT / "examples" / family / "boltz-2" / (name + ".yaml")
    example = yaml.safe_load(source.read_text())
    sequence = example["sequences"][0]["protein"]["sequence"]
    config = resolve_restraints_config(
        example["restraints_config"], base_dir=source.parent
    )
    config.pop("max_iter", None) if case != "rmsd" else None
    for section in ("distance_restraints_config", "angle_restraints_config"):
        for entry in config.get(section, []):
            for key in tuple(entry):
                if key.startswith("atom_selection"):
                    entry[key] = "chain A and (" + entry[key] + ")"
    if case == "off":
        config = None
    if case == "distance_conformer":
        config["conformer_restraints_config"] = {}
    if case == "distance_vdw":
        config["conformer_restraints_config"] = {
            key: {"weight": 0} for key in ("bond", "angle", "chiral", "cistrans")
        }
        config["conformer_restraints_config"]["vdw"] = {}
    if case == "rmsd":
        for entry in config["rmsd_restraints_config"]:
            entry["ref_cif"] = str(fixtures / Path(entry["ref_cif"]).name)
    return sequence, config, name.split("_")[0]


def make_input(model, case, seed, target, fixtures):
    seq, config, protein = case_data(case, fixtures)
    tool = MODELS[model]
    ligand = case in ("distance_conformer", "distance_vdw")
    msa = fixtures / (protein + ".a3m")
    identifier = f"{model}_{case}_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "restraints.json", config)
    if tool == "boltz":
        sequences = [{"protein": {"id": "A", "sequence": seq, "msa": str(msa)}}]
        if ligand:
            sequences.append(
                {"ligand": {"id": "B", "ccd": "ATP", "conformer_restraints": True}}
            )
        data = {"sequences": sequences}
        if config is not None:
            data["restraints_config"] = config
        path = target / "input.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    elif tool in ("protenix", "opendde"):
        entity = {"sequence": seq, "count": 1, "unpairedMsaPath": str(msa)}
        if tool == "opendde":
            entity["id"] = ["A"]
        data = {"name": identifier, "sequences": [{"proteinChain": entity}]}
        if tool == "opendde":
            data["modelSeeds"] = [seed]
        if ligand:
            lig = {"ligand": "CCD_ATP", "count": 1, "conformer_restraints": True}
            if tool == "opendde":
                lig["id"] = ["B"]
            data["sequences"].append({"ligand": lig})
        if config is not None:
            data["restraints_config"] = config
        path = target / "input.json"
        write_json(path, [data])
    elif tool == "af3":
        data = {
            "dialect": "alphafold3",
            "version": 4,
            "name": identifier,
            "modelSeeds": [seed],
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": seq,
                        "unpairedMsaPath": str(msa),
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            ],
        }
        if ligand:
            data["sequences"].append(
                {
                    "ligand": {
                        "id": "B",
                        "ccdCodes": ["ATP"],
                        "conformer_restraints": True,
                    }
                }
            )
        if config is not None:
            data["restraints_config"] = config
        path = target / "input.json"
        write_json(path, data)
    elif tool == "of3":
        query = {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": seq,
                    "main_msa_file_paths": [
                        str(fixtures / protein / "uniref90_hits.a3m")
                    ],
                }
            ]
        }
        if ligand:
            query["chains"].append(
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["B"],
                    "ccd_codes": "ATP",
                    "conformer_restraints": True,
                }
            )
        if config is not None:
            query["restraints_config"] = config
        path = target / "input.json"
        write_json(path, {"seeds": [seed], "queries": {identifier: query}})
        (target / "runner.yaml").write_text(
            yaml.safe_dump({"experiment_settings": {"seeds": [seed]}})
        )
    elif tool == "chai":
        path = target / "input.fasta"
        path.write_text(
            f">protein|name=A\n{seq}\n"
            + (f">ligand|name=B\n{ATP_SMILES}\n" if ligand else "")
        )
        if ligand:
            config = copy.deepcopy(config)
            config["conformer_restraints"] = {"B": True}
            write_json(target / "restraints.json", config)
    else:
        path = target / "input.json"
        write_json(
            path,
            {
                "sequence": seq,
                "ligand": ligand,
                "conformer": ligand,
                "seed": seed,
                "restraints_config": config,
            },
        )
    return str(path), protein


def prepare(args):
    cluster = load_cluster(args.cluster_config.resolve())
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "Use a fresh output directory; existing campaigns are immutable"
        )
    fixtures = output / "fixtures"
    fixtures.mkdir(parents=True)
    for family, protein in (("distance", "qbp"), ("angle", "adk"), ("custom", "dgot")):
        source = args.cluster_config.parent / family / "fixtures" / (protein + ".a3m")
        lines = source.read_text().splitlines()
        (fixtures / (protein + ".a3m")).write_text(
            "\n".join(line for line in lines if not line.startswith("#")) + "\n"
        )
        (fixtures / protein).mkdir()
        shutil.copy2(
            fixtures / (protein + ".a3m"), fixtures / protein / "uniref90_hits.a3m"
        )
    for name in ("1GGG.cif", "1WDN.cif"):
        shutil.copy2(
            args.cluster_config.parent / "distance" / "fixtures" / name, fixtures / name
        )
    sources = output / "sources"
    shutil.copytree(
        ROOT / "src",
        sources / "new" / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    archive = subprocess.check_output(["git", "archive", BASELINE, "src"], cwd=ROOT)
    (sources / "old").mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(sources / "old", filter="data")
    files = sorted((sources / "new" / "src").rglob("*.py"))
    digest = hashlib.sha256(
        b"".join(
            str(p.relative_to(sources / "new")).encode() + p.read_bytes() for p in files
        )
    ).hexdigest()
    old_digest = hashlib.sha256(
        b"".join(
            str(p.relative_to(sources / "old")).encode() + p.read_bytes()
            for p in sorted((sources / "old" / "src").rglob("*.py"))
        )
    ).hexdigest()
    runs = []
    for model in MODELS:
        for case in CASES:
            for seed in (0, 1, 2):
                path, protein = make_input(
                    model,
                    case,
                    seed,
                    output / "inputs" / model / case / str(seed),
                    fixtures,
                )
                runs.append(
                    dict(model=model, case=case, seed=seed, input=path, protein=protein)
                )
    manifest = dict(
        cluster=cluster,
        output=str(output),
        runs=runs,
        baseline=BASELINE,
        baseline_source_sha256=old_digest,
        timing_warmup=True,
        head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_sha256=digest,
    )
    write_json(manifest_path, manifest)
    (output / "working.patch").write_bytes(
        subprocess.check_output(["git", "diff"], cwd=ROOT)
    )
    print(manifest_path)


def command(manifest, run, out):
    model, seed = run["model"], run["seed"]
    tool, path = MODELS[model], run["input"]
    cfg = manifest["cluster"]["tools"][tool]
    workspace = Path(manifest["cluster"]["workspace_root"])
    root = Path(
        cfg.get(
            "root",
            workspace
            / {
                "boltz": "boltz_restr",
                "chai": "chai-lab_restr",
                "af3": "alphafold3_restr",
            }.get(tool, tool),
        )
    )
    if tool in ("of3", "esm"):
        pixi = root / ".pixi-bin" / "pixi"
        prefix = [
            str(pixi) if pixi.exists() else "pixi",
            "run",
            "--manifest-path",
            str(root / ("pixi.toml" if tool == "of3" else "pyproject.toml")),
        ]
        if tool == "of3":
            prefix += ["-e", cfg["pixi_env"]]
        environment = cfg["pixi_env"] if tool == "of3" else "default"
        prefix += [
            "uv",
            "run",
            "--no-project",
            "--python",
            str(root / ".pixi" / "envs" / environment / "bin" / "python"),
        ]
    else:
        prefix = [
            "uv",
            "run",
            "--no-project",
            "--python",
            str(Path(cfg["venv"]) / "bin" / "python"),
        ]
    if tool == "boltz":
        args = [
            "boltz",
            "predict",
            path,
            "--out_dir",
            str(out),
            "--model",
            model,
            "--seed",
            str(seed),
            "--diffusion_samples",
            "1",
        ]
    elif tool == "protenix":
        name = "protenix-v2" if model == "protenix2" else "protenix_base_default_v1.0.0"
        args = [
            "python",
            str(root / "runner" / "inference.py"),
            "--num_workers",
            "0",
            "--model_name",
            name,
            "--seeds",
            str(seed),
            "--dump_dir",
            str(out),
            "--input_json_path",
            path,
            "--model.N_cycle",
            "10",
            "--sample_diffusion.N_sample",
            "1",
            "--sample_diffusion.N_step",
            "200",
        ]
    elif tool == "opendde":
        args = [
            "opendde",
            "pred",
            "-i",
            path,
            "-o",
            str(out),
            "-n",
            "opendde_v1",
            "--use_msa",
            "false",
            "--use_template",
            "false",
            "--use_rna_msa",
            "false",
            "--sample",
            "1",
            "--step",
            "200",
            "--cycle",
            "4",
        ]
    elif tool == "af3":
        args = [
            "python",
            cfg["run_script"],
            "--json_path",
            path,
            "--output_dir",
            str(out),
            "--model_dir",
            cfg["model_dir"],
            "--run_data_pipeline=false",
            "--num_diffusion_samples=1",
        ]
    elif tool == "chai":
        args = [
            "python",
            "-m",
            "chai_lab.main",
            "fold",
            path,
            str(out),
            "--num-trunk-samples",
            "1",
            "--num-diffn-samples",
            "1",
            "--seed",
            str(seed),
            "--msa-directory",
            str(Path(manifest["output"]) / "chai_msa" / run["protein"]),
        ]
        if run["case"] != "off":
            args += [
                "--restraints-config-path",
                str(Path(path).with_name("restraints.json")),
            ]
    elif tool == "of3":
        args = [
            "run_openfold",
            "predict",
            "--query-json",
            path,
            "--runner-yaml",
            str(Path(path).with_name("runner.yaml")),
            "--output-dir",
            str(out),
            "--num-diffusion-samples",
            "1",
            "--use-msa-server",
            "false",
            "--use-templates",
            "false",
        ]
    else:
        args = ["python", str(HERE / "predict.py"), "esm", path, str(out)]
    return prefix + args, root


def run_model(args):
    manifest = json.loads(args.manifest.read_text())
    output = Path(manifest["output"])
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Prediction requires a Slurm allocation")
    model = args.model
    warming = getattr(args, "_warming", False)
    if args.timing_only and manifest.get("timing_warmup") and not warming:
        warmup = copy.copy(args)
        warmup._warming = True
        run_model(warmup)
    cfg = manifest["cluster"]["tools"][MODELS[model]]
    if model == "chai":
        for protein in ("qbp", "adk", "dgot"):
            directory = output / "chai_msa" / protein
            if not directory.exists():
                directory.mkdir(parents=True)
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "--no-project",
                        "--python",
                        cfg["venv"] + "/bin/python",
                        "python",
                        str(HERE / "predict.py"),
                        "chai-msa",
                        str(output / "fixtures" / (protein + ".a3m")),
                        str(directory),
                    ],
                    check=True,
                )
    selected_cases = ("distance",) if args.timing_only else args.cases.split(",")
    runs = [
        r
        for r in manifest["runs"]
        if r["model"] == model and r["case"] in selected_cases
    ]
    # Baseline and candidate use the same allocation, seed and full predictor settings.
    runs.sort(key=lambda r: (r["case"] != "distance", r["case"], r["seed"]))
    failures = []
    for run in runs:
        versions = ("old", "new") if run["case"] == "distance" else ("new",)
        if run["seed"] % 2:
            versions = tuple(reversed(versions))
        for version in versions:
            target = (
                output
                / ("warmup" if warming else "timing" if args.timing_only else "results")
                / model
                / run["case"]
                / str(run["seed"])
                / version
            )
            if target.exists():
                result_path = target / "result.json"
                previous = (
                    json.loads(result_path.read_text()) if result_path.exists() else {}
                )
                if previous.get("returncode") == 0:
                    continue
                if not args.retry_failed:
                    raise FileExistsError(
                        f"Use --retry-failed to archive and retry: {target}"
                    )
                attempt = 1
                while target.with_name(f"{version}.failed-{attempt}").exists():
                    attempt += 1
                target.rename(target.with_name(f"{version}.failed-{attempt}"))
            target.mkdir(parents=True, exist_ok=True)
            argv, cwd = command(manifest, run, target / "prediction")
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)
            paths = [str(HERE), str(output / "sources" / version / "src")]
            if model == "esm":
                paths.append(manifest["cluster"]["tools"]["esm"]["transformers_src"])
                paths.append(manifest["cluster"]["tools"]["esm"]["root"])
            env.update(
                PYTHONPATH=os.pathsep.join(paths),
                RGI_VERIFY_TRACE=str(target / "trace"),
                RGI_VERIFY_LEGACY="1" if version == "old" else "",
                RGI_VERIFY_INSTRUMENT="0" if args.timing_only else "1",
                RGI_VERIFY_JAX_CAPTURE="1" if model == "af3" else "",
                UV_CACHE_DIR=str(output / "cache" / "uv"),
                TORCHINDUCTOR_CACHE_DIR=str(
                    output / "cache" / f"inductor-{model}-{version}"
                ),
                JAX_COMPILATION_CACHE_DIR=str(
                    output / "cache" / f"jax-{model}-{version}"
                ),
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                PYTHONHASHSEED="0",
                LAYERNORM_TYPE="torch",
                USE_DEEPSPEED_EVO_ATTENTION="false",
            )
            if model == "opendde":
                env["OPENDDE_DATA_ROOT"] = cfg["data_root"]
            start = time.perf_counter()
            with (target / "predict.log").open("w") as log:
                try:
                    proc = subprocess.run(
                        argv,
                        cwd=cwd,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=7200,
                    )
                    code = proc.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            result = dict(
                run,
                version=version,
                returncode=code,
                wall_seconds=time.perf_counter() - start,
                command=argv,
                job=os.environ["SLURM_JOB_ID"],
                instrumented=not args.timing_only,
                warmup=warming,
                python_hash_seed=0,
            )
            write_json(target / "result.json", result)
            print(json.dumps(result), flush=True)
            if code:
                failures.append(str(target))
                break
        if failures:
            break
    if failures:
        raise RuntimeError(f"{len(failures)} predictor processes failed: {failures}")


def submit(args):
    manifest = json.loads(args.manifest.read_text())
    output = Path(manifest["output"])
    jobs = []
    for model in args.models.split(","):
        tool = MODELS[model]
        argv = [
            "sbatch",
            "--parsable",
            "--exclusive",
            "--gres=gpu:1",
            "--time=24:00:00",
            "--partition=" + ",".join(manifest["cluster"]["tools"][tool]["partitions"]),
            "--job-name=rgi-cg-" + model,
            "--output=" + str(output / ("slurm-" + model + "-%j.log")),
        ]
        if len(jobs) >= 4:
            argv += ["--dependency=afterany:" + jobs[-4]]
        elif args.after:
            argv += ["--dependency=afterany:" + args.after]
        wrapper = [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(HERE / "run.py"),
            "run",
            "--manifest",
            str(args.manifest.resolve()),
            "--model",
            model,
            "--cases",
            args.cases,
        ]
        if args.retry_failed:
            wrapper.append("--retry-failed")
        if args.timing_only:
            wrapper.append("--timing-only")
        argv += ["--wrap=" + shlex.join(wrapper)]
        job = subprocess.check_output(argv, text=True).strip()
        jobs.append(job)
        print(model, job, flush=True)
    write_json(
        output / ("jobs-" + jobs[0] + ".json"), dict(zip(args.models.split(","), jobs))
    )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--cluster-config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for action in ("run", "submit"):
        sub = commands.add_parser(action)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--cases", default=",".join(CASES))
        sub.add_argument("--retry-failed", action="store_true")
        sub.add_argument("--timing-only", action="store_true")
        if action == "run":
            sub.add_argument("--model", choices=MODELS, required=True)
        else:
            sub.add_argument("--after")
            sub.add_argument(
                "--models",
                default="esm,chai,protenix1,protenix2,boltz1,boltz2,af3,of3,opendde",
            )
    args = parser.parse_args()
    {"prepare": prepare, "run": run_model, "submit": submit}[args.action](args)


if __name__ == "__main__":
    main()
