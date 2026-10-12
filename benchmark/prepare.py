"""Prepare matched OFF/RGI jobs with shared inputs."""

from __future__ import annotations

import argparse
import copy
import csv
import shutil
from pathlib import Path

import yaml
from rgi_toolkit.config import RestraintsConfig
from rgi_toolkit.selection import AtomSelector

from common import (
    BENCHMARKS,
    MODELS,
    msa_query,
    protocol,
    slug,
    write_json,
)
from inputs import write_input


def restraint(config, target):
    result = copy.deepcopy(config["restraints"])
    kind = config["kind"]
    if kind in {"distance", "angle"}:
        result[f"{kind}_restraints_config"][0]["harmonic"][f"target_{kind}"] = target
    elif kind == "rmsd":
        for entry, value in zip(result["rmsd_restraints_config"], target, strict=True):
            entry["harmonic"]["target_rmsd"] = value
    elif kind == "custom":
        result["custom_restraints_config"][0]["energy"] = (
            f"((distance(A, B) - distance(C, D)) - ({target}))**2"
        )
    RestraintsConfig.from_dict(result)
    for entries in result.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            selections = [v for k, v in entry.items() if k.startswith("atom_selection")]
            selections.extend(entry.get("selections", {}).values())
            for selection in selections:
                AtomSelector(selection)
    return result


def stage_msa(source, destination, sequence):
    if msa_query(source) != sequence:
        raise ValueError(f"MSA query differs from the benchmark sequence: {source}")
    lines = Path(source).read_text().splitlines()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(line for line in lines if not line.startswith("#")) + "\n"
    )


def prepare(args):
    config = protocol(args.benchmark)
    run = Path(args.output)
    if run.exists() and any(run.iterdir()):
        raise ValueError(
            "Output must be empty; use a new directory for changed conditions"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Models must be unique")
    if args.benchmark == "conformer":
        if args.msa or args.single_sequence:
            raise ValueError("Conformer uses the MSAs supplied with its dataset")
        if not args.index:
            from fetch import conformer

            conformer(args.dataset, args.limit)
            args.index = args.dataset / "index.csv"
        index = Path(args.index)
        with index.open() as handle:
            systems = list(csv.DictReader(handle))
        if args.limit:
            systems = systems[: args.limit]
        for row in systems:
            row["msa"] = index.parent / row["msa"]
            row["sequence"] = msa_query(row["msa"])
            row["sdf"] = index.parent / row["sdf"]
            slug(row["id"])
        if not systems or len({r["id"] for r in systems}) != len(systems):
            raise ValueError("Dataset must contain nonempty, unique system IDs")
        targets = [None]
    else:
        if bool(args.msa) == bool(args.single_sequence):
            raise ValueError("Choose --msa or explicitly use --single-sequence")
        systems = [
            {"id": config["name"], "sequence": config["sequence"], "msa": args.msa}
        ]
        targets = config["targets"][: args.limit]
    for target in targets:
        restraint(config, target)
    for row in systems:
        if not args.single_sequence and msa_query(row["msa"]) != row["sequence"]:
            raise ValueError("MSA query differs from the benchmark sequence")
        if row.get("sdf") and not row["sdf"].is_file():
            raise ValueError(f"Missing SDF for {row['id']}")
    for name in config.get("references", []):
        if not (Path(args.references) / name).is_file():
            raise ValueError(f"Missing {name}; run fetch.py references first")
    run.mkdir(parents=True, exist_ok=True)
    data = run / "data"
    data.mkdir()
    for name in config.get("references", []):
        source = Path(args.references) / name
        shutil.copyfile(source, data / name)
    jobs = []
    for row in systems:
        system = slug(row["id"])
        msa = data / f"{system}.a3m"
        if args.single_sequence:
            msa.write_text(f">{system}\n{row['sequence']}\n")
        else:
            stage_msa(row["msa"], msa, row["sequence"])
        if row.get("sdf"):
            shutil.copyfile(row["sdf"], data / f"{system}.sdf")
        for model in args.models:
            # Match the OFF sample count to the complete target sweep.
            for arm in ("off", "rgi"):
                for target_index, target in enumerate(targets):
                    for seed in args.seeds:
                        actual_seed = (
                            seed
                            if arm == "rgi"
                            else seed + target_index * (max(args.seeds) + 1)
                        )
                        name = f"{system}-{model}-{arm}-t{target_index}-s{actual_seed}"
                        job = {
                            "id": name,
                            "benchmark": args.benchmark,
                            "model": model,
                            "system": system,
                            "sequence": row["sequence"],
                            "smiles": row.get("smiles"),
                            "arm": arm,
                            "target": target if arm == "rgi" else None,
                            "seed": actual_seed,
                            "msa": f"../../data/{system}.a3m",
                            "restraints": restraint(config, target)
                            if arm == "rgi"
                            else None,
                        }
                        folder = run / "jobs" / name
                        folder.mkdir(parents=True)
                        write_input(folder, job)
                        write_json(folder / "job.json", job)
                        jobs.append(f"jobs/{name}/job.json")
    (run / "protocol.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    files = [
        p.relative_to(run).as_posix()
        for sub in (data, run / "jobs")
        for p in sub.rglob("*")
        if p.is_file()
    ]
    files.append("protocol.yaml")
    write_json(
        run / "manifest.json",
        {
            "version": 1,
            "benchmark": args.benchmark,
            "jobs": jobs,
            "msa_mode": "single-sequence" if args.single_sequence else "provided",
            "files": files,
        },
    )
    print(f"Prepared {len(jobs)} jobs in {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["boltz2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--msa", type=Path)
    parser.add_argument("--single-sequence", action="store_true")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/conformer"))
    parser.add_argument("--references", type=Path, default=Path("data/references"))
    parser.add_argument(
        "--limit", type=int, help="First N targets or conformer systems"
    )
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
