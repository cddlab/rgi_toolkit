"""Stage small replay inputs onto storage accessible to CPU-only worker nodes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy(source, target):
    if target.exists() and target.stat().st_mtime_ns == source.stat().st_mtime_ns:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-structures", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source, output = Path(manifest["output"]), args.output.resolve()
    if source.resolve() == output:
        raise ValueError("Staging must use a separate output directory")
    for reference in (source / "fixtures").glob("*.cif"):
        copy(reference, output / "fixtures" / reference.name)
    for run in manifest["runs"]:
        parts = Path(run["model"]) / run["case"] / str(run["seed"])
        versions = ("old", "new") if run["case"] == "distance" else ("new",)
        for version in versions:
            trace = source / "results" / parts / version / "trace"
            target = output / "results" / parts / version / "trace"
            result = trace.parent / "result.json"
            completed = (
                result.exists() and json.loads(result.read_text())["returncode"] == 0
            )
            if completed and (trace / "spec.pkl").exists():
                target.mkdir(parents=True, exist_ok=True)
                for path in [trace / "spec.pkl", *trace.glob("step_*.npz")]:
                    copy(path, target / path.name)
                copy(result, target.parent / result.name)
            if completed and args.include_structures:
                copy(result, target.parent / result.name)
                copy(trace.parent / "predict.log", target.parent / "predict.log")
                if (trace / "trace.json").exists():
                    copy(trace / "trace.json", target / "trace.json")
                if (trace / "source.json").exists():
                    copy(trace / "source.json", target / "source.json")
                for structure in (trace.parent / "prediction").rglob("*.cif"):
                    copy(structure, target.parent / structure.relative_to(trace.parent))
        if args.include_structures and run["case"] == "distance":
            for version in ("old", "new"):
                warmup = source / "warmup" / parts / version
                warm_result = warmup / "result.json"
                if warm_result.exists():
                    copy(
                        warm_result, output / "warmup" / parts / version / "result.json"
                    )
                timing = source / "timing" / parts / version
                target = output / "timing" / parts / version
                result = timing / "result.json"
                if (
                    result.exists()
                    and json.loads(result.read_text())["returncode"] == 0
                ):
                    copy(result, target / result.name)
                    copy(timing / "predict.log", target / "predict.log")
                    if (timing / "trace" / "source.json").exists():
                        copy(
                            timing / "trace" / "source.json",
                            target / "trace/source.json",
                        )
                    for structure in (timing / "prediction").rglob("*.cif"):
                        copy(structure, target / structure.relative_to(timing))
        config = Path(run["input"]).with_name("restraints.json")
        new_input = output / "inputs" / parts / "input.json"
        new_input.parent.mkdir(parents=True, exist_ok=True)
        config_data = json.loads(config.read_text())
        for entry in (config_data or {}).get("rmsd_restraints_config", []):
            entry["ref_cif"] = str(output / "fixtures" / Path(entry["ref_cif"]).name)
        temporary = new_input.with_name(".restraints.json.tmp")
        temporary.write_text(json.dumps(config_data, indent=2) + "\n")
        temporary.replace(new_input.with_name("restraints.json"))
        run["input"] = str(new_input)
    manifest["source_manifest"] = str(args.manifest.resolve())
    manifest["output"] = str(output)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
