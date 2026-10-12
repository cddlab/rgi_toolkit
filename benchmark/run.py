"""Execute prepared jobs sequentially, or select one array task."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml
from filelock import FileLock

from common import (
    ROOT,
    read_json,
    validate_prediction,
    verify_inputs,
    write_json,
)

PATTERNS = {
    "boltz2": "**/predictions/*/*_model_0.cif",
    "protenix": "**/seed_*/predictions/*_sample_0.cif",
    "af3": "**/seed-*_sample-0/model.cif",
    "chai": "**/pred.model_idx_0.cif",
    "of3": "**/*_model.cif",
    "esm": "model.cif",
}
TERMS = {
    "distance": ("distances",),
    "angle": ("group_angle",),
    "rmsd": ("rmsd",),
    "custom": ("custom",),
    "conformer": ("bonds", "angles", "cistrans"),
}


def expand(value):
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if re.search(r"\$\{?\w+", expanded):
            raise ValueError(f"Set the environment variable in {value}")
        return expanded
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    raise ValueError("Runtime settings must contain strings, lists, or mappings")


def runtime_for(config, model, config_dir):
    runtime = expand(config[model])
    for field in ("python", "script", "model_dir"):
        if field not in runtime:
            continue
        path = Path(runtime[field])
        if not path.is_absolute():
            path = config_dir / path
        exists = path.is_dir() if field == "model_dir" else path.is_file()
        if not exists:
            raise ValueError(f"Missing {model} {field}: {runtime[field]}")
        runtime[field] = str(path.absolute())
    if model == "protenix":
        env = runtime.setdefault("env", {})
        env.setdefault("PROTENIX_ROOT_DIR", str(Path(runtime["script"]).parent.parent))
        env.setdefault("LAYERNORM_TYPE", "torch")
        env.setdefault("USE_DEEPSPEED_EVO_ATTENTION", "false")
    return runtime


def check_restraints(log, job):
    if job["arm"] != "rgi":
        return
    inventories = [line for line in log.splitlines() if "built spec:" in line]
    for term in TERMS[job["benchmark"]]:
        counts = [
            int(n) for line in inventories for n in re.findall(rf"\b{term}=(\d+)", line)
        ]
        if not counts or not all(counts):
            raise ValueError(f"No active {term} in setup log")


def run(args):
    root = args.run.resolve()
    manifest = verify_inputs(root)
    jobs = manifest["jobs"]
    if args.model:
        jobs = [path for path in jobs if read_json(root / path)["model"] == args.model]
    if args.task is not None:
        if args.task < 0 or args.task >= len(jobs):
            raise ValueError("Task index is outside the selected job list")
        jobs = jobs[args.task : args.task + 1]
    if not jobs:
        raise ValueError("No jobs selected")
    config = yaml.safe_load(args.runtime.read_text())
    probes = {}
    for relative in jobs:
        path = root / relative
        job = read_json(path)
        runtime = runtime_for(config, job["model"], args.runtime.resolve().parent)
        prefix = [
            "uv",
            "run",
            "--no-project",
            "--python",
            runtime["python"],
            "python",
            str(ROOT / "models.py"),
        ]
        print(f"{job['id']}: {shlex.join(prefix + [str(path)])}", flush=True)
        if args.dry_run:
            continue
        environment = {**os.environ, **runtime.get("env", {})}
        if job["model"] not in probes:
            result = subprocess.run(
                prefix + ["--probe"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            probes[job["model"]] = json.loads(result.stdout)
        software = probes[job["model"]]
        folder = path.parent
        prediction = folder / "prediction.cif"
        receipt = folder / "complete.json"
        identity = {
            "runtime": runtime,
            "software": software,
        }
        record = root / f"software-{job['model']}.json"
        with FileLock(str(record.with_suffix(".lock"))):
            if record.exists() and read_json(record) != identity:
                raise ValueError("Runtime or software changed; prepare a new run")
            write_json(record, identity)
        if receipt.exists() and not args.retry:
            old = read_json(receipt)
            if old["identity"] != identity:
                raise ValueError("Runtime or software changed; prepare a new run")
            validate_prediction(prediction, job)
            print("  verified; skipped")
            continue
        if (folder / "raw").exists() and not args.retry:
            raise ValueError("Incomplete raw output exists; use --retry for this task")
        if args.retry:
            receipt.unlink(missing_ok=True)
            prediction.unlink(missing_ok=True)
            if (folder / "raw").exists():
                shutil.rmtree(folder / "raw")
        write_json(folder / "runtime.json", runtime)
        started = time.monotonic()
        with (folder / "run.log").open("w") as log:
            subprocess.run(
                prefix + [str(path), "--runtime", str(folder / "runtime.json")],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env=environment,
            )
        if read_json(folder / "software.json") != software:
            raise ValueError("Predictor software changed during execution")
        matches = list((folder / "raw").glob(PATTERNS[job["model"]]))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one prediction, found {len(matches)}: {job['id']}"
            )
        validate_prediction(matches[0], job)
        check_restraints((folder / "run.log").read_text(), job)
        shutil.copyfile(matches[0], prediction)
        write_json(
            receipt,
            {
                "identity": identity,
                "seconds": time.monotonic() - started,
            },
        )
        print("  validated; complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime.yaml")
    parser.add_argument("--model")
    parser.add_argument(
        "--task", type=int, help="Zero-based index after --model filtering"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retry", action="store_true", help="Replace selected job outputs"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
