"""Write per-sample and grouped CSVs; incomplete runs exit nonzero."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from common import read_json, validate_prediction, verify_inputs, write_json
from metrics import conformer_metrics, state_metrics


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate(root):
    manifest = verify_inputs(root)
    config = yaml.safe_load((root / "protocol.yaml").read_text())
    rows = []
    for relative in manifest["jobs"]:
        folder = (root / relative).parent
        job = read_json(folder / "job.json")
        row = {
            key: job[key]
            for key in ("id", "benchmark", "model", "system", "arm", "seed")
        }
        row["target"] = json.dumps(job["target"])
        try:
            receipt = read_json(folder / "complete.json")
            path = folder / "prediction.cif"
            atoms, xyz = validate_prediction(path, job)
            values = (
                conformer_metrics(atoms, xyz, job, folder)
                if job["benchmark"] == "conformer"
                else state_metrics(atoms, xyz, job, config, folder)
            )
            if any(v is not None and not np.isfinite(v) for v in values.values()):
                raise ValueError("Non-finite metric")
            row.update(status="ok", seconds=receipt["seconds"], **values)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            row.update(status="error", error=str(exc))
        rows.append(row)
    good = [row for row in rows if row["status"] == "ok"]
    if not good:
        for row in rows[:5]:
            print(f"{row['id']}: {row.get('error')}")
        raise ValueError("No evaluable predictions; existing CSVs were preserved")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["arm"], row["target"])].append(row)
    summaries = []
    metadata = {
        "id",
        "benchmark",
        "model",
        "system",
        "arm",
        "seed",
        "target",
        "status",
        "error",
    }
    for (model, arm, target), members in groups.items():
        summary = {
            "model": model,
            "arm": arm,
            "target": target,
            "expected": len(members),
            "valid": sum(row["status"] == "ok" for row in members),
        }
        for metric in sorted(set().union(*(set(row) for row in members)) - metadata):
            values = [
                row[metric]
                for row in members
                if row["status"] == "ok" and row.get(metric) is not None
            ]
            if values:
                summary[f"{metric}_n"] = len(values)
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
        summaries.append(summary)
    write_csv(root / "results.csv", rows)
    write_csv(root / "summary.csv", summaries)
    write_json(
        root / "evaluation.json",
        {
            "rgi_toolkit": version("rgi-toolkit"),
            "rdkit": version("rdkit"),
            "numpy": version("numpy"),
            "valid": len(good),
            "expected": len(rows),
        },
    )
    print(f"Evaluated {len(good)}/{len(rows)} samples: {root / 'results.csv'}")
    return 0 if len(good) == len(rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    raise SystemExit(evaluate(parser.parse_args().run))


if __name__ == "__main__":
    main()
