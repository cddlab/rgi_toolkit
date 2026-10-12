"""Evaluate matched arms; retain valid metrics and fail visibly on missing data."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from clashes import INTER_FIELDS, INTRA_FIELDS, intermolecular, intramolecular
from common import (
    ROOT,
    read_json,
    validate_prediction,
    verify_inputs,
    write_csv,
    write_json,
)
from confidence import plddt, ptm
from metrics import EZ_METHOD, conformer_metrics, predicted_ligand, state_metrics
from pose import POSE_SETTINGS, ligand_pose_metrics
from quality import MOLPROBITY_FIELDS, molprobity, settings, tm_scores

PASS_FIELDS = ("chiral_match", "ez_match", "clash_ok", "iclash_ok")
CONFORMER_FIELDS = (
    "ideal_bond_rms_A",
    "ideal_angle_rms_deg",
    "crystal_bond_rms_A",
    "crystal_angle_rms_deg",
    "ligand_rmsd_A",
    "ligand_fit_rmsd_A",
    "protein_fit_rmsd_A",
    "paired_ca_atoms",
    "pocket_ca_atoms",
    "chiral_match",
    "ez_match",
    "chiral_sites",
    "ez_sites",
    *INTER_FIELDS,
    *INTRA_FIELDS,
)
STATE_FIELDS = (
    "distance_A",
    "angle_deg",
    "distance_in_A",
    "distance_out_A",
    "difference_A",
    "rmsd_open_A",
    "rmsd_closed_A",
    "absolute_error",
    "absolute_error_1",
    "absolute_error_2",
    "tmscore_1",
    "tmscore_2",
    *MOLPROBITY_FIELDS,
)


def summarize(rows, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["arm"], row["target"])].append(row)
    summaries = []
    for (model, arm, target), members in groups.items():
        count = len(members)
        valid = sum(row["status"] == "ok" for row in members)
        predicted = sum(row.get("prediction_status") == "ok" for row in members)
        summary = {
            "model": model,
            "arm": arm,
            "target": target,
            "expected": count,
            "predicted": predicted,
            "valid": valid,
            "prediction_success_rate": predicted / count,
            "evaluation_success_rate": valid / count,
        }
        for metric in metrics:
            values = [row[metric] for row in members if row.get(metric) is not None]
            summary[f"{metric}_n"] = len(values)
            summary[f"{metric}_missing"] = count - len(values)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_median"] = float(np.median(values)) if values else None
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
            if metric in PASS_FIELDS:
                summary[f"{metric}_passed"] = sum(bool(v) for v in values)
                summary[f"{metric}_pass_rate"] = (
                    float(np.mean(values)) if values else None
                )
        summaries.append(summary)
    return summaries


def evaluate(root, *, config_path=None, skip=(), make_report=True):
    root = Path(root)
    manifest = verify_inputs(root)
    protocol = yaml.safe_load((root / "protocol.yaml").read_text())
    config = settings(config_path)
    conformer = manifest["benchmark"] == "conformer"
    metrics = [
        "seconds",
        "plddt",
        "ptm",
        *(CONFORMER_FIELDS if conformer else STATE_FIELDS),
    ]
    rows = []
    for relative in manifest["jobs"]:
        folder = (root / relative).parent
        job = read_json(folder / "job.json")
        row = {
            key: job[key]
            for key in ("id", "benchmark", "model", "system", "arm", "seed")
        }
        row.update({key: job.get(key) for key in ("msa_depth", "msa_seed", "msa_rows")})
        row["target"] = json.dumps(job["target"])
        row.update(dict.fromkeys(metrics))
        rows.append(row)
        try:
            receipt = read_json(folder / "complete.json")
            path = folder / "prediction.cif"
            atoms, xyz = validate_prediction(path, job)
            row.update(prediction_status="ok", status="ok", seconds=receipt["seconds"])
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            row.update(prediction_status="error", status="error", error=str(exc))
            continue

        def collect(group, function, skip_group=None):
            if (skip_group or group) in skip:
                row[f"{group}_status"] = "skipped"
                return
            try:
                values = function()
                if any(v is not None and not np.isfinite(v) for v in values.values()):
                    raise ValueError("Non-finite metric")
                row.update(values)
                row[f"{group}_status"] = "ok"
            except Exception as exc:
                row.update(status="partial")
                row[f"{group}_status"] = "error"
                row[f"{group}_error"] = f"{type(exc).__name__}: {exc}"

        collect(
            "geometry",
            lambda: (
                conformer_metrics(atoms, xyz, job, folder)
                if conformer
                else state_metrics(atoms, xyz, job, protocol, folder)
            ),
        )
        collect("plddt", lambda: plddt(path), "confidence")
        collect("ptm", lambda: ptm(folder, job["model"]), "confidence")
        if conformer:
            collect(
                "pose",
                lambda: ligand_pose_metrics(
                    predicted_ligand(atoms, xyz, job), job, folder
                ),
            )
            collect(
                "inter_clash",
                lambda: intermolecular(predicted_ligand(atoms, xyz, job), path),
                "clashes",
            )
            collect(
                "intra_clash",
                lambda: intramolecular(predicted_ligand(atoms, xyz, job)),
                "clashes",
            )
        else:
            logs = folder / "evaluation"

            def tm():
                references = [
                    {**ref, "file": root / "data" / ref["file"]}
                    for ref in protocol.get("tm_references", [])
                ]
                values, output = tm_scores(path, references, config)
                logs.mkdir(exist_ok=True)
                (logs / "usalign.txt").write_text(output)
                return values

            def mp():
                values, output, retry = molprobity(path, config)
                logs.mkdir(exist_ok=True)
                (logs / "molprobity.txt").write_text(output)
                row["molprobity_bond_limit_retry"] = retry
                return values

            collect("usalign", tm)
            collect("molprobity", mp)
    predicted = sum(row.get("prediction_status") == "ok" for row in rows)
    if not predicted:
        for row in rows[:5]:
            print(f"{row['id']}: {row.get('error')}")
        raise ValueError("No evaluable predictions; existing CSVs were preserved")
    good = sum(row["status"] == "ok" for row in rows)
    write_csv(root / "results.csv", rows)
    write_csv(root / "summary.csv", summarize(rows, metrics))
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    diff = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    write_json(
        root / "evaluation.json",
        {
            "ez_method": EZ_METHOD,
            "ligand_pose": POSE_SETTINGS,
            "versions": {
                name: version(name)
                for name in ("rgi-toolkit", "rdkit", "numpy", "posebusters")
            },
            "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "git_dirty": bool(diff.stdout.strip()) if diff.returncode == 0 else None,
            "tools": {key: Path(config[key]).name for key in ("usalign", "molprobity")},
            "timeout": config["timeout"],
            "skipped": sorted(skip),
            "tm_references": protocol.get("tm_references", []),
            "predicted": predicted,
            "valid": good,
            "expected": len(rows),
        },
    )
    if make_report:
        from report import report

        report(root)
    print(f"Evaluated {good}/{len(rows)} samples completely: {root / 'results.csv'}")
    return 0 if good == len(rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", type=Path)
    for name in ("usalign", "molprobity", "clashes", "confidence"):
        parser.add_argument(f"--skip-{name}", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    skip = [
        key
        for key in ("usalign", "molprobity", "clashes", "confidence")
        if getattr(args, f"skip_{key}")
    ]
    raise SystemExit(
        evaluate(
            args.run, config_path=args.config, skip=skip, make_report=not args.no_report
        )
    )


if __name__ == "__main__":
    main()
