"""Export portable source tables and figures for manuscript authoring."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from common import read_json, write_csv, write_json
from report import numeric, read_rows

TARGET_FIELDS = {
    "distance": "target_distance",
    "angle": "target_angle",
    "custom": "target_k",
    "rmsd": "target_rmsd",
}
GEOMETRY_COLUMNS = {
    "chiral_match": "match",
    "ez_match": "ez_match",
    "ideal_bond_rms_A": "emb_brms",
    "ideal_angle_rms_deg": "emb_arms",
    "crystal_bond_rms_A": "bond_rms",
    "crystal_angle_rms_deg": "angle_rms",
    "ligand_rmsd_A": "ligand_pose_rmsd_A",
    "ligand_fit_rmsd_A": "ligand_fit_rmsd_A",
    "clash_ok": "clash_ok",
    "clash_rel": "clash_rel",
    "clash_min": "clash_min",
    "iclash_ok": "iclash_ok",
    "iclash_rel": "iclash_rel",
    "iclash_vdw_rel": "iclash_vdw_rel",
}
QUALITY_COLUMNS = (
    "plddt",
    "ptm",
    "molprobity_score",
    "clashscore",
    "rama_favored",
    "rama_outliers",
    "rotamer_outliers",
    "cbeta_deviations",
    "rms_bond",
    "rms_angle",
)


def source_tables(rows, benchmark):
    tables = defaultdict(list)
    for row in rows:
        model, arm = row["model"], row["arm"]
        base = {
            "method": model,
            "arm": arm,
            "system_id": row["system"],
            "seed": row["seed"],
            "status": row["status"],
        }
        if benchmark == "conformer":
            converted = {"pdbid": row["system"], **base}
            converted.update(
                {new: numeric(row, old) for old, new in GEOMETRY_COLUMNS.items()}
            )
            tables[f"{model}_{arm}.csv"].append(converted)
        else:
            target = json.loads(row["target"])
            base[TARGET_FIELDS[benchmark]] = (
                target[0] if isinstance(target, list) else target
            )
            if isinstance(target, list):
                base["target_rmsd_closed"] = target[1]
            base.update(
                seed_ms=row.get("msa_seed", ""),
                msa_depth=row.get("msa_depth", ""),
                seed_pd=row["seed"],
            )
            tables[f"{model}_{arm}.csv"].append(
                {
                    **base,
                    **{key: numeric(row, key) for key in ("tmscore_1", "tmscore_2")},
                }
            )
            if benchmark == "custom":
                tables[f"gate_distances_{arm}.csv"].append(
                    {
                        **base,
                        "d_cyto": numeric(row, "distance_in_A"),
                        "d_peri": numeric(row, "distance_out_A"),
                        "achieved_k": numeric(row, "difference_A"),
                    }
                )
            if benchmark == "angle":
                tables[f"achieved_angle_{arm}.csv"].append(
                    {**base, "achieved_angle": numeric(row, "angle_deg")}
                )
        tables[f"quality_{arm}.csv"].append(
            {**base, **{key: numeric(row, key) for key in QUALITY_COLUMNS}}
        )
    return tables


def export(runs, output):
    if isinstance(runs, (str, Path)):
        runs = [runs]
    runs = [Path(run) for run in runs]
    output = Path(output)
    metadata = [read_json(run / "manifest.json") for run in runs]
    names = [item["benchmark"] for item in metadata]
    if len(set(names)) != len(names):
        raise ValueError("Use one run per protocol in each manuscript bundle")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Manuscript output must be empty")
    for run in runs:
        for name in ("results.csv", "summary.csv", "protocol.yaml", "evaluation.json"):
            if not (run / name).is_file():
                raise ValueError(f"Evaluate the run before export: missing {name}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "protocols": {},
        "model_identifiers": "RGI-enabled checkout names",
    }
    for run, name in zip(runs, names, strict=True):
        destination = output / "source_data" / name
        destination.mkdir(parents=True)
        for filename in (
            "results.csv",
            "summary.csv",
            "protocol.yaml",
            "evaluation.json",
        ):
            shutil.copyfile(run / filename, destination / filename)
        for filename, rows in source_tables(
            read_rows(run / "results.csv"), name
        ).items():
            write_csv(destination / filename, rows)
        if (run / "report").is_dir():
            shutil.copytree(run / "report", output / "figures" / name)
        manifest["protocols"][name] = read_json(run / "evaluation.json")
    manifest["files"] = sorted(
        p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
    )
    write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# Manuscript source data\n\n"
        "Per-sample results, target-stratified summaries, frozen protocols, and evaluation provenance are under source_data/. "
        "Each predictor/arm CSV contains TM scores or ligand geometry; quality tables retain all ten protein quality/confidence columns. "
        "Full RGI checkout names identify models in filenames and method columns. "
        "Use the CSV tables and PNG/PDF figures directly in manuscript scripts; consumers using abbreviated model keys need to update their file/model mapping.\n\n"
        "Missing results remain blank with their status. Undefined E/Z geometry fails preservation; ligands without eligible sites remain unscored. "
        "Ligand pose RMSD uses sequence-paired receptor C-alpha atoms within 10 A of the crystal ligand, with symmetry-equivalent ligand maps and no ligand superposition. "
        "Ligand-aligned RMSD is retained as a separate diagnostic. "
        "A bundle may contain selected protocols/models; the manifest records its scope. "
        "It does not include additional experiments, pose-recovery analysis, or automatically rebuild a complete paper.\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.runs, args.output)


if __name__ == "__main__":
    main()
