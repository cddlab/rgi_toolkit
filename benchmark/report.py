"""Create comparison tables, figures, and a portable manuscript data bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from common import read_json, write_csv, write_json

PASS_FIELDS = {"chiral_match", "ez_match", "clash_ok", "iclash_ok"}
LABELS = {
    "ideal_bond_rms_A": "Bond RMS deviation from ideal (Å)",
    "ideal_angle_rms_deg": "Angle RMS deviation from ideal (degrees)",
    "crystal_bond_rms_A": "Bond RMS deviation from crystal (Å)",
    "crystal_angle_rms_deg": "Angle RMS deviation from crystal (degrees)",
    "ligand_rmsd_A": "Ligand-aligned crystal RMSD (Å)",
    "chiral_match": "R/S agreement rate",
    "ez_match": "E/Z agreement rate",
    "clash_ok": "Protein-ligand clash pass rate",
    "iclash_ok": "Internal clash pass rate",
    "clash_rel": "Minimum protein-ligand distance / VdW radius sum",
    "clash_min": "Minimum protein-ligand distance (Å)",
    "iclash_rel": "Minimum internal distance / DG lower bound",
    "iclash_vdw_rel": "Minimum internal distance / VdW radius sum",
    "plddt": "Atom-mean pLDDT",
    "ptm": "pTM",
    "molprobity_score": "MolProbity score",
    "clashscore": "MolProbity clashscore",
    "rama_favored": "Ramachandran favored (%)",
    "rama_outliers": "Ramachandran outliers (%)",
    "rotamer_outliers": "Rotamer outliers (%)",
    "cbeta_deviations": "C-beta deviations",
    "rms_bond": "MolProbity bond RMSD (Å)",
    "rms_angle": "MolProbity angle RMSD (degrees)",
    "absolute_error": "Absolute restraint error",
    "absolute_error_1": "Open-reference RMSD error (Å)",
    "absolute_error_2": "Closed-reference RMSD error (Å)",
}
STATE_VALUES = {
    "distance": "distance_A",
    "angle": "angle_deg",
    "custom": "difference_A",
    "rmsd": "rmsd_open_A",
}


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def numeric(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    if value in ("True", "False"):
        return float(value == "True")
    number = float(value)
    return number if np.isfinite(number) else None


def comparison_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["arm"])].append(row)
    output = []
    for (model, arm), members in sorted(groups.items()):
        for key in LABELS:
            values = [v for row in members if (v := numeric(row, key)) is not None]
            if not any(key in row for row in members):
                continue
            output.append(
                {
                    "model": model,
                    "arm": arm,
                    "metric": key,
                    "expected": len(members),
                    "n": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "median": float(np.median(values)) if values else None,
                    "pass_rate": float(np.mean(values))
                    if values and key in PASS_FIELDS
                    else None,
                }
            )
    return output


def report(root, output=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    output = Path(output) if output else root / "report"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root / "results.csv")
    provenance = read_json(root / "evaluation.json")
    protocol = yaml.safe_load((root / "protocol.yaml").read_text())
    table = comparison_rows(rows)
    write_csv(output / "comparison.csv", table)
    artifacts = []

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            artifacts.append(path.relative_to(output).as_posix())
        plt.close(fig)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["arm"])].append(row)
    for metric, label in LABELS.items():
        values, labels = [], []
        for (model, arm), members in sorted(groups.items()):
            observed = [v for row in members if (v := numeric(row, metric)) is not None]
            if observed:
                labels.append(f"{model}\n{arm}\nN={len(observed)}/{len(members)}")
                values.append(observed)
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(values) * 1.05), 4.5))
        if metric in PASS_FIELDS:
            ax.bar(range(1, len(values) + 1), [np.mean(v) for v in values])
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(1, len(values) + 1), labels)
        else:
            ax.boxplot(values, tick_labels=labels, showmeans=True)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
        save(fig, metric)
    if protocol["kind"] != "conformer":
        for model in sorted({row["model"] for row in rows}):
            fig, ax = plt.subplots(figsize=(5, 5))
            plotted = False
            for (name, arm), members in sorted(groups.items()):
                if name != model:
                    continue
                pairs = [
                    (numeric(r, "tmscore_1"), numeric(r, "tmscore_2")) for r in members
                ]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if pairs:
                    ax.scatter(
                        *np.asarray(pairs).T,
                        label=f"{arm} (N={len(pairs)})",
                        alpha=0.7,
                        s=20,
                    )
                    plotted = True
            if plotted:
                references = protocol["tm_references"]
                ax.set(
                    xlabel=f"TM-score: {references[0]['label']}",
                    ylabel=f"TM-score: {references[1]['label']}",
                    title=model,
                )
                ax.legend()
                save(fig, f"{model}-states")
            else:
                plt.close(fig)
        metric = STATE_VALUES[protocol["kind"]]
        fig, ax = plt.subplots(figsize=(6, 5))
        all_targets = []
        for (model, arm), members in sorted(groups.items()):
            pairs = []
            for row in members:
                target = json.loads(row["target"])
                measured = numeric(row, metric)
                if target is not None and measured is not None:
                    pairs.append(
                        (target[0] if isinstance(target, list) else target, measured)
                    )
            if pairs:
                ax.scatter(
                    *np.asarray(pairs).T, label=f"{model}: {arm}", s=18, alpha=0.7
                )
                all_targets.extend(t for t, _ in pairs)
        if all_targets:
            bounds = [min(all_targets), max(all_targets)]
            ax.plot(bounds, bounds, "--", color="black", linewidth=1)
            ax.set(xlabel="Target", ylabel=metric)
            ax.legend(fontsize=7)
            save(fig, "target-vs-achieved")
        else:
            plt.close(fig)
    lines = [
        f"# {protocol['kind']} benchmark",
        "",
        f"Predictions: {provenance['predicted']}/{provenance['expected']}. Complete evaluations: {provenance['valid']}/{provenance['expected']}.",
        "",
        "Pass rates use measured, applicable cases only. N is reported with the planned count; blanks are never treated as passes. Protein comparison boxes pool targets; summary.csv retains each target separately.",
        "",
        "Ligand RMSD is aligned on ligand atoms. It does not measure receptor-aligned binding-pose recovery.",
        "",
        f"Explicitly skipped evaluations: {', '.join(provenance['skipped']) or 'none'}.",
        "",
        "| Model | Arm | Metric | N / planned | Mean | Median |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in table:
        if not item["n"]:
            continue
        lines.append(
            f"| {item['model']} | {item['arm']} | {item['metric']} | {item['n']} / {item['expected']} | {item['mean']:.5g} | {item['median']:.5g} |"
        )
    lines.extend(["", "## Figures", ""])
    lines.extend(f"![{Path(p).stem}]({p})" for p in artifacts if p.endswith(".png"))
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(
        output / "manifest.json", {"evaluation": provenance, "figures": artifacts}
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manuscript",
        type=Path,
        help="Export manuscript CSV source data to this directory",
    )
    args = parser.parse_args()
    report(args.run, args.output)
    if args.manuscript:
        from manuscript import export

        export(args.run, args.manuscript)


if __name__ == "__main__":
    main()
