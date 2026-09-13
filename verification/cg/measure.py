"""Independent measurements of exported predictor structures and paired timing."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np


def atoms(path):
    structure = gemmi.read_structure(str(path))
    rows = []
    for chain in structure[0]:
        ordinal = 0
        for residue in chain:
            if not any(a.name == "CA" for a in residue):
                continue
            ordinal += 1
            for atom in residue:
                if atom.element.name not in ("H", "D"):
                    rows.append(
                        (
                            chain.name,
                            ordinal,
                            atom.name,
                            residue.name,
                            np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                        )
                    )
    if not rows or not np.isfinite(np.stack([r[-1] for r in rows])).all():
        raise ValueError("Missing or nonfinite protein coordinates")
    return rows


def select(rows, expression):
    """Independent parser for the maintained fixtures' chain/range/name selections."""
    ranges = [
        (int(first), int(last or first))
        for first, last in re.findall(r"resid\s+(\d+)(?:\s+to\s+(\d+))?", expression)
    ]
    chain = re.search(r"chain\s+(\w+)", expression)
    name = re.search(r"name\s+(\w+)", expression)
    chosen = [
        r[-1]
        for r in rows
        if (not chain or r[0] == chain[1])
        and (not name or r[2] == name[1])
        and (not ranges or any(lo <= r[1] <= hi for lo, hi in ranges))
    ]
    if not chosen:
        raise ValueError(f"Empty exported-structure selection: {expression}")
    return np.stack(chosen)


def aligned_rmsd(rows, reference):
    from Bio.Align import PairwiseAligner, substitution_matrices

    target = [r for r in rows if r[0] == "A" and r[2] == "CA"]
    ref = [r for r in atoms(reference) if r[0] == "A" and r[2] == "CA"]

    def sequence(rs):
        return "".join(
            gemmi.find_tabulated_residue(r[3]).one_letter_code.upper() for r in rs
        )

    aligner = PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-10,
        extend_gap_score=-0.5,
    )
    alignment = aligner.align(sequence(target), sequence(ref))[0]
    paired = [
        (i, j)
        for (lo, hi), (rl, rh) in zip(*alignment.aligned, strict=True)
        for i, j in zip(range(lo, hi), range(rl, rh), strict=True)
    ]
    if len(paired) < 200:
        raise ValueError("Too few aligned QBP reference atoms")
    x = np.stack([target[i][-1] for i, _ in paired])
    y = np.stack([ref[j][-1] for _, j in paired])
    x, y = x - x.mean(0), y - y.mean(0)
    u, _, vh = np.linalg.svd(x.T @ y)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vh)]) @ vh
    return float(np.sqrt(np.mean(np.sum((x @ rotation - y) ** 2, axis=-1))))


def observe(path, config):
    rows = atoms(path)
    result = {"protein_atoms": len(rows), "finite": True}
    if config is None:
        return result
    for entry in config.get("distance_restraints_config", []):
        centroids = [select(rows, entry[f"atom_selection{i}"]).mean(0) for i in (1, 2)]
        value = float(np.linalg.norm(centroids[0] - centroids[1]))
        result.update(
            distance=value,
            error=abs(value - entry["harmonic"]["target_distance"]),
            tolerance=0.1,
        )
    for entry in config.get("angle_restraints_config", []):
        a, b, c = [select(rows, entry[f"atom_selection{i}"]).mean(0) for i in (1, 2, 3)]
        cosine = np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b))
        value = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        result.update(
            angle=value,
            error=abs(value - entry["harmonic"]["target_angle"]),
            tolerance=1.0,
        )
    for entry in config.get("custom_restraints_config", []):
        a, b, c, d = [select(rows, entry["selections"][key]).mean(0) for key in "ABCD"]
        value = float(np.linalg.norm(a - b) - np.linalg.norm(c - d))
        result.update(distance_difference=value, error=abs(value), tolerance=0.1)
    if config.get("rmsd_restraints_config"):
        result["released_rmsds"] = [
            aligned_rmsd(rows, entry["ref_cif"])
            for entry in config["rmsd_restraints_config"]
        ]
        result["rmsd_released"] = all(
            entry["stop_sigma"] > 0 for entry in config["rmsd_restraints_config"]
        )
    return result


def rmsd_activity(trace):
    """Measure active targets separately from the released final denoising steps."""
    with (trace / "spec.pkl").open("rb") as stream:
        spec = pickle.load(stream)
    terms = spec.rmsd
    records = []
    for capture in sorted(trace.glob("step_*.npz")):
        data = np.load(capture)
        sigma, step = float(data["sigma"]), int(data["step"])
        coordinates = data["after"][..., spec.active_sites, :].reshape(
            -1, spec.n_active, 3
        )
        for i in np.flatnonzero(terms.mask):
            enabled = bool(
                terms.stop_sigma[i] <= sigma <= terms.start_sigma[i]
                and terms.start_step[i] <= step <= terms.stop_step[i]
            )
            fit = terms.fit_mask[i].astype(bool)
            calc = terms.calc_mask[i].astype(bool)
            ref = terms.fit_ref[i, fit]
            for sample in coordinates:
                x = sample[terms.fit_idx[i, fit]]
                center, ref_center = x.mean(0), ref.mean(0)
                u, _, vh = np.linalg.svd((x - center).T @ (ref - ref_center))
                rotation = u @ np.diag([1, 1, np.linalg.det(u @ vh)]) @ vh
                measured = (sample[terms.calc_idx[i, calc]] - center) @ rotation
                delta = measured - (terms.calc_ref[i, calc] - ref_center)
                value = float(np.sqrt(np.mean(np.sum(delta**2, axis=-1))))
                records.append(
                    dict(
                        step=step, sigma=sigma, term=int(i), active=enabled, rmsd=value
                    )
                )
    active = [r for r in records if r["active"]]
    released = [r for r in records if not r["active"]]
    active_ok = bool(active) and all(
        abs(r["rmsd"] - terms.target1[r["term"]]) <= 0.1 for r in active
    )
    released_ok = bool(released)
    for capture in trace.glob("step_*.npz"):
        data = np.load(capture)
        if float(data["sigma"]) < float(np.min(terms.stop_sigma)):
            released_ok &= np.array_equal(data["before"], data["after"])
    return dict(
        rmsd_capture_measurements=records,
        active_targets_ok=active_ok,
        released_steps_unchanged=bool(released_ok),
    )


def conformer_inventory(trace, case):
    with (trace / "spec.pkl").open("rb") as stream:
        spec = pickle.load(stream)
    counts = {}
    for name in ("bond", "angle", "chiral", "cistrans", "plane", "vdw"):
        term = getattr(spec, name)
        counts[name] = (
            0 if term is None else int(np.sum((term.mask > 0) & (term.weight > 0)))
        )
    fixed = spec.vdw_config
    counts["fixed_queries"] = 0 if fixed is None else len(fixed.ligand_local)
    counts["fixed_partners"] = 0 if fixed is None else len(fixed.background_global)
    passed = (
        counts["vdw"] > 0
        and counts["fixed_queries"] > 0
        and counts["fixed_partners"] > 0
    )
    if case == "distance_conformer":
        passed &= all(counts[name] > 0 for name in ("bond", "angle", "chiral"))
    else:
        passed &= all(
            counts[name] == 0
            for name in ("bond", "angle", "chiral", "cistrans", "plane")
        )
    return counts, passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--timing-report", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = Path(manifest["output"])
    report, timings = [], defaultdict(lambda: defaultdict(list))
    timing_jobs = defaultdict(set)
    patterns = {
        "af3": "**/seed-*_sample-0/*_model.cif",
        "chai": "**/pred.model_idx_0.cif",
        "of3": "**/*_model.cif",
        "esm": "model.cif",
    }
    for run in manifest["runs"]:
        if args.timing_only and run["case"] != "distance":
            continue
        for version in ("old", "new") if run["case"] == "distance" else ("new",):
            target = (
                output
                / ("timing" if args.timing_only else "results")
                / run["model"]
                / run["case"]
                / str(run["seed"])
                / version
            )
            record = dict(run, version=version, passed=False)
            result_path = target / "result.json"
            if not result_path.exists():
                record["state"] = "pending"
                report.append(record)
                continue
            process = json.loads(result_path.read_text())
            record.update(
                state="finished",
                wall_seconds=process["wall_seconds"],
                returncode=process["returncode"],
            )
            if process["returncode"]:
                record["state"] = "process_failed"
            else:
                try:
                    provenance = target / "trace" / "source.json"
                    if provenance.exists():
                        imported = json.loads(provenance.read_text())
                        expected_hash = manifest.get(
                            "source_sha256"
                            if version == "new"
                            else "baseline_source_sha256"
                        )
                        record["imported_source"] = imported
                        if expected_hash and imported["source_sha256"] != expected_hash:
                            raise ValueError(
                                "Predictor imported a different engine snapshot"
                            )
                    pattern = patterns.get(
                        run["model"],
                        "**/*model_0.cif"
                        if run["model"].startswith("boltz")
                        else "**/*sample_0.cif",
                    )
                    cifs = sorted((target / "prediction").glob(pattern))
                    if not cifs:
                        raise FileNotFoundError("No final prediction CIF")
                    if (
                        run["model"] == "of3"
                        and f"seed_{run['seed']}" not in cifs[0].parts
                    ):
                        raise ValueError(
                            "OpenFold-3 did not use the requested model seed"
                        )
                    config = json.loads(
                        Path(run["input"]).with_name("restraints.json").read_text()
                    )
                    record.update(observe(cifs[0], config), cif=str(cifs[0]))
                    log = (target / "predict.log").read_text()
                    expected = {
                        "distance": "n_distance=1",
                        "distance_conformer": "n_distance=1",
                        "distance_vdw": "n_distance=1",
                        "angle": "n_group_angle=1",
                        "rmsd": "n_rmsd=2",
                        "custom": "n_custom=1",
                    }.get(run["case"])
                    inventory = expected is None or expected in log
                    if run["case"] in ("distance_conformer", "distance_vdw"):
                        inventory = inventory and "conformer=True" in log
                        counts, populated = conformer_inventory(
                            target / "trace", run["case"]
                        )
                        record["conformer_inventory"] = counts
                        inventory &= populated
                    if run["case"] == "rmsd":
                        inventory = inventory and "conformer=False" in log
                        record.update(rmsd_activity(target / "trace"))
                    record["inventory_ok"] = inventory
                    record["passed"] = inventory and record.get(
                        "error", 0
                    ) <= record.get("tolerance", 0)
                    if run["case"] == "rmsd":
                        record["passed"] &= (
                            record["active_targets_ok"]
                            and record["released_steps_unchanged"]
                        )
                    trace_path = target / "trace" / "trace.json"
                    if trace_path.exists():
                        trace = json.loads(trace_path.read_text())
                        events = [
                            e for e in trace["events"] if e["stage"] == "minimize"
                        ]
                        record["rgi_seconds"] = (
                            sum(e["seconds"] for e in events) if events else None
                        )
                        record["rgi_calls"] = len(events) if events else None
                        record["rgi_statuses"] = {
                            str(status): sum(e.get("status") == status for e in events)
                            for status in sorted(
                                {
                                    e.get("status")
                                    for e in events
                                    if e.get("status") is not None
                                }
                            )
                        }
                        record["diffusion_seconds"] = trace["diffusion_seconds"]
                    if run["case"] == "distance" and record["passed"]:
                        if process.get("instrumented") != (not args.timing_only):
                            raise ValueError(
                                "Prediction instrumentation mode does not match the measurement"
                            )
                        timings[run["model"]][version].append(record["wall_seconds"])
                        timing_jobs[run["model"]].add(process["job"])
                except Exception as error:
                    record.update(
                        state="measurement_failed",
                        passed=False,
                        error_message=str(error),
                    )
            report.append(record)
    timing_report = {}
    for model, versions in timings.items():
        if all(len(versions[v]) == 3 for v in ("old", "new")):
            old, new = [float(np.median(versions[v])) for v in ("old", "new")]
            same_allocation = len(timing_jobs[model]) == 1
            timing_report[model] = {
                "old_median_seconds": old,
                "new_median_seconds": new,
                "ratio": new / old,
                "same_allocation": same_allocation,
                "passed": same_allocation and new <= 1.2 * old,
            }
    summary = {
        "source_sha256": manifest["source_sha256"],
        "total": len(report),
        "passed": sum(r["passed"] for r in report),
        "pending": sum(r["state"] == "pending" for r in report),
        "timing": timing_report,
        "runs": report,
    }
    if args.timing_report:
        independent = json.loads(args.timing_report.read_text())
        if (
            independent["source_sha256"] != manifest["source_sha256"]
            or independent["passed"] != 54
            or independent["total"] != 54
        ):
            raise ValueError(
                "Timing report must cover all successful paired predictions"
            )
        summary["instrumented_timing"] = timing_report
        timing_report = summary["timing"] = independent["timing"]
        summary["timing_report"] = str(args.timing_report.resolve())
    filename = "timing_measurements.json" if args.timing_only else "measurements.json"
    (output / filename).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    if args.require_complete and (
        summary["passed"] != summary["total"]
        or len(timing_report) != 9
        or not all(v["passed"] for v in timing_report.values())
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
