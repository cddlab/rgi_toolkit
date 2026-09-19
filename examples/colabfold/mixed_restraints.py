"""Validate all five notebook restraint types in real ColabFold predictions.

Use the pinned ColabFold runtime, then pass its run_alphafold.py with --runner.
Outputs and the reference generated from vanilla stay under --work-dir.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np

from rgi_toolkit.notebook_widgets import RestraintEditor


def config_for(reference):
    """Return native entries, including repeated distance and RMSD restraints."""
    return {
        "verbose": True,
        "distance_restraints_config": [
            {
                "atom_selection1": f"chain {a}",
                "atom_selection2": f"chain {b}",
                "harmonic": {"target_distance": 25},
            }
            for a, b in (("A", "B"), ("B", "C"))
        ],
        "angle_restraints_config": [
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "atom_selection3": "chain C",
                "harmonic": {"target_angle": 90},
            }
        ],
        "custom_restraints_config": [
            {
                "name": "triangle_diagonal",
                "selections": {"A": "chain A", "C": "chain C"},
                "energy": f"(distance(A, C) - {25 * np.sqrt(2)})**2",
            }
        ],
        "rmsd_restraints_config": [
            {
                "ref_cif": str(reference),
                "pairing": "identity",
                "atom_selection_target": f"chain {chain} and name CA",
                "atom_selection_ref": f"chain {chain} and name CA",
                "harmonic": {"target_rmsd": 0},
            }
            for chain in ("A", "B")
        ],
        "conformer_restraints_config": {"plane": {"weight": 1}},
    }


def coordinates(path, ca=False):
    structure = gemmi.read_structure(str(path))
    result = {}
    for chain in structure[0]:
        xyz = np.array(
            [
                [a.pos.x, a.pos.y, a.pos.z]
                for r in chain
                for a in r
                if not ca or a.name == "CA"
            ]
        )
        if not ca:
            assert xyz.size and np.isfinite(xyz).all(), path
        if xyz.size:
            result[chain.name] = xyz
    return result


def measure(path, reference):
    centers = {chain: xyz.mean(axis=0) for chain, xyz in coordinates(path).items()}
    a, b, c = (centers[chain] for chain in "ABC")
    cosine = np.dot(a - b, c - b) / np.linalg.norm(a - b) / np.linalg.norm(c - b)
    result = {
        "AB": float(np.linalg.norm(a - b)),
        "BC": float(np.linalg.norm(b - c)),
        "AC": float(np.linalg.norm(a - c)),
        "angle_ABC": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
    }
    predicted, ref = coordinates(path, ca=True), coordinates(reference, ca=True)
    for chain in "AB":
        p, q = predicted[chain], ref[chain]
        p, q = p - p.mean(axis=0), q - q.mean(axis=0)
        u, _, vt = np.linalg.svd(p.T @ q)
        rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        result[f"RMSD_{chain}"] = float(
            np.sqrt(np.mean(np.sum((p @ rotation - q) ** 2, axis=1)))
        )
    return result


def run(args):
    from colabfold.rgi.notebook import prepare_input

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sequence = "ACDEFGHIK"
    raw = {
        "name": "mixed",
        "dialect": "alphafold3",
        "version": 1,
        "modelSeeds": [42],
        "sequences": [
            {
                "protein": {
                    "id": chain,
                    "sequence": sequence,
                    "templates": [],
                    "unpairedMsa": f">query\n{sequence}\n",
                    "pairedMsa": "",
                }
            }
            for chain in "ABC"
        ]
        + [{"ligand": {"id": "D", "smiles": "O=C(O)/C=C/C(=O)O"}}],
    }
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    summary, reference = {}, None
    for arm in ("vanilla", "guided", "vanilla_again"):
        if arm == "guided":
            editor = RestraintEditor(config=config_for(reference), conformer_chains="D")
            data = prepare_input(
                raw,
                args.model,
                use_rgi=True,
                config=editor.get_config(),
                conformer_chains=editor.get_conformer_chains(),
            )
        elif arm == "vanilla_again":
            data = prepare_input(data, args.model, use_rgi=False)
        else:
            data = copy.deepcopy(raw)
        data["name"] = arm
        input_path = work / f"{arm}.json"
        input_path.write_text(json.dumps(data, indent=2))
        command = [
            sys.executable,
            *(["-m", "colabfold.rgi"] if arm == "guided" else []),
            str(Path(args.runner).resolve()),
            f"--json_path={input_path}",
            f"--model={args.model}",
            "--norun_data_pipeline",
            f"--output_dir={work / 'outputs'}",
            f"--cache_dir={work / 'cache'}",
            "--force_output_dir",
            "--flash_attention_implementation=xla",
            "--num_recycles=1",
            "--num_diffusion_samples=1",
            "--buckets=64",
            "--weights_precision=int8",
        ]
        with (work / f"{arm}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        output = work / "outputs" / arm
        paths = list(output.glob("seed-*_sample-*/*.cif"))
        assert len(paths) == 1, paths
        reference = reference or paths[0]
        summary[arm] = measure(paths[0], reference)
        if arm == "guided":
            report = json.loads((output / "rgi_report.json").read_text())
            inventory = report["seeds"][0]["inventory"]
            assert inventory["distance"] == 2 and inventory["rmsd"] == 2, inventory
            assert inventory["group_angle"] == 1 and inventory["custom"] == 1, inventory
            assert all(
                inventory[key] > 0 for key in ("bond", "angle", "plane", "cistrans")
            )
            values = summary[arm]
            assert abs(values["AB"] - 25) < 0.2 and abs(values["BC"] - 25) < 0.2, values
            assert abs(values["AC"] - 25 * np.sqrt(2)) < 0.2, values
            assert abs(values["angle_ABC"] - 90) < 1, values
            assert max(values["RMSD_A"], values["RMSD_B"]) < 0.1, values
        else:
            assert not (output / "rgi_report.json").exists()
            assert "restraints_config" not in data
        print(arm, json.dumps(summary[arm]), flush=True)
    assert summary["vanilla"] == summary["vanilla_again"], summary
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    print("MIXED RESTRAINT E2E PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--model", default="boltz2")
    parser.add_argument("--work-dir", default=".cache/colabfold-mixed")
    run(parser.parse_args())
