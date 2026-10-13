"""Native input formats for the six predictor integrations."""

import copy

import yaml

from arms import uses_rgi
from common import write_json


def write_input(folder, job):
    model, seq, msa = job["model"], job["sequence"], job["msa"]
    rc = copy.deepcopy(job["restraints"])
    smiles = job.get("smiles")
    enabled = uses_rgi(job["arm"]) and bool(smiles)
    if model == "boltz_restr":
        obj = {
            "version": 1,
            "sequences": [{"protein": {"id": "A", "sequence": seq, "msa": msa}}],
        }
        if smiles:
            obj["sequences"].append(
                {
                    "ligand": {
                        "id": "B",
                        "smiles": smiles,
                        "conformer_restraints": enabled,
                    }
                }
            )
        if rc:
            obj["restraints_config"] = rc
        (folder / "input.yaml").write_text(yaml.safe_dump(obj, sort_keys=False))
    elif model == "protenix_restr":
        obj = {
            "name": "prediction",
            "sequences": [
                {"proteinChain": {"sequence": seq, "count": 1, "unpairedMsaPath": msa}}
            ],
        }
        if smiles:
            obj["sequences"].append(
                {
                    "ligand": {
                        "ligand": smiles,
                        "count": 1,
                        "conformer_restraints": enabled,
                    }
                }
            )
        if rc:
            obj["restraints_config"] = rc
        write_json(folder / "input.json", [obj])
    elif model == "alphafold3_restr":
        obj = {
            "name": "prediction",
            "dialect": "alphafold3",
            "version": 2,
            "modelSeeds": [job["seed"]],
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": seq,
                        "unpairedMsaPath": msa,
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            ],
        }
        if smiles:
            obj["sequences"].append(
                {
                    "ligand": {
                        "id": "B",
                        "smiles": smiles,
                        "conformer_restraints": enabled,
                    }
                }
            )
        if rc:
            obj["restraints_config"] = rc
        write_json(folder / "input.json", obj)
    elif model == "openfold-3_restr":
        query = {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": seq,
                    "main_msa_file_paths": [msa],
                }
            ]
        }
        if smiles:
            query["chains"].append(
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["B"],
                    "smiles": smiles,
                    "conformer_restraints": enabled,
                }
            )
        if rc:
            query["restraints_config"] = rc
        write_json(
            folder / "input.json",
            {"seeds": [job["seed"]], "queries": {"prediction": query}},
        )
    elif model == "chai-lab_restr":
        fasta = f">protein|name=A\n{seq}\n"
        if smiles:
            fasta += f">ligand|name=B\n{smiles}\n"
        (folder / "input.fasta").write_text(fasta)
        if rc:
            if smiles:
                rc["conformer_restraints"] = {"B": True}
            (folder / "restraints.yaml").write_text(yaml.safe_dump(rc, sort_keys=False))
    elif model != "esm_restr":
        raise ValueError(f"Unknown model: {model}")
