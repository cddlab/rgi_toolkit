"""External structural validation with explicit failures and bounded execution."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from math import isfinite
from pathlib import Path

import gemmi
import yaml

MOLPROBITY_FIELDS = (
    "molprobity_score",
    "clashscore",
    "rama_favored",
    "rama_outliers",
    "rotamer_outliers",
    "cbeta_deviations",
    "rms_bond",
    "rms_angle",
)
MP_LABELS = (
    ("MolProbity score", "molprobity_score"),
    ("Clashscore", "clashscore"),
    ("Ramachandran outliers", "rama_outliers"),
    ("favored", "rama_favored"),
    ("Rotamer outliers", "rotamer_outliers"),
    ("C-beta deviations", "cbeta_deviations"),
    ("RMS(bonds)", "rms_bond"),
    ("RMS(angles)", "rms_angle"),
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"


def settings(path=None):
    result = {"usalign": "USalign", "molprobity": "phenix.molprobity", "timeout": 300}
    if os.environ.get("PHENIX_ENV"):
        result["phenix_env"] = os.environ["PHENIX_ENV"]
    if path:
        supplied = yaml.safe_load(Path(path).read_text()) or {}
        if set(supplied) - (set(result) | {"phenix_env"}):
            raise ValueError("Unknown evaluation setting")
        result.update(supplied)
    for key in ("usalign", "molprobity", "phenix_env"):
        if key not in result:
            continue
        value = os.path.expandvars(os.path.expanduser(str(result[key])))
        if "$" in value:
            raise ValueError(f"Unresolved environment variable in {key}")
        if not Path(value).is_absolute() and (key == "phenix_env" or "/" in value):
            base = Path(path).resolve().parent if path else Path.cwd()
            value = str((base / value).resolve())
        result[key] = value
    result["timeout"] = float(result["timeout"])
    if not isfinite(result["timeout"]) or result["timeout"] <= 0:
        raise ValueError("Evaluation timeout must be finite and positive")
    return result


def parse_usalign(text):
    for line in text.splitlines():
        if re.search(r"normalized by (?:length of )?Structure_1", line):
            match = re.search(rf"TM-score\s*=\s*({NUMBER})", line)
            if match and 0 <= float(match[1]) <= 1:
                return float(match[1])
    raise ValueError("USalign did not report a reference-normalized TM-score")


def tm_scores(prediction, references, config):
    if len(references) != 2:
        raise ValueError("Exactly two TM-score references are required")
    values, logs = {}, []
    for index, reference in enumerate(references, 1):
        path = Path(reference["file"]).resolve()
        if not path.is_file():
            raise ValueError(f"Missing TM-score reference: {path.name}")
        result = subprocess.run(
            [
                config["usalign"],
                str(path),
                str(Path(prediction).resolve()),
                "-chain1",
                reference.get("chain", "A"),
                "-chain2",
                "A",
            ],
            capture_output=True,
            text=True,
            timeout=config["timeout"],
            check=True,
        )
        values[f"tmscore_{index}"] = parse_usalign(result.stdout)
        logs.append(result.stdout)
    return values, "\n".join(logs)


def parse_molprobity(text):
    values = {}
    for line in text.splitlines():
        for label, field in MP_LABELS:
            match = re.search(re.escape(label) + rf"\s*[=:]\s*({NUMBER})", line, re.I)
            if match:
                values[field] = float(match[1])
                break
    missing = set(MOLPROBITY_FIELDS) - values.keys()
    if missing:
        raise ValueError("Missing MolProbity metrics: " + ", ".join(sorted(missing)))
    return values


def molprobity(prediction, config):
    with tempfile.TemporaryDirectory(prefix="rgi-molprobity-") as temporary:
        directory = Path(temporary)
        structure = gemmi.read_structure(str(prediction))
        structure.setup_entities()
        structure.write_pdb(str(directory / "prediction.pdb"))
        command = [config["molprobity"], "prediction.pdb"]
        if config.get("phenix_env"):
            command = [
                "bash",
                "-c",
                'source "$1" >/dev/null && shift && exec "$@"',
                "bash",
                config["phenix_env"],
                *command,
            ]

        def execute(extra=()):
            return subprocess.run(
                command + list(extra),
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=config["timeout"],
            )

        result = execute()
        output = directory / "molprobity.out"
        excessive = re.search(
            rf"Bond distance > max_reasonable_bond_distance:\s*({NUMBER})\s*>",
            result.stderr,
        )
        retry = None
        if not output.exists() and excessive:
            retry = float(excessive[1]) * 1.01
            result = execute(
                [f"pdb_interpretation.max_reasonable_bond_distance={retry:g}"]
            )
        if result.returncode or not output.is_file():
            raise RuntimeError(
                f"MolProbity failed (exit {result.returncode}): {result.stderr[-500:]}"
            )
        text = output.read_text()
        values = parse_molprobity(text)
        return values, text, retry
