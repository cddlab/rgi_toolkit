"""Launch one maintained predictor in its own environment (invoked by run.py)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def esm(input_path, output):
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        LigandInput,
        ProteinInput,
        StructurePredictionInput,
    )
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    data = json.loads(Path(input_path).read_text())
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
    sequences = [ProteinInput(id="A", sequence=data["sequence"])]
    if data["ligand"]:
        sequences.append(
            LigandInput(id="B", ccd=["ATP"], conformer_restraints=data["conformer"])
        )
    result = ESMFold2InputBuilder().fold(
        model,
        StructurePredictionInput(sequences=sequences),
        num_loops=20,
        num_sampling_steps=200,
        seed=data["seed"],
        restraints_config=data["restraints_config"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.cif").write_text(result.complex.to_mmcif())


def chai_msa(a3m, output):
    import tempfile

    from chai_lab.data.parsing.msas.aligned_pqt import merge_a3m_in_directory

    with tempfile.TemporaryDirectory() as temporary:
        text = "\n".join(
            line
            for line in Path(a3m).read_text().splitlines()
            if not line.startswith("#")
        )
        (Path(temporary) / "uniref90.a3m").write_text(text + "\n")
        merge_a3m_in_directory(temporary, output_directory=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["esm", "chai-msa"])
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    (esm if args.action == "esm" else chai_msa)(args.input, args.output)
