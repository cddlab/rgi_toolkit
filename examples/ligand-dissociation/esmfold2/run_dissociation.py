"""Sample MDM2-p53 using full A3Ms supplied through MSA_A3M_A and MSA_A3M_B."""

import argparse
import json
import os
from pathlib import Path

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    ProteinInput,
    StructurePredictionInput,
)
from esm.utils.msa import MSA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    input_path = args.input.resolve()
    data = json.loads(input_path.read_text())
    proteins = []
    for chain in data["sequences"]:
        msa_variable = f"MSA_A3M_{chain['id']}"
        msa_path = os.environ.get(msa_variable)
        if not msa_path or not Path(msa_path).is_file():
            raise ValueError(
                f"Set {msa_variable} to the full ColabFold A3M for chain {chain['id']}."
            )
        msa = MSA.from_a3m(path=msa_path, remove_insertions=True)
        if msa.depth < 1 or msa.query != chain["sequence"]:
            raise ValueError(f"The MSA query does not match chain {chain['id']}.")
        proteins.append(
            ProteinInput(id=chain["id"], sequence=chain["sequence"], msa=msa)
        )
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device="cuda")
    model.train(False)
    result = ESMFold2InputBuilder().fold(
        model,
        StructurePredictionInput(sequences=proteins),
        num_loops=20,
        num_sampling_steps=200,
        seed=0,
        msa_max_depth=1024,
        msa_column_mask_rate=0.1,
        restraints_config=data["restraints_config"],
    )
    output_dir = input_path.parent / "out" / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.cif").write_text(result.complex.to_mmcif())


if __name__ == "__main__":
    main()
