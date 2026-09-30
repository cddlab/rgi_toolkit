"""ESMFold2 RGI example -- centroid distance -> 25.0 A (QBP).

Set MSA_A3M to the full ColabFold A3M and run through the esm_restr environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    ProteinInput,
    StructurePredictionInput,
)
from esm.utils.msa import MSA

SEQUENCE = "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK"  # noqa: E501

RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,
    "max_iter": 100,
    "method": "CG",
    "distance_restraints_config": [
        {
            "atom_selection1": "chain A and ((resid 4 to 83) or (resid 185 to 223))",
            "atom_selection2": "chain A and ((resid 89 to 179))",
            "start_sigma": 99999999,
            "harmonic": {"target_distance": 25.0},
        }
    ],
}


def main() -> None:
    msa_path = os.environ.get("MSA_A3M")
    if not msa_path or not Path(msa_path).is_file():
        raise ValueError("Set MSA_A3M to the full ColabFold A3M for this protein.")
    msa = MSA.from_a3m(path=msa_path, remove_insertions=True)
    if msa.depth < 1 or msa.query != SEQUENCE:
        raise ValueError("The MSA query must match the example protein sequence.")
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device="cuda")
    model.train(False)

    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=SEQUENCE, msa=msa)]
    )

    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=20,
        num_sampling_steps=200,
        seed=0,
        msa_max_depth=1024,
        msa_column_mask_rate=0.1,
        restraints_config=RESTRAINTS_CONFIG,
    )
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_distance.cif")
    with open(out, "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote", out)


if __name__ == "__main__":
    main()
