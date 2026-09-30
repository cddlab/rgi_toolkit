"""ESMFold2 RGI example -- dual-reference QBP RMSD targets: open 2.65 A, closed 2.65 A.

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
    "max_iter": 1000,
    "method": "CG",
    "rmsd_restraints_config": [
        {
            "ref_cif": "1GGG.cif",
            "atom_selection_ref_fit": "chain A and name CA and resid 1 to 220",
            "atom_selection_target_fit": "chain A and name CA and resid 5 to 224",
            "atom_selection_ref_calc": "chain A and name CA and resid 1 to 220",
            "atom_selection_target_calc": "chain A and name CA and resid 5 to 224",
            "pairing": "align",
            "start_sigma": 99999999,
            "stop_sigma": 1.5,
            "harmonic": {"target_rmsd": 2.65},
        },
        {
            "ref_cif": "1WDN.cif",
            "atom_selection_ref_fit": "chain A and name CA and resid 2 to 221",
            "atom_selection_target_fit": "chain A and name CA and resid 5 to 224",
            "atom_selection_ref_calc": "chain A and name CA and resid 2 to 221",
            "atom_selection_target_calc": "chain A and name CA and resid 5 to 224",
            "pairing": "align",
            "start_sigma": 99999999,
            "stop_sigma": 1.5,
            "harmonic": {"target_rmsd": 2.65},
        },
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_rmsd.cif")
    with open(out, "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote", out)


if __name__ == "__main__":
    main()
