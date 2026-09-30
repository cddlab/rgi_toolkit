"""ESMFold2 RGI example -- custom dist-diff: Delta D = D_in - D_out -> 0.8 A (DgoT).

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

SEQUENCE = "RRRYLTLVMIFITVVICYVDRANLAVASAHIQEEFGITKAEMGYVFSAFAWLYTLCQIPGGWFLDRVGSRVTYFIAIFGWSVATLFQGFATGLMSLIGLRAITGIFEAPAFPTNNRMVTSWFPEHERASAVGFYTSGQFVGLAFLTPLLIWIQEMLSWHWVFIVTGGIGIIWSLIWFKVYQPPRLTKGISKAELDYIRDGGGLVDGDAPVKKEARQPLTAKDWKLVFHRKLIGVYLGQFAVASTLWFFLTWFPNYLTQEKGITALKAGFMTTVPFLAAFVGVLLSGWVADLLVRKGFSLGFARKTPIICGLLISTCIMGANYTNDPMMIMCLMALAFFGNGFASITWSLVSSLAPMRLIGLTGGVFNFAGGLGGITVPLVVGYLAQGYGFAPALVYISAVALIGALSYILLVGDVKRVG"  # noqa: E501

RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,
    "max_iter": 100,
    "method": "CG",
    "custom_restraints_config": [
        {
            "name": "double_distres",
            "energy": "((distance(A, B) - distance(C, D)) - 0.8)**2",
            "selections": {
                "A": "chain A and ((resid 1 to 17) or "
                "(resid 52 to 66) or (resid 69 to "
                "78) or (resid 108 to 121) or "
                "(resid 125 to 139) or (resid 168 "
                "to 178))",
                "B": "chain A and ((resid 229 to 241) or "
                "(resid 280 to 295) or (resid 298 "
                "to 311) or (resid 341 to 354) or "
                "(resid 358 to 371) or (resid 400 "
                "to 412))",
                "C": "chain A and ((resid 18 to 34) or "
                "(resid 38 to 51) or (resid 79 to "
                "87) or (resid 92 to 107) or (resid "
                "140 to 155) or (resid 158 to 167))",
                "D": "chain A and ((resid 242 to 250) or "
                "(resid 265 to 279) or (resid 312 "
                "to 323) or (resid 325 to 340) or "
                "(resid 372 to 386) or (resid 388 "
                "to 399))",
            },
            "start_sigma": 99999999,
            "weight": 1.0,
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_custom.cif")
    with open(out, "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote", out)


if __name__ == "__main__":
    main()
