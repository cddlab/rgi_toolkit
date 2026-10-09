# ESMFold2 — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

ESMFold2 + [RGI-toolkit](https://github.com/cddlab/rgi_toolkit) restraint-guided inference. `restraints_config` reference & atom-selection DSL: [`config.md`](config.md).

ESMFold2 supports single-sequence input and optional MSAs supplied through
`ProteinInput(msa=...)`. The complete example below uses single-sequence input.
The ESMFold2 scripts in [`examples/`](../examples/README.md#msa-inputs) require a
full ColabFold A3M via `MSA_A3M`.

Use **`esm_restr` on `rgi-integration`**. Since ESM 3.4.1, the native model and
diffusion loop live in `esm/models/esmfold2/`. `ESMFold2InputBuilder.fold` builds
one adapter and `CombinedRestraints` per input, then passes it through the model
to `DiffusionStructureHead.sample` in `layers.py`. The hook minimizes the
denoised coordinates before rigid alignment and integration, with the pre-churn
sigma and a zero-based step index.

Both native `EsmFold2Model` and `EsmFold2ExperimentalModel` use this hook.
The optional `EsmFold2HFAdapter` does not support RGI; use a native model when
passing `restraints_config`. The historical `transformers_restr` fork is no
longer needed by this integration.

## Installation

ESMFold2 uses a **pixi** environment. `esm_restr`'s `pyproject.toml` installs
the native model and `rgi_toolkit`. The upstream package selects PyTorch 2.11
and Transformers 4.57.6; Transformers supplies shared utilities rather than
the RGI sampling loop. Use a CUDA device compatible with the installed PyTorch
build for production inference.

```bash
git clone --branch rgi-integration https://github.com/cddlab/esm_restr.git
cd esm_restr
pixi install
```

### Co-development

Clone `RGI-toolkit` alongside `esm_restr`, then install both editably in a uv
environment. The native ESM source contains the sampling hook, so edits take
effect directly:

```bash
uv venv --python 3.12
uv pip install -e . -e ../RGI-toolkit
uv run --no-project python restr_example.py
```

## Configuration

ESMFold2's API is **Pythonic**: `restraints_config` is a plain **Python dict** passed to
`ESMFold2InputBuilder().fold(model, spi, restraints_config=...)` — not a YAML/JSON sidecar. The dict
schema is identical to the other tools.

Conformer restraints are **per-input opt-in**: set `conformer_restraints=True` on
each `LigandInput` that needs conformer restraints. Ligands left at the
default (`False`) remain unrestrained.

A ligand = one token/atom, so `token_bonds` carries intra-ligand connectivity and bond ORDERS ride
on `ChainInfo.ligand_bond_orders` (CCD via `get_ligand_ccd_bonds`, SMILES via Kekulized 3-tuples) —
so the conformer cistrans term works for both CCD and SMILES ligands.

The `RESTRAINTS_CONFIG` dict below shows the documented restraint types with concrete values (distance
/ angle / dihedral / conformer / RMSD, plus config-only `custom`); see [`config.md`](config.md) for the
alternatives (restraint types and the RMSD `atom_selection_ref` / `atom_selection_target`
shorthand). `resid` is the **per-chain 1-based ordinal** (qualify protein groups with `chain A and
(...)`). There is **no top-level `start_sigma`**.

## Complete example (Python script)

Save this as `restr_example.py`. It folds QBP with its GLN ligand and combines
distance, angle, dihedral, ligand conformer, reference RMSD, and custom restraints.
Chain IDs are explicit: protein A and ligand B.

```python
"""ESMFold2 RGI (restraint-guided inference) example via rgi_toolkit."""

from __future__ import annotations

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    LigandInput,
    ProteinInput,
    StructurePredictionInput,
)

QBP = (
    "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKK"
    "AIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAV"
    "LHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK"
)

RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,
    "method": "CG",
    "max_iter": 1000,
    "distance_restraints_config": [
        {
            "atom_selection1": "chain A and ((resid 5 to 84) or (resid 186 to 224))",
            "atom_selection2": "chain A and (resid 90 to 180)",
            "start_sigma": 99999999,
            "stop_sigma": -1,
            "move": "both",
            "weight": 1.0,
            "harmonic": {"target_distance": 25.0},
        }
    ],
    "angle_restraints_config": [
        {
            "atom_selection1": "chain A and (resid 5 to 84)",
            "atom_selection2": "chain A and (resid 90 to 180)",
            "atom_selection3": "chain A and (resid 186 to 224)",
            "start_sigma": 99999999,
            "stop_sigma": -1,
            "move": "1,3",
            "weight": 1.0,
            "harmonic": {"target_angle": 90.0},
        }
    ],
    "dihedral_restraints_config": [
        {
            "atom_selection1": "chain A and (resid 5 to 50)",
            "atom_selection2": "chain A and (resid 51 to 100)",
            "atom_selection3": "chain A and (resid 101 to 150)",
            "atom_selection4": "chain A and (resid 151 to 224)",
            "start_sigma": 99999999,
            "stop_sigma": -1,
            "move": "1,4",
            "weight": 1.0,
            "harmonic": {"target_dihedral": 180.0},
        }
    ],
    "conformer_restraints_config": {
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "bond": {"weight": 1.0, "slack": 0.0},
        "angle": {"weight": 1.0, "slack": 0.0},
        "chiral": {"weight": 1.0, "slack": 0.0},
        "cistrans": {"weight": 1.0, "slack": 0.0},
        "vdw": {"weight": 1.0},
    },
    "rmsd_restraints_config": [
        {
            "ref_pdb": "rmsd_ref.pdb",
            "harmonic": {"target_rmsd": 0.0},
            "weight": 1.0,
            "start_sigma": 99999999,
            "stop_sigma": 1.0,
            "pairing": "align",
            "best_effort": True,
            "atom_selection_ref_fit": "chain A and (resid 5 to 220)",
            "atom_selection_target_fit": "chain A and (resid 5 to 220)",
            "atom_selection_ref_calc": "chain A and (resid 90 to 180)",
            "atom_selection_target_calc": "chain A and (resid 90 to 180)",
        }
    ],
    # Define your OWN restraint as a formula (no Python beyond this dict). This one keeps
    # both lobe-halves (L1, L2) equidistant from the central domain (H) — a difference of
    # two distances, which no single built-in restraint can express. See config.md for the
    # expression vocabulary.
    "custom_restraints_config": [
        {
            "name": "equidistant",
            "energy": "(distance(L1, H) - distance(L2, H))**2",
            "selections": {
                "L1": "chain A and (resid 5 to 84)",
                "L2": "chain A and (resid 186 to 224)",
                "H": "chain A and (resid 90 to 180)",
            },
            "start_sigma": 99999999,
            "stop_sigma": -1,
            "weight": 1.0,
        }
    ],
}

def main() -> None:
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2").cuda()
    model.train(False)  # inference / eval mode

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=QBP),
            LigandInput(id="B", ccd=["GLN"], conformer_restraints=True),  # glutamine — QBP's natural ligand
        ]
    )

    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=3,
        num_sampling_steps=200,
        seed=0,
        restraints_config=RESTRAINTS_CONFIG,
    )
    with open("out_esm.cif", "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote out_esm.cif")

if __name__ == "__main__":
    main()
```

## Run

Save as `run_restr_example.sh` in `esm_restr` and run it on a GPU machine
(`bash run_restr_example.sh`):

```bash
#!/bin/bash
# ESMFold2 RGI example runner. Run on a CUDA compute node.
set -e

pixi install
pixi run python restr_example.py
```

## Verify results

With `verbose: True`, the `setup` log prints `built spec: n_active=.. bonds=.. ... distances=..
rmsd=.. group_angle=.. group_dihedral=..` — confirm the configured terms are present.
Check final residuals and measure the requested geometry from `out_esm.cif`.
If the hook is missing, confirm that `esm_restr` is on `rgi-integration` and
that the model comes from `esm.models.esmfold2`. The sampler and API tests in
`esm_restr/tests/models/` cover unchanged no-op sampling, pre-churn gates,
per-input isolation, and distance optimization for multiple samples.
