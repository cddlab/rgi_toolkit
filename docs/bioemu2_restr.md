# BioEmu2 — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

BioEmu2 uses the shared PyTorch RGI engine for distance, angle, dihedral,
conformer, RMSD, custom, and other supported restraints. Configuration fields
and atom-selection syntax are described in [config.md](config.md).

## Installation

Install the predictor fork on its RGI branch with Linux, Python 3.12 or later,
and a supported NVIDIA GPU:

```bash
git clone --branch rgi-integration https://github.com/cddlab/bioemu2_restr.git
cd bioemu2_restr
uv sync --extra cuda
```

The fork requires RGI-toolkit 0.4.7 from the `add_bioemu2` branch. For development
with a sibling toolkit checkout:

```bash
uv pip install --python .venv/bin/python -e ../RGI-toolkit
```

BioEmu2 downloads its checkpoint when needed. See the
[upstream guide](https://github.com/microsoft/bioemu2) for checkpoint and GPU
requirements. Run prediction commands on a GPU compute node.

## Configuration

Add `restraints_config` at the root of a system YAML, beside `entities`.
For conformer restraints, also set `conformer_restraints: true` on each
participating entity. Ligand and polymer entities opt in independently;
the default is false.

Use explicit `asym_ids` for predictable chain selections. `resid` is a
per-chain, one-based token ordinal. Standard polymer residues occupy one
token; atomized ligands and modified residues can occupy several. It is not
the reference structure's author residue number. `index` is the zero-based
row in the padded coordinate tensor; padding is excluded from selections.

Set `start_sigma` and `stop_sigma` on each restraint entry or on the
conformer block. There is no top-level `start_sigma`.

## Complete example (system YAML)

Save this as `restr_example.yaml`. It enables bond, angle, chirality, E/Z,
and van der Waals restraints for a SMILES ligand:

```yaml
entities:
  - entity_type: protein
    sequence: GYDPETGTWG
    asym_ids: A
  - entity_type: ligand
    smiles: "O=C(O)/C=C/[C@@H](O)C"
    asym_ids: L
    conformer_restraints: true
restraints_config:
  gpu: true
  verbose: true
  method: CG
  max_iter: 100
  conformer_restraints_config:
    start_sigma: 1.0
    bond: {weight: 1}
    angle: {weight: 1}
    chiral: {weight: 1}
    cistrans: {weight: 1}
    vdw: {weight: 1}
```

From the predictor checkout:

```bash
uv run --no-sync bioemu2 predict-structure \
  --system restr_example.yaml --out out_restr_example \
  --num-samples 1 --num-steps 200 --sampler edm --seed 0
```

An omitted or null `restraints_config` keeps ordinary sampling. An empty
object adds no restraint terms. Optional `config_path` references resolve
relative to the containing YAML; the runnable examples keep settings inline.

## Runnable examples

The same systems and restraint settings are available for the other models
in each parent directory:

| Example | Input | Run from the RGI-toolkit checkout |
|---|---|---|
| QBP distance, 25.0 Å | [YAML](../examples/distance/bioemu2/qbp_25.00.yaml) | `bash examples/distance/bioemu2/run.sh` |
| ADK angle, 72.85° | [YAML](../examples/angle/bioemu2/adk_72.85.yaml) | `bash examples/angle/bioemu2/run.sh` |
| QBP dual-reference RMSD, 2.65/2.65 Å | [YAML](../examples/rmsd/bioemu2/qbp_2.65.yaml) | `bash examples/rmsd/bioemu2/run.sh` |
| DgoT distance difference, 0.8 Å | [YAML](../examples/custom/dist-diff/bioemu2/dgot_0.80.yaml) | `bash examples/custom/dist-diff/bioemu2/run.sh` |
| MDM2–p53 dissociation, 13.0 Å | [Input and MSA setup](../examples/ligand-dissociation/README.md) | `bash examples/ligand-dissociation/bioemu2/run.sh` |

The first four examples use BioEmu2's ColabFold MSA acquisition. The dissociation
runner loads the supplied chain-A and chain-B A3Ms through the Python API.
All five runners use structure conditioning, one sample, 200 EDM steps,
and seed 0.

## Python API and precomputed MSAs

```python
from pathlib import Path

from bioemu2 import EmuSession, System, get_msa, load_model, save_frames

system = System.from_yaml(
    "../RGI-toolkit/examples/ligand-dissociation/bioemu2/mdm2_p53_13.00.yaml"
)
msa_dir = Path("../RGI-toolkit/examples/ligand-dissociation/msas")
msa = get_msa(
    system,
    {"A": msa_dir / "mdm2_p53.a3m", "B": msa_dir / "p53.a3m"},
)
session = EmuSession(system, model=load_model(), mode="structure", msa=msa)
samples = session.sample(num_samples=1, sampler="edm", seed=0)
save_frames(samples, "out_mdm2_p53")
```

Use `mode="equilibrium"` or `sample-equilibrium` for equilibrium conditioning.
RGI changes the sampling distribution; restrained samples are not an unbiased
equilibrium ensemble.

## Verify results

With `verbose: true`, check nonzero counts in `built spec:` for the requested
restraints. The conformer example has both a tetrahedral center and an E/Z
bond. The dissociation example must build one distance between chains A and B.

Measure the requested geometry in the saved structures. Reference coordinates
and RGI distance targets use angstroms; MDTraj trajectory coordinates use
nanometers. Convert those coordinates before comparing distances. A zero
energy for an absent term does not show that a restraint ran. Configuration
validation checks the schema and selection syntax; it does not establish
successful inference or satisfaction of the final target.

## Integration details

The adapter lives in `rgi_toolkit.bioemu2`. The predictor fork extracts source
chemistry and attaches metadata from the exact processed structure. For
SMILES ligands, atom correspondence through InChI normalization preserves
the original bond orders, charges, and stereochemistry. CCD components retain
their named atoms and reference stereochemistry; leaving atoms absent from
the sampled structure are excluded.

Each sampler call creates fresh restraint state. RGI minimizes the denoised
coordinates after native gradient guidance and before alignment and integration,
using the pre-churn sigma in angstroms. `edm`, `edm_gd`, and `edm_fk` support RGI.
The context-parallel path minimizes on rank zero and broadcasts coordinates;
multi-GPU validation is separate from the single-GPU examples here.
`edm_fkc` rejects RGI because its importance weights need a separate treatment
of the additional guidance.
