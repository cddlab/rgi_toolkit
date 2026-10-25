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

Save this as `restr_example.yaml` in the predictor checkout. Like the other
predictor guides, it folds QBP with its GLN ligand and combines distance, angle,
dihedral, ligand conformer, reference RMSD, and custom restraints. The custom
expression keeps the two lobe-centroid distances equal. The run section below
downloads the QBP reference structure and uses BioEmu2's ColabFold MSA acquisition.

```yaml
entities:
  - entity_type: protein
    sequence: ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK
    asym_ids: A
  - entity_type: ligand
    ccd_code: GLN
    asym_ids: B
    conformer_restraints: true
restraints_config:
  verbose: true
  gpu: true
  method: CG
  max_iter: 1000
  distance_restraints_config:
    - atom_selection1: "chain A and ((resid 5 to 84) or (resid 186 to 224))"
      atom_selection2: "chain A and (resid 90 to 180)"
      start_sigma: 99999999
      stop_sigma: -1
      move: both
      weight: 1.0
      harmonic:
        target_distance: 25.0
  angle_restraints_config:
    - atom_selection1: "chain A and (resid 5 to 84)"
      atom_selection2: "chain A and (resid 90 to 180)"
      atom_selection3: "chain A and (resid 186 to 224)"
      start_sigma: 99999999
      stop_sigma: -1
      move: "1,3"
      weight: 1.0
      harmonic:
        target_angle: 90.0
  dihedral_restraints_config:
    - atom_selection1: "chain A and (resid 5 to 50)"
      atom_selection2: "chain A and (resid 51 to 100)"
      atom_selection3: "chain A and (resid 101 to 150)"
      atom_selection4: "chain A and (resid 151 to 224)"
      start_sigma: 99999999
      stop_sigma: -1
      move: "1,4"
      weight: 1.0
      harmonic:
        target_dihedral: 180.0
  conformer_restraints_config:
    start_sigma: 99999999
    stop_sigma: -1
    bond: {weight: 1.0, slack: 0.0}
    angle: {weight: 1.0, slack: 0.0}
    chiral: {weight: 1.0, slack: 0.0}
    cistrans: {weight: 1.0, slack: 0.0}
    vdw: {weight: 1.0}
  rmsd_restraints_config:
    - ref_pdb: rmsd_ref.pdb
      harmonic: {target_rmsd: 0.0}
      weight: 1.0
      start_sigma: 99999999
      stop_sigma: 1.0
      pairing: align
      best_effort: true
      atom_selection_ref_fit: "chain A and (resid 5 to 220)"
      atom_selection_target_fit: "chain A and (resid 5 to 220)"
      atom_selection_ref_calc: "chain A and (resid 90 to 180)"
      atom_selection_target_calc: "chain A and (resid 90 to 180)"
  custom_restraints_config:
    - name: equidistant
      energy: "(distance(L1, H) - distance(L2, H))**2"
      selections:
        L1: "chain A and (resid 5 to 84)"
        L2: "chain A and (resid 186 to 224)"
        H: "chain A and (resid 90 to 180)"
      start_sigma: 99999999
      stop_sigma: -1
      weight: 1.0
```

## Run

Save this as `run_restr_example.sh` beside the YAML in the predictor checkout,
then run `bash run_restr_example.sh` on a GPU compute node. It downloads the
1GGG reference used by the RMSD entry into the working directory:

```bash
#!/usr/bin/env bash
set -euo pipefail

curl -fLsS https://files.rcsb.org/download/1GGG.pdb -o rmsd_ref.pdb

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
| MDM2–p53 dissociation, 30.0 Å | [YAML](../examples/ligand-dissociation/bioemu2/mdm2_p53_30.00.yaml) | `bash examples/ligand-dissociation/bioemu2/run.sh` |

All five examples use BioEmu2's ColabFold MSA acquisition, structure
conditioning, one sample, 200 EDM steps, and seed 0.

## Python API and precomputed MSAs

```python
from pathlib import Path

from bioemu2 import EmuSession, System, get_msa, load_model, save_frames

system = System.from_yaml(
    "../RGI-toolkit/examples/ligand-dissociation/bioemu2/mdm2_p53_30.00.yaml"
)
msa_dir = Path("inputs")
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

With `verbose: true`, check nonzero counts for distance, group angle, group
dihedral, RMSD, ligand bond/angle/chirality, and the custom expression in the
complete example. GLN has a tetrahedral center but no eligible E/Z dihedral,
so `cistrans=0` is expected even though its weight is enabled. Check that VdW
includes the ligand against the protein background. The separate dissociation
example must build one distance between chains A and B.

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
