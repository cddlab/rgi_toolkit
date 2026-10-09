# RoseTTAFold3 (Foundry) — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

RoseTTAFold3 (RF3) + [RGI-toolkit](https://github.com/cddlab/rgi_toolkit)
restraint-guided inference. This integration covers RF3 in Foundry. `restraints_config` reference & atom-selection DSL: [`config.md`](config.md).

## Installation

Install the `cddlab/foundry_restr` fork on its `rgi-integration` branch:

```bash
git clone --branch rgi-integration https://github.com/cddlab/foundry_restr.git
cd foundry_restr
uv venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend cu128 \
  -c constraints-rf3.txt -e '.[rf3]'
uv run --no-project --python .venv/bin/python foundry install rf3
```

> For co-development of the engine, install a sibling checkout in a separate step:
> `uv pip install --python .venv/bin/python -e ../RGI-toolkit`.

The fork installs the shared RGI engine. RF3 supplies PyTorch, so no separate RGI
backend extra is needed. `constraints-rf3.txt` pins the validated Linux CUDA
dependencies. Inference requires the RF3 checkpoint and a CUDA GPU; use your
cluster's scheduler when required.

## Configuration

RF3 reads RGI from a **`restraints_config` key nested inside each input JSON job**,
beside `name` and `components`. A file can contain one job or a list of jobs.
Configure restraints with:

1. **Per ligand** — add `"conformer_restraints": true` to each ligand
   component that needs conformer restraints. The default is false; replicated
   chains inherit their component's flag.
2. **The `restraints_config` object** — the shared distance / angle / dihedral /
   conformer / RMSD restraints and custom expressions.
   The example below spells out the solver, targets, weights, movement and sigma
   windows. See [`config.md`](config.md) for other restraint types, alternative
   penalty shapes, reference-anchored selections, and RMSD selection shorthands.

Set explicit component `chain_id` values for predictable selections. `resid` is
the **per-chain 1-based token ordinal**, not the author residue number. Standard
polymer residues occupy one token; ligands and atomized modified residues may
occupy one token per atom. Qualify protein groups with `chain A and (...)`.
`index` is the zero-based row in the processed coordinate array. There is **no
top-level `start_sigma`** — set it per restraint entry and once for all conformer
terms. RF3's native `ground_truth_conformer_selection` is a separate model input.

## Complete example (input JSON)

Save this as `restr_example.json`. It combines QBP and its GLN ligand with
distance, angle, dihedral, ligand conformer, reference RMSD, and custom restraints.
The custom expression keeps the two lobe-centroid distances equal. Chain IDs
are explicit: protein A and ligand B.

Place a compatible QBP reference with chain A at `rmsd_ref.pdb` in the working
directory, or remove the RMSD entry when no reference is needed. Its fit selections
use residues 5–220 and its RMSD calculation uses residues 90–180; adapt the reference
selection if its numbering differs. The example supplies sequences without an
MSA search; RF3 accepts a job-level `msa_paths` mapping for precomputed alignments.

```json
{
  "name": "qbp_rgi_example",
  "components": [
    {
      "seq": "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK",
      "chain_id": "A"
    },
    {
      "ccd_code": "GLN",
      "chain_id": "B",
      "conformer_restraints": true
    }
  ],
  "restraints_config": {
    "verbose": true,
    "gpu": true,
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
        "harmonic": {
          "target_distance": 25.0
        }
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
        "harmonic": {
          "target_angle": 90.0
        }
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
        "harmonic": {
          "target_dihedral": 180.0
        }
      }
    ],
    "conformer_restraints_config": {
      "start_sigma": 99999999,
      "stop_sigma": -1,
      "bond": {
        "weight": 1.0,
        "slack": 0.0
      },
      "angle": {
        "weight": 1.0,
        "slack": 0.0
      },
      "chiral": {
        "weight": 1.0,
        "slack": 0.0
      },
      "cistrans": {
        "weight": 1.0,
        "slack": 0.0
      },
      "vdw": {
        "weight": 1.0
      }
    },
    "rmsd_restraints_config": [
      {
        "ref_pdb": "rmsd_ref.pdb",
        "harmonic": {
          "target_rmsd": 0.0
        },
        "weight": 1.0,
        "start_sigma": 99999999,
        "stop_sigma": 1.0,
        "pairing": "align",
        "best_effort": true,
        "atom_selection_ref_fit": "chain A and (resid 5 to 220)",
        "atom_selection_target_fit": "chain A and (resid 5 to 220)",
        "atom_selection_ref_calc": "chain A and (resid 90 to 180)",
        "atom_selection_target_calc": "chain A and (resid 90 to 180)"
      }
    ],
    "custom_restraints_config": [
      {
        "name": "equidistant",
        "energy": "(distance(L1, H) - distance(L2, H))**2",
        "selections": {
          "L1": "chain A and (resid 5 to 84)",
          "L2": "chain A and (resid 186 to 224)",
          "H": "chain A and (resid 90 to 180)"
        },
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "weight": 1.0
      }
    ]
  }
}
```

## Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail

uv run --no-project --python .venv/bin/python rf3 fold \
  inputs=restr_example.json out_dir=out_restr_example \
  diffusion_batch_size=1 num_steps=50 \
  early_stopping_plddt_threshold=null seed=0
```

Omitted or null `restraints_config` retains ordinary RF3 sampling; an empty object
adds no restraints. Each job receives its own configuration and every diffusion
sample is optimized independently. Set `skip_existing=false` when comparing
configurations in an output directory that already contains predictions.

## Verify results

With `verbose: true`, the setup log prints `built spec: n_active=.. bonds=.. angles=..
chirals=.. cistrans=.. distances=.. rmsd=.. group_angle=.. group_dihedral=..
custom=..`. Confirm nonzero counts for the requested selections. Chemistry-dependent conformer counts
can be zero, for example GLN has no E/Z double bond for `cistrans`.

Inspect every output sample for finite coordinates and measure its selected
geometry against the targets. A zero residual for an absent term does not show
that the restraint ran. RF3's final integrator update follows the last minimization,
so final residuals need not be exactly zero. Schema and selection-syntax validation
does not establish that a selection matches the intended atoms or that the final
structure satisfies all targets.

The fork's `examples/rgi/` provides distance and ATP/fumarate conformer examples.
Its [validation report](https://github.com/cddlab/foundry_restr/blob/rgi-integration/models/rf3/docs/rgi-validation.md)
records real-checkpoint measurements and repeatable validation commands.

## Integration details

RF3 passes a fresh `CombinedRestraints` instance from each processed structure
through its network into the diffusion sampler. The hook minimizes the denoised
prediction before the integrator update, using the pre-churn sigma and zero-based
rollout step. Reference coordinates and atom-to-token mappings come from the same
post-transform features as the sampler. Molecular types come from AtomWorks entity
types; modified protein residues remain polymers even when marked as hetero atoms.
Source ligand chemistry preserves bond orders, formal charges and stereochemistry.

The Python `InferenceInput.from_atom_array()` and `from_cif_path()` APIs accept
`restraints_config=...` and a per-chain `conformer_restraints={"B": True}` mapping.
See the fork's [full guide](https://github.com/cddlab/foundry_restr/blob/rgi-integration/models/rf3/docs/rgi.md)
for these input forms and the validated dependency constraints.
