# RoseTTAFold3 (Foundry) — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

RoseTTAFold3 (RF3) + [RGI-toolkit](https://github.com/cddlab/rgi_toolkit)
restraint-guided inference. This integration covers RF3 in Foundry. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

> **Or generate it automatically:** the `generate-rgi-config` skill in Claude Code
> (`/generate-rgi-config`) or Codex (`$generate-rgi-config`) writes and validates
> the shared `restraints_config`. Place it in the RF3 input as shown below.

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

1. **Per component** — add `"conformer_restraints": true` to each protein, DNA,
   RNA, or ligand component whose local geometry should be restrained. The default
   is false; replicated chains inherit their component's flag.
2. **The `restraints_config` object** — the shared distance / angle / dihedral /
   plane / conformer / RMSD restraints, base-pair restraints, and custom energies.
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

Save this as `restr_example.json`. Like the other predictor guides, it combines
QBP, its GLN ligand, and short DNA and RNA duplexes with a centroid distance,
group angle, group dihedral, selected plane, ligand conformer, RMSD, a custom
formula, and Watson-Crick base pairs. Keep the entries needed for your task.
The custom term keeps both lobe halves equidistant from the central domain.
The duplex sequences are self-complementary palindromes, with chains C/D for DNA
and E/F for RNA; `chain_type` explicitly identifies each polymer type.

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
    },
    {
      "seq": "GCATGC",
      "chain_type": "POLYDEOXYRIBONUCLEOTIDE",
      "chain_id": "C"
    },
    {
      "seq": "GCATGC",
      "chain_type": "POLYDEOXYRIBONUCLEOTIDE",
      "chain_id": "D"
    },
    {
      "seq": "GCAUGC",
      "chain_type": "POLYRIBONUCLEOTIDE",
      "chain_id": "E"
    },
    {
      "seq": "GCAUGC",
      "chain_type": "POLYRIBONUCLEOTIDE",
      "chain_id": "F"
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
        "harmonic": {"target_distance": 25.0}
      }
    ],
    "base_pair_restraints_config": [
      {
        "residue1": "chain C and resid 1",
        "residue2": "chain D and resid 6"
      },
      {
        "residue1": "chain C and resid 3",
        "residue2": "chain D and resid 4"
      },
      {
        "residue1": "chain E and resid 1",
        "residue2": "chain F and resid 6"
      },
      {
        "residue1": "chain E and resid 3",
        "residue2": "chain F and resid 4"
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
        "harmonic": {"target_angle": 90.0}
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
        "harmonic": {"target_dihedral": 180.0}
      }
    ],
    "plane_restraints_config": [
      {
        "atom_selection1": "chain A and (resid 5 to 20)",
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "move": "all",
        "weight": 1.0,
        "flat-bottomed2": {"target_plane2": 0.1}
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
      "plane": {
        "weight": 1.0
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
        "harmonic": {"target_rmsd": 0.0},
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
  diffusion_batch_size=1 n_recycles=10 num_steps=50 \
  early_stopping_plddt_threshold=null seed=0
```

Omitted or null `restraints_config` retains ordinary RF3 sampling; an empty object
adds no restraints. Each job receives its own configuration and every diffusion
sample is optimized independently. Set `skip_existing=false` when comparing
configurations in an output directory that already contains predictions.

## Verify results

With `verbose: true`, the setup log prints `built spec: n_active=.. bonds=.. angles=..
chirals=.. plane=.. cistrans=.. distances=.. rmsd=.. group_angle=.. group_dihedral=..
group_plane=.. custom=..`. Confirm nonzero counts for the requested selections;
base-pair expansion also has its own log line. Chemistry-dependent conformer counts
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

## External restraint configuration

The shared `config_path` wrapper can replace the whole `restraints_config` or any
individual restraint section with a JSON/YAML file. Includes are relative to the
input JSON; nested includes and resource paths in external configurations are
relative to their own files. Inline reference paths keep the shared engine's
working-directory semantics. See
[shared file-reference syntax](config.md#external-configuration-files).

An empty `conformer_restraints_config: {}` enables bond/angle/chiral/cistrans/vdw at
weight 1 on opted-in components; plane and torsion require an explicit positive
weight. The cistrans term retains ligand E/Z; chi/omega/sp2 belong to torsion.
