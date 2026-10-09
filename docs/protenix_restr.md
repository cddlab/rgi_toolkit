# Protenix v2 — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

Protenix v2 + [RGI-toolkit](https://github.com/cddlab/rgi_toolkit) restraint-guided inference. `restraints_config` reference & atom-selection DSL: [`config.md`](config.md).

## Installation

The RGI code lives in the `cddlab/protenix_restr` fork — install **that fork**, not the upstream
PyPI `protenix`, which has no RGI hooks.

```bash
git clone https://github.com/cddlab/protenix_restr.git
cd protenix_restr
uv venv && source .venv/bin/activate           # Python 3.11+
uv pip install -e .                             # also pulls the rgi_toolkit engine (declared in requirements.txt)
```

> For co-development of the engine, override the pinned dependency with a local editable
> checkout in a SEPARATE step: `uv pip install -e ../RGI-toolkit` (sibling clone).

> **Run protenix on sm_89 (e.g. RTX 4090), NOT on Blackwell (sm_120).** On Blackwell its
> cuequivariance fused kernels **silently** emit all-NaN coordinates (no crash, exit 0) even for a
> bare fold with no restraints. A NaN output is almost always this, not the restraint; confirm by
> re-running on an sm_89 GPU. Run on a machine with a CUDA GPU.

## Configuration

protenix reads RGI from a **`restraints_config` key nested inside each fold-input object** of the
input JSON (the input is a JSON *list* of fold jobs; the key sits beside `name`/`sequences`). Turn
restraints on with:

1. **Per ligand** — `"conformer_restraints": true` on each ligand object
   enables its conformer restraints.
2. **The `restraints_config` object** — the distance / angle / dihedral / conformer /
   RMSD restraints, plus config-only `custom` restraints (define your own — see config.md). The example below shows the documented restraint types with concrete values; see
   [`config.md`](config.md) for what each does, the alternative restraint types
   (`flat-bottomed` etc.), and the RMSD `atom_selection_ref`/`atom_selection_target` shorthand.

`resid` is the **per-chain 1-based ordinal** (qualify protein groups with `chain A and (...)`).
There is **no top-level `start_sigma`**.

## Complete example (input JSON)

Save this as `restr_example.json`. It folds QBP with its GLN ligand and combines
distance, angle, dihedral, ligand conformer, reference RMSD, and custom restraints.
The custom expression keeps the two lobe-centroid distances equal.
Protenix assigns chain letters by sequence order: protein A and ligand B.
The run command enables its MSA search.

```json
[
  {
    "name": "qbp_rgi_example",
    "sequences": [
      { "proteinChain": { "sequence": "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK", "count": 1 } },
      { "ligand": { "ligand": "CCD_GLN", "count": 1, "conformer_restraints": true } }
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
          "harmonic": { "target_distance": 25.0 }
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
          "harmonic": { "target_angle": 90.0 }
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
          "harmonic": { "target_dihedral": 180.0 }
        }
      ],
      "conformer_restraints_config": {
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "bond": { "weight": 1.0, "slack": 0.0 },
        "angle": { "weight": 1.0, "slack": 0.0 },
        "chiral": { "weight": 1.0, "slack": 0.0 },
        "cistrans": { "weight": 1.0, "slack": 0.0 },
        "vdw": { "weight": 1.0 }
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
]
```

## Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`):

```bash
#!/bin/bash
# protenix RGI example runner. Run on an sm_89 CUDA GPU (Blackwell sm_120 emits silent all-NaN coords).
set -e
source .venv/bin/activate

protenix pred -i restr_example.json -o out_restr_example \
    --model_name protenix-v2 --use_default_params true --use_msa true \
    --seeds 0 --step 200 --sample 1 --cycle 10
```

## Verify results

With `verbose: true`, `setup` logs `built spec: n_active=.. bonds=.. angles=.. chirals=..
cistrans=.. distances=.. rmsd=.. group_angle=.. group_dihedral=..` — confirm the counts are non-zero for what you requested.
Cross-check the result with the workspace helpers (any gemmi/rdkit venv): `../check_dist.py
<pred.cif>` (centroid distance vs 25 Å) and `../check_conf.py <pred.cif> GLN` (ligand geometry). If
the output is all-NaN, you almost certainly ran on Blackwell — re-run on an sm_89 GPU.
