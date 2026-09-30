# rgi_toolkit RGI examples

Representative inputs from the paper's four protein state-control benchmarks.
Each `<type>/<tool>/` directory contains one restraint setting and a `run.sh` for
one prediction. The paper uses five target settings and nine seeds
per setting. These examples retain the same restraint objective, atom selections,
activation window, and minimization settings at the representative target.

| Directory | System | Representative restraint |
|---|---|---|
| `distance/` | QBP, 226 residues | Interdomain centroid distance of 25.0 Å |
| `angle/` | ADK, 214 residues | NMP–CORE–LID centroid angle of 72.85° |
| `rmsd/` | QBP | Open/closed reference RMSD targets of 2.65/2.65 Å on 220 common Cα atoms |
| `custom/dist-diff/` | DgoT, 419 residues | ΔD = D_in − D_out = 0.8 Å |

The six paper predictors are `boltz-2`, `protenix-v2`, `alphafold3`, `openfold-3`,
`chai`, and `esmfold2`. The `opendde` directories demonstrate the same restraints
in another supported predictor; OpenDDE was not included in the paper benchmark.

## Run

```bash
bash examples/distance/boltz-2/run.sh

# ESMFold2 requires the full ColabFold A3M for the selected protein.
MSA_A3M=inputs/qbp.a3m bash examples/rmsd/esmfold2/run.sh
```

Run these commands from the RGI-toolkit checkout on a GPU compute node. Each
runner locates the matching predictor fork as a sibling of `RGI-toolkit/` and
uses its existing venv or Pixi environment. See [the integration guides](../docs/)
for installation. Protenix-v2 is selected explicitly and requires a supported
sm_89 GPU in this integration.

All runners request one diffusion sample. OpenFold3 and Chai also request one
model/trunk sample. The restraint minimizer is CG with the default strong-Wolfe
line search and `gtol=1e-5`; `max_iter` is 1000 for RMSD and 100 otherwise.

## Restraint settings

Distance, angle, and custom restraints use `start_sigma=99999999` and remain
active through the final diffusion step. The paper's distance targets are
25.00, 26.02, 27.05, 28.08, and 29.10 Å. Its angle targets are 60.90, 66.88,
72.85, 78.82, and 84.80°.

RMSD uses two simultaneous harmonic restraints with `pairing: align`,
`start_sigma=99999999`, and `stop_sigma=1.5`. The five (open, closed) target pairs
are (0, 5.3), (1.325, 3.975), (2.65, 2.65), (3.975, 1.325), and (5.3, 0) Å.
Both fitting and RMSD calculation use these chain-A Cα selections:

| Coordinates | Per-chain residue ordinals |
|---|---|
| Prediction | 5–224 |
| 1GGG, open reference | 1–220 |
| 1WDN, closed reference | 2–221 |

The two entries therefore use the same 220 corresponding atoms. No protein
conformer restraints or conformer opt-in flags are added. The runners download
the two reference CIFs from RCSB when needed; downloaded files are ignored.

For DgoT, D_in = distance(A, B) is the cytoplasmic domain-centroid distance and
D_out = distance(C, D) is the periplasmic distance. The harmonic custom loss is
`((distance(A, B) - distance(C, D)) - 0.8)**2`. The paper's ΔD targets are −4.3,
−1.75, 0.8, 3.35, and 5.9 Å. Group D excludes query residues 251–258 and 263–264,
which are absent from the outward-occluded reference 6E9O. The query spans native
residues 27–445; selections always use query-local, per-chain residue ordinals.

## MSA inputs

The paper supplies the same full ColabFold A3M for each protein across all six
predictors. Restraint settings alone do not reproduce a particular prediction:
the same MSA, model checkpoint, and seed are also required.

The examples retain convenient MSA acquisition for Boltz-2, Protenix-v2,
OpenFold3, and Chai. Those runners use an MSA server. AlphaFold3 uses its local
data pipeline and requires `MODEL_DIR` and `DB_DIR`. ESMFold2 reads the full A3M
from `MSA_A3M`, checks its query against the example sequence, and uses the paper
settings `msa_max_depth=1024` and `msa_column_mask_rate=0.1`, with 20 recurrent
loops and 200 diffusion steps. It does not fall back to a single-sequence input.

To reuse a fixed paper MSA, provide it through the predictor's native input:

| Predictor | Precomputed MSA input |
|---|---|
| Boltz-2 | Protein `msa` field; omit `--use_msa_server` |
| Protenix-v2 | Protein-chain `unpairedMsaPath` |
| AlphaFold3 | Protein `unpairedMsaPath`, `pairedMsa: ""`, and `templates: []`; use `--run_data_pipeline=False` |
| OpenFold3 | Chain `main_msa_file_paths`; use `--use-msa-server false` |
| Chai | Aligned Parquet files through `--msa-directory`; omit `--use-msa-server` |
| ESMFold2 | `MSA_A3M` |

OpenDDE disables external MSA and template searches in its extension examples.

## Validation

All protein selections are qualified with `chain A`. `resid` means a per-chain
1-based ordinal, not the reference structure's author residue number. The
[configuration guide](../docs/config.md) describes the schema and selection DSL.

`verbose: true` enables setup diagnostics. Confirm a nonzero restraint count:
one distance, group-angle, or custom term, or two RMSD terms with
`conformer=False`. Schema and selection-syntax checks do not establish that a
selection matches the intended atoms in a different input structure.
