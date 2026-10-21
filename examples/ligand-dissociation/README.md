# MDM2–p53 ligand dissociation

Each model directory contains the same two protein sequences and ten harmonic
distance settings. Chain A is the 107-residue MDM2 fragment; chain B is the
17-residue p53 peptide. The distance is between the arithmetic centroids of
all atoms selected by `chain A` and `chain B`, not a closest-contact distance.
The peptide is a protein entity in every model.

| Setting | Value |
|---|---|
| Distance targets (Å) | 13.00, 19.33, 25.67, 32.00, 38.33, 44.67, 51.00, 57.33, 63.67, 70.00 |
| Selections | `chain A` and `chain B` |
| Penalty | Harmonic, default weight 1, both groups movable |
| Minimizer | `CG`, `max_iter: 1000` |
| Activation | `start_sigma: 99999999` on the distance entry; active through the final step |
| Conformer restraints | Disabled |

These sequences, targets, selections, and explicit optimizer settings follow
the reference MDM2–p53 sampling inputs. The legacy input's top-level
`start_sigma` is placed on the distance entry for the current RGI schema.
The current engine's default CG line search applies.

## MSA files

Place the reference run's full, unpaired alignments at these paths:

```text
examples/ligand-dissociation/msas/mdm2_p53.a3m
examples/ligand-dissociation/msas/p53.a3m
```

The first file corresponds to `out_msa/mdm2_p53.a3m` in the reference sampling
directory, and the second to `out_msa_2/p53.a3m`. Keep their query sequences
unchanged. Every runner requires both files and performs no MSA search.
The shared `prepare_msa.sh` creates copies under `msas/a3m/` with only
ColabFold's `#` metadata lines removed, for native FASTA parser compatibility.
All FASTA headers and alignment rows are preserved, and the originals are
unchanged. Alignments, converted Parquet files, and predictions are ignored by Git.

Chai converts the unpaired rows to its aligned-Parquet format under
`msas/chai/`, without pairing keys. All predictors read the same prepared
A3M rows through their native interfaces. Model-specific MSA featurization
and sampling defaults still differ.

## Run one setting

Use the matching predictor environment on a GPU compute node. As with the
other examples, the predictor forks are siblings of the RGI-toolkit checkout.
Run from the toolkit directory:

```bash
bash examples/ligand-dissociation/boltz-2/run.sh mdm2_p53_13.00.yaml
bash examples/ligand-dissociation/rf3/run.sh mdm2_p53_13.00.json
bash examples/ligand-dissociation/bioemu2/run.sh mdm2_p53_13.00.yaml
```

The optional argument selects a configuration filename in that model's
directory; the default is `mdm2_p53_13.00` with the appropriate extension.
Outputs go to `<model>/out/<input-stem>/`. Each runner requests one diffusion
sample; repeat seeds separately when reproducing the reference sampling
campaign. The provided examples do not launch that full campaign.

| Model directory | Input | Alignment input |
|---|---|---|
| `boltz-2` | YAML | Protein `msa` paths |
| `protenix-v2` | JSON job list | Protein `unpairedMsaPath` |
| `alphafold3` | JSON | `unpairedMsaPath`, empty paired MSA, no templates |
| `openfold-3` | JSON query | `main_msa_file_paths` |
| `chai` | YAML plus `mdm2_p53.fasta` | Converted aligned-Parquet directory |
| `esmfold2` | JSON consumed by `run_dissociation.py` | `MSA.from_a3m` for each protein |
| `opendde` | JSON job list | Protein `unpairedMsaPath` |
| `rf3` | JSON job | Chain-keyed `msa_paths` |
| `bioemu2` | System YAML | `get_msa` with chain-keyed paths in the runner |

AlphaFold3 also requires `MODEL_DIR`. Its data pipeline is disabled because
both alignments are supplied. Protenix-v2 requires a supported sm_89 GPU in
this integration. RF3 uses 50 diffusion steps; BioEmu2, ESMFold2, Protenix-v2,
and OpenDDE use 200. See the [integration guides](../../docs/README.md) for
installation and the [main example guide](../README.md) for general settings.

## Verify

With `verbose: true`, confirm that `built spec:` reports one distance term
and that both selected chains contain atoms. Measure the chain-centroid
distance in each output and compare it with that file's target. Schema and
input checks do not demonstrate that a sampled structure reached the target.
