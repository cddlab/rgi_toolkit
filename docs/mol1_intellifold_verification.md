# Mol-1 and IntelliFold small-case verification

Verified on 2026-11-07 with RGI-toolkit 0.5.0. These are functional integration
checks with one seed and one sample per condition, not an accuracy benchmark.

## Tested revisions and environment

| Component | Inference revision |
|---|---|
| RGI-toolkit | `c08766fdd668d95bbeed85e046d7ed3ea78dfeab` |
| Mol-1 | `d4b6de899d4102b2b5fd536f3581deafb0cb06b9` |
| IntelliFold v2 | `74f3132d8b588455ff5739852994ed7997d2ad32` |
| bench-rgi fixtures/configuration | `70dd0a722b1f450d553c5b83883a33942bc3bdd3` |

IntelliFold subsequently received a serialization fix in `5eaae93`: identical
molecules with different conformer opt-ins remain distinct when writing JSON.
All 14 CPU tests pass on that revision. The fix does not change the sampler or
the single-ligand inputs measured below.

Both predictors ran on an RTX 5090 with Python 3.12.13. Mol-1 used PyTorch
2.13.0/CUDA 13.0, cuEquivariance 0.10.0, RDKit 2026.9.1, and Pandas 2.3.3.
IntelliFold used JAX/jaxlib 0.9.1 and RDKit 2025.9.4. Each prediction used seed 0,
one diffusion sample, 200 diffusion steps, and one recycle. Both models consumed
the same source MSAs through their native featurizers. Template search was off.

Mol-1's native Fast LayerNorm extension was built with a CUDA 13 toolchain.
`TORCH_NO_COMPILER_WRAPPER=1` bypassed a failing local sccache wrapper; inference
used the native CUDA extension. The RGI compiler was enabled during inference.

## Distance guidance and per-input configuration

The 226-residue QBP fixture comes from `bench-rgi/distance/fixtures/qbp.a3m`.
The distance uses all atoms in residues 4-83 plus 185-223 versus residues 89-179,
all in chain A. Both adapters selected 946 and 686 atoms, respectively, and
registered one distance restraint. CG used `max_iter: 100` and
`start_sigma: 100000000`.

| Predictor | No restraint (A) | Target 25 A | Target 35 A |
|---|---:|---:|---:|
| Mol-1 | 24.900717 | 25.000793 | 34.997422 |
| IntelliFold v2 | 25.225381 | 25.002289 | 34.998089 |

Distances were measured independently from the written CIF coordinates.
Each predictor processed the three different configurations in one process.
Both restrained targets were within 0.003 A, confirming that the second target
was not replaced by cached settings from the first input.

For Mol-1, the precomputed unpaired MSA path was supplied and `pairedMsaPath`
was omitted. An empty string for that optional path triggers another MSA search.
For IntelliFold, `pairedMsa: ""`, `templates: []`, and `--norun_data_pipeline`
disabled searches. The ColabFold metadata comment was removed from the QBP A3M;
its aligned sequences were retained.

## Ligand conformer guidance

The ct209 case `1cz2__1__1.A__1.B` contains a 90-residue protein and a 24-heavy-atom
ligand. Its source MSA, SMILES, and SDF came from the pinned
`wasarou/plinder-cistrans-209` v1.0 data used by bench-rgi. ON and OFF used the
same input sequence, ligand, MSA, seed, and sampling settings.

ON used bench-rgi's conformer settings: CG with 100 iterations, UFF target
relaxation, `start_sigma: 1.0`, and unit weights for bond, angle, chiral,
cis/trans, and VdW terms. The optional plane term was disabled. Each predictor
registered 24 bond, 29 angle, one chiral, and two cis/trans restraints, plus
ligand-background VdW interactions against 667 protein atoms.

| Predictor | Source chirality matches OFF / ON | Source E/Z matches OFF / ON | Ligand ON/OFF aligned RMSD (A) |
|---|---|---|---:|
| Mol-1 | 0/1 -> 1/1 | 1/2 -> 2/2 | 3.868602 |
| IntelliFold v2 | 0/1 -> 1/1 | 1/2 -> 2/2 | 0.680232 |

Stereo was assigned independently from the written ligand coordinates and
compared with the source SMILES. A mismatch includes an unassigned center.
The RMSD uses an explicit atom-name correspondence and ligand-only rigid
alignment; it measures conformational change, not docking accuracy.
The final weighted restraint objectives were approximately 0.00002 for Mol-1
and 0.00022 for IntelliFold. These are toolkit objective values, not physical
energies. All coordinates in all ten output structures were finite.

## Regression and installation checks

- Toolkit: 1,800 eager CPU tests and 22 actual CPU compiler tests passed.
- Mol-1: 196 CPU tests passed, 14 skipped; 203 GPU tests passed, 7 skipped.
  Both runs also passed 27 subtests. The six new RGI cases passed in both runs.
- IntelliFold: all 14 tests passed, including input round-trips, independent
  opt-ins for identical molecules, pipeline preservation, pre-churn sigma,
  native steering order, and finite coordinates with or without RGI.
- All ten new example configurations passed schema and selection-syntax
  validation. All ten shell runners passed syntax checks. These checks do not
  replace full predictions for the larger angle, RMSD, custom, or dissociation
  examples.
- Toolkit lint, formatting, and lock consistency passed. Both installed
  predictor environments passed dependency checks, and installation resolution
  succeeded against the published toolkit branch.

The Mol-1 fork constrains Pandas below 3 after its existing vectorization tests
exposed incompatible string/NaN behavior, and its CUDA extras include Ninja.
IntelliFold clamps diffusion noise variance to zero before taking its square
root, preventing negative roundoff at zero churn on CPU.

## Artifact provenance

Generated inputs, output CIFs, logs, environment checks, `manifest.json`, and
independent measurement scripts are kept under
`$BENCH_VERIFICATION_ROOT/mol1_intellifold`. The root is configured by bench-rgi;
these generated artifacts are outside the source repositories.

| Artifact | SHA-256 |
|---|---|
| Original QBP A3M | `aebbfcd7c42af3e00f0e01b9f33fdd2f11e13bde8267602b986d7075a223d156` |
| Normalized QBP A3M | `188ed7137e01921f90e5fed691fe034978603473fc7d474954ed37779c3335f8` |
| Ligand-case A3M | `6610b711d244bf4b8eb0024a4fb4cc34cb72a1b48a8246731fc5dec1d7ece583` |
| Ligand-case source SDF | `e27b5648dc8c82a1186c3defa29f3a092491c691beacd0739f5cb2808da6a187` |
| Mol-1 base weights | `d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6` |
| IntelliFold v2 JAX weights | `16ae6c2002989a3eb3fae60714f2802fc336f6b1fdfb1a90a74a91745cdcdef2` |
| IntelliFold Fourier weights | `4453bf29ac0a0f921a67a15155d0dd61e3f7214ec124d6deb91ce935d6a93201` |
