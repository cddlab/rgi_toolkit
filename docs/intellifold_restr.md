# IntelliFold v2 — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

This integration supports the current `intellifold predict` JAX path with
IntelliFold v2 weights and the vendored AlphaFold 3 engine. Its framework-free
adapter is `rgi_toolkit.intellifold.adapter.IntelliFoldAdapter`; it reuses the
shared AF3 atom layout and ligand mapping. The legacy PyTorch runner is outside
this integration.

## Installation

Install both the fork's vendored engine and its wrapper. A stock AlphaFold 3
installation does not contain the RGI sampling hooks.

```bash
git clone --branch rgi-integration https://github.com/cddlab/IntelliFold_restr.git
cd IntelliFold_restr
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ./third_party/alphafold3
uv run --active --no-project build_data
uv pip install -e .
```

The engine requires a C++ build toolchain. Build native components on a machine
compatible with the inference node. The wrapper depends on the toolkit's
`add_mol1_intellifold` branch until that branch is merged. For co-development,
run `uv pip install -e ../RGI-toolkit` after installing the wrapper.

IntelliFold v2 weights download on first use into `--model-dir`; no AlphaFold 3
weights are needed. Inputs use AF3 JSON with explicit chain IDs. Put
`restraints_config` beside `sequences` and `modelSeeds`. A ligand opt-in is:

```json
{"ligand": {"id": "B", "smiles": "N[C@@H](CCC(N)=O)C(O)=O", "conformer_restraints": true}}
```

SMILES and single-CCD ligands support conformer restraints. Opting a multi-CCD
ligand into conformer restraints raises an error rather than silently skipping
it. The RGI configuration and per-chain flags survive input serialization and
the MSA/template pipeline.

The pure JAX minimizer runs inside `hk.scan`. Numeric targets and other restraint
arrays are runtime arguments to the outer JIT, so compatible inputs can reuse
compiled programs. Each input and seed builds its own restraint state using that
seed's reference features. `gpu` does not move JAX restraints off the model's
device. Native `--steering` is optional; when enabled, steering runs first and
RGI runs second within each denoising step.

## Configuration

The shared [configuration reference](config.md) defines distance, angle, dihedral,
improper, chiral, plane, RMSD, base-pair and custom restraints. Each input keeps
its own `restraints_config`. Omitting it preserves native sampling.

Conformer restraints additionally require `"conformer_restraints": true` on each
ligand or polymer entity to be restrained. Other restraints need no entity opt-in.
`resid` is the per-chain 1-based residue/token ordinal; ligand atoms and atomized
modified residues occupy separate tokens. `index` refers to the model coordinate
array, so it can include padding gaps. Qualify protein selections with `chain A`.

The engine applies restraints after denoising and before the diffusion update,
using the pre-churn sigma. Each restraint can use either sigma windows or step
windows. There is no top-level `start_sigma`.

## Examples

The five runnable example types are distance, angle, RMSD, custom distance
difference, and MDM2-p53 dissociation. Their input files contain all settings
inline; reference structures for RMSD are downloaded by the runner.

```bash
DB_DIR=databases bash examples/distance/intellifold/run.sh
DB_DIR=databases bash examples/angle/intellifold/run.sh
DB_DIR=databases bash examples/rmsd/intellifold/run.sh
DB_DIR=databases bash examples/custom/dist-diff/intellifold/run.sh
DB_DIR=databases bash examples/ligand-dissociation/intellifold/run.sh
```

Run from the toolkit checkout with the predictor installed as a sibling. Set
`DB_DIR` to the local AF3 sequence databases. `MODEL_DIR` optionally overrides
the model cache. All runners accept an optional input JSON path.

For precomputed MSAs, give each protein `unpairedMsaPath`, `pairedMsa: ""`, and
`templates: []`, then set `RUN_DATA_PIPELINE=false`. A query-only input can use
`unpairedMsa: ""` instead of a path. This explicitly disables MSA search.

```bash
RUN_DATA_PIPELINE=false bash examples/distance/intellifold/run.sh input_with_msa.json
```

Use a new `OUT_DIR` for a changed restraint configuration, because the native
batch runner can skip completed predictions.

## Verification

Use `verbose: true` and check the `built spec:` counts before interpreting final
energies. A zero energy with zero registered restraints does not demonstrate
guidance. Measure the selected distance or other geometry in the written CIF,
check that all coordinates are finite, and compare conformer ON/OFF predictions
with the same seed. Test two inputs with different targets in one process to
check that settings do not leak between structures.
