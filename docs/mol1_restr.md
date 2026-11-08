# Mol-1 — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

Mol-1 uses the PyTorch optimizer. Its adapter lives in
`rgi_toolkit.mol1.adapter.Mol1Adapter` and reuses the shared Protenix-shaped
chemistry extraction while following Mol-1's actual atom-to-token map.

## Installation

Install the RGI fork and its engine dependency. This integration uses the
toolkit's `add_mol1_intellifold` branch until that branch is merged.

```bash
git clone --branch rgi-integration https://github.com/cddlab/Mol-1_restr.git
cd Mol-1_restr
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[cuda]'
export MOL_ROOT_DIR="${MOL_ROOT_DIR:-$PWD/data}"
bash scripts/database/download_mol_data.sh
mkdir -p "$MOL_ROOT_DIR/checkpoint"
curl -fL https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
    -o "$MOL_ROOT_DIR/checkpoint/mol-1-base.pt"
```

Use `.[cuda12]` for a CUDA 12 driver, or `.[cpu]` for CPU inference. During
co-development, install the sibling toolkit with `uv pip install -e ../RGI-toolkit`
after installing the predictor.

GPU inference builds Fast LayerNorm on first use. Install a C++ compiler and a
CUDA toolkit matching the PyTorch CUDA major version, and set `CUDA_HOME` when
the default toolkit differs. The CUDA extras install Ninja. On systems without
that build toolchain, set `LAYERNORM_TYPE=torch` to use the native PyTorch
LayerNorm implementation. The fork constrains Pandas below 3 because its
existing dataset transformations rely on the Pandas 2 string/NaN behavior.

Mol-1 takes a JSON list of jobs. Put `restraints_config` beside each job's `name`
and `sequences`. A ligand opt-in belongs inside its `ligand` object:

```json
{"ligand": {"ligand": "N[C@@H](CCC(N)=O)C(O)=O", "count": 1, "conformer_restraints": true}}
```

SMILES bond orders and source stereochemistry are preserved for conformer
restraints. The adapter takes targets from the reference conformer; the shared
engine performs the configured relaxation and stereo checks.

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
bash examples/distance/mol1/run.sh
bash examples/angle/mol1/run.sh
bash examples/rmsd/mol1/run.sh
bash examples/custom/dist-diff/mol1/run.sh
bash examples/ligand-dissociation/mol1/run.sh
```

Run from the toolkit checkout with the predictor installed as a sibling.
The runners use Mol-1's MSA server. To supply existing alignments, set
`unpairedMsaPath` on each protein-chain object and optionally `pairedMsaPath`.
Omit `pairedMsaPath` when no paired alignment is available: an empty string is
treated as a missing file and triggers a new MSA search. To run without
MSA search, invoke `mol pred` directly with `--use_msa false`.

## Verification

The [small-case verification report](mol1_intellifold_verification.md) records
bench-rgi inputs, measured distances, source-stereo checks, and tested revisions.

Use `verbose: true` and check the `built spec:` counts before interpreting final
energies. A zero energy with zero registered restraints does not demonstrate
guidance. Measure the selected distance or other geometry in the written CIF,
check that all coordinates are finite, and compare conformer ON/OFF predictions
with the same seed. Test two inputs with different targets in one process to
check that settings do not leak between structures.
