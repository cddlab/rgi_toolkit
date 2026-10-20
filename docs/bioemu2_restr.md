# BioEmu2 — Restraint-Guided Inference

[Documentation index](README.md) · [Configuration reference](config.md)

The BioEmu2 integration uses the shared PyTorch restraint engine. Its adapter
lives in `rgi_toolkit.bioemu2`; the predictor fork contains input plumbing,
source-molecule extraction and sampling-loop hooks.

## Installation

Use Linux, Python 3.12 or later, and a supported NVIDIA GPU:

```bash
git clone --branch rgi-integration https://github.com/cddlab/bioemu2_restr.git
cd bioemu2_restr
uv sync --extra cuda
```

For development with the sibling toolkit checkout:

```bash
uv pip install --python .venv/bin/python -e ../RGI-toolkit
```

The integration requires RGI-toolkit 0.4.7 from the `add_bioemu2` branch.
BioEmu2 downloads its own checkpoint. See the
[upstream documentation](https://github.com/microsoft/bioemu2) for model
requirements, MSA acquisition and sampling modes.

## Inputs and sampling

Add `restraints_config` at the root of a BioEmu2 system YAML. Each entity can
opt into conformer restraints with `conformer_restraints: true`; the default
is false. This applies independently to ligand and polymer entities.

```yaml
entities:
  - entity_type: protein
    sequence: GYDPETGTWG
    asym_ids: A
  - entity_type: ligand
    smiles: "O=C(O)/C=C/C(=O)O"
    asym_ids: L
    conformer_restraints: true
restraints_config:
  gpu: true
  verbose: true
  conformer_restraints_config:
    start_sigma: 1.0
```

Run the bundled QBP distance example from the predictor checkout:

```bash
uv run --no-sync bioemu2 predict-structure \
  --system ../RGI-toolkit/examples/bioemu2/qbp_distance.yaml \
  --num-samples 1 --out outputs/qbp
```

The Python API accepts the same field through `System`:

```python
from bioemu2 import EmuSession, System, load_model, save_frames

system = System.from_yaml("../RGI-toolkit/examples/bioemu2/qbp_distance.yaml")
model = load_model()
samples = EmuSession(system, model=model, mode="structure").sample(num_samples=1)
save_frames(samples, "outputs/qbp")
```

Use `mode="equilibrium"` or the `sample-equilibrium` command when that
conditioning mode is intended. Restraint-guided samples are biased by the
chosen restraints; they should not be interpreted as an unbiased equilibrium
ensemble.

Root and section `config_path` references are resolved relative to the system
YAML. The same dictionary supports distance, angle, dihedral, RMSD, custom,
conformer and other shared engine terms.

## Mapping and sampler behavior

- `index` is the zero-based row of the padded coordinate tensor. Padding is
  excluded from selections and receives atomic number zero.
- `resid` is the per-chain, one-based token ordinal. Standard polymer residues
  occupy one token; atomized ligands and modified residues can occupy several.
  Chain names are the input `asym_ids`.
- Atom names and reference geometry come from the exact structure used for
  sampling. Reference coordinates are in angstroms. Saved trajectories use
  nanometers and must be converted before comparison with restraint targets.
- Ligand chemistry follows BioEmu2's source processing: SMILES pass through
  its InChI representation; CCD components retain their named atoms, charges
  and reference stereochemistry. Named leaving atoms absent from the sampled
  structure are removed before constructing restraints.
- Each sampler call creates a fresh `CombinedRestraints` instance. RGI runs
  after denoising and native gradient guidance, before reverse-diffusion
  alignment and integration. Gates use the pre-churn sigma in angstroms.
- `edm`, `edm_gd` and `edm_fk` accept RGI. The context-parallel path applies
  minimization on rank zero and broadcasts corrected coordinates.
  `edm_fkc` rejects RGI explicitly because its importance-weight calculation
  requires a separate derivation for the added guidance.

Check both the nonzero restraint counts in `built spec:` and the achieved
geometry in saved structures. A zero final energy alone does not establish
that the intended atoms were restrained.

The real-checkpoint smoke check on the preview checkpoint used 200 EDM steps,
structure conditioning, and step scale 1.5. Saved QBP distances for targets
25.0 and 29.1 angstrom were 25.003 and 29.097 angstrom. A ligand with one
tetrahedral center and one E/Z bond preserved both, with nonzero conformer
restraint counts. Repeating an unrestrained sample with the same seed after
the restrained runs reproduced identical coordinates. Context-parallel
execution is wired but has not been validated in this single-GPU check.
