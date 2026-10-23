<p align="center">
  <img src="assets/rgi-toolkit-logo.svg" alt="RGI-toolkit" width="640">
</p>

# `RGI-toolkit`: Differentiable Restraints for Controllable Biomolecular Structure Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ColabFold2_preview.ipynb)

[Documentation](docs/README.md) | [Examples](examples/README.md) | [Colab guide](docs/colabfold.md) | [Paper](https://doi.org/10.64898/2026.10.05.756905)

`RGI-toolkit` applies **Restraint-Guided Inference (RGI)** to diffusion-based
biomolecular structure predictors. Define geometric restraints on selected atoms
or atom groups, then optimize their coordinates during prediction with a shared
PyTorch or JAX engine.

At each active diffusion step, the predictor applies:

**Denoising prediction → Restraint minimization → Coordinate update**

Try RGI in [Google Colab](#google-colab), or use one of the
[supported predictor integrations](#supported-predictors) locally.

## Features

- Control **distances, angles and dihedrals** between selected atom groups.
- Restrain **ligand geometry** with bond, angle, chiral, cis/trans and VdW terms.
- Guide structures toward one or more **PDB/mmCIF references** with superposed RMSD restraints.
- Define **custom differentiable energies** as expressions over named atom selections.
- Combine restraint types, adjust weights and set activation windows during denoising.
- Use the same configuration and atom-selection syntax across **PyTorch and JAX** integrations.

The [configuration reference](docs/config.md) describes each restraint type,
its targets, penalties and defaults.

## Installation

For local prediction, follow the installation guide for your
[chosen predictor](#supported-predictors). Each integration installs `rgi-toolkit`
as a dependency and provides the model-specific inference command.
The shared engine requires **Python 3.12 or later** and a PyTorch or JAX backend;
model weights and GPU requirements are covered in each predictor's guide.

Clone this repository to use the example inputs or develop the shared engine:

```bash
git clone https://github.com/cddlab/rgi_toolkit.git
cd rgi_toolkit
```

See [Development](#development) for the engine's development environment.

## Quick Examples

The following example uses **Boltz-2** to guide QBP toward an interdomain centroid
distance of **25 Å**. Install the [Boltz integration](docs/boltz_restr.md), activate
its environment, and run from this repository's root on a GPU compute node:

```bash
boltz predict examples/distance/boltz-2/qbp_25.00.yaml \
    --model boltz2 --use_msa_server --seed 0 \
    --out_dir examples/distance/boltz-2/out
```

The input file contains both the protein sequence and its `restraints_config`.
The command fetches an MSA from the server and writes results under the selected
output directory.

The bundled examples cover four protocols across seven predictors:

| Example | System | Restraint |
| --- | --- | --- |
| [Distance](examples/distance/) | QBP | Interdomain centroid distance of 25 Å |
| [Angle](examples/angle/) | ADK | NMP–CORE–LID centroid angle of 72.85° |
| [RMSD](examples/rmsd/) | QBP | RMSD targets of 2.65 Å to both open and closed references |
| [Custom energy](examples/custom/dist-diff/) | DgoT | Difference between two domain distances of 0.8 Å |

See the [example guide](examples/README.md) for predictor-specific environments,
MSA inputs, reference structures and run commands. RF3 examples are maintained in
the [Foundry integration](https://github.com/cddlab/foundry_restr/tree/rgi-integration/examples/rgi).

## Google Colab

**RGI is available in the upstream ColabFold notebooks.** The forms support
distance, conformer, angle, custom and RMSD restraints, including multiple
restraints and native atom selections.

1. Open a notebook below and choose an RGI-supported model.
2. Enter your molecules in **Input sequences**.
3. Enable **use_rgi** in the **RGI** form and fill in the restraints you need.
4. Select **Runtime → Run all**.

| Notebook | Use case |
| --- | --- |
| [ColabFold2 preview](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ColabFold2_preview.ipynb) | Choose among the RGI-supported preview models |
| [AlphaFold3 / OpenFold3](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold3_of3.ipynb) | AlphaFold3, OpenBind-0 and OpenFold3 |
| [Boltz-1](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/Boltz1.ipynb) | Native Boltz-1 inference |

Leave `use_rgi` off for vanilla prediction. See the [Colab guide](docs/colabfold.md)
for supported models, form examples, reference uploads and result downloads.

## Supported Predictors

The backends below refer to the local predictor integrations.

| Model | Integration | Backend | Guide |
| --- | --- | --- | --- |
| **Boltz-1** | boltz | PyTorch | [Boltz-1](docs/boltz_restr.md) |
| **Boltz-2** | boltz | PyTorch | [Boltz-2](docs/boltz_restr.md) |
| **AlphaFold3** | alphafold3 | JAX | [AlphaFold3](docs/alphafold3_restr.md) |
| **Protenix v1** | protenix | PyTorch | [Protenix v1](docs/protenix_restr.md) |
| **Protenix v2** | protenix | PyTorch | [Protenix v2](docs/protenix_restr.md) |
| **ESMFold2** | esmfold2 | PyTorch | [ESMFold2](docs/esmfold2_restr.md) |
| **OpenFold-3** | openfold-3 | PyTorch | [OpenFold-3](docs/openfold-3_restr.md) |
| **Chai-1** | chai-lab | PyTorch | [Chai-1](docs/chai-lab_restr.md) |
| **OpenDDE v1** | opendde | PyTorch | [OpenDDE](docs/opendde_restr.md) |
| **RF3** | foundry | PyTorch | [RF3](docs/foundry_restr.md) |
| **BioEmu2** | bioemu2 | PyTorch | [BioEmu2](docs/bioemu2_restr.md) |

Each guide covers installation, input configuration and the prediction command.
The [documentation index](docs/README.md) also summarizes where each predictor
accepts `restraints_config`.

## Configuration

For a protein chain A with at least 50 residues, this block targets a centroid
distance of 25 Å between two residue groups:

```yaml
restraints_config:
  verbose: true
  distance_restraints_config:
    - atom_selection1: chain A and resid 1 to 10
      atom_selection2: chain A and resid 40 to 50
      harmonic:
        target_distance: 25.0
```

Place the block in your predictor's input as described in its integration guide.
Residue selections use **per-chain, 1-based ordinals**; qualify them with a chain.
The [atom-selection guide](docs/config.md#atom-selection-dsl) covers atom names,
residue ranges, molecule types and boolean expressions.

Use `start_sigma` and `stop_sigma` on each restraint entry, or on the shared
conformer block, to control its activation window. Conformer restraints also
require a per-ligand opt-in in the predictor's input. See the
[configuration reference](docs/config.md) for penalty shapes, weights, reference
groups and custom expressions.

The default optimizer is nonlinear conjugate gradient with a strong-Wolfe line
search; Armijo and L-BFGS are also available. For PyTorch, `gpu: false` runs the
restraint optimizer on CPU. JAX uses its selected device. Optimizer settings and
convergence criteria are described in the
[configuration reference](docs/config.md#top-level-keys).

## Python API

Predictor integrations construct one `CombinedRestraints` instance per structure,
prepare its restraints, and call it after each denoising prediction:

```python
from rgi_toolkit.combined import CombinedRestraints

restraints = CombinedRestraints()
restraints.setup(adapter, nbatch=multiplicity, config=restraints_config)

# Inside the predictor's denoising loop.
coords = restraints.minimize(coords, step, sigma)

# After sampling, emit optional per-term diagnostics.
restraints.finalize(coords, step)
```

Here, `adapter`, `multiplicity`, `coords`, `step` and `sigma` come from the
predictor, and `restraints_config` is the configuration dictionary. JAX integrations
inside `lax.scan` use the pure closure returned by `get_minimizer()`.

## Output

The selected predictor writes the predicted structures and confidence metrics.
With `verbose: true`, RGI logs the built restraint counts at setup and per-term
energies at finalization. Check the counts and selected atoms to confirm that the
requested restraints were applied.

Colab result archives include the effective input and restraint configuration;
the preview notebook also includes `rgi_report.json`. See the
[Colab guide](docs/colabfold.md#predictors-and-results) for details.

## Documentation

| Topic | Guide |
| --- | --- |
| Predictor setup and input formats | [Integration guides](docs/README.md) |
| Restraints, atom selections and defaults | [Configuration reference](docs/config.md) |
| Browser-based prediction | [Colab notebooks](docs/colabfold.md) |
| Example systems and run commands | [Examples](examples/README.md) |
| Benchmark protocols and analysis | [Benchmark guide](benchmark/README.md) |
| Common problems | [FAQ](docs/FAQ.md) |

## Agent Skills

Bundled skills live in [`.claude/skills/`](.claude/skills/) and are shared with
Codex through [`.agents/skills/`](.agents/skills/).

| Skill | Purpose |
| --- | --- |
| [generate-rgi-config](.claude/skills/generate-rgi-config/SKILL.md) | Create and validate restraint configurations |
| [implement-rgi](.claude/skills/implement-rgi/SKILL.md) | Add RGI to a predictor's sampling loop |
| [sync-upstream](.claude/skills/sync-upstream/SKILL.md) | Update supported integration forks from upstream |
| [create-github-release](.claude/skills/create-github-release/SKILL.md) | Prepare version tags and GitHub releases |

## Getting Help

Check the [FAQ](docs/FAQ.md) for common restraint and runtime issues, or open an
[issue](https://github.com/cddlab/rgi_toolkit/issues) with the predictor, configuration
and relevant logs.

## Development

Prepare the engine's development environment from the repository root:

```bash
uv sync --extra torch --extra jax --extra notebook
```

```bash
task lint       # Check style.
task format     # Format and fix style.
task test-ci    # Run the short CI selection.
task test-local # Run CPU regressions and real compilation checks.
task test-gpu   # Run the complete GPU selection.
task test       # Alias for test-local.
```

Test tasks use an activated environment when present and otherwise use the project
environment. Run GPU checks on a GPU compute node.

## Citation

If you use RGI-toolkit, please cite:

Hori, T., Moriwaki, Y. & Ishitani, R. (2026). *RGI-Toolkit: Differentiable Restraints
for Controllable Biomolecular Structure Prediction.* bioRxiv, preprint.
[doi:10.64898/2026.10.05.756905](https://doi.org/10.64898/2026.10.05.756905).

```bibtex
@article{hori2026rgi,
  author  = {Hori, Tatsuki and Moriwaki, Yoshitaka and Ishitani, Ryuichiro},
  title   = {{RGI-Toolkit}: Differentiable Restraints for Controllable Biomolecular Structure Prediction},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.05.756905},
  url     = {https://www.biorxiv.org/content/10.64898/2026.10.05.756905},
  note    = {Preprint}
}
```

## License

[MIT License](LICENSE). Third-party notices are included in the license file.
