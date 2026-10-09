# RGI-toolkit

Restraint-Guided Inference (RGI) toolkit for diffusion-based structure predictors
(PyTorch and JAX).

The documentation covers the restraint types described in the RGI-Toolkit
paper and the predictor integrations listed below.

The [ColabFold notebooks](docs/colabfold.md) provide forms for **distance, conformer,
angle, custom and RMSD**, including repeated and mixed restraints with native atom
selections. Leave `use_rgi` off for vanilla prediction.

| Model | Integration | Backend | Details |
| --- | --- | --- | --- |
| **Boltz-2** | boltz | torch | [Boltz-2](docs/boltz_restr.md) |
| **AlphaFold3** | alphafold3 | jax | [AlphaFold3](docs/alphafold3_restr.md) |
| **Protenix v2** | protenix | torch | [Protenix v2](docs/protenix_restr.md) |
| **ESMFold2** | esmfold2 | torch | [ESMFold2](docs/esmfold2_restr.md) |
| **OpenFold-3** | openfold-3 | torch | [OpenFold-3](docs/openfold-3_restr.md) |
| **Chai-1** | chai-lab | torch | [Chai-1](docs/chai-lab_restr.md) |
| **OpenDDE v1** | opendde | torch | [OpenDDE](docs/opendde_restr.md) |
| **RF3** | foundry | torch | [RF3](docs/foundry_restr.md) |

See each tool's guide in [`docs/`](docs/) for install / run details, and
[`docs/config.md`](docs/config.md) for the `restraints_config` reference. For common failure modes
and troubleshooting guidance, see the [`FAQ`](docs/FAQ.md).
The [`implementation specification`](docs/SPEC.md) covers API contracts, energy and
gradient conventions, optimizer references, and independent SciPy/E2E validation.

**Ready-to-run samples live in [`examples/`](examples/)** — 4 restraint types (`distance/`,
`angle/`, `rmsd/`, `custom/dist-diff/`) × 7 documented predictors, each a real system with a `run.sh`
that finds the matching fork's env and folds. Start there rather than from the snippets
below: `bash examples/distance/boltz-2/run.sh`. It needs the matching fork checked out as a
sibling of `RGI-toolkit/` and a GPU node; see [`examples/README.md`](examples/README.md) for the
per-tool prerequisites.

RF3 examples and their validation runner are maintained in the
[`foundry_restr` fork](https://github.com/cddlab/foundry_restr/tree/rgi-integration/examples/rgi).

The documented restraint types are minimized during the denoising loop:

- **conformer** — ligand bond lengths, bond angles, chiral volumes, acyclic
  double-bond E/Z geometry (`cistrans`), and intermolecular VdW repulsion.
  Geometry targets come from an ideal reference conformer relaxed with RDKit UFF.
  Each term has an independently adjustable weight.
- **RMSD** — Kabsch-superposed RMSD of selected atoms to a reference PDB or mmCIF.
  Multiple reference restraints can be applied simultaneously.
- **distance** — centroid distance between two atom groups.
- **angle** — angle between three atom-group centroids, with group 2 at the vertex.
- **dihedral** — dihedral angle between four atom-group centroids, about groups 2–3.
- **custom** — a differentiable mathematical expression over named atom selections;
  see [Custom restraints](#custom-restraints).

The default `method='CG'` solver (a nonlinear conjugate gradient with autodiff gradients)
runs on GPU or CPU through torch or jax. For Torch, `gpu: false` forces CPU optimization;
otherwise the input coordinates' device is used. JAX uses its selected device and ignores
`gpu`. All restraints, including distance, use the selected solver.
CG uses SciPy-style PR+ with `line_search: strong-wolfe` (default) or `armijo`,
without a per-atom displacement cap. Mixed distance/conformer CG uses a fixed coordinate transformation
to improve conditioning while preserving every energy and weight. Dynamic VdW caches
remain exact at every trial point. Each restraint is gated by an optional `start_sigma` (active once
`sigma <= start_sigma`) and `stop_sigma` (released once `sigma < stop_sigma`).

## Installation

RGI-toolkit is the shared engine; each integrated tool **declares it as a dependency**, so installing
a tool (`uv pip install -e .` / `pixi install`) pulls it automatically — see the tool's guide in
[`docs/`](docs/). To hack on the engine itself, in this checkout:

```bash
uv sync          # dev environment for this repo
```

### Migrating from rgi-utils

The project is now RGI-toolkit, hosted at
[`cddlab/rgi_toolkit`](https://github.com/cddlab/rgi_toolkit). The distribution is
`rgi-toolkit`, and the Python namespace is `rgi_toolkit`. Update imports from
`rgi_utils` to `rgi_toolkit` and use the updated `rgi-integration` branches of the
predictor forks. Existing restraint configurations and API signatures carry over.
The old import namespace is no longer provided.

The workspace examples use `RGI-toolkit` as the local checkout directory:

```bash
git clone https://github.com/cddlab/rgi_toolkit.git RGI-toolkit
```

For an existing environment, replace the old distribution with the new checkout:

```bash
uv pip uninstall rgi-utils
uv pip install -e ../RGI-toolkit
```

Run these commands from the predictor checkout in its active environment. Recreate
the engine's development environment with `uv venv --clear` followed by `uv sync`
after moving its checkout; virtual-environment entry points retain their original
paths. Remove any generated `src/rgi_utils.egg-info` left by an old editable install.
Refresh dependency locks against the new repository URL instead of retaining a
pre-rename commit.

## Usage

```python
from rgi_toolkit.combined import CombinedRestraints

restraints_config = {
    "gpu": True,                 # False forces Torch CPU optimization; JAX ignores this flag.
    "method": "CG",
    "max_iter": 200,
    "verbose": True,
    # start_sigma / stop_sigma are rejected as top-level keys.
    # They are set per distance/rmsd/group entry and once inside
    # conformer_restraints_config. start_sigma omitted -> active at every step (set it,
    # e.g. 1.0, to act only late); stop_sigma omitted -> never released.
    "distance_restraints_config": [          # a list of entries
        {
            "atom_selection1": "chain A and resid 10",
            "atom_selection2": "chain B and resid 20",
            "harmonic": {"target_distance": 5.0},
            "start_sigma": 1.0,              # optional; active when sigma <= start_sigma
            # "move": "both",               # which group the centroid shift moves: both / 1 / 2
        }
    ],
    "angle_restraints_config": [             # group-centroid angle: 3 groups, vertex = group 2
        {
            "atom_selection1": "chain A and resid 1 to 10",
            "atom_selection2": "chain A and resid 40 to 50",
            "atom_selection3": "chain A and resid 80 to 90",
            "harmonic": {"target_angle": 90.0},   # degrees
        }
    ],
    "conformer_restraints_config": {
        # Applied to ligand objects with conformer_restraints: true.
        # An empty mapping enables bond/angle/chiral/cistrans/vdw at weight 1.
    },
    "custom_restraints_config": [            # custom energy formula (DSL)
        {"name": "symmetric",               # keep two inter-domain distances equal
         "energy": "(distance(A, B) - distance(C, D))**2",
         "selections": {"A": "chain A and resid 10", "B": "chain B and resid 10",
                        "C": "chain A and resid 90", "D": "chain B and resid 90"}},
    ],
    # "rmsd_restraints_config": [{"ref_pdb": "ref.pdb", "harmonic": {"target_rmsd": 0.0}}],
    # "dihedral_restraints_config": [...],   # group-centroid dihedral: 4 groups, axis = 2-3
}

# Create one instance per structure. setup() takes the config dict.
restr = CombinedRestraints()
restr.setup(adapter, nbatch=multiplicity, config=restraints_config)

# Inside the denoising loop, right after the network's denoised x0 prediction:
coords = restr.minimize(coords, step, sigma)   # torch/numpy: mutates in place + returns
# After sampling (optional per-term energy log when verbose):
restr.finalize(coords, step)
```

For a **JAX** tool whose loop runs inside `lax.scan` (no Python callbacks), build the
spec outside the scan and grab the pure closure with `restr.get_minimizer()`
(`(flat_coords, sigma) -> flat_coords`), then call it inside the compiled loop instead
of `minimize`.

The default CG uses SciPy 1.17.1 PR+ with strong-Wolfe searches. Set
`line_search: armijo` for historical backtracking and the small-energy-change
stop, or `method: l-bfgs` with no `line_search` key for L-BFGS. For CG, set
`return_info=True` on `minimize` or `get_minimizer` to obtain `(coords, CGInfo)`
and distinguish gradient convergence, a small energy change, search failure and an
iteration limit. A failed CG search keeps the last accepted coordinates.
See the [solver specification](docs/SPEC.md#nonlinear-conjugate-gradient)
and [diagnostic fields](docs/SPEC.md#public-lifecycle).

### Atom selection syntax

Distance restraints use a selection DSL to specify atom groups. Its keyword / range /
boolean vocabulary is **MDTraj-like** (`resid 1 to 5`, `and` / `or` / `not`,
`protein` / `backbone` / …), though `chain` takes letter ids and `resid` is the
per-chain 1-based ordinal:

| Example                                  | Meaning                                               |
| ---------------------------------------- | ----------------------------------------------------- |
| `chain A`                                | all atoms in chain A                                  |
| `resid 10`                               | residue 10 (1-based, per-chain ordinal)               |
| `resid 1 to 5`                           | residues 1–5                                          |
| `resid 1 3 7`                            | residues 1, 3, 7                                      |
| `index 42`                               | atom at padded index 42                               |
| `name CA`                                | atoms named CA (case-insensitive)                     |
| `protein` / `dna` / `rna`                | polymer-type selectors                                |
| `backbone` / `sidechain`                 | MDTraj-like polymer selectors (gated on polymer type) |
| `chain A and resid 1 to 5`               | boolean AND                                           |
| `chain A or chain B`                     | boolean OR                                            |
| `not chain A`                            | negation                                              |
| `(chain A or chain B) and resid 1 to 10` | parenthesized expressions                             |

### Distance restraint types

| Type             | Parameters                             | Behavior                           |
| ---------------- | -------------------------------------- | ---------------------------------- |
| `harmonic`       | `target_distance`                      | Quadratic penalty at all distances |
| `flat-bottomed`  | `target_distance1`, `target_distance2` | No penalty between d1–d2           |
| `flat-bottomed1` | `target_distance1`                     | Penalty only below d1              |
| `flat-bottomed2` | `target_distance2`                     | Penalty only above d2              |

Distance is calculated between the centroids (unweighted geometric centers) of the two selected atom groups (`calc_method: "unfixed-absolute"`).

The per-entry `move` key picks which group the CG moves toward the target: `both`
(default, both move — minimal-displacement split) / `1` / `2` (pin the other group via
`stop_gradient` — e.g. move only a ligand toward a fixed pocket).

A distance entry may use one external reference group: keep `atom_selection1/2`, write the
reference-side value as `ref1 and <selection>`, and define `refs.ref1` with `ref_pdb` or
`ref_cif`. The ref group stays fixed; `move` may name only prediction-side group indices
(`all`/omitted = all prediction groups).

### Angle / dihedral restraints

`angle_restraints_config` uses three groups with group 2 at the vertex.
`dihedral_restraints_config` uses four groups with groups 2–3 defining the axis.
Both restrain centroid geometry and accept the same four penalty types as distance.
Targets are in **degrees** (`target_angle` / `target_dihedral`); `weight` defaults
to 1.0. `move` selects which groups are free. By default, the arms move and the
anchor groups are pinned. Reference groups stay fixed; with references,
omitted/`all`/`both` moves every prediction group.

### Custom restraints

Define a custom restraint as a differentiable mathematical expression over named
selections. The same formula runs on Torch and JAX:

```yaml
custom_restraints_config:
  - name: symmetric                       # keep two inter-domain distances equal
    energy: "(distance(A, B) - distance(C, D))**2"
    selections: {A: "chain A and resid 10", B: "chain B and resid 10",
                 C: "chain A and resid 90", D: "chain B and resid 90"}
    move: [A, C]                              # B and D are pinned for this term
    weight: 1.0
```

The formula exposes geometry (`distance` `angle` `dihedral` `centroid`
`norm` `dot`), penalty (`harmonic` `flat_bottomed` `flat_bottomed1` `flat_bottomed2`), and math
(`sqrt` `exp` `log` `abs` `sin` `cos` `clip` `minimum` `maximum` `where` `sum` + arithmetic). Branch with
`where(cond, a, b)` or the conditional expression `a if cond else b` (the same thing — `if` is lowered to
`where`), combining conditions with `and` / `or` / `not`; branching is elementwise and **evaluates both
branches**, so a `NaN` in the dead branch poisons the gradient. The energy (× `weight`) is added to the CG objective with the
usual `start_sigma` / `stop_sigma` gating. `move` accepts a prediction selection name or list
of names; omitted/`all`/`both` moves every prediction selection, while ref-backed selections stay fixed. Formulas are parsed safely (no
`eval`). A custom selection can use
`refN and <selection>`; all geometry functions accept it, and `rmsd(A,B)` requires prediction
selection A and reference-backed selection B. Full reference:
[`docs/config.md`](docs/config.md) (the `custom_restraints_config` section).

### Implementing a framework adapter

```python
from rgi_toolkit.atom_context import AtomRecord, LigandConf
from typing import Iterator

class MyAdapter:
    # Required for distance restraints:
    def iter_atoms(self) -> Iterator[AtomRecord]:
        for atom in self.real_atoms:                 # skip padding
            yield AtomRecord(
                chain=atom.chain_id,
                resid=atom.per_chain_ordinal,        # 1-based, resets at each chain
                index=atom.row_in_coord_tensor,      # global flat index into the coord tensor
            )

    # Optional — add these for conformer / VdW restraints:
    def num_atoms(self) -> int: ...                  # padded coord-tensor length
    def get_elements(self): ...                      # (num_atoms,) atomic numbers, 0 = padding
    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        for lig in self.ligands:                     # one LigandConf per ligand
            yield LigandConf(
                mol,
                conf_coords,
                global_indices,
                stereo_mol=source_mol_in_coordinate_order,
            )
```

For a SMILES ligand, pass its source-graph molecule as `stereo_mol` after renumbering it
to `mol`/`global_indices` order. This keeps the input `@`/`@@` and E/Z labels available
when the framework's reference conformer has already inverted them.

Tool-side adapters are tiny — see `src/rgi_toolkit/{boltz,protenix,chai,openfold3}/adapter.py`
for worked examples, and the shared `implement-rgi` skill under `.claude/skills/` and
`.agents/skills/` for the full integration recipe.

## Development

```bash
task lint       # check style
task format     # auto-fix style
task test-ci    # short CI selection with critical correctness checks
task test-local # comprehensive CPU regressions and real compilation checks
task test-gpu   # complete GPU tests in a CUDA-enabled environment
task test       # alias for test-local
```

Prepare the development environment with
`uv sync --extra torch --extra jax`. Test tasks use an activated
environment when present and otherwise use the project environment, without
resolving or replacing its dependencies. The short CI selection includes numerical
parity and representative public-API minimization tests; exhaustive combinations,
compiler checks, and GPU tests remain in the local suites. See
[testing](docs/testing.md) for coverage and GPU environment setup.

## Citation

If you use RGI-toolkit, please cite:

Hori, T., Moriwaki, Y. & Ishitani, R. (2026). *RGI-Toolkit: Differentiable Restraints
for Controllable Biomolecular Structure Prediction.* bioRxiv, preprint, version 1.
[doi:10.64898/2026.10.05.756905](https://doi.org/10.64898/2026.10.05.756905).

```bibtex
@article{hori2026rgi,
  author  = {Hori, Tatsuki and Moriwaki, Yoshitaka and Ishitani, Ryuichiro},
  title   = {{RGI-Toolkit}: Differentiable Restraints for Controllable Biomolecular Structure Prediction},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.05.756905},
  url     = {https://www.biorxiv.org/content/10.64898/2026.10.05.756905v1},
  note    = {Preprint, version 1}
}
```
