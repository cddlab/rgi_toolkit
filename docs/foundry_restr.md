# RF3 (Foundry)

RGI is integrated into **RF3**, the structure predictor in
[`foundry_restr`](https://github.com/cddlab/foundry_restr/tree/rgi-integration).
The other Foundry models are outside this integration.

## Install and run

From the Foundry fork on `rgi-integration`:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend cu128 \
  -c constraints-rf3.txt -e '.[rf3]'
uv run --no-project --python .venv/bin/python foundry install rf3
bash examples/rgi/run.sh distance.json distance
```

During co-development, install the matching sibling toolkit checkout into the
same environment with `uv pip install --python .venv/bin/python -e ../RGI-toolkit`.
RF3 supplies PyTorch; no separate RGI backend extra is needed. Inference requires
the usual RF3 model checkpoint. Use your cluster's scheduler for GPU work.

## Configuration

Each RF3 JSON job carries one shared [`restraints_config`](config.md):

```json
{
  "name": "example",
  "components": [{"seq": "GLKEMALQ", "chain_id": "A"}],
  "restraints_config": {
    "verbose": true,
    "distance_restraints_config": [{
      "atom_selection1": "chain A and resid 1 and name CA",
      "atom_selection2": "chain A and resid 8 and name CA",
      "harmonic": {"target_distance": 12.0}
    }]
  }
}
```

Run with `uv run --no-project --python .venv/bin/python rf3 fold inputs=job.json
out_dir=predictions`. Omitted/null configuration retains ordinary RF3 sampling.
All shared restraint types, custom energies, references, solver settings and
sigma/step windows use the common engine without tool-specific parsing.

For conformer geometry, add `"conformer_restraints": true` to each component
to restrain, and supply `conformer_restraints_config` in the job. This covers
ligands and polymers, including replicated chains. The default is opt-out.
RF3's native `ground_truth_conformer_selection` is a separate model input.

Selections use processed chain IDs, one-based token ordinals within each chain,
and zero-based coordinate-row atom indices. Ligands and atomized modified
residues can have one token per atom. Molecular types come from AtomWorks entity
types, so a modified protein residue is not treated as a ligand solely because
it has `hetero=True`.

External `config_path` references are relative to the input JSON; resource paths
inside external configurations are relative to their own files. Inline reference
paths keep the shared engine's working-directory semantics.

The Python `InferenceInput.from_atom_array()` and `from_cif_path()` APIs accept
`restraints_config=...` and a per-chain `conformer_restraints={"B": True}` mapping.
See the fork's [full guide](https://github.com/cddlab/foundry_restr/blob/rgi-integration/models/rf3/docs/rgi.md)
for complete examples and the validated CUDA dependency constraints.

## Lifecycle and validation

Each processed structure gets a fresh `CombinedRestraints` instance. RF3 passes
the same instance through its trainer and network into the diffusion sampler.
The hook minimizes each denoised prediction **before** the integrator update,
using the pre-churn schedule sigma and zero-based rollout step. All samples are
optimized independently. `finalize` measures the final integrated coordinates.
An active final restraint can have a small nonzero residual after that last update.

Reference coordinates and atom-to-token mappings must come from the same
post-transform features as the sampler. Source ligand chemistry preserves bond
orders, formal charges and stereochemistry; the toolkit builds and optimizes all
restraint terms.

The fork's `examples/rgi/` covers different distance targets in one batch and
ATP/fumarate conformer restraints. Verify nonzero spec counts, finite coordinates
and measured geometry for every output sample. Confirm that the unrestrained
control and empty configuration follow the ordinary RF3 path. Generated outputs
and local model caches are ignored.

The [validation report](https://github.com/cddlab/foundry_restr/blob/rgi-integration/models/rf3/docs/rgi-validation.md)
records real-checkpoint measurements for all 22 sample CIFs, CPU regression
results and commands to repeat the checks.
