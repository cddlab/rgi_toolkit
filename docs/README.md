# RGI documentation

Use the tool-specific guide for installation, config placement, and the run command. Use the
shared [`restraints_config` reference](config.md) for restraint semantics and defaults. See the
[FAQ](FAQ.md) for common failure modes and troubleshooting guidance.
The [implementation specification](SPEC.md) describes the toolkit's API, mathematical
contracts, optimizer algorithms, and verification against independent implementations.

## Tool guides

These guides cover the six predictors described in the paper.

| Predictor         | Backend | Config placement                                              | Guide                              |
| ----------------- | ------- | ------------------------------------------------------------- | ---------------------------------- |
| Boltz-2 | PyTorch | `restraints_config` in the input YAML                         | [Boltz](boltz_restr.md)            |
| AlphaFold 3       | JAX     | `restraints_config` in the fold-input JSON                    | [AlphaFold 3](alphafold3_restr.md) |
| Protenix v2  | PyTorch | `restraints_config` in each fold-input JSON object            | [Protenix](protenix_restr.md)      |
| ESMFold2          | PyTorch | Python dict passed to `fold()`                                | [ESMFold2](esmfold2_restr.md)      |
| OpenFold 3        | PyTorch | `queries.<name>.restraints_config` in the input JSON          | [OpenFold 3](openfold-3_restr.md)  |
| Chai-1            | PyTorch | Top-level sidecar YAML passed with `--restraints-config-path` | [Chai-1](chai-lab_restr.md)        |

## Recommended workflow

1. Open the guide for your predictor and follow its installation and config-placement rules.
2. Define the restraints using the [configuration reference](config.md).
3. Check the input syntax and atom selections before submitting a GPU job.
4. Run with `verbose: true` and verify that every requested restraint has a non-zero count in the
   `built spec:` log.

## Conventions shared by every tool

- `resid` is the per-chain, 1-based residue or token ordinal. It is not an author residue number
  or a global index. Qualify residue selections with `chain`.
- Use `start_sigma` and `stop_sigma` to set each restraint's activation window.
- Conformer terms require both `conformer_restraints_config` and a per-ligand opt-in. The opt-in
  location differs by tool.
- The backend is inferred from the predictor. `gpu` selects the PyTorch device and is inert for
  AlphaFold 3's JAX path.
- Config validation checks schema and selection syntax. Runtime `built spec:` counts confirm
  which restraints were built; verify the selected atom identities separately against the input
  structure to check that the selections match the intended atoms.
