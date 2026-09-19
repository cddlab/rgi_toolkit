# ColabFold notebooks

[ColabFold's RGI branch](https://github.com/th2ch-g/ColabFold/tree/rgi-integration)
provides beginner forms for the toolkit's existing predictor families. Start with its
[Colab notebook](https://colab.research.google.com/github/th2ch-g/ColabFold/blob/rgi-integration/ColabFold2_preview.ipynb)
and [usage guide](https://github.com/th2ch-g/ColabFold/blob/rgi-integration/docs/rgi.md).

The preview supports AlphaFold3, OpenFold3, Boltz2, Protenix2, RoseTTAFold3, Chai1,
OpenDDE and the three exposed ESMFold2 variants through ColabFold's shared JAX port.
The legacy Boltz-1 notebook uses the existing native PyTorch RGI integration.
The predictor's **use_rgi** switch preserves vanilla execution.

`rgi_toolkit.notebook` contains framework-independent form helpers:

- `residue_selection(chain, residues, atoms)` translates one-based ranges into the
  ordinary chain-qualified selection DSL.
- `make_config(...)` constructs distance/ligand presets or resolves a complete
  YAML/JSON configuration using the shared parser. A positive distance tolerance
  creates a flat-bottomed interval; it is not an optimizer tolerance.
- `restraint_inventory(restraints)` reports actual built rows, with dynamic VdW
  inventory separately identified.
- `distance_report(restraints, coords)` measures final centroid distances per sample.

The forms do not implement energies, optimizers or their own selection parser. The
ColabFold shim resolves its ligand chemistry and reuses `AF3RestraintAdapter`.
OpenDDE's structural-token map is handled at the coordinate boundary, retaining
per-chain residue numbering. JAX minimizers are passed as pytrees to avoid carrying
numeric restraint settings from one prediction into another.

Conformer presets still require explicit entity opt-in. The UI selects ligand chains
when the user requests ligand geometry; it never silently opts protein chains in.
Custom configurations retain all toolkit options, including external files and reference
structures. Config validation does not replace inspecting actual built counts and final
geometry.
