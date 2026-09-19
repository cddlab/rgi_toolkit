# RGI in ColabFold notebooks

Open [ColabFold2 preview](https://colab.research.google.com/github/th2ch-g/ColabFold_restr/blob/rgi-integration/ColabFold2_preview.ipynb).
The original notebook headings and workflow are retained. **RGI (optional)** is an
ordinary Colab form after **Input sequences**: its fields are visible and editable
**before any cell runs**. It needs no widget, Add button or separate confirmation.

1. Choose an RGI-supported model, for example **boltz2**, in **Install dependencies**.
2. Enter your molecules in **Input sequences**.
3. In **RGI (optional)**, check **use_rgi** and fill the fields for the restraints you need.
4. Use **Runtime → Run all**.

For vanilla, leave **use_rgi** off. The upstream default model, OpenBind0, is vanilla
only, as are AF2 and IntelliFold2; choose a supported model to enable RGI.
An enabled but empty RGI form raises an error before inference.
After editing a form, rerun that cell and prediction, or use **Run all**.
The form values live in notebook source and are retained when you save the notebook.

## First example: a distance of 25 Å

For protein chain A with at least 50 residues, set these fields in **RGI (optional)**:

| Field | Enter |
| --- | --- |
| `use_rgi` | Checked |
| `distance_atom_selection1` | `chain A and resid 1 to 10` |
| `distance_atom_selection2` | `chain A and resid 40 to 50` |
| `target_distance` | `25` |

Leave the other restraint fields empty. The two selections identify atom groups;
RGI restrains their **centroid distance**. No quotes or dictionary syntax are needed
in the two selection fields. The helper generates an ordinary
`distance_restraints_config` entry with `harmonic: {target_distance: 25}`.

## The five types

All five can be used together in the same form. Basic distance, angle and RMSD fields
create **harmonic** restraints. Use the native `restraints_config` field for bounds,
weights, activation windows, reference geometry and every other toolkit option.

| Type | Fields to fill | Leave disabled |
| --- | --- | --- |
| distance | `distance_atom_selection1`, `distance_atom_selection2`, `target_distance` (Å) | Both selections empty |
| conformer | `conformer_chains`, e.g. `B,C` or `ligands` | Chains empty |
| angle | `angle_atom_selection1`, `angle_atom_selection2`, `angle_atom_selection3`, `target_angle` (degrees) | All three selections empty |
| custom | `custom_selections` and `custom_energy` | Energy empty |
| RMSD | `ref_pdb` or `ref_cif`, optional `atom_selection_target`/`atom_selection_ref`, `target_rmsd` (Å) | Both reference filenames empty |

For angle, selection 2 is the vertex. For custom, an example energy is
`harmonic(distance(A, B), 25.0)` with the named selections A and B in
`custom_selections`. Custom angular functions use radians.

For RMSD, use Colab's **Files → Upload**, then put the uploaded filename in `ref_pdb`
or `ref_cif`. These are paths in the runtime, not paths on your computer.
Empty RMSD selections use the whole structure; `chain A and name CA` selects the
C-alpha atoms in chain A. Multiple references and separate fit/calc selections are
also supported through native configuration.

Conformer is **one shared configuration** with multiple chain opt-ins. Protein, DNA,
RNA and ligand chains can be selected. Its native defaults enable bond, angle, chiral,
cistrans and VdW; plane and torsion default to zero. For example, set the advanced field
to `{"conformer_restraints_config": {"use_esd": True, "plane": {"weight": 1}}}` and
set `conformer_chains` to the actual chains to restrain. A conformer config without
chain opt-ins is rejected.

## Multiple restraints in the same fields

Enter a list in any field to create multiple entries. A single value is reused for
every entry; lists must have either one item or the same length as the longest list.
Selection/filename/energy lists use JSON syntax with double quotes. Numeric fields
accept a number or a list of numbers.

Example: restrain chain A residues 1–10 to two different regions:

| Field | Enter |
| --- | --- |
| `distance_atom_selection1` | `chain A and resid 1 to 10` |
| `distance_atom_selection2` | `["chain A and resid 20 to 30", "chain A and resid 40 to 50"]` |
| `target_distance` | `[15, 25]` |

This makes two distance restraints, targeting 15 Å and 25 Å respectively. It does not
combine the second selections into one group. The same list convention works for
angle, custom and RMSD. For different custom selection mappings, use a list of dicts
in `custom_selections`. Conformer uses a comma-separated list of chain IDs instead.

For different penalties or advanced settings per entry, put any number of entries
in the ordinary native lists under `restraints_config`. These append to the basic
fields rather than overwriting them. Rerunning a cell does not accumulate duplicates.

## Selections and chain IDs

The fields accept the complete [RGI-toolkit selection DSL](config.md#atom-selection-dsl):

```text
chain A and resid 1 to 10
chain A and (resid 1 to 10 or resid 40 to 50)
chain A B and name CA
protein and backbone and not resid 1
```

Residues are numbered from **1 within each chain**, not by author numbering in a
reference structure. Qualify residue selections with a chain. The input cell prints
actual chain IDs; check those after input processing. The engine validates actual
atom matches and reference pairing before the model forward pass.

## Full native configuration and files

`restraints_config` accepts the native Python dict. For example, to allow a distance
up to 8 Å instead of targeting one exact value:

```python
{"distance_restraints_config": [{"atom_selection1": "chain A", "atom_selection2": "chain B", "flat-bottomed2": {"target_distance2": 8}}]}
```

Clear the basic distance fields if you only want this entry. All native sections,
including dihedral, improper, chiral, plane and base pairing, remain available.
See [the configuration reference](config.md) for the complete schema.

To reuse a YAML/JSON file, upload it with Colab's Files panel and enter
`{"config_path": "restraints.yaml"}` in `restraints_config`. Reference paths inside
the file are relative to that file; upload the referenced structures too. Set
`conformer_chains` when a file includes conformer restraints.

## Predictors and results

| Notebook | RGI models |
| --- | --- |
| ColabFold2 preview | AlphaFold3, OpenFold3, Boltz2, Protenix2, RoseTTAFold3, Chai1, OpenDDE, ESMFold2 and LM600M/LM300M variants |
| [AlphaFold3 / OpenFold3](https://colab.research.google.com/github/th2ch-g/ColabFold_restr/blob/rgi-integration/AlphaFold3_of3.ipynb) | AlphaFold3 and OpenFold3 |
| [Boltz-1](https://colab.research.google.com/github/th2ch-g/ColabFold_restr/blob/rgi-integration/Boltz1.ipynb) | Native Boltz-1 |

Preview uses `alphafold3-colabfold==3.1.11` and the shared JAX adapter. ESMFold2's
language-model route requires exactly one protein chain. The Boltz-1 form is after
its input cell and before installation; when switching `use_rgi`, rerun installation
as well to select the matching Boltz environment.

Result names include effective settings to avoid stale output reuse. The result ZIP
contains the effective input, all configured entries and conformer opt-ins. Preview
also includes `rgi_report.json` with per-seed inventories and final distances.
Inspect built counts and geometry as well as confidence. Turning `use_rgi` off strips
RGI settings and entity opt-ins, even if the form contains unfinished entries.

## Shared implementation and verification

`notebook_colab.COLAB_FORM` is the source for the standard form in all three notebooks;
`config_from_fields` converts its values into the shared native configuration.
ColabFold supplies only molecule input and predictor-specific execution. Configuration,
selection, geometry and optimization remain in RGI-toolkit. The optional
`notebook_widgets.RestraintEditor` is still available for separate Jupyter applications;
ColabFold does not use it or require `ipywidgets`.

`examples/colabfold/mixed_restraints.py` exercises vanilla, all five types together,
then vanilla again and checks written CIF geometry. ColabFold scripts cover the
model matrix and batch isolation. Generated results belong under `.cache`. Release
Colab sessions after collecting E2E results.
