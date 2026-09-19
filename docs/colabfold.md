# RGI in ColabFold notebooks

Open [ColabFold2 preview](https://colab.research.google.com/github/th2ch-g/ColabFold_restr/blob/rgi-integration/ColabFold2_preview.ipynb). The notebook shows the setup steps before anything is run.

1. In **1. Install and choose RGI**, check **use_rgi** and click the play button on the left.
2. Enter your sequences and ligands in **2. Enter molecules**, then run that cell.
3. Find **3. Configure RGI — click ▶ to open the settings** and click its play button.
   **The editable form appears directly below this cell after it runs.** The notebook
   introduction also links straight to this cell.
4. Click **Add distance**, **Add conformer**, **Add angle**, **Add custom** or **Add RMSD**.
   Fill in the entry below the buttons. Repeat to add more restraints.
5. Click **Check RGI settings**, then run **4. Predict structure**.

For RGI, stop at the configuration step and edit the form before prediction.
A new form has no restraints; running prediction first explains how to open and fill it.
For vanilla, leave **use_rgi** unchecked. If the form says RGI is OFF, enable the switch
in step 1, rerun installation, then rerun step 3. When changing a finished configuration,
edit its controls and rerun prediction; rerunning the form cell preserves your entries.
In Boltz-1, the switch is in **1. Enter molecules and choose RGI**, and the form is
**4. Configure RGI** after installation and MSA generation.

### First example: one distance

For a protein chain A with at least 50 residues, click **Add distance** and fill in:

| Visible field | Value | Meaning |
| --- | --- | --- |
| `atom_selection1` | `chain A and resid 1 to 10` | The first group of atoms |
| `atom_selection2` | `chain A and resid 40 to 50` | The second group of atoms |
| `penalty` | `harmonic — exact target` | Pull the groups to one distance |
| `target_distance` | `25` | Desired centroid distance in Angstrom |

Use the chain IDs and residue ranges from **Your input chains** for your own molecules.
Click **Check RGI settings** and then run **Predict structure**. For a second distance,
click **Add distance** again or **Duplicate** on the first entry and edit its selections.

The editor and config helpers live in **RGI-toolkit**, in `notebook_widgets.py` and
`notebook.py`. ColabFold supplies the molecule input and its predictor-specific hooks.
The editor uses the toolkit's native configuration names and selection DSL.

## Add, combine and repeat restraints

The five **Add** buttons show each native type and what it does. Each entry can be edited, duplicated,
disabled or removed. For example, add two distances, an angle, two RMSDs and a custom
energy together. They become separate entries in the corresponding native lists;
adding another restraint never replaces a previous entry.

| Type | Native section | Form controls |
| --- | --- | --- |
| distance | `distance_restraints_config` | Two atom selections; centroid distance in Angstrom |
| conformer | `conformer_restraints_config` | Chain opt-ins and bond, angle, chiral, cistrans, VdW, plane and torsion settings |
| angle | `angle_restraints_config` | Three atom selections; centroid angle with group 2 as the vertex |
| custom | `custom_restraints_config` | An energy expression and any number of named atom selections |
| RMSD | `rmsd_restraints_config` | Reference PDB/mmCIF, target/ref selections, optional separate fit/calc selections |

**conformer is one shared mapping**, as in RGI-toolkit. Select multiple target chains in
that card; do not create a conformer entry per chain. The other four types support any
number of entries. Explicitly disabling a card excludes it from prediction.

Click **Check RGI settings** before prediction. The next step is shown below the button.
Expand **View the native RGI configuration** to inspect the YAML. This check validates syntax. Actual atom matches, reference pairing and nonzero built counts
are checked by the engine during setup, before the model forward pass.

## Selections

Selection fields accept the complete toolkit DSL. They are not limited to one chain,
one residue range, or a fixed atom menu:

```text
chain A and resid 1 to 10
chain A and (resid 1 to 10 or resid 40 to 50)
chain A B and name CA
protein and backbone and not resid 1
```

Residues are numbered from **1 within each chain**, independently of author numbering in
a reference structure. Check the chain table above the form. Distance and angle act on
the selected groups' centroids. See the [selection reference](config.md#atom-selection-dsl)
for the complete language.

## distance and angle

Enter `atom_selection1`, `atom_selection2`, and, for angle, `atom_selection3`.
Choose a native penalty: `harmonic` targets one value, `flat-bottomed` permits a range,
`flat-bottomed1` sets a lower bound, and `flat-bottomed2` sets an upper bound.
The form shows the corresponding `target_distance` or `target_angle` fields.
Distance units are Angstrom. Angle units default to degrees and can be changed to radians.

Expand **Weight, activation window and additional settings** for `weight`, `move`,
sigma/step windows, or native fields such as `refs`. Leaving a window at `always` uses
the toolkit default. Additional settings cannot silently override a visible field.

## conformer

Enter comma-separated chain IDs in **conformer_chains**, for example `B,C`.
`ligands` selects all ligand chains. Protein, DNA and RNA chains can also be selected
explicitly. The notebook writes each selected entity's `conformer_restraints: true` flag.

Defaults match RGI-toolkit: bond, angle, chiral, cistrans and VdW have weight 1;
plane and torsion have weight 0. Expand the terms to change their weights and slack,
or choose VdW mode/scale. `use_esd` defaults to false. For polymer dictionary geometry,
`monomer_library` accepts YAML such as `true`, a library path, or a mapping. Consult
[conformer settings](config.md#conformer_restraints_config-single-dict) for units,
dictionary behavior and chemical coverage. A monatomic ion has no internal bond/angle
restraints; inspect the actual built counts.

## custom

**custom means a custom energy**, not a whole-config input mode. Add named selections
with **Add selection**, then use those names in `energy`, for example:

```text
A: chain A and resid 1 to 10
B: chain A and resid 40 to 50
C: chain B and name CA

energy: (distance(A, B) - distance(A, C))**2
```

The expression is parsed by RGI-toolkit, not evaluated as Python. Custom angular functions
return radians. `use` selects a registered custom function instead of an expression.
See the [custom reference](config.md#custom_restraints_config-list) for supported functions
and reference-backed selections. Each custom card has its own selections and label.

## RMSD

Click **Upload reference** inside the RMSD entry to choose a PDB/mmCIF file. The
reference format and file path are filled in automatically. Alternatively, choose
`ref_pdb` or `ref_cif` and enter an existing runtime path in **reference_file**. Use `atom_selection_target` and `atom_selection_ref` for
both fit and measurement, or expand **Separate fit and calc selections** to set the four
native `_fit`/`_calc` fields independently. Empty selections retain the toolkit's
whole-structure behavior. `pairing` and `best_effort` use the native defaults.

`harmonic` with `target_rmsd: 0` pulls the selection onto the superposed reference.
`flat-bottomed2` allows RMSD up to `target_rmsd2`. Multiple RMSD cards can use different
references and regions. See the [RMSD reference](config.md#rmsd_restraints_config-list).

## Full YAML/JSON and files

**Input method** selects the form, pasted YAML/JSON, or a YAML/JSON file. Whole-config text and files
are separate from the custom energy type. They support all toolkit sections, including
dihedral, improper, chiral, plane and base pairing. A config file's reference paths are
relative to that file. Upload any referenced structures too.

Use **Load into form** to edit the five supported types as cards. Other native sections
remain under **Advanced global settings**, so importing does not discard them.
For conformer in YAML/file mode, also set **conformer_chains** to opt the intended entities in.

## Predictors, results and vanilla

| Notebook | RGI models |
| --- | --- |
| ColabFold2 preview | AlphaFold3, OpenFold3, Boltz2, Protenix2, RoseTTAFold3, Chai1, OpenDDE, ESMFold2 and LM600M/LM300M variants |
| [AlphaFold3 / OpenFold3](https://colab.research.google.com/github/th2ch-g/ColabFold_restr/blob/rgi-integration/AlphaFold3_of3.ipynb) | AlphaFold3 and OpenFold3 |
| [Boltz-1](https://colab.research.google.com/github/th2ch-g/ColabFold_restr/blob/rgi-integration/Boltz1.ipynb) | Native Boltz-1 |

Preview uses the shared JAX port pinned to `alphafold3-colabfold==3.1.11` (Python 3.13).
It reuses `AF3RestraintAdapter` and checks the sampler source before injecting hooks.
AlphaFold2, OpenBind0 and IntelliFold2 support vanilla only; enabling RGI raises an error.
ESMFold2's language-model route requires exactly one protein chain.

The result ZIP includes structures and the effective input with all configured entries
and entity opt-ins. Preview also saves `rgi_report.json` with per-seed inventories and
measured final distances. The engine log reports final per-term energies. Check nonzero
counts and geometry as well as confidence; a proposed restraint is not evidence of
biological correctness. Released restraints can deviate from their targets at the end.

Switch **use_rgi** off to run vanilla; prediction strips RGI settings and opt-ins.
For Boltz-1, rerun installation when changing this switch because guided and vanilla
packages use separate environments. Result names include effective settings to avoid
reusing outputs from a different configuration.

## Reuse and verification

Install the optional `notebook` extra to use the toolkit's editor outside Colab:

```python
from rgi_toolkit.notebook_widgets import RestraintEditor
editor = RestraintEditor()
editor.display()
```

After adding and editing entries, read them in a separate prediction cell:

```python
from rgi_toolkit.notebook_widgets import read_editor

config, chains = read_editor(editor)
```

`notebook.make_config` validates a native mapping, YAML/JSON text, or file;
`notebook.compose_config` appends repeated native entries without replacing prior ones.
Selections and all mathematics remain in the shared engine.

`examples/colabfold/mixed_restraints.py` runs vanilla, all five types together, then
vanilla again. It checks written CIF geometry independently, including both distance
entries, angle, the custom diagonal, and two reference RMSDs. Generated results belong
under `.cache`. Existing ColabFold scripts cover the model matrix and batch isolation.
Colab CLI 0.6.0 requires `jupyter-kernel-client<1`; a separate uv environment can provide
that compatibility pin. Release Colab sessions after collecting E2E results.
