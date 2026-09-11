---
name: generate-rgi-config
description: >-
  Create and validate a ready-to-run restraints_config for Restraint-Guided
  Inference (RGI), placing it correctly for boltz, protenix, OpenDDE, chai-lab,
  alphafold3, openfold-3, or esmfold2. Use when a user wants to run RGI, write
  a restraint file, constrain a distance, angle, dihedral, chiral volume, ligand conformer,
  RMSD, or custom energy, or translate a plain-language structural goal into
  the selection DSL, target, and activation window. Use only for an
  already-integrated tool; use implement-rgi when adding RGI support to a new
  predictor.
---

# Generate an RGI restraints_config

## What you are producing

Every RGI tool is driven by **one `restraints_config` dict** (YAML for boltz/chai, JSON
for protenix/OpenDDE/AF3/openfold, a Python dict for esmfold2). Your job is to turn a user's
plain-language goal into that dict, place it where the tool reads it, and **validate it
before they spend a GPU run on it**. The engine (`rgi_toolkit`) does all the maths — you
only write config.

RGI nudges the atoms during diffusion sampling so the final structure satisfies the
restraints. There are **eight direct built-in restraint types**, a base-pair macro, and a
custom one:

| the user wants to… | restraint type | block |
|---|---|---|
| keep two parts of the structure at a set distance | **distance** | `distance_restraints_config` |
| set the angle / twist between three / four parts | **angle** / **dihedral** | `angle_` / `dihedral_restraints_config` |
| restrain an improper angle between four groups | **improper** | `improper_restraints_config` |
| set the signed volume or handedness of four atom groups | **chiral** | `chiral_restraints_config` |
| keep a group flat, or two groups coplanar / stacked | **plane** | `plane_restraints_config` |
| keep a ligand at a chemically sensible shape | **conformer** | `conformer_restraints_config` |
| pull a region onto a reference structure (PDB/mmCIF) | **RMSD** | `rmsd_restraints_config` |
| pair two nucleotides in Watson–Crick geometry | **base-pair macro** | `base_pair_restraints_config` |
| anything else, as a math formula | **custom** | `custom_restraints_config` |

Full mapping + worked phrasings: **`references/restraint-recipes.md`** (read it when you
are unsure which type a goal needs). Full schema (every key, default, allowed value):
the repo's **`docs/config.md`** — it is the source of truth; do not guess defaults.

## The audience is a beginner — interview, then translate

Assume the user is new to RGI *and* to structure prediction. **Do not** ask them for a
selection DSL string or a penalty type by name. Ask in plain language, then translate and
**show your translation back** so they can sanity-check it. For example:

- "Which two parts should be held apart, and how far?" → a `distance` restraint, harmonic,
  `target_distance`.
- "Should the distance be *exactly* X, or just *at least / at most* X?" → harmonic vs
  flat-bottomed (a band / floor / ceiling). See recipes.
- "Should it apply the whole time, or only once the fold has roughly formed?" → the
  `start_sigma` window (omit = always on; set late only if they want it late).

Explain the *why* as you go (this is the point of the skill): e.g. "I'm qualifying the
selection with `chain A` because `resid` numbering restarts on every chain, so a bare
`resid 90 to 180` would also grab the ligand."

## Workflow

### 1. Which tool? (decides the file format + where the config goes)

If the user hasn't said, ask. The tool decides three things that are **easy to get wrong**
— file format, where the `restraints_config` sits, and how a ligand opts into conformer
restraints. OpenDDE can also opt polymer entities in. The per-tool table is in
**`references/tools.md`**; the essentials:

- **boltz / protenix / OpenDDE / alphafold3 / openfold-3** — the config is **nested under a
  `restraints_config` key** inside the tool's own input file (the same file that lists the
  sequences). protenix and OpenDDE inputs are JSON *lists* of jobs; openfold nests it under
  `queries.<name>`.
- **chai-lab** — the **odd one out**: a **separate sidecar YAML** whose top level **IS**
  the `restraints_config` (do **not** nest it). Sequences live in a separate FASTA.
- **esmfold2** — a **Python dict** passed to `ESMFold2InputBuilder().fold(...,
  restraints_config=...)`.

### 2. What does the user want? → restraint type + penalty shape

Map the goal to one or more restraint types using `references/restraint-recipes.md`. The
two decisions that recur:

- **Pin vs bound.** `harmonic` drives a quantity *to* a target. The `flat-bottomed`
  family leaves it free inside a band (`flat-bottomed`), above a floor (`flat-bottomed1`),
  or below a ceiling (`flat-bottomed2`). "exactly 25 Å" → harmonic; "no closer than 25 Å"
  → flat-bottomed1.
- **Angles are in degrees** in the config (not radians) — `target_angle: 90`,
  `target_dihedral: 180`.
- **Chiral targets are Angstrom cubed** — `target_chiral`, with group 1 as the center
  and no division by six. Four selections are required; each may select one atom or a
  group. All groups move by default. Confirm the ordered selections and signed target.

### 3. Which atoms? → selection DSL

Translate the user's description of the parts into selection strings. The grammar +
beginner walk-through is **`references/selection-dsl.md`**. The one rule that causes most
silent failures:

> **`resid` is the per-chain 1-based ordinal** (it restarts at each chain; the ligand gets
> its own ordinal). It is NOT the author residue number and NOT global. **Always qualify a
> protein group with `chain A and (...)`** or a bare `resid` range also sweeps in the
> ligand chain.

### 4. Targets, gating, weight

- **Target**: the distance (Å), angle/dihedral (degrees), chiral volume (Å³), or `target_rmsd` (Å).
- **Sigma window** (`start_sigma` / `stop_sigma`): omit for "active every step" (the usual
  case). There is **no top-level `start_sigma`** — it goes on each entry (and once for all
  conformer terms). Setting one at the top level is an error.
- **Weight**: default `1.0` is right for almost everything. For a single distance restraint
  `weight` is a *no-op* (it reaches the target exactly regardless) — don't present it as a
  strength knob there. See `docs/config.md` for the exact semantics.
- **Polymer dictionary geometry**: `conformer_restraints_config.monomer_library: true`
  acquires and caches the public CCP4 library when setup needs it. Use a path for an existing
  snapshot; `{on_missing: error}` requests the automatic cache with strict coverage. Keep
  the entity opt-in explicit. An empty conformer block enables bond/angle/chiral/cistrans/vdw at weight 1; plane requires an explicit positive weight. Dictionary bond/angle/chiral/plane and
  chi/omega/sp2 torsions use ESD-based inverse-variance weights automatically, with default
  slack zero. ESD is relative strength, not a tolerance band. Do not copy ESD into `slack`;
  consult `docs/config.md` for units, peptide state selection and offline cache behavior.
- **Without a dictionary**: protein chi/omega and acyclic sp2 torsions use documented
  RDKit-based approximations; omitting `monomer_library` never acquires a dictionary.
  VdW uses chemical contact rules and ESD 0.2 A (dummy atoms 0.3 A), with `scale` default
  1.0. Existing VdW weights may need retuning against unnormalized reference geometry;
  do not assume the former 0.75-scale objective. Read `docs/config.md` before migrating one.

### 5. Write the config in the right place

- If the user **already has an input file**, inject the `restraints_config` (and the
  per-entity opt-in flag if conformer is used) into it.
- If they **don't**, scaffold a minimal runnable input from the closest example in the
  repo (`examples/<type>/<tool>/`, `bench_in_<tool>_*`, or the `docs/<tool>.md` "Full config"
  example) and fill in
  their sequences/ligand. Tell them which fields are theirs to replace.
- For **chai**, write the sidecar YAML *and* remind them it pairs with a FASTA.
- For **esmfold2**, write the Python dict + the `.fold(..., restraints_config=...)` call.

Use `config_path` to reuse the entire restraints config or one whole section. The
reference wrapper contains only this key; never merge local overrides into it. JSON/YAML
paths are relative to their containing file, including external reference structures.
The validator resolves the same files as inference. Read `docs/config.md` under
*External configuration files* for the file shapes and examples.

### 6. CONFORMER OPT-IN — the #1 silent no-op

A `conformer_restraints_config` block does **nothing** unless the intended sequence entity is
*also* flagged to opt in. This flag lives **outside** the config block and its placement differs
per tool:

| tool | how the entity opts in |
|---|---|
| boltz / protenix / alphafold3 / openfold-3 | `conformer_restraints: true` on the ligand object |
| OpenDDE | `conformer_restraints: true` on the sequence entity |
| chai-lab | a `conformer_restraints: {<chain_id>: true}` map in the sidecar |
| esmfold2 | `conformer_restraints=True` on the `LigandInput` |

If you write a conformer block, you **must** also write the matching opt-in, or the run
looks fine but applies no conformer restraint.

### 7. Validate before running

Resolve `SKILL_DIR` to the directory containing this `SKILL.md`, then run the
bundled validator on the file you produced:

```bash
uv run --project <rgi-toolkit-dir> --frozen --with pyyaml \
  python "$SKILL_DIR/scripts/validate_config.py" <file>
```

It runs the real `RestraintsConfig.from_dict` (catching unknown/misspelled section names,
a top-level `start_sigma`, a leftover `backend` key, mixed sigma+step windows, empty windows, …),
syntax-checks every selection string, and warns when a conformer block has no opt-in.

### 8. Hand off with the run command AND the validation ceiling

Give the user the exact run command for their tool (in `references/tools.md`). Then state
plainly what validation does **not** prove:

> The validator confirms the config is well-formed and the selections *parse*. It **cannot**
> confirm a selection matches the atoms you meant — a syntactically valid
> `chain A and resid 5 to 84` that resolves to **zero atoms** (wrong range, forgotten chain
> qualifier) passes validation but does nothing. To confirm the real run, set
> **`verbose: true`** and read the setup log line `built spec: ... distances=N angles=N
> bonds=N ...`: the count must be **non-zero** for every restraint you asked for. A
> `finalize` energy of `0.00000` with a count of 0 is a silent no-op, not "satisfied".

## Sanity checklist before you hand it over

- [ ] Tool identified; config placed correctly (nested vs chai sidecar vs esmfold2 dict).
- [ ] Every protein selection is qualified with `chain ...`.
- [ ] Angles/dihedrals in **degrees**; distances/RMSD in **Å**.
- [ ] Chiral volume in **Å³**, centered on selection 1; custom `chiral(A,B,C,D)` uses the same units.
- [ ] If a conformer block exists, the intended sequence entity has its opt-in flag.
- [ ] `verbose: true` is set (so the user can confirm the spec counts at run time).
- [ ] The validator passes.
- [ ] You told the user the run command and the "validation ≠ correct selection" caveat.

## Reference files

- `references/restraint-recipes.md` — goal → restraint type + penalty + example block. The
  interview's hardest step. **Read this first** when the goal is vague.
- `references/selection-dsl.md` — the atom-selection language, beginner-first, with the
  `resid` / `chain` gotcha and ready-made patterns.
- `references/tools.md` — per-tool placement, file format, conformer opt-in, run command.
- the repo's `docs/config.md` — the full, authoritative schema (every key/default/value).
  Always defer to it for anything not spelled out above.
