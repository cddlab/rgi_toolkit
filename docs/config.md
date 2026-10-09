# `restraints_config` reference

[Documentation index](README.md)

Every RGI tool is driven by one `restraints_config` dict: a YAML/JSON block
for Boltz, Protenix, Chai, AlphaFold3, OpenFold3, OpenDDE, and RF3, or a Python
dict for ESMFold2. See each tool's guide for where to place it.
This page documents configuration for the restraint types described in the paper.
Each documented key includes its type, default, allowed values, and meaning.

Omitted keys fall back to their documented defaults. Unknown top-level keys and unknown conformer
keys raise an error. Unknown keys inside individual restraint entries are logged and ignored, so
check their spelling against this page. Source of truth:
`src/rgi_toolkit/{config,distance_restr_data,group_geom_restr_data,ref_geom_restr_data,ref_config,featurizer,rmsd_restr_data,pdb_ref,_align,selection}.py`.

## Quick navigation

- [Config shape](#shape) and [top-level keys](#top-level-keys)
- [Computational cost](#computational-cost-current-implementation)
- [Activation windows](#sigma-gating-start_sigma--stop_sigma)
- [Atom-selection DSL](#atom-selection-dsl) and [penalty shapes](#penalty-shapes-shared)
- [Distance](#distance_restraints_config-list), [group angle](#angle_restraints_config-list),
  and [group dihedral](#dihedral_restraints_config-list)
- [Conformer geometry and VdW](#conformer_restraints_config-single-dict)
- [RMSD](#rmsd_restraints_config-list)
- [Custom restraints](#custom_restraints_config-list)

## Shape

```yaml
restraints_config:
  # --- top-level knobs ---
  verbose: ...        # bool
  gpu: ...            # bool
  compile_cpu: ...    # bool, PyTorch CPU compilation (default true)
  method: ...         # "CG" | "l-bfgs"
  line_search: ...    # CG only: "strong-wolfe" (default) | "armijo"
  max_iter: ...       # int
  gtol: ...           # nonnegative finite float, default 1e-5
  # --- restraints (each block optional) ---
  distance_restraints_config: [ ... ]   # list
  angle_restraints_config:    [ ... ]   # list  (group-centroid angle)
  dihedral_restraints_config: [ ... ]   # list  (group-centroid dihedral)
  conformer_restraints_config: { ... }  # single dict (ligand local geometry)
  rmsd_restraints_config:     [ ... ]   # list
  custom_restraints_config:   [ ... ]   # list  (define your OWN restraint — see below)
```

A restraint type is active only if its block is present (and, for conformer terms, the term's
`weight > 0`).

The documented restraint types are distance, angle, dihedral, conformer, RMSD,
and custom expressions. `custom_restraints_config` defines a differentiable loss
as a mathematical formula.

## Top-level keys

| key | type | default | meaning |
|---|---|---|---|
| `verbose` | bool | `false` | Log the built spec (per-restraint counts) at setup and per-term energies at finalize. Strongly recommended — it is how you confirm a restraint was actually built. |
| `gpu` | bool | `true` | Torch **device**: `true` = accelerator (default), `false` = CPU. It does **not** change the backend. (Inert for AF3, which always runs the JAX minimizer on the model's device.) Accepts `true/false` and the strings `1/0/yes/no/on/off`. |
| `compile_cpu` | bool | `true` | Compile the PyTorch CPU objective, gradient, and CG trial statistics. Set `false` to use eager evaluation. The first call can be slower; repeated calls with the same shapes can amortize compilation. Does not change the device or JAX execution. |
| `method` | str | `"CG"` | Optimizer: `"CG"` (nonlinear conjugate gradient) or `"l-bfgs"` (opt-in). |
| `line_search` | str | `"strong-wolfe"` | CG only: `"armijo"` or `"strong-wolfe"`. Omit this key with L-BFGS. |
| `max_iter` | int | `100` | Nonnegative maximum optimizer iterations per denoising step, shared by all methods. |
| `gtol` | float | `1e-5` | Nonnegative finite gradient tolerance for CG and L-BFGS on both backends. Smaller values request tighter optimization. |

Torch compiles eligible objective and gradient calculations. Compilation failures
fall back to eager evaluation. `compile_cpu: false` disables CPU compilation;
`RGI_DISABLE_COMPILE=1` disables Torch compilation. The first compiled call can
be slower, while repeated calls reuse compiled artifacts. JAX uses its JIT path.

Choose one of these three configurations inside `restraints_config`:

```yaml
method: CG
line_search: armijo
```

```yaml
method: CG
line_search: strong-wolfe
```

```yaml
method: l-bfgs
```

Omitting both keys selects CG with Strong Wolfe. Armijo restores the historical
backtracking search and stops when the relative energy change is below `1e-9`;
this is reported as `FUNCTION_TOLERANCE`, distinct from gradient convergence.
Strong Wolfe retains the SciPy-style PR+ solver and its gradient-based stopping rule.
CG and L-BFGS default to a gradient tolerance of `1e-5` on both backends.
Set `gtol: 1e-8` inside `restraints_config` to request a tighter threshold.
CG and Torch L-BFGS use the gradient infinity norm; JAX L-BFGS retains its
library's Euclidean gradient norm. For CG the default matches SciPy;
Armijo additionally has the energy-change stop.
This threshold bounds the gradient, not the distance or angle error; large groups
can retain a measurable residual because their centroid gradients are divided by
the number of atoms.
Tightening `gtol` does not increase `max_iter`, remove other stopping rules,
or guarantee a zero loss for competing restraints or a local minimum.
L-BFGS uses the backend library's line search; an explicit `line_search` key with
L-BFGS raises an error. See the [solver specification](SPEC.md#optimizers).

**There is no `backend` key** — the compute backend (torch / jax) is **inferred from how the
engine is invoked**, not configured: a JAX tool (AF3) grabs the pure minimizer via
`get_minimizer()` → jax; every other tool calls `minimize(coords)`, where a torch/numpy array →
torch. A leftover `backend:` key raises with a migration hint. (`gpu` above still selects the
torch *device*.) There is no numpy optimizer (numpy is the energy reference only).

**There is no top-level `start_sigma` / `stop_sigma`** — setting one at the top level raises. They
are per-restraint (see below).

Parsing rejects multiple penalty-type blocks on one entry, fractional or boolean
`move` indices, malformed blocks, and nonfinite weights, targets or slacks. Activation
windows may use infinite bounds but cannot contain NaN or be empty. These checks run
before any adapter or reference-structure processing.

## Computational cost (current implementation)

The orders below describe the tensors actually evaluated by the current torch/JAX kernels, not
only the underlying geometry formula. In particular, variable-length groups are padded to the
widest group in the same restraint category, so the maximum padded width appears in the cost.
The SVD used by RMSD/Kabsch is always on a **3 x 3** matrix and is therefore
constant work per entry; gathering, centring, and covariance construction remain
linear in the number of selected atoms.

Let `Q` be the number of objective/gradient evaluations made by the optimizer in one diffusion
step, and let `N_active` be the number of optimized atoms. `Q` is usually a multiple of the
CG/L-BFGS iteration count because line search may evaluate the objective more than once. Apart from
the dynamic VdW neighbour-list build, the repeated cost of one diffusion step is

```text
O(Q * (N_active + sum(active restraint-kernel costs below))).
```

The common `N_active` term covers gradient-vector production and optimizer vector algebra; it is
not attributable to one restraint. Reverse-mode autodiff changes constant factors and retained
working storage, but not these asymptotic time orders. A batch dimension multiplies every listed
time and transient working-tensor cost by the batch size; the prepared restraint/spec arrays
themselves are shared across the batch.

| config path / term | notation | one objective + gradient evaluation | stored term arrays | implementation detail |
|---|---|---:|---:|---|
| `distance_restraints_config` | `R` entries, padded group width `G` | `O(RG)` | `O(RG)` | Two centroids per entry; the factor 2 is constant. |
| `angle_restraints_config` | `R` entries, padded group width `G` | `O(RG)` | `O(RG)` | Three group centroids plus one constant-size angle. |
| `dihedral_restraints_config` | `R` entries, padded group width `G` | `O(RG)` | `O(RG)` | Four group centroids plus one constant-size torsion. |
| conformer `bond` | `B` bond tuples | `O(B)` | `O(B)` | Each tuple gathers two atoms. |
| conformer `angle` | `A` angle tuples | `O(A)` | `O(A)` | Each tuple gathers three atoms. |
| conformer `chiral` | `C` chiral tuples | `O(C)` | `O(C)` | Each tuple gathers four atoms and evaluates one scalar triple product. |
| conformer `cistrans` | `T` tuples per term | `O(T)` | `O(T)` | Each tuple gathers four atoms. |
| `rmsd_restraints_config` | `R` entries, maximum padded fit/calc widths `F` / `C` | `O(R(F + C))` | `O(R(F + C))` | One 3 x 3 Kabsch SVD per entry; it is not `O(F^3)` or `O(C^3)`. |
| built-in `refN and ...` variant | total prediction/reference group atoms `G`, reference-fit atoms `F` | `O(G + F)` per reference access | `O(G + F)` | These entries use a closure. A configured fit recomputes a 3 x 3 Kabsch transform; without a fit the `F` term is absent. |
| `custom_restraints_config` | `C_expr` = sum of the executed primitive/array costs | `O(C_expr)` | expression-dependent | Centroid geometry is linear in the selected group sizes; `kabsch` and `rmsd` are also linear because their matrix decomposition is 3 x 3. Repeated calls are recomputed, and `where`/conditional expressions evaluate **both** branches. |

The table starts after atom selections and reference pairings have been resolved. That one-time
setup scans the target atom records once per selection (`O(S * N_target)` for `S` selections).
Sequence-aligned references additionally have worst-case `O(UV)` dynamic-programming work for
target/reference polymer lengths `U` and `V`. Ligand conformer setup may also run the selected
RDKit force field for at most 200 iterations; that library-dependent cost does not recur inside CG.

### VdW cost breakdown

For the protein–ligand setting, restrained ligand atoms are queried against the
fixed protein background. Let `L` be the number of moving ligand atoms, `B` the
background atom count, and `K = max_neighbors` the sparse capacity. At ordinary
density, cell-list construction costs `O(B log B + L log B)` and an objective
evaluation costs `O(LK)`. Overflow rows use complete background sums. A collapsed
structure can require `O(LB)` work, without allocating a dense distance matrix.

When every restraint is outside its sigma window, minimization is skipped.

## Sigma gating: `start_sigma` & `stop_sigma`

Diffusion runs from a high noise level (`sigma`) down to ~0. Each restraint is gated to a `sigma`
window:

- **`start_sigma`** (float) — the restraint is active only once `sigma <= start_sigma`. **Omitted →
  `+inf`** (active at *every* step). Set e.g. `1.0` to act only late in denoising. The example value
  `99999999` is "always on".
- **`stop_sigma`** (float) — the restraint is **released** once `sigma < stop_sigma`. **Omitted →
  `-1`** (never released; any value `<= 0` means off).
- **Active window:** `stop_sigma <= sigma <= start_sigma`. `stop_sigma > start_sigma` is an empty
  window and **raises** (a silent no-op would read as "satisfied").

Where they live: **once per `distance` / `angle` / `dihedral` / `rmsd` / `custom` entry**, and **once
for all conformer terms** (`conformer_restraints_config.start_sigma` / `.stop_sigma`). When `sigma`
exceeds every active restraint's `start_sigma`, the whole minimization step is skipped (cheap at high
noise).

The RMSD restraint's `stop_sigma` has a specific use: releasing it late (e.g. `1.0`) lets the model
re-idealise geometry the restraint held distorted — the fix for a peptide bond broken at a
restrained-residue / free unmodeled-tail boundary.

## Atom-selection DSL

Atom groups are chosen with a small boolean language (`selection.py`). Precedence: `not` > `and` >
`or`; parenthesise to override.

| token | matches | example |
|---|---|---|
| `chain <ids>` | atoms in the listed chain IDs | `chain A` · `chain A B` |
| `resid <n…>` / `resid A to B` | residues by **per-chain 1-based ordinal** | `resid 5` · `resid 1 to 84` · `resid 1 3 7` |
| `index <n…>` | atoms by **flat row** in the coordinate tensor | `index 42` |
| `name <names>` | atom name, case-insensitive; a prime may be written `'`, `*` or `"` and matches either spelling in the structure | `name CA` · `name N CA C O` · `name C1'` · `name O2*` |
| `protein` / `dna` / `rna` | atoms of that polymer type | `protein` |
| `backbone` / `sidechain` | polymer backbone / sidechain heavy atoms (gated on polymer type; a ligand atom named "C" never matches) | `backbone and chain A` |
| `and` / `or` / `not` / `( )` | boolean composition | `chain A and (resid 5 to 84 or resid 186 to 224)` |

**Critical convention:** `resid` is the **per-chain 1-based ordinal** — it resets at each chain, and
a ligand atom gets its own ordinal. It is **not** 0-based and **not** the author residue number.
Always qualify protein groups with `chain A and (...)`, or a bare `resid` range will also match the
ligand chain's atoms with the same ordinal. This convention is identical across all tools.

## Penalty shapes (shared)

Every built-in restraint is defined by the same squared penalty,

```math
E = \sum_i w_i\,\delta_i^2
```

where $w_i$ is the entry `weight` and $\delta_i$ is the deviation of a measured quantity $x$ (a
distance, angle, volume, …) from its target. Four block names choose how $\delta$ is shaped:

| block | $\delta$ | effect |
|---|---|---|
| `harmonic` | $x - t$ | penalise any deviation from $t$ |
| `flat-bottomed` | $0$ for $t_1 \le x \le t_2$; $x - t_1$ below; $x - t_2$ above | no penalty inside the window |
| `flat-bottomed1` | $`\min(0,\, x - t_1)`$ | lower bound — penalise only $x \lt t_1$ |
| `flat-bottomed2` | $`\max(0,\, x - t_2)`$ | upper bound — penalise only $x \gt t_2$ |

The same four shapes drive the `distance`, `angle`, `dihedral`, and `rmsd` blocks.
Their target keys are `target_distance`, `target_angle`, `target_dihedral`, and
`target_rmsd`, with `…1` / `…2` for the flat-bottomed bounds.
The **conformer** geometry terms use the flat-bottomed shape with a symmetric `slack`:
$\delta = 0$ within $\pm$`slack` of the RDKit-ideal value, quadratic outside (`slack = 0` $\Rightarrow$ pure harmonic).

`distance` is part of the same CG objective as every other restraint. Centroids use
ordinary mean derivatives (`1/N` per atom), so the supplied gradient agrees with the
reported scalar energy. An unrestricted line search chooses the step length and can
move large groups at the default weight. For a single disjoint distance restraint,
equal per-atom mobility gives the minimum squared-displacement split (`s1:s2=N2:N1`).
Weights balance competing restraints; they can also change convergence speed under
a finite iteration budget. Pinned groups retain their coordinates.

## `distance_restraints_config` (list)

Pulls the **centroid distance** between two atom groups toward a target. Each free
atom in one group receives the same mean derivative, giving rigid translation when
no other term acts on that group. CG optimizes all active energies together.

The measured quantity is the distance between the two groups' centroids,

```math
d = \lVert c_2 - c_1 \rVert, \qquad c_k = \frac{1}{|G_k|}\sum_{a \in G_k} x_a
```

(a plain masked-mean centroid), shaped by one of the penalty blocks below (see Penalty shapes).
`harmonic` drives the distance toward $d = t$ with CG (gradient tolerance `gtol`;
see [termination semantics](SPEC.md#nonlinear-conjugate-gradient)).

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1` | str | — (required) | group 1 (selection DSL) |
| `atom_selection2` | str | — (required) | group 2 |
| `start_sigma` | float | `+inf` | activation upper bound (see Sigma gating) |
| `stop_sigma` | float | `-1` | release lower bound |
| `move` | `"both"`/`"all"`/`1`/`2`/`[1,2]`/`"1,2"` | `"both"` | which group the correction moves: `both` (= `all` = `[1,2]`) = minimal-displacement split; `1` / `2` (or `[1]` / `[2]`) move only that group and **pin** the other (e.g. move a ligand toward a fixed pocket). Shares the list/comma vocabulary of the angle/dihedral `move`; distance has 2 groups so only indices 1–2 are valid (`[1,3]` raises) |
| `weight` | float | `1.0` | relative strength. **No-op for a single restraint or disjoint groups** (each reaches its exact target regardless). Only bites when an atom is the **sole mover** of two **over-constrained coupled** restraints (each pinning its other group), where the shared atom settles `w₁:w₂` between the two targets (e.g. `2` vs `1` → 2:1). Not a soft "strength" knob in the common case |
| one restraint-type block | dict | — (required) | the penalty (below) |

Restraint-type block (exactly one):

| block | params | behaviour |
|---|---|---|
| `harmonic` | `target_distance` | quadratic penalty everywhere toward the target |
| `flat-bottomed` | `target_distance1`, `target_distance2` | no penalty inside `[d1, d2]` (needs `d1 < d2`) |
| `flat-bottomed1` | `target_distance1` | penalise only below `d1` |
| `flat-bottomed2` | `target_distance2` | penalise only above `d2` |

### Reference groups

A distance entry may use one external structure. Keep both normal group keys; prefix the
reference-side value with `ref1 and`, then define `refs.ref1`. The suffix is evaluated on
the reference structure. At least one group must remain a prediction selection. Reference
groups are fitted independently, held fixed, and do not pull their fit anchors.
`move` controls prediction groups only: omit it or use `all`/`both` to move every prediction
group, or give prediction-side group indices. Selecting a reference group raises.

```yaml
distance_restraints_config:
  - atom_selection1: "chain A and resid 120"
    atom_selection2: "ref1 and chain A and resid 200"
    refs:
      ref1:
        ref_cif: template.cif
        atom_selection_target_fit: "chain A and resid 1 to 80 and backbone"
        atom_selection_ref_fit: "chain A and resid 1 to 80 and backbone"
        pairing: align
        best_effort: true
    harmonic: {target_distance: 5.0}
```

## `angle_restraints_config` (list)

The **angle of 3 group centroids**, with the vertex at group 2 — the group-centroid analogue of the
distance restraint, distinct from the per-ligand-atom conformer `angle` term. CG-solved; rigid group
motion (the centroid-only energy gives every atom in a free group the same gradient, so the group
translates as a unit) means `weight: 1.0` drives any group size.

The measured quantity is the angle at centroid $c_2$ (with $c_k$ the centroid of group $k$),

```math
\theta = \arccos\left( \frac{(c_1 - c_2)\cdot(c_3 - c_2)}{\lVert c_1 - c_2 \rVert\,\lVert c_3 - c_2 \rVert} \right)
```

penalised by $`E = \sum w\,\delta^2(\theta)`$ with the usual shapes (see Penalty shapes). Targets are in **degrees**
by default — set `unit: radians` on the entry to give them in radians (stored internally as radians
either way).

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1..3` | str | — (required) | the three groups; group 2 is the vertex |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating |
| `unit` | `"degrees"`/`"radians"` | `"degrees"` | unit of the target angle(s) for this entry |
| `weight` | float | `1.0` | energy scale |
| `move` | `"all"` / int / list / `"1,3"` | arms (1,3) free, vertex (2) pinned | which groups are free; the rest are pinned (stop-gradient). `"all"` frees every group |
| one restraint-type block | dict | — (required) | `harmonic {target_angle}` or `flat-bottomed{,1,2}` with `target_angle1` / `target_angle2` (degrees, or radians if `unit: radians`) |

### Reference groups

An angle entry may use up to two distinct references. Write each reference group as
`refN and <selection>` and define the corresponding `refs.refN`. Each reference has its own
structure and optional fit configuration, so groups from different fitted structures can be
combined in one angle. At least one of the three groups must use prediction atoms. `move`
accepts prediction-side group indices; omitted/`all`/`both` moves every prediction group, while
selecting a
reference group raises.

```yaml
angle_restraints_config:
  - atom_selection1: "chain A and resid 10"
    atom_selection2: "ref1 and chain A and resid 20"
    atom_selection3: "chain B and resid 30"
    refs:
      ref1: {ref_cif: state1.cif}
    move: 1  # move group 1; group 3 is pinned and group 2 is a fixed reference
    harmonic: {target_angle: 90.0}
```

## `dihedral_restraints_config` (list)

The **dihedral of 4 group centroids**, about the axis through groups 2–3 — the group-centroid
analogue of the distance restraint, distinct from the per-atom conformer `cistrans` term.
CG-solved; `weight: 1.0` drives any group size, as for the angle.

The measured quantity is the dihedral angle $\phi$ of the four centroids $c_1, c_2, c_3, c_4$ about
the axis through $c_2$ and $c_3$ (with $c_k$ the centroid of group $k$),

```math
\phi = \mathrm{atan2}\big( (n_1 \times \hat{b}_2)\cdot n_2,\; n_1 \cdot n_2 \big)
```

with the bond and normal vectors

```math
b_1 = c_2 - c_1, \quad b_2 = c_3 - c_2, \quad b_3 = c_4 - c_3
```

```math
n_1 = b_1 \times b_2, \quad n_2 = b_2 \times b_3, \quad \hat{b}_2 = b_2 / \lVert b_2 \rVert
```

This is the signed `atan2` convention (range $\pm 180^\circ$); it is penalised by
$`E = \sum w\,\delta^2(\phi)`$ (see Penalty shapes). The `harmonic` shape is **periodicity-safe**: the
deviation $\phi - t$ is wrapped to $[-180^\circ, 180^\circ]$ before squaring, so e.g. $+179^\circ$
and $-179^\circ$ count as a $2^\circ$ difference. The `flat-bottomed` shapes use the raw angle and
therefore **cannot straddle $\pm 180^\circ$** (`target_dihedral1 < target_dihedral2` is enforced).
Targets in **degrees** by default (`unit: radians` to override).

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1..4` | str | — (required) | the four groups; groups 2–3 are the axis |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating |
| `unit` | `"degrees"`/`"radians"` | `"degrees"` | unit of the target dihedral(s) for this entry |
| `weight` | float | `1.0` | energy scale |
| `move` | `"all"` / int / list / `"1,4"` | ends (1,4) free, axis (2,3) pinned | which groups are free; the rest are pinned (stop-gradient). `"all"` frees every group |
| one restraint-type block | dict | — (required) | `harmonic {target_dihedral}` or `flat-bottomed{,1,2}` with `target_dihedral1` / `target_dihedral2` (degrees, or radians if `unit: radians`) |

### Reference groups

A dihedral entry may use up to three distinct references. Write each reference group as
`refN and <selection>` and define the corresponding `refs.refN`. Each reference is fitted
independently, so one dihedral may combine prediction atoms with groups from up to three
external structures. At least one group must use prediction atoms. `move` accepts only
prediction-side group indices; omitted/`all`/`both` moves all prediction groups. Selecting a
reference
group raises.

```yaml
dihedral_restraints_config:
  - atom_selection1: "chain A and resid 10"
    atom_selection2: "ref1 and chain A and resid 20"
    atom_selection3: "chain B and resid 30"
    atom_selection4: "chain C and resid 40"
    refs:
      ref1: {ref_cif: state1.cif}
    move: [1, 4]  # groups 1 and 4 move; group 3 is pinned
    harmonic: {target_dihedral: 180.0}
```

## `conformer_restraints_config` (single dict)

Ligand conformer restraints use a single dictionary. Set `conformer_restraints: true`
on each ligand sequence/chain object, or `conformer_restraints=True` on an
ESMFold2 `LigandInput`. Chai uses a sidecar map keyed by ligand chain ID.

```yaml
conformer_restraints_config: {}  # Enable the five default terms at weight 1.
```

An omitted or null block disables conformer restraints. An empty or partial
mapping enables bond, angle, chiral, cistrans, and VdW at weight 1 on opted-in
ligands. Set an individual term's `weight` to zero to disable it. The shared
`start_sigma` and `stop_sigma` keys control all conformer terms together.

### Reference geometry

Targets are measured from the predictor's ligand reference conformer after RDKit
UFF relaxation. The source molecular graph supplies tetrahedral chirality and
acyclic double-bond E/Z labels. If the relaxed coordinates disagree with these
labels, the toolkit retries stereo-aware ETKDG embedding and UFF relaxation with
up to four deterministic seeds. If UFF changes an initially correct conformer
and all retries fail, that original conformer is retained. If no reference with
the source stereochemistry can be recovered, setup raises an error.

Relaxation operates on a copy. It requires evidence of real bond orders
(an aromatic or double bond); stereo validation also covers saturated chiral
ligands. These targets are distinct from an independently generated conformer
used to evaluate predictions.

| term | keys (default) | meaning |
| --- | --- | --- |
| `bond` | `weight` (1.0), `slack` (0.0 Å) | Bond lengths toward reference values |
| `angle` | `weight` (1.0), `slack` (0.0 rad) | Bond angles toward reference values |
| `chiral` | `weight` (1.0), `slack` (0.0 Å³) | Signed chiral volumes toward reference values |
| `cistrans` | `weight` (1.0), `slack` (0.0 rad) | Dihedrals about acyclic double bonds toward reference E/Z geometry |
| `vdw` | `weight` (1.0), `mode` (`"intermolecular"`), `scale` (0.75), `dmax` (5.0 Å), `max_neighbors` (32), `neighbor_skin` (2.0 Å) | Intermolecular repulsion based on elemental radius sums |

### Energy terms

Bond, angle, chiral, and cistrans terms use squared deviations from their
reference values, with a symmetric interval of width `slack` on either side of
the target where the loss is zero.

| term | measured quantity | target |
| --- | --- | --- |
| `bond` | Bond length | Reference bond length |
| `angle` | Bond angle in radians | Reference bond angle |
| `chiral` | `(a1-a0) dot ((a2-a0) cross (a3-a0))` | Reference signed volume, without division by six |
| `cistrans` | Dihedral in radians about an acyclic double bond | Reference dihedral; the deviation wraps to `[-pi, pi]` |

Cumulated double bonds, such as those in allenes, are excluded from cistrans
restraints because the linear endpoint does not define the required dihedral.
Conformer angles strictly within 0.5 degrees of 180 degrees use a stable cosine
residual. At zero slack their energy is `2 * weight * (1 + cos(theta))`.
With nonzero slack the residual is
`max(2*sin((pi-theta)/2) - 2*sin(slack/2), 0)`. Bond norms in the cosine
denominator have a 0.02 Å floor. Group-angle and custom-angle terms are unchanged.

### Intermolecular van der Waals repulsion

The default `vdw.mode` is `intermolecular`. In a protein–ligand system, this
penalizes ligand contacts with the protein using

```math
E = w \sum_{(i,j)} \min(0, d_{ij} - \mathrm{scale}(r_i + r_j))^2.
```

The radii are RDKit elemental values from
`Chem.GetPeriodicTable().GetRvdw(atomic_number)`. They are not adjusted for
hydrogen bonds or ionic environments. Covalent 1–2, 1–3, and 1–4 pairs are
excluded using topology, even when the corresponding geometry energies are
disabled. The default contact threshold is `0.75 * (r_i + r_j)`.

```yaml
restraints_config:
  conformer_restraints_config:
    vdw:
      mode: intermolecular
      weight: 1.0
      scale: 0.75
```

### Dynamic intermolecular neighbor lists

Every objective/gradient evaluation checks the neighbor cache against its own
trial coordinates, including rejected line-search trials. The protein background
is held fixed for one minimization invocation. The search radius is
`max(dmax, max_contact + neighbor_skin)`. With a fixed background, the cache is
rebuilt when the maximum ligand displacement exceeds `neighbor_skin`.

`max_neighbors` controls sparse buffer capacity. Rows that exceed the capacity
use complete pair sums in bounded chunks, so no eligible contacts are discarded.
Increasing capacity or the skin can change performance but does not change the
objective. Inactive conformer windows skip list construction.

CG, L-BFGS, and energy diagnostics evaluate the same dynamic VdW objective.
Rebuilds preserve the objective, conjugate direction, and counters. Exact or
near-exact overlaps use a deterministic pair-dependent separation direction in
the gradient. See the [solver specification](SPEC.md#nonlinear-conjugate-gradient).

## `rmsd_restraints_config` (list)

Drives the **Kabsch-superposed RMSD** of a moving group versus a reference structure, shaped by one
of the penalty blocks (see Penalty shapes) on the RMSD value. The reference must be generated first
(a vanilla prediction → PDB or mmCIF via gemmi); see each tool's page. Supply it as **either**
`ref_pdb` (legacy PDB) **or** `ref_cif` (mmCIF) — mutually exclusive; both parse via gemmi
(lazy-imported) to the same atom records, so a `.cif` prediction can be used directly without
converting to PDB first.

The energy is

```math
E = \sum w\,\delta^2, \qquad \mathrm{RMSD} = \sqrt{\tfrac{1}{n}\sum_{a} \lVert P_a - \hat{R}\,Q_a \rVert^2}
```

where $\delta$ is the penalty-block deviation of the RMSD (see Penalty shapes; `harmonic`
$\Rightarrow \delta = \mathrm{RMSD} - t$), $P_a$ / $Q_a$ are the prediction / reference **calc**
atoms centred on their fit-atom centroids, $n = n_\text{calc}$, and $\hat{R}$ is the optimal rotation
from a Kabsch SVD on the **fit** atoms. $\hat{R}$ (and the centroids) are treated as **fixed**
(stop-gradient), so the gradient pulls the moving atoms, not the rotation —
`harmonic: {target_rmsd: 0}` drives the group onto the reference, while
`flat-bottomed2: {target_rmsd2: X}` keeps it within $X$ Å of the reference.

| key | type | default | meaning |
|---|---|---|---|
| `ref_pdb` / `ref_cif` | str | — (one required, mutually exclusive) | path to the reference structure — `ref_pdb` = legacy PDB, `ref_cif` = mmCIF; both read via gemmi to the same atoms |
| `harmonic` / `flat-bottomed` / `flat-bottomed1` / `flat-bottomed2` | block | — (one required) | restraint-type block on the RMSD (Å); target keys `target_rmsd` (harmonic) / `target_rmsd1` / `target_rmsd2` (see Penalty shapes). `harmonic: {target_rmsd: 0}` = match the reference; `flat-bottomed2: {target_rmsd2: X}` = stay within $X$ Å |
| `weight` | float | `1.0` | energy scale |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating (set `stop_sigma`, e.g. `1.0`, to release late and heal a strained terminus) |
| `pairing` | `"align"` / `"identity"` | `"align"` | how reference and prediction residues correspond. `align` = sequence-align polymer chains (BLOSUM62, via biopython `Bio.Align`, lazy-imported) so a **homolog** ref maps on despite substitutions/indels/renumbering; non-polymer atoms always pair by ordinal. `identity` = strict (chain, resid, name) ordinal pairing. |
| `best_effort` | bool | `true` | skip atoms with no match in the ref (instead of raising); `false` = strict |

### Atom pairing and selection

All selection keys are optional; omit all of them to use the whole structure in best-effort mode.
The superposed ("fit") and measured ("calc") atom sets are chosen independently:

| key | sets |
|---|---|
| `atom_selection_ref` / `atom_selection_target` | shorthand: **both** fit and calc, on the ref / target side |
| `atom_selection_ref_fit` / `atom_selection_target_fit` | atoms used for the Kabsch superposition |
| `atom_selection_ref_calc` / `atom_selection_target_calc` | atoms over which the RMSD is measured |

Use the shorthand for the common case (fit = calc); use the four `_fit`/`_calc` keys to e.g.
superpose on the backbone but measure over a pocket. Under `pairing: align`, restrict the fit to
`name CA` or `backbone` so a substituted homolog's side chain is not pinned.

## `custom_restraints_config` (list)

### Formula definition

Write a differentiable energy as a mathematical formula over named atom
selections. The same expression runs on Torch and JAX.

Each entry's energy (× `weight`) is added to the CG objective, gated by the usual
`start_sigma` / `stop_sigma` window.

| key | type | default | meaning |
|---|---|---|---|
| `energy` | str | — | the required formula (DSL) |
| `selections` | dict | `{}` | `name -> selection string`; use `refN and <selection>` for a group selected from `refs.refN` |
| `refs` | dict | `{}` | `refN -> {ref_pdb`\|`ref_cif, atom_selection_ref_fit, atom_selection_target_fit, pairing, best_effort}`; names are reserved as `ref1`, `ref2`, ... |
| `move` | `"all"` / `"both"` / str / list[str] | `"all"` | prediction selection name(s) that receive this restraint's gradient. Omitted/`all`/`both` moves every prediction selection. Reference-backed selections are always fixed and cannot be named |
| `weight` | float | `1.0` | scales the whole energy |
| `name` | str | `"custom"` | label shown in the `finalize` per-term log |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating (as everywhere) |

### Evaluation model

Each entry compiles to a closure `energy(active_coords) → scalar`. A selection name
in a formula (`A`) is resolved from the entry's `selections` map to that group's atoms; a string
literal (`"chain A"`) is a raw selection. Selection names resolve to their group **centroids** at
setup (a dry run records which names the energy touches). The formula must reduce to a **scalar** (a
`sum` over any batch dimension). `move` applies stop-gradient to every unlisted prediction
selection for this custom term; another restraint may still move the same atoms. The formula is parsed safely — no `eval`,
and
`import` / attribute access /
subscripting / `lambda` all raise — so only the vocabulary below is callable.

### Vocabulary

The vocabulary has three groups — **geometry** (coordinates → a number), **penalty** (a number → an
energy), and **math** (elementwise / reduction helpers) — plus operators.

#### Geometry

Geometry functions operate on selection **centroids**; angular results are in **radians** (note: the
built-in `angle` / `dihedral` configs take *degrees*, but a custom formula is in radians).
$\lVert\cdot\rVert$ is the Euclidean norm:

| call | result | definition | use it for |
|---|---|---|---|
| `centroid(A)` | vector | $c_A$ = mean of $A$'s atoms | a building block — subtract two, or feed one to `norm` / `dot` |
| `distance(A,B)` | scalar | $\lVert c_A - c_B \rVert$ | a separation between two groups; a **difference of two distances** encodes symmetry / equidistance |
| `angle(A,B,C)` | scalar (rad) | $`\arccos\big( (c_A - c_B)\cdot(c_C - c_B) / (\lVert c_A - c_B \rVert\,\lVert c_C - c_B \rVert) \big)`$, vertex $B$ | the bend of three groups about the vertex $B$ |
| `dihedral(A,B,C,D)` | scalar (rad) | torsion about the B–C centroid axis, range $\pm\pi$ | the twist / handedness across four groups — a **periodic** quantity: wrap its deviation, see below |
| `norm(v)` | scalar | $\lVert v \rVert$ | the length of a vector you built, e.g. `centroid(A) - centroid(B)` |
| `dot(u,v)` | scalar | $u \cdot v$ | projections and cosine-like terms |
| `coords(A)` | block $(k,3)$ | $A$'s atom coordinates | feed a bare selection into arithmetic with `kabsch` output (a bare name alone is a *selection identifier*, not coordinates) |
| `kabsch(A,B)` | block $(k,3)$ | $A$ rigid-body-superposed onto $B$ (Kabsch) | align two moving groups, then measure the leftover per-atom deviation — see **Reference-backed selections** |
| `rmsd(A,B)` | scalar | superposed RMSD of prediction selection `A` vs reference-backed selection `B` | pull a group onto an external reference; `B` must map to `refN and <selection>` — see **Reference-backed selections** |

Most geometry consumes selection **centroids**; `coords` / `kabsch` instead flow a whole
**$(k,3)$ coordinate block**, so they compose: `centroid(kabsch(A,B))`, `norm(kabsch(A,B) - coords(B))`.
#### Reference-backed selections and superposition

A reference-backed selection is declared in the ordinary `selections` map:

```yaml
selections:
  moving: "chain A and resid 1 to 80"
  state1: "ref1 and chain A and resid 1 to 80"
refs:
  ref1:
    ref_cif: state1.cif
    atom_selection_target_fit: "chain A and backbone"
    atom_selection_ref_fit: "chain A and backbone"
    pairing: align
    best_effort: true
```

`ref1`, `ref2`, ... are reserved entry-local names. Each definition requires exactly one of
`ref_pdb` / `ref_cif`. Its optional target/ref fit selections place all selections from that
reference into the current prediction frame. The fitted reference coordinates and transform are
stop-gradient values: they act as fixed landmarks and do not pull the fit anchor. If both fit
selections are omitted, the reference is used in its own coordinate frame.

Reference-backed selections work with the normal geometry vocabulary: `distance(A,B)`,
`angle(A,B,C)`, `dihedral(A,B,C,D)`, `centroid(A)`, `coords(A)`, and `kabsch(A,B)`. At least one
selection in the custom entry must come from the prediction; an all-reference expression is a
constant and raises.

- **`kabsch(A, B)`** returns A after rigid-body superposition onto B. Both arguments are bare
  selection identifiers and must contain the same number of atoms. Either may be reference-backed.
- **`rmsd(A, B)`** returns Kabsch-superposed RMSD from prediction selection A to reference-backed
  selection B. Atom correspondence is resolved at setup with `pairing: align` (default) or
  `identity`; `best_effort: true` skips atoms missing from the reference. A must be prediction-backed
  and B must start with `refN and`; both must be bare selection identifiers.

The frozen rotation makes `kabsch` / `rmsd` gradients torch/jax-consistent but different from a
numpy finite-difference through the SVD, matching the built-in RMSD restraint.

#### Degenerate geometry

`angle` and `dihedral` are ill-defined when the centroids collapse —
coincident centroids, a collinear A–B–C for `angle`, or a `dihedral` whose central axis runs parallel
to an arm. The value stays finite but its gradient is near-zero, so the term cannot push the
structure; choose groups whose centroids are distinct and non-collinear. (`distance` has no such caveat.)

#### Dihedral periodicity

Wrap every `dihedral` deviation. These angles are periodic ($\phi$ and $\phi + 2\pi$
are the same geometry), so a penalty on its **deviation** must fold that deviation into
$[-\pi, \pi]$ first. Write `wrap(dihedral(A,B,C,D) - t)**2` (harmonic), **not** the naïve
`harmonic(dihedral(A,B,C,D), t)`: the naïve form counts $\phi = +179^\circ$ against $t = -179^\circ$
as a $358^\circ$ deviation (huge energy, and a gradient pointing the *long way* round) instead of the
correct $2^\circ$. `wrap(x)` $= \mathrm{atan2}(\sin x, \cos x)$ is exactly the fold the
built-in `dihedral_restraints_config` / conformer `cistrans` apply internally — see the Math table.
For a window, wrap relative to the centre: `flat_bottomed(wrap(dihedral(...) - centre), -w, w)` — and
because the deviation is wrapped, this window **can straddle $\pm 180^\circ$** (the built-in
flat-bottomed dihedral cannot). Note `t` / `centre` are in **radians** (a custom formula does no degree
conversion, unlike the built-in `dihedral_restraints_config`). (`angle` is bounded to $[0, \pi]$ by
`arccos`, so it needs no wrap.)

#### Penalties

These are convenience squared penalties; you may also write the algebra directly. Use
`harmonic` to drive a quantity **to** a value, and the `flat_bottomed` family to **bound** it — leave
it free inside a window, above a floor, or below a ceiling:

| call | definition | effect — use when |
|---|---|---|
| `harmonic(x, t)` | $(x - t)^2$ | quadratic toward $t$ — pin $x$ at a target |
| `flat_bottomed(x, lo, hi)` | $`\min(0,\, x - \text{lo})^2 + \max(0,\, x - \text{hi})^2`$ | zero inside $[\text{lo}, \text{hi}]$ — keep $x$ within a band |
| `flat_bottomed1(x, lo)` | $`\min(0,\, x - \text{lo})^2`$ | lower bound — enforce $x \ge \text{lo}$ only |
| `flat_bottomed2(x, hi)` | $`\max(0,\, x - \text{hi})^2`$ | upper bound — enforce $x \le \text{hi}$ only |

`flat_bottomed` / `flat_bottomed1` / `flat_bottomed2` are the same maths (and names) as the built-in
`flat-bottomed` / `flat-bottomed1` / `flat-bottomed2` blocks.

#### Math and operators

Math functions are dispatched to the active backend:

| group | names |
|---|---|
| elementwise | `sqrt` `exp` `log` `abs` `sin` `cos` `clip(x, lo, hi)` `wrap(x)` = $\mathrm{atan2}(\sin x, \cos x)$, folds an angle/deviation into $[-\pi, \pi]$ (use on `dihedral` deviations — see Periodicity above) |
| reductions | `sum` `minimum` `maximum` |
| branching | `where(cond, a, b)`, or the conditional expression `a if cond else b` — **the same thing** (`if` is lowered to `where`) |

Supported operators are `+ - * / ** %`, unary `-`, comparisons (`<` `<=` …), and the logical
`and` / `or` / `not`. `&` and `|` also work and mean exactly the same as `and` / `or`.

> ⚠️ **Branching is elementwise and never short-circuits — both branches are always evaluated.**
> That is what keeps the closure traceable inside `lax.scan` (a real Python branch on a traced
> value is impossible), but it has one sharp edge: if the branch that is *not* selected produces
> a `NaN`/`inf` — `log` of a non-positive, `sqrt` of a negative, a division by zero — the **value**
> stays correct while the **gradient** becomes `NaN`, and the CG then moves nothing. Guard the
> operand instead of the result: write `log(clip(x, 1e-8, 1e8))`, not `log(x) if x > 0 else 0.0`.

Because both branches are evaluated, a conditional formula resolves the atom selections of
**both** of them at setup — the `built spec` selection count covers the whole formula, not just
the branch that happens to be live.

### Examples

```yaml
custom_restraints_config:
  # symmetry: keep two inter-domain distances equal
  - name: symmetric
    energy: "(distance(A, B) - distance(C, D))**2"
    selections:
      A: "chain A and resid 10"
      B: "chain B and resid 10"
      C: "chain A and resid 90"
      D: "chain B and resid 90"
    move: [A, C]  # B and D are pinned for this custom term
    weight: 1.0
  # periodicity-safe dihedral toward 180deg (pi rad): wrap the deviation, NOT harmonic(dihedral, t)
  - name: planar_dihedral
    energy: "wrap(dihedral(A, B, C, D) - 3.14159)**2"
    selections: {A: "resid 10", B: "resid 11", C: "resid 12", D: "resid 13"}
  # nearest of two equivalent pockets: only the closer one pulls (homodimer A/B, ligand C)
  - name: nearest_pocket
    energy: "flat_bottomed2(distance(L, PA), 4.0) if distance(L, PA) < distance(L, PB)
             else flat_bottomed2(distance(L, PB), 4.0)"
    selections:
      L:  "chain C"
      PA: "chain A and resid 50 to 60"
      PB: "chain B and resid 50 to 60"
    move: L        # only the ligand moves; both pockets are pinned
    # NOTE: the choice is re-made every step from the CURRENT coordinates — it is not
    # latched at the moment the restraint activates.
  # composable RMSD: compare one moving domain with two reference states
  - name: rmsd_compose
    energy: "rmsd(dom, state1) - rmsd(dom, state2)"
    selections:
      dom: "chain A and resid 1 to 80"
      state1: "ref1 and chain A and resid 1 to 80"
      state2: "ref2 and chain A and resid 1 to 80"
    refs:
      ref1: {ref_cif: "state1.cif", pairing: align}
      ref2: {ref_pdb: "state2.pdb", pairing: align}
```
