# RGI-toolkit implementation specification

This document describes the toolkit's implemented contracts and their verification.
The [configuration reference](config.md) defines accepted keys, defaults, selection
syntax, and complete examples. The [predictor guides](README.md) describe where each
host invokes RGI; the maintained [example workflows](../examples/README.md) are the
starting point for predictor runs. Tests here exercise the shared toolkit without
loading predictor weights or running a complete structure predictor.

## Scope and architecture

RGI corrects coordinates during a predictor's denoising loop by minimizing a scalar
restraint objective. It does not implement the predictor, a physical dynamics
integrator, or a calibrated thermodynamic energy. Weights combine quantities with
different units; reducing this objective alone does not establish structural validity.

| Layer | Responsibility | Source |
| --- | --- | --- |
| Input and adapters | Validate config, expose atom metadata, resolve selections and reference correspondences | [`config.py`](../src/rgi_toolkit/config.py), [`atom_context.py`](../src/rgi_toolkit/atom_context.py), [`selection.py`](../src/rgi_toolkit/selection.py) |
| Specification | Derive targets and pack masked NumPy arrays with local atom indices | [`featurizer.py`](../src/rgi_toolkit/featurizer.py), [`spec.py`](../src/rgi_toolkit/spec.py), [`polymer.py`](../src/rgi_toolkit/polymer.py) |
| Energy | Evaluate shared geometry, penalties, and activation masks through a backend facade | [`_geometry.py`](../src/rgi_toolkit/_geometry.py), [`energy/_kernels.py`](../src/rgi_toolkit/energy/_kernels.py), [`energy/_runtime.py`](../src/rgi_toolkit/energy/_runtime.py) |
| Optimization | Gather active coordinates, differentiate, minimize, and scatter the correction | [`optim/torch_optim.py`](../src/rgi_toolkit/optim/torch_optim.py), [`optim/jax_optim.py`](../src/rgi_toolkit/optim/jax_optim.py) |

The array kernels are implemented once against `_array_ops.py`. NumPy, Torch, and
JAX energy modules are thin adapters. The `TermDef` registry in
[`energy/_terms.py`](../src/rgi_toolkit/energy/_terms.py) drives packing, dispatch,
gating, and breakdown of twelve array terms. Custom and reference-dependent
closures and dynamic VdW complete the objective outside that array registry.
NumPy supplies an energy reference; optimization requires Torch or JAX. SciPy is
a development dependency used by tests, not a runtime optimizer.

## Public lifecycle

Construct one `CombinedRestraints` per structure. The preferred sequence is:

```python
restraints = CombinedRestraints()
restraints.setup(adapter, nbatch=nbatch, config=restraints_config)
coords = restraints.minimize(coords, istep=step, sigma=sigma)
restraints.finalize(coords, istep=step)
```

`add_custom(...)` is called before `setup` when adding an in-process callable or
formula. `set_config(dict)` remains the adapter-independent parse/validation API;
the two-call `set_config` then `setup` form is the deprecated equivalent of
`setup(config=...)`.

`setup` clears the previous derived spec and optimizer before resolving the new
input. A successful setup leaves backend selection lazy. Pending `add_custom`
entries belong to the instance and survive another setup. `is_active()` reports
whether the constructed spec has work; empty configurations leave coordinates
unchanged. `nbatch` is accepted for host compatibility; coordinate arrays determine
the actual batch shape.

| Invocation | Backend and result |
| --- | --- |
| `minimize(torch_tensor, istep, sigma)` | Torch; updates and returns the same tensor |
| `minimize(numpy_array, istep, sigma)` | CPU Torch in float64; writes back to a writable floating array, otherwise returns the resulting array |
| `minimize(jax_array, istep, sigma)` | JAX; requires an explicit sigma and returns a new array |
| `get_minimizer()` | Selects JAX and returns pure `(coords, sigma, step) -> coords`, or `None` for an inactive spec; suitable for `jit` and `lax.scan` |

One instance cannot switch backend after selection; a conflicting invocation
raises. `gpu: false` moves a Torch accelerator input to CPU for the correction and
copies it back. `gpu: true` uses the Torch input's device. This setting does not
select a backend or move a JAX computation. Half-precision Torch coordinates are
optimized in float32 and cast back; float32/float64 retain their working dtype.
Torch explicitly enables autograd for this correction inside host inference mode.
This API is a coordinate correction, not a promise of differentiation through the
optimizer into the predictor.

For CG, opt in to termination diagnostics without changing coordinate semantics:

```python
from rgi_toolkit import CGStatus

coords, info = restraints.minimize(coords, sigma=sigma, return_info=True)
converged = info.status == CGStatus.CONVERGED
# A JAX factory fixes its output structure at construction, including inside scan.
minimize_with_info = jax_restraints.get_minimizer(return_info=True)
```

The two calls illustrate separate Torch/NumPy and JAX instances, respectively;
one instance still cannot switch backends. `CGInfo` is a framework-independent
named tuple and a native JAX pytree. Its scalar fields are `status`, `nit`
(accepted iterations), `nfev`, `njev`, `fun`, and `grad_norm` (infinity norm of the
gradient actually used, including RGI gradient modifications). One record covers
the **entire batch** and all neighbor blocks. JAX returns traced scalar arrays;
decode the integer status on the host or compare it within JAX control flow.

| `CGStatus` | Meaning |
| --- | --- |
| `INACTIVE` (0) | No active restraint window; no evaluations, with zero counters/value/norm |
| `CONVERGED` (1) | Initial or accepted gradient meets `gtol` |
| `MAX_ITER` (2) | The iteration budget ended before gradient convergence |
| `LINE_SEARCH_FAILED` (3) | No acceptable strong-Wolfe step within the search/bound budget |
| `NONFINITE` (4) | A nonfinite initial evaluation or the final failed search trial/slope |
| `NO_PROGRESS` (5) | The final failed trial cannot change representable coordinates |

Failure diagnostics describe the last accepted point, or the initial evaluation
when none was accepted. Rejected trial calls still count. An initially nonfinite
value/norm remains nonfinite in the report. Empty specs retain `get_minimizer() is
None`, including with the flag; direct `minimize` reports `INACTIVE`. Re-setup
clears both JAX factories. Requesting diagnostics with L-BFGS raises `ValueError`;
its existing coordinate-only behavior is unchanged.

`finalize` is an optional verbose diagnostic, not another minimization. It reports
ungated energies at the supplied coordinates, including custom terms and static
and dynamic VdW, without moving atoms. Its failures are reported as warnings and
do not abort inference. A zero final energy can mean a satisfied restraint; verify
construction with the setup counts. A Torch/NumPy diagnostic with dynamic VdW can
construct the optimizer and select Torch before the first `minimize`; the JAX
diagnostic evaluates directly without locking an unused instance to JAX.

## Atom and array contracts

The minimal adapter implements `iter_atoms()` over non-padding `AtomRecord`s.
`index` is the zero-based row in the predictor's flattened, possibly padded atom
axis. `resid` is a one-based ordinal within each chain, not an author residue
number. Atom names, normalized molecule types, residue names, and per-chain
conformer opt-ins support the richer selectors and polymer/reference paths.

Conformer adapters additionally expose the padded atom count, atomic numbers
(`0` for padding), ligand conformers, and polymer reference positions when needed.
Each `LigandConf` maps RDKit atom order to coordinates through `global_indices`;
different molecules must have disjoint global indices. Source chemistry retains
formal charges, hydrogen counts, isotopes, and stereochemistry when a coordinate
graph is reconstructed. A coordinate-free `stereo_mol` may supply that source graph.
The three biotite integrations share `_biotite_adapter.py`; framework-specific
adapters translate features rather than implementing restraint energies.

Runtime coordinates have shape `(..., N, 3)` in Angstrom. `active_sites` stores
global rows participating in restraints. Array-term indices address this gathered
subset, so local row `i` means global row `active_sites[i]`. Noncontiguous global
rows and independent batch members are valid. All batch energies are summed into
one scalar, so solvers can share line-search decisions across a batch even though
the objective contains no cross-sample contacts.

Restraint rows and variable-size atom groups are padded with explicit masks. The
prepared shapes remain static during minimization; padding must contribute no
energy or gradient. Reference atoms are constants, not additional prediction rows.
Atoms outside `active_sites` are unchanged. `move` controls one term's gradient:
an atom pinned by one term can still move under another term.

Torch's prepared arrays, gate cache, custom closures, and compiled artifacts are
keyed by both device and dtype. Per-entry gate decisions use the host NumPy spec
arrays, avoiding a device-to-host read per restraint category at each diffusion
step. Peptide-state masks are invocation-specific and never stored in these caches.

## Objective and units

The objective sums all enabled terms and batch members; there is no common `1/2`
factor or normalization by the total number of restraints. For the four shared
penalty shapes, `E = weight * delta**2`:

| Shape | Residual for measured value `q` |
| --- | --- |
| `harmonic` | `q - target` |
| `flat-bottomed` | `q - clip(q, lower, upper)` |
| `flat-bottomed1` | `min(q - lower, 0)` |
| `flat-bottomed2` | `max(q - upper, 0)` |

Conformer slack instead removes a symmetric interval from a target deviation:
`delta = sign(q - target) * max(abs(q - target) - slack, 0)`. Plane slack is
one-sided because its target is zero. Numerical norm floors, angular clipping,
and a small RMS regularizer keep geometry finite near singular configurations;
the exact constants and branch rules live in `_geometry.py` and `_kernels.py`.

| Array term | Measured quantity and target | Units and penalty | Gate |
| --- | --- | --- | --- |
| `bond` | Interatomic length; conformer or dictionary length | Angstrom; symmetric slack, or stretch-only for `half` link rows | Shared conformer |
| `angle` | Three-atom angle; conformer or dictionary angle | Radians internally; symmetric slack with the linear-target branch below | Shared conformer |
| `chiral` | Signed scalar triple product about the first atom; reference or dictionary target | Angstrom cubed; no division by six; symmetric slack; dictionary `both` accepts either sign | Shared conformer |
| `plane` | RMS distance from the group's own least-squares plane; target zero | Angstrom; `max(q - slack, 0)` | Shared conformer |
| `cistrans` | Ordered torsion and periodicity `n`; chemical/reference/dictionary target | Radians; `wrap(n * (phi - target)) / n`, then symmetric slack | Shared conformer |
| `vdw` | Pair distance relative to a chemical contact | Angstrom; repulsive overlap divided by pair ESD | Shared conformer |
| `distance` | Distance between two geometric centroids; user target/bounds | Angstrom; four shared shapes | Per entry |
| `rmsd` | Proper-rotation Kabsch fit followed by RMS measurement; reference structure and user target/bounds | Angstrom; four shared shapes | Per entry |
| `group_angle` | Three centroids, vertex at group 2; user target/bounds | Radians internally; config defaults to degrees; four shared shapes | Per entry |
| `group_dihedral` | Ordered four-centroid torsion about groups 2-3 | Same angular units; harmonic deviation wraps at pi | Per entry |
| `group_improper` | The same ordered torsion convention as `group_dihedral`, with separate config and diagnostics | Same angular units and periodicity behavior | Per entry |
| `group_chiral` | Signed scalar triple product of four geometric centroids about group 1; explicit user target/bounds | Angstrom cubed; no division by six; four shared shapes | Per entry |
| `group_plane` | RMS from one plane fitted to the pooled selected atoms | Angstrom; four shared shapes, default harmonic target zero | Per entry |

`group_improper` is not a separate arcsine elevation or chiral-volume formula.
Dihedral/improper flat intervals are ordinary ordered intervals and cannot cross
the minus-pi/pi boundary. `group_plane` pools one to four contiguously numbered
groups; pinning is per atom and every group is free by default. The conformer
plane and standalone plane use the same least-squares measurement but different
target construction, weighting, and gates.

For conformer angles strictly within 0.5 degrees of a 180-degree target, the
zero-slack penalty is `2 * weight * (1 + cos(theta))`. Nonzero slack uses a chord
residual with the same angular free interval, and each bond norm in this cosine
has a 0.02 Angstrom floor. This branch does not alter standalone or custom angles.

### Target provenance and weights

Ligand targets normally come from a copy of the predictor reference relaxed by
RDKit UFF, followed by geometric measurement. `relax_force_field.ligand` selects
UFF, MMFF94, MMFF94s, or `none`. Ordinary relaxation requires an aromatic or double
bond as evidence of real bond orders. Stereo validation also covers saturated
chiral molecules and can retry four deterministic ETKDG seeds. UFF can retain a
correct reference when relaxation fails; explicitly selected MMFF variants raise
on failure. An unrecoverable reference with incorrect source stereochemistry
raises instead of enforcing the wrong isomer. Relaxation operates on a copy and
does not mutate source aromaticity. Plane membership is confirmed on relaxed
coordinates, so the selected force field can change the plane count.

Polymer conformer restraints require per-chain opt-in. With no monomer library,
reference bond/angle/chiral/plane geometry is supplemented by template-derived
chi, omega, and acyclic sp2 torsions. Approximate chi periods depend on axis
hybridization (3/6/2); their ESDs are 10/10/5 degrees, and approximate omega and
sp2 ESDs are 5 degrees. Ligand E/Z keeps period one. No complete backbone phi/psi
or nucleic backbone torsion potential is implied by `cistrans`.

An enabled CCP4 monomer library replaces reference-derived tuples wholly inside
covered residues. Link add/change/delete operations are applied before deriving
geometry, chiral volumes, and propagated ESDs. Dictionary torsion signs are
negated to match RGI's ordered-torsion convention; nonpositive periods become one.
Dictionary weights are `user_weight / ESD**2`, with angles and their ESDs converted
to radians. A dictionary plane additionally multiplies by its atom count, making
the squared-RMS kernel equal the sum of per-atom squared distances. ESD is not
slack. Dictionary slack defaults to zero; explicit slack remains independent.
Disabled/nonpositive-ESD terms retain needed topology exclusions, while nonfinite
active targets or ESDs raise.

Automatic library acquisition is lazy and process-locked, validates a temporary
snapshot, then publishes it atomically under the user's configuration cache.
Explicit paths never download; complete snapshots are reused offline without
automatic updates. The source commit is logged. Missing entries follow the
configured `on_missing` policy. Library acquisition and RDKit preparation happen
at setup, not within objective evaluations.

Peptide cis/trans alternatives are selected separately for each local link and
batch member from the coordinates at the start of a minimization. Ties and
degenerate states choose trans. The selected link geometry and modifications
remain fixed through all line-search trials and neighbor-list blocks; the next
denoising invocation can select again.

### Reference, custom, and macro restraints

RMSD references accept mutually exclusive PDB or mmCIF paths, parsed with Gemmi
into the same atom records. Target/reference fit and calculation selections are
independent. Pairing defaults to polymer sequence alignment through Biopython;
identity pairing uses chain, per-chain residue ordinal, and normalized atom name.
Ligands/nonpolymer inputs use identity-style fallback. Atom prime spellings are
normalized, and polymer selectors do not classify ligands by atom name alone.

Built-in geometry containing `refN and ...` is routed to a reference closure.
A configured reference fit can align its coordinates to the live prediction.
A reference plane is fitted to the reference atoms alone and measures prediction
distances from that plane; it is a different objective from pooling both groups
into a freely fitted plane.

Standalone `chiral_restraints_config` uses four selections, each of any nonzero size,
with the same signed scalar triple product as conformer chiral. All groups are free
by default; reference groups are fixed. Targets/bounds are `target_chiral`/`target_chiral1`/
`target_chiral2` in Angstrom cubed, and `unit` is rejected. It has its own `group_chiral`
array term and entry windows, independent of conformer opt-in and conformer windows.

Custom entries may be a restricted formula, a registered Python function, or an
`add_custom` callable. They resolve named selections and references during setup
and create backend closures returning a scalar, including reduction over batches.
The formula parser permits the documented geometry/math/penalty vocabulary and
rejects arbitrary Python evaluation. Python callables are trusted code and should
use the supplied context to remain portable across backends. Custom angular
functions return radians; `chiral(A,B,C,D)` / `ctx.chiral(...)` returns the signed
scalar triple product about A in Angstrom cubed. Custom `move` stops gradients through unlisted
prediction selections; reference coordinates remain fixed. Custom centroid
functions use ordinary mean derivatives, without the built-in group rescaling.

Torch custom closures prepare reference-coordinate tensors before differentiation
and compilation, avoiding NumPy conversion inside a grad transform. Evaluation casts
them to the coordinate dtype; the optimizer rebuilds closures on device/dtype changes.

The base-pair configuration is a macro for named nucleotide H-bond distances and
optional pooled coplanarity. It expands into ordinary distance and plane entries
with their weights, movement choices, and windows; it has no separate optimizer.

### Gradient conventions

Gradients come from Torch/JAX autodiff. Several deliberate transformations affect
how they should be checked:

- Built-in centroid terms preserve energy values while scaling centroid gradients.
  Group angle/dihedral/improper/chiral multiply by the group atom count `N`, removing the
  mean's `1/N` dilution. Distance uses `N` when only one group moves and
  `N1*N2/(N1+N2)` when both move. For an isolated pair this gives equal translation
  within each group and a displacement ratio `N2:N1`, preserving the atom-count
  weighted center. These are not ordinary derivatives of the reported scalar.
- `move` pins the selected term's gradient without removing pinned atoms from its
  measured geometry. For a plane they still influence the fit. Plane terms have
  no centroid rescaling.
- Kabsch rotations and fitted plane normals are recomputed at each evaluation but
  detached from autodiff. Centers remain differentiable. For a nondegenerate
  least-squares fit measured on the same atoms, the envelope theorem makes this
  consistent with differentiating the minimized scalar; it need not agree when
  RMSD fit/calc sets differ or a plane fitted to one moving group measures another.
- Near-exact VdW overlaps use a deterministic pair-dependent separation direction
  with a straight-through derivative. Finite differences of the raw radial
  function do not validate this escape rule at the singularity.

Backend agreement, ordinary finite differences, and these movement conventions
are distinct checks. CG can stall on a surrogate direction even when its reported
energy is finite; general convergence proofs for smooth gradients do not establish
convergence for every RGI combination.

Chiral geometry is implemented once in `_geometry.chiral_points` for conformer,
standalone, reference and custom paths. An odd permutation of the four points reverses
the sign; proper rotations and translations preserve it. Collinear or coincident
centroids have zero volume and zero gradient; no escape displacement is introduced.
Single-atom standalone harmonic/interval restraints match conformer chiral at the same
target, weight and slack. Existing dictionary `both` still acts on the absolute volume;
custom formulas express that objective using `abs(chiral(...))`.

## Activation windows

A restraint uses either `stop_sigma <= sigma <= start_sigma` or
`start_step <= step <= stop_step`, with both endpoints included. Sigma defaults
are positive infinity and minus one; step defaults are negative and positive
infinity. Mixed axes, NaN bounds, and empty windows are rejected at parsing.
The conformer terms share a window; distance, RMSD, standalone group terms, and
custom/reference entries have individual windows. Base-pair expansion propagates
its configured window to the generated entries.

Gates multiply masks/energies; they do not rebuild the spec. If sigma exceeds all
start thresholds, minimization returns immediately. Inactive terms contribute no
update even when another term keeps the solver active. Torch's `sigma=None`
omits sigma gating, while the public call still supplies its `istep` value. The
JAX pure minimizer requires the actual schedule values. `finalize` ignores windows
so its energy reports are not necessarily the objective of the last denoising step.

## VdW execution

VdW is an opt-in conformer term. `mode` selects intramolecular, intermolecular, or
both categories. For an eligible pair, its contribution is
`weight * (min(d - scale * contact, 0) / ESD)**2`. Chemical contact priority is
1-4, hydrogen bond, metal, dummy, then ordinary radius sum. Pair ESD is 0.2 Angstrom
except dummy contacts at 0.3 Angstrom; hydrogen-inclusive radii are capped at
2 Angstrom. Dictionary energy types take priority over approximate source/template
chemistry, with a warning and elemental fallback when unavailable. All atoms,
including fixed background and nonrestrained ligands, are typed.

| Pair path | Construction and movement |
| --- | --- |
| Within a restrained ligand | Static eligible pairs; excludes covalent 1-2/1-3 and same-plane 1-4 pairs; independent of reference distance |
| Between restrained ligands | Static all-cross-pairs rows; both endpoints move; no cutoff in unrelated reference coordinate frames |
| Restrained atoms against background | Dynamic two-set list; background is non-padding atoms outside `active_sites`, held at the invocation's coordinates |
| Eligible active-active contacts | Dynamic list covering polymer contacts and other moved atoms with conformer-restrained participation; static ligand pairs are excluded to prevent double counting |

Topology and plane exclusions survive disabled geometry weights. A sorted cell
list filters topology, molecule mode, and moving participation before retaining
`max_neighbors` candidates by VdW clearance. It avoids a dense atom-pair distance
matrix; see [computational cost](config.md#computational-cost-current-implementation)
for ordinary-density and worst-case bounds. Neighbor truncation can discard
contacts, so equivalence to a dense all-pair objective requires adequate capacity.

CG caps per-atom trial displacement at `max_atom_step` whenever VdW is active.
Staleness is checked every `neighbor_rebuild_interval` iterations. Let
`M = max_atom_step * neighbor_rebuild_interval`; the fixed-background search radius
includes at least `max_contact + M + neighbor_skin`, and the active-active radius
includes `max_contact + 2*M + neighbor_skin`. Measured displacement since the last
build triggers rebuilding above the skin for fixed-background lists or half the
skin for active-active lists. Defaults are 0.1 Angstrom, 10 iterations, and
2 Angstrom respectively. A check without a rebuild preserves the CG state. An
actual rebuild invalidates it because truncation can change the objective.

L-BFGS rebuilds dynamic pairs at every objective evaluation, including line-search
trials, since it has no CG displacement cap. Every search and diagnostic radius
is at least the largest contact even if `dmax` is smaller. Peptide-state choices
remain fixed through these rebuilds.

## Optimizers

### Nonlinear conjugate gradient

`method: CG` uses one algorithm in three execution forms:

| Implementation | Execution |
| --- | --- |
| `TorchRestraintOptimizer._minimize_cg` | Eager Torch autograd and a Python loop |
| `_torch_cg_gpu._cg_minimize_torch` | Functional gradient/value callable; CUDA normally uses an Inductor-compiled energy/gradient with host-controlled early exits |
| `jax_optim._cg_minimize` | Pure JAX autodiff and `lax.while_loop`, usable inside JIT/scan |

Torch energy compilation explicitly uses `dynamic=False`: each artifact specializes
the spec and neighbor-list shapes. This also avoids automatic symbolic-shape
generalization across different structures or VdW modes.

All three forms call the shared PR+ loop in
[`optim/_cg.py`](../src/rgi_toolkit/optim/_cg.py) and the shared scalar line-search
transitions in [`optim/_cg_linesearch.py`](../src/rgi_toolkit/optim/_cg_linesearch.py).
The reference is **SciPy 1.17.1**, specifically
[`_minimize_cg`](https://github.com/scipy/scipy/blob/v1.17.1/scipy/optimize/_optimize.py),
[Wolfe1/Wolfe2 searches](https://github.com/scipy/scipy/blob/v1.17.1/scipy/optimize/_linesearch.py),
and [`DCSRCH`/`dcstep`](https://github.com/scipy/scipy/blob/v1.17.1/scipy/optimize/_dcsrch.py).
The adapted code retains the SciPy BSD notice in [`LICENSES/scipy.txt`](../LICENSES/scipy.txt),
shipped in source and wheel distributions, and the MINPACK attribution in the source.
SciPy remains a pinned development oracle; runtime minimization imports no SciPy.

For objective `f`, autodiff vector `g`, and search direction `d`:

```text
d0 = -g0
previous_f0 = f0 + norm(g0, 2) / 2
alpha0 = min(1, 1.01 * 2 * (f - previous_f) / dot(g, d))
beta = max(0, dot(g_new, g_new - g) / dot(g, g))
d_new = -g_new + beta * d
```

A negative initial step guess is replaced by one, following SciPy; the applicable
step bounds then limit it. The previous accepted objective, rather than the previous
step length, supplies the next initial guess. No epsilon is added to the PR+
denominator: a nonfinite or nonpositive squared gradient cannot start a search.

The first search uses More--Thuente DCSRCH with its four safeguarded `dcstep`
interpolation cases. If it fails or its candidate fails the prospective PR+
sufficient-descent check, the solver invokes SciPy's Wolfe2 bracketing/zoom
procedure. Both require strong Wolfe (`c1=1e-4`, `c2=0.4`):

```text
f(x + alpha*d) <= f(x) + c1 * alpha * dot(g, d)
abs(dot(g_new, d)) <= -c2 * dot(g, d)
```

An accepted candidate additionally satisfies
`dot(g_new, d_new) <= -0.01 * dot(g_new, g_new)`, unless its gradient already meets
`gtol`. That exception bypasses only the prospective-direction check, never Wolfe.
DCSRCH permits 100 iterations and uses `xtol=1e-14`; the Wolfe2 expansion budget is
10, and its zoom loop follows SciPy's `i > 10` exhaustion rule (up to 11 trials).
The current trial's value and gradient are cached across search phases. Evaluation
counters count actual combined value/gradient calls, including rejected trials.
Constants live in [`optim/_cg_config.py`](../src/rgi_toolkit/optim/_cg_config.py).

Only `max(abs(g)) <= 1e-7` reports gradient convergence. There is no energy-change
stop or restart latch, no accepted-step doubling, and no steepest-descent retry
after failed searches. Failure returns the last accepted coordinates and terminates
that minimization, even if earlier iterations moved atoms. The next denoising
invocation starts a fresh CG search.

Strict RGI acceptance differs from SciPy's exceptional exits: Wolfe2's unverified
last trial on iteration exhaustion is rejected, and nonfinite coordinates, values,
gradients, or a step with no representable movement cannot be accepted. Invalid
DCSRCH trials trigger Wolfe2; invalid high-bracket trials are bisected toward the
last finite lower endpoint. Scalar bounds `1e-100..1e100` are restricted to the
working dtype's representable range (upper bound at most `finfo.max / 8`). These
safeguards and different floating-point evaluation orders preclude a promise of
bitwise agreement on every objective.

When VdW is active, every trial stays on one straight line. A single scalar bound
`alpha_max = max_atom_step / max_i(norm(d_i, 2))` limits all atoms together; no
per-atom clipping bends the direction. If no point within the bound satisfies
Wolfe, the invocation fails without an Armijo fallback. For example, minimizing
`(x-1)**2` from zero with an absolute step bound of `0.1` cannot reduce the slope
magnitude to `0.4` of its initial value. Increasing `max_iter` cannot resolve that
infeasible line search. RGI's modified gradients (centroid rescaling, pinned atoms,
and stop-gradient fits) can also make strong Wolfe harder or impossible to satisfy;
convergence of ordinary smooth SciPy objectives does not certify those cases.

The state carries `f`, `g`, `d`, `dot(g,g)`, the previous objective, a validity flag,
and cumulative `CGInfo` across neighbor-list blocks. Only block-budget exhaustion
is resumable; convergence or failure ends the whole call. An actual neighbor
rebuild invalidates search history and recomputes value/gradient while retaining
the aggregate counters. Unchanged lists retain the state without another initial
evaluation. Coordinates and the objective must remain unchanged to reuse a state.

The mathematical references are [Polak and Ribiere (1969)](https://numdam.org/item/M2AN_1969__3_1_35_0/)
and [Gilbert and Nocedal (1992)](https://epubs.siam.org/doi/10.1137/0802003), with the
More--Thuente and bracketing algorithms documented in the pinned SciPy sources.
Their smooth-objective assumptions do not establish a convergence theorem for
RGI's capped, sometimes modified-gradient molecular objective.

### L-BFGS

L-BFGS is delegated to existing libraries rather than reimplemented. The method
is described by [Liu and Nocedal (1989)](https://link.springer.com/article/10.1007/BF01589116).

| Backend | Delegation and explicit RGI options | Other stopping/history settings |
| --- | --- | --- |
| Torch | [`torch.optim.LBFGS`](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/lbfgs.py), `max_iter`, `line_search_fn="strong_wolfe"` | Upstream defaults; the locked Torch 2.6 uses gradient tolerance `1e-7`, change tolerance `1e-9`, history size 100 |
| JAX | [`jaxopt.LBFGS`](https://jaxopt.github.io/stable/_autosummary/jaxopt.LBFGS.html), `maxiter`, `linesearch="backtracking"`, `implicit_diff=False` | Upstream defaults; the locked JAXopt 0.8.5 uses tolerance `1e-3`, history size 10, maximum 30 line-search steps |

These defaults and stopping quantities differ across libraries and can change
when dependencies are updated. `uv.lock` defines the repository test environment;
predictor environments may pin other versions. Torch's source attributes its
implementation to minFunc; JAXopt documents the standard limited-memory inverse
Hessian method. RGI's use of either does not imply matching their defaults to CG.
The independent SciPy comparison uses unbounded `L-BFGS-B`; this is a comparison
of final solutions and residuals, not a claim of identical implementations.

## Verification contract

The optimizer comparison modules require SciPy 1.17.1, Torch, JAX, and JAXopt on CPU;
missing dependencies fail collection rather than silently skipping a backend.
Fixtures are small, deterministic, offline, and generated in memory or pytest
temporary directories. They do not require an external monomer library or
predictor installation. The oracle calculations in these modules do not call
RGI's geometry or energy kernels.

[`test_optimizer_reference.py`](../tests/test_optimizer_reference.py) compares
all three CG forms and both L-BFGS backends against independent analytic/SciPy
objectives: isotropic and diagonal quadratics, a coupled quadratic with condition
number 256, Rosenbrock plus a decoupled quadratic coordinate, and an already-solved
initial point. SciPy CG and unbounded L-BFGS-B references are themselves checked
against analytic optima and gradients. Tests use float64 and at most 10,000
iterations. For these unique solutions, the acceptance limits are:

- Energy difference at most `1e-6 * (1 + abs(reference_energy))`.
- Coordinate maximum absolute difference at most `1e-3`.
- Independent gradient maximum absolute value below `1e-6` for CG or `1e-3`
  for the existing L-BFGS settings.

The same module checks previous-objective/state preservation across blocks,
nonfinite trial gradients, and unrepresentable updates. Separately marked GPU
tests check the compiled Torch path and JAX CG inside JIT/scan against SciPy.
[`test_cg_linesearch.py`](../tests/test_cg_linesearch.py) compares safeguarded
interpolation cases, search-phase transitions and representative accepted
trajectories with the pinned SciPy implementation, using the same tolerance and
iteration budget. It independently checks Wolfe and prospective descent.
[`test_optimizer_info.py`](../tests/test_optimizer_info.py) verifies the public
diagnostics, inactive windows, batch aggregation, reset, JAX scan output, strict
bounded failure, and termination/counter preservation across neighbor blocks.

[`test_e2e.py`](../tests/test_e2e.py) follows config parsing, adapter selections,
spec construction, minimization, and verbose finalization. It drives Torch's
public mutation API and JAX's public minimizer inside JIT/scan, with both methods.
It checks initial as well as final energies so an incorrect coefficient cannot
hide behind a zero-valued minimum. Coverage includes all distance penalty shapes,
unequal centroid groups and move modes, noncontiguous atom rows, batch inputs,
sigma/step boundaries, repeated setup, group geometry, conformer geometry,
PDB/mmCIF RMSD, custom formulas/callables, and static/dynamic VdW. Dynamic contacts
are compared with direct dense evaluation at both the energy and stationarity
levels. Their convergence fixtures use a nonbinding displacement cap; separate
fixtures prove that an infeasible cap returns failure and retains coordinates.
Nonunique geometric solutions are compared through energy, measured
geometry, rigid-group motion, and pins instead of arbitrary Cartesian equality.
Nonconvex fixtures keep starts in the same basin; these tests do not certify a
global minimum for general molecular objectives.

| Additional contract | Existing verification |
| --- | --- |
| Backend energy/gradient agreement and fixed-fit geometry | [`test_backend_parity.py`](../tests/test_backend_parity.py), [`test_shared_geometry.py`](../tests/test_shared_geometry.py) |
| Chiral volume, independent determinant, conformer equivalence, custom derivatives, and CPU/GPU entry paths | [`test_chiral.py`](../tests/test_chiral.py) |
| Selection grammar, atom names, reference pairing | [`test_selection.py`](../tests/test_selection.py), [`test_reference_atom_names.py`](../tests/test_reference_atom_names.py), [`test_align.py`](../tests/test_align.py), [`test_ref_config.py`](../tests/test_ref_config.py) |
| Config rejection and activation bounds | [`test_config_validation.py`](../tests/test_config_validation.py), [`test_window_params.py`](../tests/test_window_params.py) |
| Public state, adapters, and scan wrapper | [`test_combined_restraints.py`](../tests/test_combined_restraints.py), [`test_adapters_shared.py`](../tests/test_adapters_shared.py), [`test_scan_runner.py`](../tests/test_scan_runner.py) |
| Chemical targets, relaxation, stereochemistry | [`test_featurizer.py`](../tests/test_featurizer.py), [`test_conformer_chemistry.py`](../tests/test_conformer_chemistry.py), [`test_relax_force_field.py`](../tests/test_relax_force_field.py), [`test_ideal_conformer.py`](../tests/test_ideal_conformer.py) |
| Dictionary targets, links, ESDs, cache publication | [`test_monlib_geom.py`](../tests/test_monlib_geom.py), [`test_monlib_dictionary.py`](../tests/test_monlib_dictionary.py), [`test_monlib_esd.py`](../tests/test_monlib_esd.py), [`test_monlib_cache.py`](../tests/test_monlib_cache.py) |
| Standalone geometry, macro expansion, custom move/gradient rules | [`test_group_geom_data.py`](../tests/test_group_geom_data.py), [`test_improper.py`](../tests/test_improper.py), [`test_plane_restr_data.py`](../tests/test_plane_restr_data.py), [`test_base_pair.py`](../tests/test_base_pair.py), [`test_custom.py`](../tests/test_custom.py), [`test_custom_move.py`](../tests/test_custom_move.py) |
| Dtype/device caches, VdW lists, compiled objectives | [`test_optim.py`](../tests/test_optim.py) |

Run the mandatory CPU suite and code checks with:

```bash
uv sync --frozen --extra torch --extra jax
JAX_PLATFORMS=cpu uv run pytest -m "not gpu" -q
uv run ruff check src tests
uv run ruff format --check src tests
```

The existing CI installs both backend extras and executes the non-GPU suite.
GPU-marked cases require a CUDA/JAX accelerator environment and a compute-node
allocation; CPU execution of the functional CUDA loop does not validate GPU
compilation, device synchronization, or accelerator performance. Full predictor
sampling and structural/scientific validation remain separate from toolkit E2E.

## Conformer activation and torsion priority

`RestraintsConfig.conformer_config` is `None` for an absent/null block and a dictionary
for an explicit block, including `{}`. The shared `conformer_weight` helper supplies
weights 1 for bond/angle/chiral/cistrans/vdw and 0 for plane. Explicit nonpositive/null
weights disable a term. Molecule opt-in remains mandatory. Dictionary collection,
reference featurization and VdW consume the same effective weights.

After collecting enabled torsions and snapshotting topology exclusions, discard any
conformer plane containing all four atoms of an enabled torsion. Local dictionary
conditions are subtracted as disjoint conjunctions, without whole-chain enumeration;
surviving reference planes retain their original weights and slack. The existing
per-invocation peptide selector binds these conditions once per minimization. No backend
kernel, standalone group-plane, base-pair or custom energy semantics change.

## External configuration resolution

`resolve_restraints_config(config, *, base_dir=None)` is public from `rgi_toolkit` and
`rgi_toolkit.config`. `RestraintsConfig.from_dict` accepts the same keyword. A mapping
containing only `config_path` replaces the root configuration or one whole registered
restraint section with a JSON/YAML value. File-local paths propagate through nested
includes and external structure/dictionary references; inline resource behavior is
preserved. Python dictionaries default to the working directory. The resolver copies
input data, rejects mixed wrappers and cycles, and leaves schema validation to the
existing parser. Predictor file loaders resolve before input location is discarded or
preprocessed inputs are serialized. Chai extracts its chain opt-in map after expansion.
