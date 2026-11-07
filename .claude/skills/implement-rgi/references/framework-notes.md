# Framework notes

## PyTorch (eager loop)

Boltz, Protenix, OpenDDE, Chai-1, OpenFold-3, ESMFold2, RF3, BioEmu2, and Mol-1
use the PyTorch path.

- **autograd under `inference_mode`**: predict loops often run inside
  `torch.inference_mode()`, where autograd backward fails with "element does not
  require grad / no grad_fn". rgi_toolkit' torch optimizer handles this for you: it
  wraps the minimization in `torch.inference_mode(False) + torch.enable_grad()`
  and clones the active coords into a normal leaf (`empty_like + copy_`) so
  backward works. You don't write this — but know it's why passing an
  inference-mode coordinate tensor to `minimize` is fine.
- **adapter placement**: `rgi_toolkit/<tool>/adapter.py` (receives a plain
  dict/array, imports no framework code — EXCEPT boltz, whose feats are native torch
  tensors, so its adapter imports torch, read at batch 0).
- **VdW flavours**: the conformer `vdw` term has two flavours — static
  intramolecular (ligand-internal pairs) and dynamic fixed-background / intermolecular
  (a per-step radius search). `mode` defaults to **`both`** (run them together); both now run
  on **torch AND jax**, so a JAX tool gets the full VdW too.

ESMFold2's native implementation is under `esm_restr/esm/models/esmfold2/`:
`processor.py` builds the restraint state, `model.py` and `experimental.py`
forward it, and `layers.py` applies the sampling hook. The old `transformers_restr` model
implementation and installed-package copy workaround are no longer needed.

## JAX (JIT / `lax.scan`) — AlphaFold3 and IntelliFold v2

- **adapter placement**: the framework-free adapter lives in rgi_toolkit
  (`rgi_toolkit/alphafold3/adapter.py`), like the torch tools; only a thin in-tool shim
  (`alphafold3_restr/.../restraints/adapter.py` `build_af3_adapter`) does the one
  alphafold3-coupled step — CCD/SMILES mol resolution + `fold_input` read — and feeds
  the rgi_toolkit adapter plain data.
  IntelliFold uses the same arrangement through `intellifold.rgi.build_restraints`
  and `rgi_toolkit/intellifold/adapter.py`.
- **no Python callbacks in the scan**: build the spec outside the scan (numpy, at
  build time) and inject `get_minimizer()`'s pure closure inside. Never call the
  Python `minimize` per step in a compiled loop.
- **line search is load-bearing**: the jax optimizer uses jaxopt
  `NonlinearCG`/`LBFGS` with **`linesearch="backtracking"`** (Armijo). This is
  not cosmetic. The default `zoom`/`hager-zhang` searches can accept a huge first
  step that collapses atoms onto each other, where the eps-regularized distance
  `sqrt(0 + eps)` makes the gradient vanish — a false stationary point the solver
  then "converges" to. Backtracking rejects that step. A plain fixed-step
  gradient descent diverges (NaN) on degenerate geometry, which is why a real
  line-searched solver is required, not hand-rolled GD.
- **coordinate reshape**: AF3 coords are `(num_tokens, max_atoms, 3)`; reshape to
  `(-1, 3)` before the minimizer and back after. That reshape is the only glue.
- **non-finite guard**: the jax minimizer returns the input coords unchanged if a
  step produces NaN/Inf, so one bad geometry can't poison the whole trajectory.

## Why the old AF3 path was slow (and the fix)

The original AF3 restraint code minimized with `jaxopt.ScipyMinimize` (scipy,
*outside* JIT, with O(n) finite-difference gradients) called via
`jax.pure_callback` — a CPU round-trip on every denoise step, which was
extremely slow. Replacing it with rgi_toolkit' in-JIT `make_minimizer` (analytic
`jax.grad`, running entirely on the accelerator) is the "slow → fast" fix. When
you integrate a JAX tool, use `get_minimizer()`; do not reach for scipy or
`pure_callback`.

## Backend selection recap

The backend is **inferred from invocation, not a config key**: a JAX tool grabs the
pure minimizer via `get_minimizer()` → jax; a PyTorch tool calls `minimize(coords)`,
where a torch/numpy array → torch (resolved lazily on the first call). `config.gpu`
selects the *device*, not the backend: `gpu: false` runs the torch optimizer on CPU.
There is **no numpy optimizer** — numpy remains only as the energy reference for
`tests/test_backend_parity.py` (torch and jax compute identical energies/gradients).
A leftover `backend:` config key raises with a migration hint.
