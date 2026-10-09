# Testing

Prepare the CPU development environment with:

```bash
uv sync --extra torch --extra jax
```

The test tasks use `uv run --active --no-sync`: an activated virtual environment
takes precedence over the project environment, and running a test does not change
installed dependencies. Commands are run from the repository root. On a cluster,
run tests in an appropriate compute allocation.

| Command | Purpose | Coverage |
|---|---|---|
| `task test-ci` | Short CI validation, targeting about one minute of test execution | Important numerical contracts and small representative integration cases |
| `task test-local` | Comprehensive local CPU validation | Every non-GPU test, including real Torch compilation |
| `task test-gpu` | Comprehensive GPU validation | Every test marked `gpu`; requires a CUDA-enabled environment |
| `task test` | Alias for `task test-local` | Complete CPU validation |

## Short CI selection

GitHub Actions and `task test-ci` execute the same explicit selection in
[`tests/ci.txt`](../tests/ci.txt), using pytest's
[argument-file support](https://docs.pytest.org/en/9.0.x/how-to/usage.html#read-arguments-from-file).
Missing or renamed selected tests cause an error rather than silently reducing
coverage. The selection includes:

- Configuration validation, defaults, selection, reference pairing, adapter atom
  order and chemistry, restraint construction, and activation windows.
- NumPy/PyTorch/JAX energy agreement, autodiff versus finite differences,
  Kabsch/RMSD behavior, and fixed versus moving atom groups.
- CG line-search agreement with SciPy, nonfinite-gradient handling, rejected
  steps, and preservation of the last accepted coordinates.
- VdW topology exclusions, dense versus neighbor-list energies and gradients,
  overflow, exact overlaps, and contact-cache validity.
- Small public-API minimizations for distance, angle, dihedral, RMSD,
  ligand conformer stereochemistry, custom energies, and intermolecular VdW.
  These retain independent SciPy/geometry assertions and selected JAX execution.
- CPU compilation defaults, explicit opt-out, and simulated compiler failure
  fallback without changing CUDA failure state.

Torch compilation is disabled for CI execution to avoid repeated compiler startup;
the failure-path test injects a failing artifact and does not invoke a compiler.
JAX numerical and selected JIT/minimization checks still run. Real Torch compilation
is verified locally. CI uses two file-grouped workers. CPU tasks enable JAX float64
support explicitly so available precision does not depend on test order.
This is a curated subset, not a replacement for full validation.

## Comprehensive local validation

`task test-local` runs the complete CPU suite in two stages:

1. All ordinary numerical regressions, exhaustive solver/parameter combinations,
   large atom groups, and public-API end-to-end tests, using two file-grouped workers.
2. All `cpu_compile` tests with compilation enabled, including compiled energy and
   gradient parity, dynamic VdW, custom restraints, CG/L-BFGS, and dtype changes.

The two stages together include every CI case. No existing regression is deleted or weakened.
Actual compiler checks require a C++ compiler and Python development headers;
inspect skips before treating local validation as complete.

For GPU tests, activate an existing predictor environment with CUDA-enabled
PyTorch and JAX plus the test dependencies, then run `task test-gpu` in a GPU
allocation. The development lockfile installs CPU-only PyTorch, so it is not a
GPU test environment. The task does not sync or replace an activated environment's
framework packages. GPU tests are additional to the complete CPU suite.
All compiler-heavy checks remain outside short CI.

## Maintaining coverage

Run `task test-local` before delivering code changes, and `task test-gpu` when GPU
behavior changes. A correctness fix should have a small representative CI regression
where practical, with broader combinations retained locally. Keep assertions and
tolerances unchanged when selecting cases. Inspect `--durations` output and keep
compiler-heavy or large-system sweeps out of the short selection. Tests that need
real CPU compilation carry `pytest.mark.cpu_compile`.

Additional pytest options can follow `--`, for example
`task test-ci -- --durations=20` or `task test-gpu -- --collect-only -q`.
