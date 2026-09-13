# CG predictor verification

This harness tests the default distance workflow that previously stopped at the
VdW step bound, then exercises combined restraints in real diffusion predictors.
It uses maintained `examples/` inputs and an external cluster configuration;
generated inputs, checkpoints, caches, captures and structures belong outside the
tracked source tree or under an ignored directory.

The matrix contains Boltz-1/2, Protenix v1/v2, Chai-1, AlphaFold3, OpenFold-3,
ESMFold2 and OpenDDE. Each runs three seeds for seven cases: unrestrained,
distance, distance with an empty conformer block, distance with VdW alone, angle,
RMSD and custom distance difference. Distance also runs the successful historical
engine at `11de8b4aa49c9298a7de4fdc112afa322886aa40`: 216 predictions in total.

Distance and conformer weights, CG limits and VdW controls use library defaults.
The maintained RMSD fixture explicitly requests `max_iter: 1000` and releases its
two restraints below `sigma: 1.0`; that fixture setting is retained. The VdW-only
arm explicitly disables other conformer terms. Predictor settings are identical
between paired historical and candidate runs.
OpenFold-3 receives its model seed through a generated `runner.yaml`; its query
JSON alone does not override the runner's default seeds. Measurement verifies the
seed encoded in the exported structure path.

## Run

Use `uv` with a development environment containing NumPy, PyYAML, Gemmi,
Biopython, RDKit, Torch and SciPy 1.17.1. Predictor environments, model weights,
partition policies, precomputed MSAs and reference structures come from the
adjacent benchmark workspace's `config.yaml` and maintained fixtures. No model
weights or external databases are downloaded by this harness.

```bash
uv run python verification/cg/run.py prepare \
  --cluster-config ../bench-rgi/config.yaml --output .cache/cg-campaign
uv run python verification/cg/run.py submit \
  --manifest .cache/cg-campaign/manifest.json
uv run python verification/cg/run.py submit \
  --manifest .cache/cg-campaign/manifest.json --timing-only
```

`prepare` snapshots both engine sources and records the candidate source hash.
`submit` requests exclusive GPU allocations with at most four concurrent jobs.
Successful runs are retained; `--retry-failed` archives failed or interrupted
attempts before retrying. `--models`, `--cases` and `--after` allow bounded
resubmissions without overwriting successful predictions.

Submit timing after the capture campaign, preserving the four-job limit.
The separate 54 paired timing predictions disable all capture and diagnostic
wrappers, while keeping the historical import bridge. Both versions must still
satisfy the exported-structure distance gate.
Each timing input first runs once into a separate `warmup/` directory (54
additional predictions). A fixed Python hash seed keeps process-dependent
iteration order from producing different compilation-cache keys.

Run the following analysis commands on a CPU allocation with access to the
campaign storage. If CPU nodes use separate storage, `stage_replay.py MANIFEST
--output STAGING --include-structures` copies completed captures and CIF files
without moving the original artifacts. The repository's CPU and GPU pytest
suites are separate gates.

```bash
uv run python verification/cg/measure.py .cache/cg-campaign/manifest.json \
  --timing-only --require-complete
uv run python verification/cg/measure.py .cache/cg-campaign/manifest.json \
  --timing-report .cache/cg-campaign/timing_measurements.json --require-complete
uv run python verification/cg/replay.py .cache/cg-campaign/manifest.json \
  --model boltz2
```

Repeat replay for each model. It compares actual denoising inputs at steps 0,
100 and the final low-sigma call using double precision, the same objective and
CG iteration limit. The final index differs by predictor: for example, ESMFold2
crops its schedule and Chai includes a second-order corrector.
Dynamic VdW reference pairs are enumerated solely from chemical eligibility,
without coordinate cutoffs or neighbour capacities. SciPy termination, objective,
gradient norm and iteration count are recorded separately from the predictor's
captured result. Mixed distance/conformer cases use an independently constructed
version of the same fixed affine coordinate map. Nonstationary termination is
reported explicitly. Stationary results require energy agreement within
`1e-5*(1+abs(reference_energy))`. Unfinished nonlinear trajectories can amplify
roundoff, so replay also compares **each actual line search** against SciPy from
the identical direction, scalar objective, previous objective and PR+ state.
Step lengths must agree to relative `1e-8`; prospective descent is checked
independently with NumPy. Iteration-limited endpoints are reported as unfinished,
must lower the objective, and must pass this search audit. They are not treated
as converged solutions. Distance differences must remain within 0.1 Angstrom for
stationary or low-sigma (`sigma <= 1`) results; earlier finite-iteration distances
are reported separately. Exported-structure distance gates remain unchanged.
At initial, reference, candidate and captured coordinates, production VdW values
and gradients must match the complete-pair oracle to double-precision tolerances.
Replay `candidate_seconds` includes the SciPy search audit and is not a production
speed measurement; use the separate prediction timing campaign for performance.

A complete-pair reduction and a sparse reduction can differ at roundoff level,
which can change whether a high-sigma trajectory reaches stationarity within 100
iterations. If the complete-pair SciPy reference converges while the candidate
exhausts its budget, replay adds a conservative numerical control: SciPy must also
exhaust the same budget using the exact production floating-point callback, and
the candidate must have no larger residual objective. One additional candidate
invocation must reach the original reference minimum under the unchanged energy
and gradient tolerances. Such a record is labeled `finite_budget_roundoff`, keeps
the original `MAX_ITER` result and failed `stationary_match`, and still requires
every line-search audit and the original distance gate to pass. This diagnostic
does not change predictor coordinates, iteration limits or exported-CIF gates.

## Gates and evidence

`measure.py` parses the exported CIF independently of the optimization kernels.
Distance and custom residuals must be at most 0.1 Angstrom; angle residuals must
be at most 1 degree. Restraint inventory logs must show the expected active terms.
The ATP conformer cases additionally require positive bond, angle, chiral and
intramolecular VdW rows plus populated fixed-background VdW queries/partners.
The VdW-only cases require those other conformer terms to be disabled.
RMSD captures must reach both active targets within 0.1 Angstrom and leave
coordinates unchanged after release; final released RMSDs are reported separately.

Historical/candidate distance timings alternate run order and use the same seed
and exclusive GPU allocation. All three historical predictions must satisfy the
distance gate before their median can serve as the speed baseline. The candidate
median must be at most 1.2 times that baseline. Measured wall time includes process
startup and loading compiled executables; initial compilation is recorded in the
separate warmup runs. Torch traces additionally record total RGI and diffusion time.

`trace_rgi.py` is an opt-in test hook loaded only through this harness's
`PYTHONPATH`. It records actual source imports, specs, coordinate snapshots and
termination diagnostics. A startup source digest is also saved for uninstrumented
timing predictions; measurement rejects a digest that differs from the manifest.
The historical bridge aliases the old package name and
passes through fully expanded configurations. It also accepts the newer
OpenFold-3 caller's empty `smiles_by_chain` argument for protein-only baselines;
a nonempty mapping raises. It does not change the historical optimizer.
The AlphaFold3 hook uses test-only JAX callbacks to save three snapshots.
Production optimization has no host callback or SciPy dependency.

Keep `manifest.json`, `measurements.json`, `replay/*.json`, job logs and prediction
artifacts together. A successful scheduler exit alone does not satisfy these gates.
