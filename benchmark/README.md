# RGI benchmarks

Five benchmark protocols, six predictors, and three steps: prepare, run, evaluate.

| Protocol | System | Evaluation |
|---|---|---|
| `conformer` | PLINDER cis/trans 209 | Ligand bond/angle deviations, R/S and E/Z agreement, crystal ligand RMSD |
| `distance` | QBP | Domain centroid distance |
| `angle` | ADK | NMP–CORE–LID centroid angle |
| `rmsd` | QBP | RMSD to open and closed references on 220 common Cα atoms |
| `custom` | DgoT | Cytoplasmic minus periplasmic centroid distance |

## Quick start

Use Python 3.12+ and [uv](https://docs.astral.sh/uv/). Install the desired
[RGI-enabled predictor](https://github.com/cddlab/rgi_toolkit/tree/main/docs)
and its model weights separately. This environment installs the parent toolkit
checkout for preparation and evaluation; inference uses the predictor's existing
environment.
Run inference on a compatible GPU. On a cluster, run installation, preparation,
tests, and evaluation on compute nodes as well.

```bash
cd benchmark
uv sync --frozen

# Set this to your predictor's Python; relative paths start at runtime.yaml.
export BOLTZ_PYTHON=../predictors/boltz/.venv/bin/python

uv run python prepare.py conformer --models boltz2 --limit 1 --output runs/conformer-small
uv run python run.py runs/conformer-small --dry-run
uv run python run.py runs/conformer-small
uv run python evaluate.py runs/conformer-small
```

`prepare.py conformer` automatically downloads
[`wasarou/plinder-cistrans-209`](https://huggingface.co/datasets/wasarou/plinder-cistrans-209)
from the latest `main` when preparing each new run, preserving dataset order.
Downloaded files are cached locally. `--limit 1` fetches one system;
omit it for all 209. No private dataset, token, or original PLINDER installation
is required. Use `--dataset data/conformer` to choose the cache, or `--index`
for an already materialized CSV with `id,smiles,msa,sdf` columns and relative paths.

## Protein benchmarks

Supply the same A3M for all predictors. Its query must equal the sequence in the
protocol file. Downloaded or generated MSAs belong under the ignored `data/` directory.

```bash
uv run python prepare.py distance --msa data/qbp.a3m --models boltz2 af3 --seeds 0 1 2 --output runs/distance
uv run python prepare.py angle --msa data/adk.a3m --output runs/angle
uv run python prepare.py custom --msa data/dgot.a3m --output runs/custom
uv run python fetch.py references
uv run python prepare.py rmsd --msa data/qbp.a3m --output runs/rmsd

uv run python run.py runs/distance
uv run python evaluate.py runs/distance
```

For a quick input/integration check, replace `--msa ...` with `--single-sequence`.
This choice is recorded in the manifest and changes the experiment; it does not
reproduce an MSA-conditioned benchmark. `--limit N` selects the first N targets.
The default is one prediction seed and all five targets.

Every protocol includes OFF and RGI arms. Protein sweeps give OFF the same number
of samples as the full RGI sweep, using distinct baseline seeds. Conformer arms
use the same system and seed. Native guidance and MSA-subsampling controls,
TM-score, MolProbity, and PoseBusters analyses are outside this prototype.

## Predictor environments

`runtime.yaml` reads environment variables only for selected models. Alternatively,
copy it to `runtime.local.yaml`, fill in your interpreter paths, and pass
`--runtime runtime.local.yaml`. Paths are relative to that YAML file.
Each model accepts optional `args` and `env` entries for its native CLI. File paths
inside extra CLI arguments are interpreted from the individual job directory.

| Model key | Required variables |
|---|---|
| `boltz2` | `BOLTZ_PYTHON` |
| `protenix` | `PROTENIX_PYTHON`, `PROTENIX_ROOT` |
| `af3` | `AF3_PYTHON`, `AF3_ROOT`, `AF3_MODEL_DIR` |
| `chai` | `CHAI_PYTHON` |
| `of3` | `OPENFOLD3_PYTHON` (the installed Pixi environment's Python) |
| `esm` | `ESMFOLD2_PYTHON` (the installed Pixi environment's Python) |

All six use prepared MSAs; no model silently switches to an MSA server. AF3 skips
its data pipeline. Chai converts the A3M to `aligned.pqt`, verifies the query
sequence, and logs its row count. Both arms use `--msa-directory`.
Protenix selects `protenix-v2`; this integration's verified GPU
architecture is sm_89. ESMFold2 uses 20 loops, 200 diffusion steps, MSA depth 1024,
and column masking 0.1. Each job requests one sample. Model and toolkit versions
are saved alongside predictions.
Each sample starts a fresh model process. Recorded seconds include startup and
compilation and are not a controlled measurement of RGI overhead.

## Outputs and resuming

```text
runs/<name>/
  manifest.json          # Input paths and ordered job list
  protocol.yaml          # Frozen conditions
  data/                  # Shared MSAs, SDFs, and references
  jobs/<id>/             # Native input, log, raw output, prediction.cif, receipt
  results.csv            # Per-sample metrics and errors
  summary.csv            # Counts, means, and sample standard deviations
  evaluation.json        # Evaluation versions and sample counts
```

Run the same command to verify and skip completed jobs. Changed prepared inputs,
software, or runtime settings require a new run directory. After a failed job,
`--task N --retry` replaces only that job's outputs. `--task` is zero-based and is
applied after an optional `--model` filter. Jobs run sequentially by default.

For Slurm, submit the same command with a site-appropriate partition and GPU
request. A two-job quick start can use `--array=0-1%1` and
`uv run python run.py runs/conformer-small --task "$SLURM_ARRAY_TASK_ID"`
inside the job script. No cluster settings are embedded in the benchmark.

Completion requires finite coordinates, the full chain-A Cα set and ligand heavy
atoms, and nonzero requested restraint counts in the RGI setup log. Evaluation
rejects missing outputs and mismatched ligand atom names. A partially evaluated
run writes error rows and exits nonzero;
a run with no valid samples preserves existing CSVs. Blank stereochemistry values
mean no annotated sites, not success.
E/Z scores cover acyclic bonds whose stereochemistry can be determined from heavy atoms.

## Development

```bash
uv sync --frozen --group dev
uv run pytest -q
uv run ruff check .
```

Conformer references use RGI ETKDGv3/UFF preparation and stereo validation. All
generated inputs, predictions, downloads, and local runtime settings are ignored.
