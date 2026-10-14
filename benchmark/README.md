# RGI benchmarks

Five protocols, six RGI-enabled predictor checkouts, matched comparison arms, and
structural evaluation. Model identifiers are the full checkout names throughout
preparation, runtime settings, results, figures, and manuscript exports.

| Protocol | System | Measurements |
|---|---|---|
| `conformer` | PLINDER cis/trans 209 | Seven ligand geometry/stereo metrics, six clash metrics, pLDDT/pTM |
| `distance` | QBP | Domain centroid distance, two reference TM-scores, protein quality |
| `angle` | ADK | NMP–CORE–LID centroid angle, two reference TM-scores, protein quality |
| `rmsd` | QBP | RMSD on 220 common Cα atoms, two reference TM-scores, protein quality |
| `custom` | DgoT | Cytoplasmic minus periplasmic centroid distance, two reference TM-scores, protein quality |

## Installation

Use Python 3.12+ and [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync --frozen
```

This installs the parent RGI-toolkit checkout, RDKit, gemmi, PoseBusters, and
Matplotlib. Predictor environments and weights are separate; use the
`rgi-integration` branches of the `*_restr` checkouts, following the
[predictor integration guides](../docs/README.md). Plain upstream installations do not
provide the RGI hooks required by these benchmarks. ESMFold2 uses the
native model from `esm_restr`, as described in its [guide](../docs/esmfold2_restr.md).
On a cluster, run installation, preparation, tests, evaluation, and inference
on compute nodes; inference requires a compatible GPU.

[USalign](https://github.com/pylelab/USalign) needs a C++ compiler and `make`.
One local installation is:

```bash
mkdir -p tools
git clone https://github.com/pylelab/USalign.git tools/USalign
make -C tools/USalign
cp evaluation.yaml evaluation.local.yaml
# In evaluation.local.yaml, set: usalign: tools/USalign/USalign
./tools/USalign/USalign -h
```

Record the USalign revision used for a campaign. A package-manager alternative
is `conda install -c conda-forge -c bioconda usalign` in a separate tools environment;
its `USalign` executable must be on `PATH`.

For MolProbity, obtain Phenix from the [official download page](https://www.phenix-online.org/download).
The download requires the appropriate academic or industrial access. Follow the
[platform-specific installer instructions](https://phenix-online.org/version_docs/2.2-6161/install-setup-run.html);
for a current Linux/macOS command-line installer, `bash <downloaded-installer>.sh`
starts installation. This benchmark calls `phenix.molprobity`, which includes the
required validation programs. Set the installed environment script in the local
configuration, for example:

```yaml
usalign: tools/USalign/USalign
molprobity: phenix.molprobity
phenix_env: tools/phenix/phenix_env.sh
timeout: 300
```

Replace the example Phenix directory with the one selected during installation.
Paths in `evaluation.local.yaml` are relative to that file. Alternatively export
`PHENIX_ENV` to an existing environment script, or source that script so
`phenix.molprobity` is on `PATH`. Confirm it with `phenix.molprobity --help` in
the sourced shell. Phenix uses its own environment; do not install it into the
benchmark Python environment. Pass `--config evaluation.local.yaml` to evaluation.

## Quick start

With a sibling `boltz_restr` checkout and its environment already installed:

```bash
export BOLTZ_PYTHON=../../boltz_restr/.venv/bin/python
uv run python prepare.py conformer --models boltz_restr --limit 1 --output runs/conformer-small
uv run python run.py runs/conformer-small --dry-run
uv run python run.py runs/conformer-small
uv run python evaluate.py runs/conformer-small
```

`prepare.py conformer` downloads
[`wasarou/plinder-cistrans-209`](https://huggingface.co/datasets/wasarou/plinder-cistrans-209)
from `main` for each new run and caches files locally, preserving dataset order.
Pocket-fit references are fetched from the public PLINDER `2024-06/v2` release.
Only the requested receptor and its matching ligand are read from each remote ZIP
using HTTP range requests. Pass `--receptor-root` to use an offline PLINDER systems
directory instead. Receptor and ligand coordinates must share the crystal frame;
evaluation verifies this against the matching PLINDER ligand before fitting.
`--limit 1` fetches one system; omit it for all 209. No private dataset, token,
or original PLINDER installation is required. `--dataset` changes the cache;
`--index` accepts an existing CSV with `id,smiles,msa,sdf` and relative paths.
Optional `receptor,reference_ligand` columns provide local pose references;
otherwise references are fetched for the PLINDER system ID. Custom system IDs
require a `receptor` column in the same coordinate frame as `sdf`.
Prepared runs include these files and their SHA-256 checksums.

## Comparison arms and sampling

| Arm | RGI | Native guidance | MSA |
|---|---|---|---|
| `off` / `vanilla` | No | No | Supplied full MSA |
| `ms` | No | No | Subsampled homolog rows |
| `rgi` | Yes | No | Supplied full MSA |
| `g-only` | No | Yes | Supplied full MSA |
| `rgi+g` | Yes | Yes | Supplied full MSA |

Conformer defaults to `off g-only rgi`, with seed 0. Guidance arms run only on
`boltz_restr` (Boltz2 steering, `--use_potentials`) and `protenix_restr`
(Protenix-v2 TFG). Use `--arms off g-only rgi rgi+g` to include the combination.
The same system and prediction seed are used across conformer arms. Native
guidance cannot be overridden through runtime `args`.

Protein defaults are `vanilla ms rgi`. RGI uses nine seeds (0–8) for each of five
targets, giving **45 predictions per model**. Vanilla uses 45 distinct prediction
seeds (0–44). `ms` uses depths 16, 32, 64, 128, 256 × three MSA seeds × three
prediction seeds, also **45 predictions per model**. Depth counts homologs,
excluding the query; the query is always first. Sampling is without replacement
and capped by available homologs. The manifest jobs record requested depth,
actual retained row count, MSA seed, and prediction seed. The sampled files are
shared across models. Shallow inputs can yield identical subsets at large depths.

The `ms` arm uses one recycle for every model (`num_loops=1` for ESMFold2).
Other arms retain their existing model settings.

Override with `--seeds`, `--msa-depths`, `--msa-seeds`, and
`--msa-prediction-seeds`. `--seeds` changes RGI/full-MSA arms only.
`--limit N` restricts targets (or conformer systems); MSA subsampling has no target.

## Protein runs

Supply a common A3M whose query equals the protocol sequence. Keep generated
or downloaded MSAs under the ignored `data/` directory. Fetch evaluation references
before preparing any protein run:

```bash
uv run python fetch.py references
uv run python prepare.py distance --msa data/qbp.a3m --models boltz_restr alphafold3_restr --output runs/distance
uv run python prepare.py angle --msa data/adk.a3m --models openfold-3_restr --output runs/angle
uv run python prepare.py custom --msa data/dgot.a3m --output runs/custom
uv run python prepare.py rmsd --msa data/qbp.a3m --output runs/rmsd
uv run python run.py runs/distance
uv run python evaluate.py runs/distance --config evaluation.local.yaml
```

`--single-sequence` replaces `--msa` for input/integration checks and defaults to
`vanilla rgi`. It is recorded as a different experiment and cannot use `ms`.

## Predictor environments

`runtime.yaml` reads environment variables for selected models only. Alternatively,
copy it to ignored `runtime.local.yaml` and pass `--runtime runtime.local.yaml`.
Interpreter/script/model-directory paths are relative to that YAML file.
The examples below assume sibling checkouts beside RGI-toolkit.

| Model identifier | Interpreter example | Other variables |
|---|---|---|
| `boltz_restr` | `BOLTZ_PYTHON=../../boltz_restr/.venv/bin/python` | — |
| `protenix_restr` | `PROTENIX_PYTHON=../../protenix_restr/.venv/bin/python` | `PROTENIX_ROOT=../../protenix_restr` |
| `alphafold3_restr` | `ALPHAFOLD3_PYTHON=../../alphafold3_restr/.venv/bin/python` | `ALPHAFOLD3_ROOT=../../alphafold3_restr`, `ALPHAFOLD3_MODEL_DIR` |
| `chai-lab_restr` | `CHAI_PYTHON=../../chai-lab_restr/.venv/bin/python` | — |
| `openfold-3_restr` | `OPENFOLD3_PYTHON=../../openfold-3_restr/.pixi/envs/openfold3-cuda12/bin/python` | Use the installed Pixi environment |
| `esm_restr` | `ESMFOLD2_PYTHON=../../esm_restr/.pixi/envs/default/bin/python` | Use the native model in the installed Pixi environment |

Native CLI `args` and `env` are optional entries per model. Paths inside extra
CLI arguments are interpreted from the job directory. All models consume the
prepared MSA; none silently switches to an MSA server. Chai verifies the query
after A3M-to-parquet conversion. AlphaFold3 skips its data pipeline. Boltz2 and
Protenix-v2 are the selected variants. ESMFold2 uses 20 loops, 200 diffusion steps,
maximum MSA depth 1024, and column masking 0.1. Each job requests one sample.
A fresh process is used per sample; recorded time includes startup and compilation.

## Evaluation and reports

USalign compares chain A against both references, normalized by the reference
length (`Structure_1`): QBP open/closed (1GGG/1WDN), ADK open/closed (4AKE/1AKE),
and DgoT outward/inward (6E9O/6E9N). The reference definitions are frozen in each run.

Protein quality includes MolProbity score, clashscore, Ramachandran favored and
outliers, rotamer outliers, C-beta deviations, bond RMSD, and angle RMSD.
pLDDT is the atom-mean CIF B-factor (0–100). pTM is read from the exact matching
raw prediction's confidence file; a coordinate-file mismatch is an error.
ESMFold2 writes its scalar confidence sidecar during inference.

The six ligand clash columns are `clash_ok`, `clash_rel`, `clash_min`,
`iclash_ok`, `iclash_rel`, and `iclash_vdw_rel`. PoseBusters uses 0.75 times the
VdW radius sum for protein-ligand clashes and 0.70 times RDKit's distance-geometry
lower bound for internal clashes. The separate internal VdW ratio excludes
1–2 and 1–3 pairs, retaining 1–4 pairs. Pass booleans are true when no clash is
found. No eligible internal pairs or stereocenters gives a blank, not a pass.
Ligand crystal RMSD aligns the ligand itself; it is not receptor-aligned pose RMSD.

External failures, missing confidence, and parse errors are explicit per-group
errors. Other successful metrics are retained, and incomplete evaluation exits
nonzero. `--skip-usalign`, `--skip-molprobity`, `--skip-clashes`, and
`--skip-confidence` explicitly omit groups and record that choice in provenance.
MolProbity runs in an isolated directory with a timeout. Its excessive-bond
recovery attempt, when needed, is recorded in the result row.

```text
runs/<name>/
  manifest.json          # Ordered jobs and prepared input paths
  protocol.yaml          # Frozen conditions and reference states
  data/                  # Shared MSAs, ligand SDFs, references
  jobs/<id>/             # Inputs, raw output, prediction, receipt, evaluation logs
  results.csv            # Per-sample values, group statuses, and errors
  summary.csv            # Per-target N, missing counts, mean, median, SD, pass rates
  evaluation.json        # Versions, Git revision, counts, skipped groups
  report/
    report.md            # Comparison tables and figures
    comparison.csv       # Per-model/arm mean, median, N, pass rates
    figures/             # PNG and PDF comparisons, state maps, target-vs-achieved
```

Evaluation generates reports automatically; use `--no-report` to omit plotting.
Rebuild figures or export selected evaluated protocols for a manuscript:

```bash
uv run python report.py runs/distance
uv run python manuscript.py runs/distance runs/angle runs/conformer-small --output reports/manuscript
```

The manuscript bundle contains portable source CSVs, frozen protocols, provenance,
and available PNG/PDF figures. It retains full checkout names in filenames and
method columns; existing manuscript scripts using abbreviated identifiers need
their file/model mapping updated. Native guidance and RGI+guidance stay separate.
A bundle records exactly the protocols/models exported; it does not silently
supply additional paper experiments.

## Resuming and cluster jobs

Repeat `run.py` to verify and skip completed jobs. Changes to conditions,
software, or runtime settings require a new run directory. `--task N --retry`
replaces only that task's outputs. Tasks are zero-based after optional `--model`
filtering; without `--task`, jobs execute sequentially. For a Slurm array, pass
`--task "$SLURM_ARRAY_TASK_ID"` and choose the array range from `manifest.json`.
A one-system Boltz2 conformer run has three jobs with the default arms.
No cluster partitions or site paths are embedded here.

Completion requires finite coordinates, the full chain-A Cα sequence, ligand
heavy atoms, and nonzero requested RGI restraint counts. If no prediction can be
evaluated, existing CSVs are preserved. Success/pass rates always state their
measured denominator; missing values do not become successes.

## Development

```bash
uv sync --frozen --group dev
uv run pytest -q
uv run ruff check .
```

Conformer references use RGI ETKDGv3/UFF preparation and stereo validation.
E/Z preservation compares the input graph's CIP labels with labels assigned by
RDKit from the predicted 3D conformer. `evaluation.json` records the method.
Ligands without eligible E/Z sites remain unscored.

`ligand_rmsd_A` is the ligand pose RMSD used in Figure S1F: fit sequence-paired
receptor C-alpha atoms within 10 Å of any crystal ligand heavy atom, apply that
transform to the predicted ligand, and minimize over symmetry-equivalent atom
maps without fitting the ligand again. The sequence alignment uses global mode,
match/mismatch scores 1/0, gap opening/extension scores -10/-0.5, and free end gaps.
At least three non-collinear paired pocket atoms are required. Missing pose metadata
or an unmeasurable fit leave pose RMSD unavailable and preserve the other metrics.
`ligand_fit_rmsd_A` retains the separate ligand-aligned geometry diagnostic.
The export column `ligand_pose_rmsd_A` contains the receptor-aligned value.

Existing runs prepared before pose references were added must be prepared again
in a new directory to include the reference files. Existing predictions can be
reused; updating the evaluation does not require new inference.
Inputs, downloads, predictions, reports, local settings, and external tools are ignored.
