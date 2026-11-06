# MO-OBAM implementation

This repository implements Wei et al., *Multi-Objective Optimization-Based
Anonymization of Structured Data for Machine Learning*, using the current
[arXiv v2 (15 May 2025)](https://arxiv.org/abs/2501.01002). Version 2 changes
the formulation and adds PSO pseudocode, so results produced with the January
2025 v1 objective are not interchangeable with this implementation.

The consolidated data, methodology, implementation, result, and limitation
record for the PRTI-GH adaptation is
[`PRTI_MO_OBAM_DOCUMENTATION.md`](PRTI_MO_OBAM_DOCUMENTATION.md).
The repository documentation and recorded evidence were last cross-checked on
5 November 2026.

## Reproducible environment

The verified environment is Python 3.12 with the exact package versions in
`requirements.txt`. From this directory, create or refresh the project-local
virtual environment with:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The terminal wrappers use `.venv/bin/python` by default. To use an activated
environment or a server-managed interpreter instead, set `PRTI_PYTHON` to its
executable path before running a wrapper.

### Shared server: `python3.12` is not available

The pinned NumPy and SciPy versions require Python 3.12. If a shared server
provides only Python 3.10 or 3.11, do not replace its system Python and do not
run `sudo` or `apt` commands for this project. The following fallback installs
`uv`, a Python 3.12 runtime, the package cache, and the virtual environment
inside the current repository. It does not modify `/usr/bin/python3`, shell
profiles, or installations used by other server users.

From the repository root, create the project-local directories:

```bash
mkdir -p .local-tools .local-python .local-python-bin .uv-cache
```

Install an unmanaged, project-local copy of `uv` without changing `PATH` or a
shell profile:

```bash
curl -LsSf https://astral.sh/uv/install.sh |
  env UV_UNMANAGED_INSTALL="$PWD/.local-tools" sh
```

Use it to download Python 3.12 into the repository and create `.venv`:

```bash
UV_PYTHON_INSTALL_DIR="$PWD/.local-python" \
UV_PYTHON_BIN_DIR="$PWD/.local-python-bin" \
UV_CACHE_DIR="$PWD/.uv-cache" \
./.local-tools/uv venv --python 3.12 .venv
```

Install the pinned packages into that environment:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" \
./.local-tools/uv pip install \
  --python .venv/bin/python \
  -r requirements.txt
```

Verify both the isolated environment and the unchanged system interpreter:

```bash
.venv/bin/python --version
/usr/bin/python3 --version
readlink -f .venv/bin/python
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -q
```

The first command must report Python 3.12.x. The system command may continue
to report Python 3.10.x; that is expected. `uv` is only an installation tool
for this fallback and is not used by the anonymization algorithm. After setup,
run the project through `.venv/bin/python` and the supplied shell wrappers.
The relevant upstream references are the
[`uv` installer options](https://docs.astral.sh/uv/reference/installer/) and
[`uv` storage controls](https://docs.astral.sh/uv/reference/storage/).

`xlrd` is required because PRTI-GH is an Excel 97-2003 `.xls` workbook. SciPy
provides the compiled pairwise-distance kernels used by the PSO implementation.
`openpyxl` writes the single multi-objective Excel report; it is not an
optimizer dependency.
Every new PRTI audit and anonymization summary records the Python, NumPy,
pandas, SciPy, xlrd, and openpyxl versions used. It also records the exact paper
identifier, revision date, and URL. The January 2025 PDF previously downloaded
beside the other project is arXiv v1; the implementation authority is
[`arXiv:2501.01002v2`](https://arxiv.org/abs/2501.01002v2), revised 15 May 2025.

The active implementation is the `mo_obam/` package and the documented
`run_*.py`, `generate_*.py`, and cluster-study shell entry points. The root
`MO-OBAM.ipynb`, `constants.py`, `utils.py`, `model_train.py`, and
`particle_swarm.py` files are the retained authors' reference bundle. They are
not imported by the active PRTI-GH implementation and must not be mistaken for
the server workflow.

## Implemented paper formulation

`mo_obam.MOOBAMFormulation` implements:

- one-cluster-per-record membership (Equation 9);
- numerical squared deviation, categorical mismatch, centroid distance, and
  normalized information loss (Equations 10-14);
- cluster entropy, the minimum cluster entropy for each sensitive attribute,
  and their sum (Equations 15-19);
- hard k-anonymity (Equation 20);
- the active v2 objective `IL - lambda * sum(minimum SA entropy)`; and
- QID replacement by cluster centroids while non-QID columns remain unchanged.

`mo_obam.ExactMOOBAMSolver` enumerates every feasible unlabeled partition for
small validation datasets. It derives numerical centroids as cluster means and
categorical centroids as modes and returns the global optimum of the active v2
objective over that enumerated feasible set.

## Scalable search and its boundary

`mo_obam.MOOBAMParticleSwarmOptimizer` follows v2 Algorithm 1: centroids are
particles, initial centroid entries are sampled from observed QI values,
records are assigned to their nearest centroid using Equation 12, and one PSO
run minimizes the active objective. The continuous/categorical position update
mechanics, population 100, and 100 iterations are taken from the authors'
public repository and notebook. PSO is stochastic and returns a best-found
solution, not a proof of a global optimum.

For speed, duplicate QID tuples are evaluated once with their record counts as
weights. Identical QID tuples have identical distances to every centroid, so
this changes neither assignments nor any information-loss, entropy, or
k-anonymity equation. A full seeded Adult run is regression-checked against the
uncompressed computation.

Sensitive-value counts are stored sparsely by QID profile and accumulated for
all SAs in one vectorized calculation. The result is regression-tested against
the direct Equations 15-19 implementation; it changes runtime, not entropy or
the objective.

Two source inconsistencies are deliberately not guessed around:

- Equation 20 is enforced as a hard constraint. Algorithm 1 prints a negative
  k-violation penalty, which would reward violations during minimization; the
  authors' code instead adds a positive normalized penalty but does not publish
  the Adult penalty multiplier needed for exact reproduction. With large
  cluster counts, random initialization can therefore contain no hard-k
  feasible particle. The experiment runner records such a configuration as
  `blocked`; it does not invent a penalty or centroid-repair rule.
- Section II-B prints linkage risk as `1 / class_size >= tau`, but says
  "exceeds" and the Adult Table 6 baseline values are reproduced only by the
  strict `>` operator. Reports therefore use `>` and state that boundary.

The paper also says objective normalization "may be required" without defining
a formula. The baseline implementation therefore uses the active mathematical
objective literally and does not introduce an undocumented normalization.
Algorithm 1's observed-QI initialization is used even though the public code
samples numeric integers from within column bounds.
See [`PAPER_REVIEW_V2.md`](PAPER_REVIEW_V2.md) for the equation-by-equation
review and the measured effect of the linkage-threshold ambiguity.

## Adult experiment

The paper's Adult configuration used here is:

- numerical QID: `age`;
- categorical QIDs: `race`, `sex`, and `marital_status`;
- sensitive attribute: `occupation`;
- `k = 5`, `C = 4`, and `lambda = 1`.

`AdultDataPreparer` combines the supplied UCI training and test files and
removes records containing `?`. It validates the paper's 30,162 retained
training rows, 15,060 retained test rows, and 45,222 total rows. It performs no
imputation, category recoding, or duplicate removal. The test-file target
suffix (`.`) is removed so train and test labels use the same original
categories.

Run the complete experiment from this directory:

```bash
.venv/bin/python run_adult.py
```

The command saves these files under `outputs/`:

- `adult_prepared_original.csv`: the 45,222 prepared records before
  anonymization;
- `adult_before_after_anonymization.csv`: aligned records with each QID shown
  as adjacent `_before` and `_after` columns, unchanged attributes shown once,
  and the assigned cluster;
- `adult_anonymized_final.csv`: the complete 45,222-row, 15-column release
  dataset in original column order, with only the four QIDs replaced;
- `adult_model_evaluation.csv`: observed information loss, entropy, and the
  linkage/homogeneity attack counts used in the paper;
- `adult_paper_comparison.csv`: observed values beside the expected Adult
  values from arXiv v2 Table 6(b), including the unmodified baseline checks;
- `adult_cluster_details.csv`: cluster sizes, QID centroids, and SA diversity;
- `adult_run_summary.json`: paper revision, configuration, final metrics,
  attack evaluation, PSO evidence, data-preparation counts, and validation
  checks.

The result tables are printed automatically after a run. Display them again
without rerunning PSO:

```bash
.venv/bin/python run_adult.py --show-tables
```

For a completed **v2** run that already has the prepared, before/after, and JSON
files, recreate the final release and tables without rerunning PSO:

```bash
.venv/bin/python run_adult.py --report-existing
```

The command refuses v1 summaries so obsolete results cannot be silently
relabeled as v2.

The paper comparison is diagnostic only. Published values are never supplied
to the optimizer or used to alter the observed result.

Raw files in `data/adult/` are read-only inputs and are never overwritten.

## PRTI-GH experiment

### Quick start: choose what you want to produce

Do not run every command in this README. Choose one of these three tasks:

| What you need | Command to use | Parameters you control | Patient-level anonymized CSV? |
|---|---|---|---|
| Anonymize once with settings you already chose | `run_prti.py` | `k`, clusters (`C`), lambda, seed, population, iterations | Yes |
| Compare candidate values of `k`, `C`, and lambda before selecting one | `run_prti_experiments.py parameter-grid` | Lists of `k`, `C`, and lambda values plus comparison/held-out seeds | Comparison is aggregate-only; the selected held-out command produces the final CSV |
| Run the Pareto multi-objective workflow used by V3 | `run_prti_multiobjective.py`, then `run_prti_multiobjective_selected_candidate.py` | `k` and `C`; lambda is analysed continuously after optimization | Only the final held-out command produces the CSV |

#### Produce one anonymized dataset with your own k, C, and lambda

Use this when the parameter values are already decided and you only need the
anonymized records. Change the values and output label in this command:

```bash
.venv/bin/python run_prti.py \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory outputs/prti/custom-k10-c5-lambda1-seed12 \
  --run-label custom-k10-c5-lambda1-seed12 \
  --k 10 \
  --clusters 5 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seed 12
```

The anonymized dataset will be:

```text
outputs/prti/custom-k10-c5-lambda1-seed12/prti_anonymized_final.csv
```

Always use a new `--output-directory` and matching `--run-label` when changing
`k`, `C`, lambda, or seed. This one-run command does not compare settings or
prove that the chosen configuration is best. To compare candidate values of
all three parameters, use Section 6. To reproduce the completed V3 Pareto
workflow and its reserved seed 12, use Section 7.3; do not replace its selected
configuration manually.

In scalar mode, `--lambda-weight` is a required scientific choice in the
objective `IL - lambda * entropy_sum`. In Pareto multi-objective mode, there is
no `--lambda-weight` input: the optimizer retains the IL/entropy frontier and
calculates lambda crossover intervals over `[0, 1]` afterward.

### 1. Input and attribute roles

By default, `run_prti.py` reads the unchanged source workbook from:

```text
data/PRTI-GH-study/PRTI-GH-study.xls
```

The `data/` directory is intentionally ignored by Git. Place the reviewed
workbook at that path after cloning; it is not published with the repository.
The adapter validates the reviewed 1,485-row by 106-column shape and the
workbook SHA-256 fingerprint before running. A different location for the same
reviewed workbook can be supplied with `--source`.

The first documented baseline uses 29 QIDs: two numerical QIDs (`childsAge`
and `year`) and 27 categorical QIDs. `religiousAffiliation` is a categorical
QID. The other 77 attributes are SAs; this means `temperatureDegreeCelsius`
and `oxygenSaturation` remain SAs in this baseline. These are project threat-
model decisions, not roles assigned by the MO-OBAM paper.

Preparation is deliberately lossless:

- numerical values are unchanged;
- every exact categorical value receives a deterministic one-to-one identifier;
- the identifiers have no numerical or ordinal meaning in the MO-OBAM distance;
- categorical QIDs use mismatch distance, so one-hot encoding is not applied;
- spelling, case, whitespace, and category labels are not cleaned or merged;
- missing values become an explicit model state and are never imputed;
- known duplicate source columns remain separate attributes and are never
  automatically merged or removed; and
- decoding must reproduce every original distribution with zero frequency
  difference before optimization is allowed to run.

### 2. Choose the workflow before running commands

The commands below are not one continuous sequence. Choose the workflow that
matches the research task:

| Workflow | Use it when | Recommended order |
|---|---|---|
| Fresh installation preflight | The repository or dataset is new on a machine | Section 3 audit, then smoke test |
| Main fixed-parameter cluster study | `k` and lambda are already fixed and only `C` must be selected | Sections 3, 5, then 8 |
| Main joint-parameter study | `k`, `C`, and lambda must all be compared | Sections 3, 6, then 8 |
| Faster multi-objective study | Search `k` and `C` jointly while analysing the full lambda domain without separate lambda runs | Sections 3 and 7 |
| Historical reproducibility | A recorded baseline or earlier experiment must be reproduced | Section 3, then the required command in Section 4 |

Do not run the historical seed-9 baseline merely because the audit succeeded.
It is optional reproduction evidence and does not select the best
configuration.

### 3. Fresh installation preflight

Run this section after installing the project on a new machine, after
installing a new copy of the workbook, or after changing data-preparation code.
The audit is required. The smoke test is strongly recommended once per fresh
installation before an expensive main study. Neither is a thesis experiment.

#### 3.1 Audit the source and preparation

Prepare and inspect the data without anonymizing it:

```bash
.venv/bin/python run_prti.py --audit-only
```

The audit writes private preparation artifacts under
`outputs/prti/audit-only/`. In particular, `prti_prepared_input.csv` is coded
patient-level input, not anonymized data. Keep the directory private and do not
commit it.

#### 3.2 Recommended small end-to-end smoke test

After the audit passes, verify optimization, entropy, privacy metrics,
checkpoint writing, and aggregate reporting with a deliberately small run:

```bash
./run_prti_cluster_range_study.sh 1 3 \
  --k 10 \
  --lambda-weight 1 \
  --population 10 \
  --iterations 5 \
  --seeds 0,1 \
  --held-out-seed 11 \
  --workers 1 \
  --output-directory evidence/prti-smoke-fresh-v1
```

Smoke-test results are installation evidence only. Never merge their
checkpoints or metrics into a main study directory or thesis result.

### 4. Historical reproducibility and diagnostic workflows

Everything in this section is optional unless the research task is explicitly
to reproduce a recorded result. These commands are not prerequisites for the
fresh main workflows in Sections 5 and 6.

#### 4.1 Reproduce the recorded seed-9 baseline

Run the recorded PRTI result explicitly:

```bash
.venv/bin/python run_prti.py \
  --output-directory outputs/prti/baseline-v1-seed9 \
  --run-label baseline-v1-seed9 \
  --k 10 \
  --clusters 30 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seed 9
```

This executes only `k=10`, `C=30`, `lambda=1`, and seed 9. It produces
patient-level anonymization artifacts for internal validation, but it does not
compare configurations or establish that this configuration is best.

#### 4.2 Reproduce the exploratory seed screen

Recreate and retain the exploratory 20-iteration seed screen as aggregate-only
evidence:

```bash
.venv/bin/python run_prti_experiments.py seed-screen \
  --output-directory evidence/prti-baseline-v1 \
  --k 10 \
  --clusters 30 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 20 \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

#### 4.3 Compare the labelled and legacy baseline outputs

Compare a fresh labelled result with the legacy result without exporting any
patient rows:

```bash
.venv/bin/python run_prti_experiments.py compare \
  --legacy-directory outputs/prti/archive/legacy-seed9 \
  --fresh-directory outputs/prti/baseline-v1-seed9 \
  --evidence-directory evidence/prti-baseline-v1
```

#### 4.4 Reproduce the paper-derived sensitivity plan

Write the complete 220-point sensitivity plan without starting optimization:

```bash
.venv/bin/python run_prti_experiments.py sensitivity --dry-run
```

Execute the plan in resumable batches while retaining aggregate results only:

```bash
.venv/bin/python run_prti_experiments.py sensitivity --max-new-runs 1
```

The sensitivity grid uses exactly Wei et al.'s Appendix A values: four `k`
values, eleven cluster counts, and five lambda values. Applying that grid to
PRTI-GH is a labelled project adaptation, not a paper-reported PRTI-GH
experiment. Settings for which `k * clusters > 1,485` are recorded as
mathematically infeasible without running PSO or inventing result values. The
sensitivity workflow never writes assignments, centroids, or patient rows.

#### 4.5 Reproduce the four-configuration robustness study

Recreate or resume the predeclared 40-run robustness study for the baseline,
privacy, lowest-IL, and compromise configurations:

```bash
.venv/bin/python run_prti_experiments.py robustness \
  --output-directory evidence/prti-robustness-v1 \
  --population 100 \
  --iterations 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --workers 4
```

The ten-seed design is a labelled project robustness protocol, not a parameter
specified by Wei et al. It retains one aggregate checkpoint per seed and no
patient-level rows. Matching checkpoints are reused when the command resumes.

#### 4.6 Diagnose one chosen cluster count

Run or resume one chosen cluster count with explicit comparison seeds:

```bash
./run_prti_cluster_study.sh 8 \
  --k 10 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --held-out-seed 11
```

Pass all model and optimizer settings explicitly when needed:

```bash
./run_prti_cluster_study.sh 12 \
  --k 15 \
  --lambda-weight 0.1 \
  --population 150 \
  --iterations 200 \
  --seeds 9 \
  --held-out-seed 11 \
  --workers 4
```

The model, optimizer and comparison seeds are explicit in the terminal
history. Its optional controls are `--workers N`, `--output-directory PATH`,
and `--dry-run`. The range wrapper additionally supports `--max-new-runs N`,
`--reuse-directory PATH`, and `--no-reuse`. A declared held-out seed must not
appear in the comparison seeds.
The cluster count must not exceed `floor(1485/k)`. Each default
output directory includes the cluster count, k, lambda, population, and
iteration and seed values. This prevents
different configurations from overwriting or resuming one another. The
wrapper is aggregate-only: it never writes patient rows, assignments,
centroids, or QID representatives.

### 5. Main workflow A: select C with fixed k and lambda

Use this workflow only when the study has already fixed `k=10` and `lambda=1`
and needs to select the best observed eligible cluster count. It cannot select
the best joint `k`, `C`, and lambda combination.

For the primary inclusive comparison, use the range wrapper. This example
tests every mathematically possible cluster count for `k=10`, reserves seed 11,
and writes no patient rows:

```bash
./run_prti_cluster_range_study.sh 1 148 \
  --k 10 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --held-out-seed 11 \
  --workers 4 \
  --output-directory evidence/prti-primary-range-k10-fresh-v1 \
  --no-reuse \
  --dry-run
```

After inspecting the plan, execute an initial resumable batch with the same
scientific configuration:

```bash
./run_prti_cluster_range_study.sh 1 148 \
  --k 10 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --held-out-seed 11 \
  --workers 4 \
  --output-directory evidence/prti-primary-range-k10-fresh-v1 \
  --no-reuse \
  --max-new-runs 20
```

Repeat that exact command to resume. After the server proves stable,
`--max-new-runs 20` may be increased or omitted. Selection is privacy-first:
only configurations passing all seeds and thresholds are ranked by mean
information loss. The self-contained
fresh-start instructions, including installation, a smoke test, guaranteed
no-reuse execution, monitoring, held-out generation, outputs, and remaining
reporting work, are in
[`PRTI_CLUSTER_RANGE_STUDY.md`](PRTI_CLUSTER_RANGE_STUDY.md).

### 6. Main workflow B: jointly select k, C, and lambda

Use this workflow when the research question requires comparison of all three
model parameters. The fixed range in Section 5 is insufficient for this goal.
First inspect the self-contained help:

```bash
.venv/bin/python run_prti_experiments.py parameter-grid --help
```

Write the proposed server plan without starting optimization:

```bash
.venv/bin/python run_prti_experiments.py parameter-grid \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-parameter-grid-k10-15-20-25-30-v3 \
  --minimum-clusters 1 \
  --maximum-clusters 148 \
  --k-values 10 15 20 25 30 \
  --lambda-weights 0.0001 0.001 0.01 0.1 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --held-out-seed 11 \
  --workers 32 \
  --worker-backend process \
  --no-reuse \
  --dry-run
```

Inspect `prti_parameter_grid_plan.json`, including the source, parameter sets,
seed separation, feasibility bounds, and `patient_level_outputs: false`. After
approving that protocol, limit each worker process to one internal numerical-
library thread before executing a real server batch:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

The study already parallelizes independent runs with `--workers` and
`--worker-backend process`. NumPy, SciPy, OpenBLAS, MKL, or NumExpr may
otherwise create additional threads inside every worker process. For example,
32 process workers that each create several library threads can oversubscribe
the server's CPUs, increase memory use and context switching, and make the
study slower or unstable. Setting these variables to `1` keeps parallelism at
the runner level; it does not change k, C, lambda, the PSO population or
iterations, seeds, numerical formulas, privacy rules, or selection results.
Set them once in every new shell before starting or resuming a real study.

Then execute the first server batch:

```bash
.venv/bin/python run_prti_experiments.py parameter-grid \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-parameter-grid-k10-15-20-25-30-v3 \
  --minimum-clusters 1 \
  --maximum-clusters 148 \
  --k-values 10 15 20 25 30 \
  --lambda-weights 0.0001 0.001 0.01 0.1 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --held-out-seed 11 \
  --workers 32 \
  --worker-backend process \
  --no-reuse \
  --max-new-runs 64
```

Repeat that exact command to resume. Worker count and batch size may be
adjusted after a stable batch without changing the declared scientific grid.
The selection CSV and JSON retain mean, standard deviation, minimum, and
maximum entropy sums and identify the privacy-eligible information-loss/entropy
Pareto frontier. Automatic selection remains privacy-first and then chooses the
lowest mean information loss; Pareto status is an additional analysis, not a
replacement rule.

That mode bounds C separately for each k, writes aggregate-only resumable
checkpoints, requires an unseen held-out seed and an explicit reuse policy,
prevents concurrent writers to one study directory, and withholds selection
until every declared configuration/seed outcome is recorded. The expanded
full-grid command and its 21,450-run count are documented in
[`PRTI_CLUSTER_RANGE_STUDY.md`](PRTI_CLUSTER_RANGE_STUDY.md#16-joint-k-cluster-count-and-lambda-study-on-a-server).
On a Linux server, `--worker-backend process` runs independent configurations
in forked processes; `thread` remains the portable fallback. The backend
changes execution only, not the optimizer budget or selection protocol.
SciPy's compiled distance kernel and compiled entropy counting retain every
declared run while removing the dominant temporary-array and Python dispatch
overhead.

### 7. Faster multi-objective extension across C=1 through C=30

The original scalar PSO and its recorded `v3` results remain the baseline. The
additive multi-objective extension searches information loss and sensitive-
attribute entropy jointly and retains a bounded Pareto archive. It does not run
PSO separately for every lambda. Instead, after each run it calculates the
scalar objective `IL - lambda * entropy_sum` and its crossover intervals over
the complete lambda domain `[0,1]` represented by the discovered archive.

The complete declared study contains:

```text
30 cluster counts (C=1 through C=30)
5 k values (10, 15, 20, 25, 30)
10 comparison seeds (0 through 9)
= 1,500 optimizer runs
```

Every declared cluster count from 1 through 30 is included. The command never
runs the held-out seed during configuration selection.

First write and inspect the immutable plan. This performs no optimization and
does not read patient rows:

```bash
.venv/bin/python run_prti_multiobjective.py \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-multiobjective-v1 \
  --minimum-clusters 1 \
  --maximum-clusters 30 \
  --k-values 10 15 20 25 30 \
  --population 100 \
  --iterations 100 \
  --archive-size 100 \
  --initialization-attempts 20 \
  --random-coefficient-max 0.01 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --held-out-seed 12 \
  --maximum-information-loss 0.60 \
  --maximum-homogeneity-fraction 0.20 \
  --maximum-zero-entropy-sensitive-attributes 1 \
  --workers 32 \
  --dry-run
```

After checking the plan, run or resume all missing comparison runs by using the
same command without `--dry-run`. The SQLite checkpoint database makes an
interrupted invocation resumable. `--max-new-runs N` may be used to stop cleanly
after a declared batch; a partial study writes diagnostics but cannot select a
configuration.

The study directory contains only four persistent artifacts:

```text
prti_multiobjective_plan.json
prti_multiobjective_selection.json
prti_multiobjective_report.xlsx
prti_multiobjective_checkpoints.sqlite3
```

The Excel workbook contains the readable summary, configuration comparison,
and lambda intervals. SQLite is internal resumable evidence. The study is
aggregate-only and produces no anonymized patient rows. To regenerate the JSON
selection and Excel report from SQLite without optimization, repeat the command
with `--report-only`.

The project-specific homogeneity and IL thresholds are explicit required
arguments so they cannot be silently hidden in a new study. The resulting
selection is a best-observed stochastic Pareto result, not proof of a global
mathematical optimum and not approval for unrestricted data release.

The PSO random-coefficient bound is also a required argument. Use `0.01` to
match the authors' implementation. It is written into the immutable plan and
SQLite metadata, so changing it requires a new versioned output directory.

#### 7.1 Correct and regenerate an existing report

Earlier versions did not recompute `objective_mean_at_midpoint` after adjacent
lambda intervals with the same candidate signature were merged. The interval
boundaries, candidate signatures, IL, entropy, k/C selection, and SQLite
checkpoints were unaffected. After pulling the corrected code, regenerate the
v1 JSON and Excel report without rerunning PSO by repeating the exact v1
scientific configuration with `--report-only`:

```bash
.venv/bin/python run_prti_multiobjective.py \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-multiobjective-v1 \
  --minimum-clusters 1 \
  --maximum-clusters 30 \
  --k-values 10 15 20 25 30 \
  --population 100 \
  --iterations 100 \
  --archive-size 100 \
  --initialization-attempts 20 \
  --random-coefficient-max 0.01 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --held-out-seed 12 \
  --maximum-information-loss 0.60 \
  --maximum-homogeneity-fraction 0.20 \
  --maximum-zero-entropy-sensitive-attributes 1 \
  --workers 32 \
  --report-only
```

#### 7.2 Exploratory refinement before any held-out run

The completed v1 comparison selected `k=20`, `C=5` under its declared
privacy-first, lowest-mean-IL rule. No held-out run is authorized while
refinement continues. A separate exploratory v2 study expands k to every
integer from 5 through 30 and concentrates C on 2 through 10. The v1 evidence
showed no privacy-eligible configuration at C=1 and none beyond C=7; including
C=8 through C=10 retains a short boundary check without repeating the
unproductive C=11 through C=30 region.

This follow-up contains 234 k/C configurations and 2,340 comparison runs. It
uses the same population, iterations, archive size, initialization attempts,
random-coefficient bound, seeds, and privacy rules as v1. Use a new output
directory; never overwrite or relabel v1. First run this command with
`--dry-run`, then remove only `--dry-run` after inspecting the plan:

```bash
.venv/bin/python run_prti_multiobjective.py \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-multiobjective-refinement-k5-30-c2-10-v2 \
  --minimum-clusters 2 \
  --maximum-clusters 10 \
  --k-values 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
  --population 100 \
  --iterations 100 \
  --archive-size 100 \
  --initialization-attempts 20 \
  --random-coefficient-max 0.01 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --held-out-seed 12 \
  --maximum-information-loss 0.60 \
  --maximum-homogeneity-fraction 0.20 \
  --maximum-zero-entropy-sensitive-attributes 1 \
  --workers 32 \
  --dry-run
```

This is an adaptive exploratory refinement informed by v1, not a
paper-reported experiment or an independent confirmation. Seed 12 remains
unseen. Selection remains withheld until all 2,340 outcomes are recorded.

#### 7.3 Fresh-seed validation of the v2 frontier

After v2 is complete, validate only its three declared frontier configurations
with fresh comparison seeds. The `0.01` absolute mean-IL margin is applied
before minimizing maximum homogeneity; seed 12 remains held out. Write and
inspect the immutable 30-run plan first:

```bash
.venv/bin/python run_prti_multiobjective.py \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-multiobjective-confirmation-v3 \
  --configuration-pairs 20:5 8:5 12:3 \
  --population 100 \
  --iterations 100 \
  --archive-size 100 \
  --initialization-attempts 20 \
  --random-coefficient-max 0.01 \
  --seeds 13 14 15 16 17 18 19 20 21 22 \
  --held-out-seed 12 \
  --maximum-information-loss 0.60 \
  --maximum-homogeneity-fraction 0.20 \
  --maximum-zero-entropy-sensitive-attributes 1 \
  --selection-information-loss-margin 0.01 \
  --workers 32 \
  --dry-run
```

Remove only `--dry-run` after inspecting the plan. This validation remains
aggregate-only and does not execute seed 12.

After V3 reports `selection_complete`, predeclare the one held-out run. This
reads the selected k/C pair and seed 12 from the saved V3 evidence; it accepts
no manual k, C, lambda, or seed override:

```bash
.venv/bin/python run_prti_multiobjective_selected_candidate.py \
  --study-directory evidence/prti-multiobjective-confirmation-v3 \
  --output-directory outputs/prti/multiobjective-v3-heldout-seed12 \
  --evidence-directory evidence/prti-multiobjective-final-selection-v3 \
  --plan-only
```

Inspect the plan, then generate the anonymized data by running the same command
without `--plan-only`:

```bash
.venv/bin/python run_prti_multiobjective_selected_candidate.py \
  --study-directory evidence/prti-multiobjective-confirmation-v3 \
  --output-directory outputs/prti/multiobjective-v3-heldout-seed12 \
  --evidence-directory evidence/prti-multiobjective-final-selection-v3
```

The anonymized data is written to
`outputs/prti/multiobjective-v3-heldout-seed12/prti_anonymized_final.csv`.
The output directory contains only that CSV and its run summary; the immutable
plan and acceptance decision remain in the evidence directory.

### 8. Generate one held-out candidate after selection

The commands below apply to the existing scalar `dense-clusters` and
`parameter-grid` workflows. Multi-objective V3 uses the separate held-out
command in Section 7.3 because it must select from a Pareto archive rather than
run the scalar optimizer.

Do not manually choose a configuration after looking at the comparison
results. Wait until the selected study reports completion, then let the
candidate runner read `k`, `C`, lambda, population, iterations, and held-out
seed from the saved selection evidence.

After the range is complete, generate the recorded selected configuration with
the prospectively reserved held-out seed:

```bash
.venv/bin/python run_prti_selected_candidate.py \
  --study-directory evidence/prti-primary-range-k10-fresh-v1 \
  --output-directory outputs/prti/range-selected-heldout-seed11-fresh-v1 \
  --evidence-directory evidence/prti-range-final-selection-fresh-v1 \
  --plan-only
```

Remove `--plan-only` only after checking the final plan. The final command does
not accept k, C, lambda, population, iterations, or seed overrides; it derives
them from the completed selection evidence and refuses seed substitution.

For a completed joint workflow, use:

```bash
.venv/bin/python run_prti_selected_candidate.py \
  --study-directory evidence/prti-parameter-grid-k10-15-20-25-30-v3 \
  --output-directory outputs/prti/parameter-grid-selected-heldout-seed11-v3 \
  --evidence-directory evidence/prti-parameter-grid-final-selection-v3 \
  --plan-only
```

For either workflow, inspect the generated plan and then repeat its exact
command without `--plan-only`. The same candidate runner supports both
completed study types.

### 9. Maintain previously recorded results

The commands in this section operate on existing recorded candidates and
reports. They are not steps in a new main study.

Generate the prospectively declared four-cluster held-out candidate. The
default command records the plan before running seed 10 and refuses to
overwrite an existing candidate:

```bash
.venv/bin/python run_prti_release.py
```

Recheck the saved candidate against the recorded project criteria without
rerunning PSO:

```bash
.venv/bin/python run_prti_release.py --evaluate-only
```

Regenerate the write-up-ready aggregate tables from the recorded baseline,
sensitivity, and robustness evidence:

```bash
.venv/bin/python generate_prti_results.py
```

Generate the consolidated aggregate results package, including the held-out
candidate, eight CSV tables, five PNG/SVG figure pairs, and a provenance
manifest:

```bash
.venv/bin/python generate_prti_results_package.py
```

This command does not rerun anonymization. It requires the recorded ignored
patient-level outputs only as local inputs for aggregate calculations; no
patient rows or centroids are copied into the evidence package. GNUPlot is
required for the figures.

The interpreted findings and claim boundaries are in
[`PRTI_MO_OBAM_RESULTS.md`](PRTI_MO_OBAM_RESULTS.md). Machine-readable tables
from the first aggregate report are retained under
[`evidence/prti-results-v1/`](evidence/prti-results-v1/). The current
consolidated package is indexed at
[`evidence/prti-results-v2/README.md`](evidence/prti-results-v2/README.md).

Generate the aggregate-only release-readiness register and decision record:

```bash
.venv/bin/python generate_prti_release_readiness.py
```

The current decision is documented in
[`PRTI_GH_RELEASE_READINESS.md`](PRTI_GH_RELEASE_READINESS.md). The remaining
181 sensitivity-grid points are not required for the bounded current claims,
and downstream utility is deliberately not claimed. The generated attribute
register is technical triage only: clinical, privacy, data-custodian, and
independent disclosure-control approvals remain external release gates.

The full run is stochastic and can take several minutes. Display its saved
tables without rerunning PSO:

```bash
.venv/bin/python run_prti.py --show-tables
```

Refresh the source, code, and artifact hashes for the saved run without
rerunning PSO or modifying any research-data CSV:

```bash
.venv/bin/python run_prti.py --refresh-manifest
```

### 10. Saved outputs and privacy boundary

Full baseline runs, saved-table display, and manifest refresh use
`outputs/prti/baseline-v1-seed9/` by default. Audit-only and smoke-test modes
use `outputs/prti/audit-only/` and `outputs/prti/smoke-test/`, respectively.
`--output-directory` overrides these labelled defaults. The original unlabeled
seed-9 files are preserved under `outputs/prti/archive/legacy-seed9/` only for
the recorded fresh-versus-legacy comparison.

- `prti_prepared_input.csv`: the 106 model-input columns after lossless
  representation; it is not anonymized data;
- `prti_input_audit.csv`: role, type, cardinality, missingness, treatment, and
  round-trip distribution checks for all 106 attributes;
- `prti_category_mappings.csv`: every categorical identifier and its exact
  source value, count, proportion, role, and missing-state meaning;
- `prti_preparation_summary.json`: source fingerprint, policies, role counts,
  baseline equivalence classes, and audit validation;
- `prti_before_after_anonymization.csv`: aligned records with each QID shown as
  adjacent `Before` and `After` columns, each SA shown once, and the cluster ID;
- `prti_anonymized_final.csv`: the complete 106-column release in original
  column order, with QIDs replaced and all SAs preserved;
- `prti_cluster_details.csv`: cluster sizes, QID representatives, and compact
  entropy summaries;
- `prti_sensitive_entropy.csv`: one row for every cluster/SA combination;
- `prti_model_evaluation.csv`: compact baseline/result privacy and utility
  metrics with configured cluster count and observed equivalence-class count
  reported in separate fields;
- `prti_run_summary.json`: paper and runtime provenance, configuration, full
  metrics, privacy evaluation, PSO evidence, validations, and output names; and
- `prti_artifact_manifest.json`: source and saved-artifact hashes plus the
  generation-time Git state and separate implementation, test, and
  documentation hashes. Refreshing artifact hashes preserves a run-start
  generation snapshot.

Before a full anonymization summary is saved, every CSV is reopened and
cross-checked for row alignment, column order, complete mappings, unchanged
SAs, matching QID after-values, cluster totals, hard k-anonymity, and complete
entropy rows.

### 11. Interpretation boundary

`C=30`, `k=10`, and `lambda=1` define the first PRTI model configuration. Seed
0 remains the historical first baseline. A deterministic 20-iteration screen
of seeds 0-9 identified seed 9 for a complete 100-iteration run. Relative to
seed 0, the seed-9 run reduced information loss from `0.4540905901` to
`0.4366537386`, increased entropy sum from `52.2712468208` to `52.3112654127`,
and improved the objective from `-51.8171562306` to `-51.8746116741`.

The retained replacement screen is available as
[`evidence/prti-baseline-v1/prti_seed_screen.csv`](evidence/prti-baseline-v1/prti_seed_screen.csv)
with its configuration and provenance in the adjacent JSON file. Seven seeds
completed; seeds 1, 6, and 8 are explicitly recorded as having no feasible
particle in their first evaluated populations.

The fresh labelled 100-iteration run from clean commit `e8d8659` exactly
reproduced the legacy selected metrics, ordered cluster sizes, privacy counts,
search counts, best iteration, and four principal row-level result tables. The
aggregate comparison and the fresh artifact manifest are retained under
[`evidence/prti-baseline-v1/`](evidence/prti-baseline-v1/).

That improvement is not universal across every privacy measure: records above
the strict linkage threshold `tau=0.05` increased from 51 to 67, although
homogeneity-risk records decreased from 1,259 to 1,249 and the number of SAs
with zero minimum entropy decreased from 33 to 30. The paper neither uses
PRTI-GH nor publishes a seed for it. A `full` run means 100 particles and 100
PSO iterations, not exhaustive optimization or proof of a global optimum.
Missing-value representation, the 29/77 role split, and the use of raw
high-cardinality SAs remain documented research policies that should be
examined in later sensitivity experiments rather than silently changed.
The exact duplicate source columns `xray` and `xray1` are likewise retained as
separate SAs. Their entropy terms are counted separately in this source-faithful
baseline; a deduplicated comparison would be a new labeled scenario, not a
correction to the workbook.

Raw inputs and generated row-level outputs are excluded by `.gitignore`.
All versioned `evidence/prti-*` directories are aggregate-only. They may hold
configurations, metrics, hashes, screening checks, tables, and figures, but not
patient rows, assignments, QID representatives, or cluster centroids.

The held-out seed-10 candidate used $k=10$, four clusters, $\lambda=1$, 100
particles, and 100 iterations. It produced class sizes 471, 238, 339, and 437;
IL 0.5342360914; entropy sum 101.0762772244; zero linkage records at all three
reported thresholds; zero any-SA homogeneity records; and zero SAs with
minimum entropy zero. It passed the criteria declared in
`evidence/prti-final-selection-v1/prti_final_selection_plan.json`. Those are
project criteria, not Wei et al. release thresholds, and passing them does not
approve unrestricted publication.

## Paper experiment runner (machine learning paused)

Appendix A specifies 220 Adult MO-OBAM combinations: four `k` values, eleven
cluster counts, and five lambda values. Table 6 reports two boundary settings
for each `k`, giving eight highlighted MO-OBAM runs. Inspect that exact plan
without starting optimization:

```bash
.venv/bin/python run_adult_experiments.py --dry-run
```

Run the eight Table 6 settings with the authors' population and iteration
counts. Matching checkpoints are reused, and the results CSV is updated after
every configuration:

```bash
.venv/bin/python run_adult_experiments.py --preset table-6
```

Long runs can be batched and resumed:

```bash
.venv/bin/python run_adult_experiments.py \
  --preset full-grid \
  --max-new-runs 5
```

The full grid is intentionally not called an exact replay: arXiv v2 does not
publish a random seed, its Algorithm 1 k-penalty has a sign conflict, and its
Adult penalty multiplier is absent. A blocked configuration is retained and
shown in `adult_mo_obam_grid_results.csv`; use `--retry-blocked` only when you
deliberately want to retry the same documented settings.

Each completed run saves compact `centroids.csv`, `assignments.csv`, and
`summary.json` checkpoints. Export one selected final solution into full
prepared-original, before/after, anonymized-release, and evaluation CSVs with:

```bash
.venv/bin/python run_adult_experiments.py \
  --export-run k05_c004_lambda1
```

The current repository contains only the Adult source data. Consequently, the
German Credit and Sepsis experiments cannot be executed here. The paper's two
comparison algorithms are also not silently approximated: the MO-OBAM article
cites them but does not specify their complete implementation and tie-breaking
details, and no matching comparison implementation is supplied in the authors'
repository.

## Verification

Run the tests with the project virtual environment:

```bash
.venv/bin/python -m unittest discover -v
```

Verification on 5 November 2026 completed with 95 passing tests.

The tests cover the paper's nine-record example, multiple sensitive attributes
with `sum(min(entropy))`, the direct v2 objective, hard membership and k
constraints, exact small-data enumeration, PSO determinism against that exact
reference, the lossless duplicate-QID acceleration, Appendix A/Table 6 plans,
and the Adult preparation and published baseline counts.
The PRTI tests additionally preserve the known duplicate-column policy, pin the
recorded seed-9 result, verify the prospective held-out policy and seed-10
candidate, and confirm that manifest refreshes do not alter inputs or result
artifacts. The results-package tests verify its 29-QID, 77-SA, four-cluster,
and ten-screening-check coverage; reject centroid leakage; preserve explicit
missing-state accounting; and confirm that its tables remain aggregate-only.
The release-readiness tests verify that the value-free attribute screen flags
identifier patterns and potential free text, exports no source values, and
keeps external approval and downstream-utility claims explicitly pending.
The server-readiness tests additionally cover joint k/C/lambda plan bounds,
held-out handoff, atomic checkpoint writes, corrupt-checkpoint recovery, and
single-writer locking. The multi-objective tests cover Pareto dominance,
deterministic hard-k-feasible execution, the complete C=1 through C=30 plan,
lambda crossover intervals, immutable PSO settings, SQLite metadata rejection,
and the documented three-sheet Excel report.

## Small exact experiment

For a small custom dataset, use the exhaustive reference solver:

```bash
.venv/bin/python run_mo_obam.py \
  --input data.csv \
  --output-directory output \
  --numeric-qid age \
  --categorical-qid personal_status \
  --categorical-qid job \
  --sensitive-attribute checking_status \
  --sensitive-attribute savings_status \
  --k 2 \
  --clusters 2 \
  --lambda-weight 1
```
