# PRTI-GH fresh cluster-range study

## 1. Purpose and status

This is the self-contained runbook for starting a new PRTI-GH cluster-count
study without depending on historical result or output directories. It covers
installation, configuration, a smoke test, an immutable dry-run plan,
resumable execution, automatic privacy-first selection, one held-out candidate,
and release-readiness evidence.

| Item | Status |
|---|---|
| Range-study code | Complete and tested |
| Joint k/C/lambda grid code | Complete and tested separately from the range study |
| Verified repository tests | 94 passing tests |
| Selection-stage patient rows | Disabled |
| Mathematical bound | `C <= floor(1485/k)` enforced in shell and Python |
| Automatic selection | Privacy and feasibility first, then lowest mean information loss |
| Held-out seed | Declared before selection and prohibited from comparison seeds |
| Additive multi-objective protocol | Implemented separately for C=1 through C=30; no result recorded yet |
| External release approval | Not granted by this workflow |

This is a project protocol rather than an experiment reported in the source
paper. Its valid conclusion is the **best observed eligible cluster count within
the declared range, comparison seeds, and PSO budget**. It is not proof of a
mathematical global optimum.

## 2. Required files

A new study needs only:

1. this repository at implementation commit `d8235d7` or later;
2. Python 3.12 and the pinned packages in `requirements.txt`; and
3. authorized access to the unchanged reviewed PRTI-GH workbook.

Historical `evidence/` and `outputs/prti/` directories are not required for a
new from-zero study.

The default workbook location is:

```text
data/PRTI-GH-study/PRTI-GH-study.xls
```

The required source contract is:

| Property | Required value |
|---|---|
| Rows | 1,485 |
| Columns | 106 |
| SHA-256 | `0deb1fc54a3dccba3a4c0339696ed1fb4f124a0349afde9f06e4b79dbf89a37e` |

Verify the source before running:

```bash
sha256sum \
  data/PRTI-GH-study/PRTI-GH-study.xls
```

The runner also verifies the shape and fingerprint and stops if they differ.

## 3. Primary configuration

The recommended from-zero primary study is:

| Parameter | Value | Meaning |
|---|---:|---|
| Cluster interval | 1 through 148 | Every mathematically feasible C for k=10 |
| `k` | 10 | Minimum records in every equivalence class |
| Lambda | 1 | Full declared entropy/privacy weight |
| Population | 100 | PSO particles per run |
| Iterations | 100 | PSO iterations per run |
| Comparison seeds | 0 through 9 | Ten independent comparisons per C |
| Held-out seed | 11 | Evaluated once after selection |
| Workers | 2 | Concurrent independent runs; not a model parameter |
| Planned runs | 1,480 | 148 C values times 10 comparison seeds |

For 1,485 records, `k=10` permits at most `floor(1485/10)=148`
non-empty clusters. For another k, the code calculates and enforces the
corresponding maximum before optimization.

### Predeclared eligibility rules

Every configuration must record every comparison seed and satisfy hard k. Its
worst result across those seeds must also have:

- zero linkage-risk records at each reported threshold;
- at most 20% of records homogeneous for at least one sensitive attribute;
- at most one sensitive attribute with zero minimum cluster entropy; and
- information loss no greater than 0.60.

Only eligible configurations are ranked. Rank 1 is the lowest mean information
loss, with exact ties broken by lower maximum homogeneity exposure and then the
smaller cluster count.

## 4. Prepare and verify the environment

Run from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover
```

Show the range-runner help at any time:

```bash
./run_prti_cluster_range_study.sh --help
```

## 5. Optional smoke test

Run this small installation check before starting the primary study:

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

This is a smoke test, not a thesis result. Never merge its checkpoints or
metrics into the primary directory.

## 6. Guarantee a from-zero primary study

Use `--no-reuse` to prohibit historical checkpoint imports. The immutable plan
will then record `exact_checkpoint_reuse.permitted: false`. Do not replace this
flag after the study begins.

## 7. Write and inspect the immutable dry-run plan

```bash
.venv/bin/python run_prti_experiments.py dense-clusters \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-primary-range-k10-fresh-v1 \
  --minimum-clusters 1 \
  --maximum-clusters 148 \
  --k 10 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --held-out-seed 11 \
  --workers 2 \
  --no-reuse \
  --dry-run
```

Inspect:

```text
evidence/prti-primary-range-k10-fresh-v1/prti_dense_cluster_plan.json
```

Confirm that it records:

- C=1 through C=148;
- 148 configurations and 1,480 planned runs;
- k=10, lambda=1, population=100, and iterations=100;
- comparison seeds 0 through 9;
- held-out seed 11;
- `patient_level_outputs: false`; and
- the predeclared privacy and feasibility rules above.

The dry plan initially has `source_sha256: null` because `--dry-run` does not
read the workbook. The first real execution verifies the source and immutably
fills in its SHA-256 before optimization starts.

## 8. Execute in resumable batches

Run the identical configuration without `--dry-run`:

```bash
.venv/bin/python run_prti_experiments.py dense-clusters \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-primary-range-k10-fresh-v1 \
  --minimum-clusters 1 \
  --maximum-clusters 148 \
  --k 10 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --held-out-seed 11 \
  --workers 2 \
  --no-reuse \
  --max-new-runs 20
```

Repeat this exact command. Each invocation verifies and skips existing matching
checkpoints, then calculates at most 20 missing cluster/seed runs. After the
machine has proved stable, `--max-new-runs 20` may be changed to
`--max-new-runs 100`, or omitted to attempt every remaining run. Batch size and
worker count are execution controls; they do not change the scientific model.

Do not run two processes against the same output directory. Do not change k,
the C interval, lambda, population, iterations, comparison seeds, held-out
seed, thresholds, or source after inspecting results.

Do not use `--retry-blocked` merely because a checkpoint reports
`no_feasible_particle`. That is a valid recorded stochastic outcome, not a
missing result.

## 9. Monitor progress and completion

Count recorded checkpoints:

```bash
find evidence/prti-primary-range-k10-fresh-v1/runs \
  -name 'seed_*.json' | wc -l
```

The final count must be 1,480.

Inspect status:

```bash
rg -n '"status"|"selected_configuration"' \
  evidence/prti-primary-range-k10-fresh-v1/prti_dense_cluster_selection.json
```

Until every planned result is recorded, the correct state is:

```json
"status": "selection_incomplete",
"selected_configuration": null
```

After all outcomes are recorded, the status becomes `selection_complete`.
Completion does not mean that every configuration passed; it means that every
declared cluster/seed outcome exists and can be screened.

The study directory contains:

- `prti_dense_cluster_plan.json`: immutable protocol;
- `prti_dense_cluster_runs.csv`: one aggregate row per recorded C/seed;
- `prti_dense_cluster_summary.csv`: aggregate statistics by C;
- `prti_dense_cluster_selection.csv`: every pass/fail check and rank;
- `prti_dense_cluster_selection.json`: completion and selected setting;
- `prti_dense_cluster_summary.json`: provenance and interpretation boundary;
- `runs/`: resumable aggregate-only checkpoints.

No patient row, assignment, representative, or centroid is written during
selection.

## 10. Inspect the selected configuration

```bash
.venv/bin/python -c \
'import json; p="evidence/prti-primary-range-k10-fresh-v1/prti_dense_cluster_selection.json"; d=json.load(open(p)); print(json.dumps(d["selected_configuration"], indent=2))'
```

If `selected_configuration` remains null after `selection_complete`, no tested
configuration passed every declared rule. Stop there. Do not relax thresholds
or substitute seeds after seeing the result; that requires a separate new
protocol.

## 11. Declare and generate the held-out candidate

First create only the immutable held-out plan:

```bash
.venv/bin/python run_prti_selected_candidate.py \
  --study-directory evidence/prti-primary-range-k10-fresh-v1 \
  --output-directory outputs/prti/range-selected-heldout-seed11-fresh-v1 \
  --evidence-directory evidence/prti-range-final-selection-fresh-v1 \
  --plan-only
```

Inspect:

```text
evidence/prti-range-final-selection-fresh-v1/prti_dense_final_selection_plan.json
```

The final command derives k, C, lambda, population, iterations, thresholds,
and seed from the completed study. It deliberately does not accept manual
overrides for those settings.

Remove `--plan-only` to evaluate held-out seed 11 exactly once:

```bash
.venv/bin/python run_prti_selected_candidate.py \
  --study-directory evidence/prti-primary-range-k10-fresh-v1 \
  --output-directory outputs/prti/range-selected-heldout-seed11-fresh-v1 \
  --evidence-directory evidence/prti-range-final-selection-fresh-v1
```

Do not replace seed 11 if its result is unfavorable. The local anonymized
research candidate will be:

```text
outputs/prti/range-selected-heldout-seed11-fresh-v1/prti_anonymized_final.csv
```

The before/after file and other patient-level artifacts are internal audit
material and must not be placed in `evidence/` or distributed.

## 12. Generate release-readiness evidence

```bash
.venv/bin/python generate_prti_release_readiness.py \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --candidate-directory outputs/prti/range-selected-heldout-seed11-fresh-v1 \
  --evidence-directory evidence/prti-release-readiness-fresh-v1
```

Automated screening does not authorize external release. Direct-identifier and
free-text review, QID/SA and threat-model approval, data-custodian and
governance approval, and independent disclosure-control review still remain.

## 13. Expected fresh directory structure

Before held-out generation:

```text
evidence/
  prti-primary-range-k10-fresh-v1/

outputs/prti/
  # no new candidate yet
```

After held-out generation and readiness review:

```text
evidence/
  prti-primary-range-k10-fresh-v1/
  prti-range-final-selection-fresh-v1/
  prti-release-readiness-fresh-v1/

outputs/prti/
  range-selected-heldout-seed11-fresh-v1/
```

## 14. Historical and current local directories

Never mix a fresh study with historical or interrupted directories. Use the
new `*-fresh-v1` paths above.

The existing local `evidence/prti-dense-clusters-v1/` directory is an
interrupted earlier pilot, not a final selection. The existing
`evidence/prti-primary-range-k10-v1/` directory is a separate in-progress
study that permits exact historical checkpoint reuse. Neither belongs in the
from-zero `prti-primary-range-k10-fresh-v1` study.

## 15. Remaining reporting task

`generate_prti_results_package.py` currently builds the historical results
package. It must not be presented as the fresh range result. After the new
seed-11 candidate finishes, the reporting package must be updated to consume:

- `prti-primary-range-k10-fresh-v1`;
- `prti-range-final-selection-fresh-v1`;
- `range-selected-heldout-seed11-fresh-v1`; and
- `prti-release-readiness-fresh-v1`.

Only then should new final tables and figures be generated. Until that work is
complete, the range CSV/JSON reports are the authoritative fresh-study results.

## 16. Joint k, cluster-count, and lambda study on a server

The C=1 through C=148 workflow above holds `k=10` and `lambda=1` fixed. It can
select only the best observed **cluster count conditional on those fixed
values**. It cannot determine the best joint `k`, C, and lambda configuration.

Use the separate `parameter-grid` mode when the server is available. It keeps
the existing C-only protocol unchanged, automatically applies
`C <= floor(1485/k)` separately for each k, uses every comparison seed for
every feasible configuration, and ranks configurations with the same
privacy-first eligibility rules.

Show its self-contained help and example:

```bash
.venv/bin/python run_prti_experiments.py parameter-grid --help
```

The following approved expanded grid retains the project privacy floor `k>=10`
and uses the paper-derived lambda values. The dry plan records the protocol that
was used for the external v3 server execution:

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
  --workers 48 \
  --worker-backend process \
  --no-reuse \
  --dry-run
```

This plan contains:

- 148 feasible C values for k=10;
- 99 feasible C values for k=15;
- 74 feasible C values for k=20;
- 59 feasible C values for k=25;
- 49 feasible C values for k=30;
- five lambda values;
- 2,145 configurations; and
- 21,450 aggregate comparison runs across ten seeds.

The server execution completed, but its complete v3 evidence directory is not
tracked in this checkout. Therefore, this runbook records the protocol but does
not treat the absent server files as locally verified results.

Inspect `prti_parameter_grid_plan.json` before removing `--dry-run`. For the
52-physical-core, 93-GiB server reviewed for this project, begin a repeat execution
with `--workers 32 --worker-backend process --max-new-runs 64`. If that batch
is stable, use `--workers 48 --worker-backend process --max-new-runs 480` and
repeat the otherwise identical command. Worker count, backend, and batch size
are execution controls; they do not alter the grid, seeds, PSO budget, privacy
rules, or selection. The single-writer lock deliberately rejects a second
command using the same output directory.

The plan predeclares a second, descriptive analysis alongside the automatic
selection rule. Among configurations passing every privacy and feasibility
gate, the reports identify the Pareto frontier that minimizes mean information
loss and maximizes mean sensitive-attribute entropy sum. Selection remains
privacy-first and then lowest mean information loss; Pareto membership does not
silently replace that rule.

The optimized distance kernel uses SciPy's compiled squared-Euclidean and
Hamming implementations while retaining Equations 10-12. The entropy kernel
uses an equivalent compiled grouped count. A recorded C=30, seed-9,
population-100, iteration-100 aggregate run decreased from approximately
332.7 seconds in the retained historical checkpoint to 33.7 seconds locally;
its selected iteration, feasibility counts, cluster sizes, privacy counts, and
reported metrics were unchanged at the documented precision. Actual server
throughput must still be taken from the first real batch.

Progress and selection files are:

```text
evidence/prti-parameter-grid-k10-15-20-25-30-v3/
  prti_parameter_grid_plan.json
  prti_parameter_grid_runs.csv
  prti_parameter_grid_summary.csv
  prti_parameter_grid_selection.csv
  prti_parameter_grid_selection.json
  runs/
```

After `prti_parameter_grid_selection.json` reports `selection_complete` and a
non-null `selected_configuration`, the same `run_prti_selected_candidate.py`
command accepts this directory and derives the selected k, C, lambda, PSO
budget, and held-out seed without terminal overrides.

The joint-grid result is still the best observed eligible point in a finite
declared grid, not proof of a continuous or mathematical global optimum.

## 17. Server preflight before copying the real study command

Use one Linux/POSIX server and a local persistent filesystem. The runner uses a
single-writer file lock and supports local thread or forked-process workers; it
is not a multi-node scheduler. A four-worker optimized benchmark found
`process` approximately 9% faster than `thread`, so the Linux server command
uses `process`; `thread` remains the portable fallback. Never start two
commands against the same output directory.

From a clean checkout at commit `e6a1bad` or a later readiness commit, run:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover
sha256sum \
  data/PRTI-GH-study/PRTI-GH-study.xls
.venv/bin/python run_prti_experiments.py parameter-grid --help
nproc
free -h
df -h .
```

The tests must pass, and the workbook hash must be:

```text
0deb1fc54a3dccba3a4c0339696ed1fb4f124a0349afde9f06e4b79dbf89a37e
```

Then write the full dry plan from Section 16. Confirm its configuration count,
run count, seed separation, `patient_level_outputs: false`, and
`exact_checkpoint_reuse.permitted: false` before removing `--dry-run`.

The runner writes checkpoints and aggregate reports with atomic replacement.
An interrupted command can be resumed with the identical command; valid
checkpoints are retained, malformed or truncated checkpoints are treated as
pending, and a changed plan is rejected rather than overwritten. Individual
checkpoints are saved immediately. To avoid server-scale quadratic I/O, the
combined CSV/JSON reports refresh every 25 completed runs and again at clean
exit; use the checkpoint-count command while a batch is still running. The
worker queue is bounded to the active worker count, so an interrupt cancels
work that has not started and preserves checkpoints already written.

## 18. Additive multi-objective C=1 through C=30 protocol

The scalar parameter grid above runs the optimizer separately for each declared
lambda. The additive multi-objective workflow instead minimizes information
loss and maximizes the sensitive-attribute entropy sum in one Pareto PSO search
for each k/C/seed combination. It leaves the existing scalar solver and its
recorded evidence unchanged.

The documented protocol uses five k values, every C from 1 through 30, and ten
comparison seeds, producing 150 configurations and 1,500 optimizer runs. Lambda
is not another run dimension: after optimization, the report calculates the
scalar objective and exact candidate crossover intervals over lambda in [0,1].
Seed 12 is prospectively reserved and is not executed during selection.

Show the complete command and required parameters with:

```bash
.venv/bin/python run_prti_multiobjective.py --help
```

Run the example with `--dry-run` first and inspect the immutable plan. Removing
`--dry-run` executes or resumes missing comparison runs. The workflow persists
only:

```text
evidence/prti-multiobjective-v1/
  prti_multiobjective_plan.json
  prti_multiobjective_selection.json
  prti_multiobjective_report.xlsx
  prti_multiobjective_checkpoints.sqlite3
```

The SQLite file is an internal aggregate checkpoint store that prevents
duplicate k/C/seed runs and supports safe resume. It stores no patient rows,
assignments, centroids, prepared records, or anonymized records. The JSON and
Excel reports cannot select a configuration until every declared comparison
run is recorded. This workflow currently stops at aggregate selection; it does
not generate a held-out anonymized dataset and has no recorded result yet.
