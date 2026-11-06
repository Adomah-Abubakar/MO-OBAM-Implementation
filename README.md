# MO-OBAM anonymization

This repository implements the MO-OBAM formulation from Wei et al.,
*Multi-Objective Optimization-Based Anonymization of Structured Data for
Machine Learning* ([arXiv:2501.01002v2](https://arxiv.org/abs/2501.01002v2)).

Use this README to install the project, configure an experiment, run it, and
find its outputs. Detailed research material is kept separately:

- [`PRTI_MO_OBAM_DOCUMENTATION.md`](PRTI_MO_OBAM_DOCUMENTATION.md): dataset
  roles, equations, implementation decisions, and limitations;
- [`PRTI_MO_OBAM_RESULTS.md`](PRTI_MO_OBAM_RESULTS.md): recorded results and
  interpretation;
- [`PRTI_CLUSTER_RANGE_STUDY.md`](PRTI_CLUSTER_RANGE_STUDY.md): detailed and
  historical server study protocols; and
- [`PAPER_REVIEW_V2.md`](PAPER_REVIEW_V2.md): paper ambiguities and the chosen
  implementation boundaries.

## 1. Install

Python 3.12 and the pinned packages in `requirements.txt` are the verified
environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Place the reviewed PRTI-GH workbook at:

```text
data/PRTI-GH-study/PRTI-GH-study.xls
```

The adapter verifies the expected 1,485 rows, 106 columns, and source SHA-256
before optimization. The workbook and patient-level outputs are intentionally
excluded from Git.

Run the preparation audit once after a new installation:

```bash
.venv/bin/python run_prti.py --audit-only
```

## 2. Choose one task

Do not run every command in this README. Choose the task that matches your
goal:

| Goal | Command | Produces anonymized patient rows? |
|---|---|---|
| Run one configuration you already chose | `run_prti.py` | Yes |
| Compare explicit `k`, cluster, and lambda values | `run_prti_experiments.py parameter-grid` | Comparison is aggregate-only; a separate held-out command produces the final CSV |
| Search information loss and entropy jointly | `run_prti_multiobjective.py` | Comparison is aggregate-only; a separate held-out command produces the final CSV |
| Generate the current V3-selected candidate | `run_prti_multiobjective_selected_candidate.py` | Yes |

For most users who simply want anonymized data with chosen settings, start
with Section 4.

## 3. Configuration guide

| Setting | Meaning | Guidance |
|---|---|---|
| `k` | Minimum records required in every released equivalence class | Larger values generally strengthen group-size protection but may increase information loss or make PSO initialization harder |
| `clusters` or `C` | Number of anonymized QID groups | Must satisfy `k × C <= 1485` |
| `lambda-weight` | Scalar weight in `information loss - lambda × entropy sum` | Used only by the scalar optimizer |
| `seed` | Reproducible PSO initialization | Keep it fixed when comparing configuration changes; use multiple seeds when measuring stability |
| `population` | Particles evaluated per iteration | The recorded studies use `100` |
| `iterations` | PSO update iterations | The recorded studies use `100` |
| `workers` | Independent runs executed concurrently | Operational setting; choose for the available server resources |

The Pareto multi-objective workflow does not accept one fixed lambda. It
optimizes information loss and entropy jointly, then reports lambda crossover
intervals over `[0, 1]`.

Seeds have distinct purposes:

- the same configuration and seed reproduce the same deterministic run;
- the same seed across different configurations supports a controlled
  comparison;
- several comparison seeds measure stochastic stability; and
- a held-out seed is reserved for one final confirmation after selection.

Always use a new output directory when changing scientific settings. Do not
overwrite an earlier result and relabel it as a different experiment.

## 4. Run one configurable anonymization

Use this command when `k`, `C`, and lambda are already chosen. Change the
values and the matching output label as needed:

```bash
.venv/bin/python run_prti.py \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory outputs/prti/custom-k10-c5-lambda1-seed9 \
  --run-label custom-k10-c5-lambda1-seed9 \
  --k 10 \
  --clusters 5 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seed 9
```

Run again with a new label and directory after changing a configuration. This
single-run mode does not claim that the chosen configuration is optimal.

### Important outputs

The selected output directory contains:

- `prti_anonymized_final.csv`: the complete anonymized dataset in original
  column order;
- `prti_run_summary.json`: configuration, information loss, entropy, objective,
  privacy metrics, validation, and provenance;
- `prti_model_evaluation.csv`: readable before/after metrics;
- `prti_before_after_anonymization.csv`: aligned QID changes and unchanged
  sensitive attributes; and
- `prti_reidentification_risk.csv`: numerical before/after prosecutor and
  marketer risk;
- `prti_l_diversity.csv`: achieved distinct and entropy-effective diversity;
- `prti_distribution_analysis.csv` and `prti_summary_statistics.csv`:
  information-loss context and typed distribution metrics;
- `prti_anonymization_report.md` and `figures/`: the consolidated report and
  graphs generated from the saved analysis tables; and
- `prti_data_dictionary.csv` plus supporting audit and manifest files.

Journalist risk is recorded as unavailable unless population QID frequencies
and the sampling fraction are supplied; the code does not invent a sample-only
substitute. These analyses measure the Wei result and do not add another
anonymization transformation.

The terminal also prints the run's information loss, entropy sum, and
objective. For the example above, the anonymized data will be:

```text
outputs/prti/custom-k10-c5-lambda1-seed9/prti_anonymized_final.csv
```

## 5. Compare k, clusters, and lambda

Use the scalar parameter grid when lambda itself is one of the values being
compared. This broad first-stage screen emphasizes the small lambda values that
may reduce information loss while retaining larger values to expose the
privacy-utility trade-off. It declares eight k values, `C=2..10`, fourteen
lambda values, comparison seeds `0,1,2`, and held-out seed `12` (3,024 runs):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

.venv/bin/python run_prti_experiments.py parameter-grid \
  --source data/PRTI-GH-study/PRTI-GH-study.xls \
  --output-directory evidence/prti-parameter-grid-broad-v1 \
  --minimum-clusters 2 \
  --maximum-clusters 10 \
  --k-values 5 8 10 12 15 20 25 30 \
  --lambda-weights 0 0.0001 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5 0.75 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0 1 2 \
  --held-out-seed 12 \
  --workers 32 \
  --worker-backend process \
  --no-reuse \
  --dry-run
```

Inspect the generated plan, then repeat the exact command without `--dry-run`.
The study is resumable and retains aggregate metrics for every run, including
information loss. Adjust `--workers` for available server resources. Use
`--max-new-runs N` to execute a limited batch. Population and iterations should
be raised to 200 only in a new refinement study containing the shortlisted
configurations, not by changing this immutable broad-study directory.

After the selection JSON reports `selection_complete`, predeclare and run the
selected held-out candidate:

```bash
.venv/bin/python run_prti_selected_candidate.py \
  --study-directory evidence/prti-parameter-grid-broad-v1 \
  --output-directory outputs/prti/parameter-grid-broad-v1-heldout-seed12 \
  --evidence-directory evidence/prti-parameter-grid-broad-v1-final \
  --plan-only
```

Inspect the final plan, then repeat the command without `--plan-only`. The
candidate runner reads `k`, `C`, lambda, optimizer settings, and the held-out
seed from the completed study; it does not accept manual scientific overrides.

## 6. Run a Pareto multi-objective study

Use this mode to search `k` and `C` while treating information loss and entropy
as separate objectives. Choose either a k/C range or explicit
`--configuration-pairs K:C`.

The current confirmation study used three selected pairs and fresh comparison
seeds:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

The four thread variables prevent each worker process from creating additional
numerical-library threads. Adjust `--workers` for the server; it does not
change the scientific plan.

Inspect the immutable plan, then repeat the exact command without `--dry-run`.
The study resumes missing runs from SQLite and writes:

- `prti_multiobjective_plan.json`;
- `prti_multiobjective_selection.json`;
- `prti_multiobjective_report.xlsx`; and
- `prti_multiobjective_checkpoints.sqlite3`.

This comparison stage never writes patient rows. Its report contains every
k/C/seed result, information loss, entropy, privacy checks, and lambda
intervals.

The earlier broad exploration and refinement stages remain historical evidence
and are described in the detailed documentation. A new user does not need to
rerun them before using a completed confirmation study.

## 7. Generate the current V3 held-out anonymized data

The completed V3 evidence selected `k=8`, `C=5`; seed `12` remained held out.
The runner reads those values from the saved selection rather than accepting
manual replacements.

First write the final plan without running seed 12:

```bash
.venv/bin/python run_prti_multiobjective_selected_candidate.py \
  --study-directory evidence/prti-multiobjective-confirmation-v3 \
  --output-directory outputs/prti/multiobjective-v3-heldout-seed12 \
  --evidence-directory evidence/prti-multiobjective-final-selection-v3 \
  --plan-only
```

Inspect the plan, then generate the anonymized data:

```bash
.venv/bin/python run_prti_multiobjective_selected_candidate.py \
  --study-directory evidence/prti-multiobjective-confirmation-v3 \
  --output-directory outputs/prti/multiobjective-v3-heldout-seed12 \
  --evidence-directory evidence/prti-multiobjective-final-selection-v3
```

The main output is:

```text
outputs/prti/multiobjective-v3-heldout-seed12/prti_anonymized_final.csv
```

Its information loss and other observed checks are saved in:

```text
outputs/prti/multiobjective-v3-heldout-seed12/prti_run_summary.json
```

The command refuses to overwrite an existing candidate or substitute another
seed after seeing the held-out result.

## 8. Results and evidence

Use [`PRTI_MO_OBAM_RESULTS.md`](PRTI_MO_OBAM_RESULTS.md) for the written result
record. It currently contains the historical scalar baseline, sensitivity,
robustness, and seed-10 held-out analysis. The exploration, refinement,
confirmation, and seed-12 result should be added there as clearly labelled
new sections after the final held-out run is complete.

Machine-readable aggregate evidence belongs under `evidence/`. Private
patient-level releases belong under `outputs/` and must not be committed.
Passing the project's technical screening checks is not approval for an
unrestricted public release; custodian and independent disclosure-control
review remain external requirements.

## 9. Other supported workflows

Run the supplied Adult experiment:

```bash
.venv/bin/python run_adult.py
```

Inspect available historical and maintenance commands through their help
instead of treating them as required steps:

```bash
.venv/bin/python run_prti_experiments.py --help
.venv/bin/python run_adult_experiments.py --help
.venv/bin/python run_mo_obam.py --help
```

## 10. Verification

Run the repository test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover the formulation, privacy metrics, deterministic PSO behavior,
data preparation, experiment plans, resumable checkpoints, held-out handoff,
result exports, and aggregate reporting. Do not hard-code the test count in
documentation; use the command's current result.
