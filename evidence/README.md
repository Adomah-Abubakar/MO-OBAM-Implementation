# Reproducibility evidence

This directory is for versioned, aggregate-only experiment evidence.

Allowed contents include configurations, seeds, timestamps, metrics, Git and
file hashes, comparison tables, and artifact manifests. Raw source data,
prepared row-level data, before/after records, and final anonymized patient
records belong under the ignored `data/` or `outputs/` directories and must not
be copied here.

The `prti-baseline-v1/` evidence is generated with
`run_prti_experiments.py`. Its files describe the search and result but cannot
reconstruct patient records.

The baseline evidence directory contains:

- `prti_seed_screen.csv` and `.json`: all requested seeds, including failed
  initial populations, the explicit selection rule, configuration, timestamps,
  and generation commit;
- `prti_fresh_vs_legacy_comparison.csv` and `.json`: selected aggregate metrics
  from the fresh and legacy runs; and
- `prti_fresh_artifact_manifest.json`: hashes and generation provenance for the
  ignored row-level fresh-run artifacts.

The original unlabeled seed-9 files are preserved locally under
`outputs/prti/archive/legacy-seed9/`. They are not a third result. The labelled
`outputs/prti/baseline-v1-seed9/` bundle is the active baseline, while the
archive exists only to reproduce the saved fresh-versus-legacy comparison.
The move and the SHA-256 hash of every archived file are recorded in
`prti-baseline-v1/prti_legacy_output_archive.json`.

The `prti-sensitivity-v1/` directory records a resumable PRTI-GH adaptation of
the 220 parameter combinations printed for Adult in Wei et al. Appendix A. It
contains the declared plan, aggregate result table, and one aggregate JSON
checkpoint per attempted configuration. It deliberately contains no
assignments, centroids, before/after data, or patient-level release data.
The current table records 39 grid points: 10 completed runs, 9
`no_feasible_particle` outcomes, and 20 settings rejected directly as
mathematically infeasible because $k\times C>1{,}485$. The remaining 181 grid
points have not been evaluated. The 18-point one-factor subset overlaps this
grid and must not be counted as a separate experiment total.

The `prti-robustness-v1/` directory records the predeclared four-configuration,
ten-seed robustness study. It contains the plan, 40 aggregate JSON checkpoints,
the run-level CSV, and the configuration summary CSV/JSON. Thirty-four runs
completed and six recorded no feasible initial particle. It contains no
assignments, centroids, or patient rows.

The tracked `prti-dense-clusters-v1/` and `prti-primary-range-k10-v1/`
directories are incomplete historical follow-up studies. Their selection JSON
files report `selection_incomplete`; the primary-range study has no selected
configuration. Any interim selection present in the older dense-cluster file is
not a completed-study conclusion and must not be used as the current result.
These directories are retained only as aggregate historical evidence.

The scalar v3 parameter-grid execution was performed on the external server,
but `prti-parameter-grid-k10-15-20-25-30-v3/` is not tracked in this checkout.
Its results must not be inferred or regenerated from the older incomplete
directories above; install and verify the complete v3 evidence separately
before incorporating it into a reporting package.

The `prti-final-selection-v1/` directory records the prospective held-out
candidate plan and its aggregate screening result. The plan fixes the selected
configuration, held-out seed rule, and project thresholds before execution.
The acceptance CSV/JSON records only checks, metrics, hashes, and status. The
patient-level before/after and candidate files remain under the ignored
`outputs/prti/privacy-candidate-heldout-seed10/` directory.

The `prti-results-v1/` directory contains write-up-ready aggregate tables built
from the recorded baseline, sensitivity, and robustness evidence. Its JSON
summary records source hashes, output hashes, and a code/test/documentation
snapshot. It is retained as an archive; use the results-v2 package generator
for current reporting and do not place patient rows in this directory.

The `prti-results-v2/` directory is the consolidated reporting package. It
integrates the held-out candidate with the baseline, sensitivity, robustness,
QID transformation, SA diversity, cluster, and screening evidence. It contains
eight aggregate CSV tables, five figures in PNG and SVG, an index, and a
machine-readable provenance manifest. Regenerate it with
`generate_prti_results_package.py`. It must remain free of patient rows and
cluster centroids. The package and this evidence index were cross-checked on
3 November 2026.

The `prti-release-readiness-v1/` directory contains a 106-row aggregate-only
attribute review register and a machine-readable decision record. The screen
uses attribute names, types, cardinality, missingness, string lengths, token
counts, and identifier-pattern counts; it exports no patient values. All
clinical, privacy, data-governance, custodian, and independent-review fields
remain `PENDING` until named reviewers supply evidence. The decision record
also explains why the remaining 181 grid points and downstream utility testing
are not required for the bounded current claims.

The additive multi-objective workflow writes a new versioned directory such as
`prti-multiobjective-v1/` only when the user runs it. Its four persistent files
are an immutable JSON plan, a selection JSON, one human-readable Excel report,
and an internal SQLite checkpoint database. SQLite stores aggregate run and
Pareto-candidate metrics only; it contains no patient rows, assignments,
centroids, prepared records, or anonymized records. The reviewed v1 server
study recorded all 1,500 outcomes and selected k=20, C=5, but those external
aggregate files are not tracked in this checkout. A separate refinement must
use a new versioned evidence directory.

The repository verification snapshot on 5 November 2026 contains 95 passing
tests.
