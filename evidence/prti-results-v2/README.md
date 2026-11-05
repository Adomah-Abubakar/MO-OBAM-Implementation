# PRTI-GH aggregate results package v2

This directory contains reporting-ready aggregate evidence for the recorded
PRTI-GH MO-OBAM experiments. It contains no patient rows, QID representatives,
cluster centroids, prepared records, or anonymized records.

## Held-out candidate recorded here

- Configuration: k=10, 4 clusters, lambda=1, population=100, iterations=100.
- Held-out seed: 10.
- Information loss: 0.5342360914448178.
- Sum of minimum sensitive-attribute entropy: 101.07627722444862.
- Released class sizes: 471, 238, 339, 437.
- Linkage records above tau=0.05, 0.075, and 0.1: 0, 0, 0.
- Homogeneity-exposed records for any SA: 0.
- Sensitive attributes with zero minimum cluster entropy: 0.

All ten predeclared project checks passed. This is not approval for public or
unrestricted release; formal role review and independent disclosure-control
review are still required.

## Evidence coverage

- Sensitivity grid: 39 of 220 declared points recorded:
  10 completed, 9 `no_feasible_particle`,
  20 mathematically infeasible, and
  181 not yet evaluated.
- One-factor subset: 18 of those sensitivity points — 9 completed and 9
  `no_feasible_particle`; it is not an additional experiment total.
- Robustness study: 34 of 40
  seed runs completed; the remaining 6
  are retained as `no_feasible_particle` outcomes.
- Held-out candidate: one prospectively declared seed-10 run completed and
  assessed against ten recorded project checks.

## Tables

1. `tables/table_01_dataset_and_roles.csv` — dataset size, reviewed roles, source hash, and paper identity.
2. `tables/table_02_baseline_vs_final_candidate.csv` — prepared data, recorded seed-9 baseline, and held-out seed-10 results.
3. `tables/table_03_qid_transformation.csv` — aggregate transformation counts for all 29 QIDs. A missing QID state can disappear after anonymization when it is not selected as a cluster representative; this is not preprocessing imputation.
4. `tables/table_04_sensitive_attribute_results.csv` — source missingness, value-preservation status, entropy, and homogeneity exposure for all 77 SAs.
5. `tables/table_05_final_cluster_summary.csv` — four class sizes and entropy/risk summaries; centroids are excluded.
6. `tables/table_06_robustness_summary.csv` — four configurations across the predeclared seeds 0-9.
7. `tables/table_07_final_screening_checks.csv` — the ten predeclared held-out checks.
8. `tables/table_08_experiment_coverage.csv` — planned, recorded, completed, infeasible, and pending experiment counts. The one-factor subset overlaps the full sensitivity grid and must not be added to it.

## Figures

The five figures are supplied as 1600 x 1000 PNG files and editable SVG files:

- information loss for the two recorded single runs;
- linkage-threshold exposure;
- homogeneity exposure;
- the multi-seed privacy-utility trade-off; and
- final equivalence-class sizes.

Error bars in the robustness figure are sample standard deviations across
completed seeds only. `no_feasible_particle` runs remain explicit failures and
are not converted into metric values.

## Metric boundaries

- Information loss is Wei Equation (14), not a percentage of deleted data.
- The linkage measure is the proxy 1/|E|, not a real external-data attack.
- Homogeneity exposure means a record belongs to a class homogeneous for at
  least one configured SA.
- Objective values are comparable here only where lambda is the same.
- A seeded `no_feasible_particle` result does not prove mathematical
  infeasibility.

## Validation snapshot

On 3 November 2026, all 54 repository tests passed. The eight CSVs reloaded
through the spreadsheet validation tooling, every output hash matched the
manifest, all PNGs were 1600 x 1000 pixels, and visual inspection found no
clipping or unreadable labels. These checks validate implementation and
artifact consistency; they do not replace disclosure-control review.

## Regeneration

From the repository root, with the recorded ignored outputs installed:

```bash
.venv/bin/python generate_prti_results_package.py
```

GNUPlot is required for the PNG/SVG figures. The manifest records all input and
output hashes, the renderer version, and the code/test/documentation snapshot.
