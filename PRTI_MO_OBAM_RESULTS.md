# PRTI-GH MO-OBAM Results

## Document status

| Field | Value |
|---|---|
| Status | Write-up-ready results; held-out candidate passed project screening; external release not approved |
| Results version | 1.6.0 |
| Analysis date | 5 November 2026 |
| Method | Wei et al. MO-OBAM, arXiv:2501.01002v2 |
| Dataset | PRTI-GH, 1,485 records and 106 attributes |
| Current roles | 29 QIDs and 77 sensitive attributes |
| Machine learning | Excluded from this anonymization analysis |

This document reports the recorded PRTI-GH anonymization results, a
predeclared one-factor sensitivity study, a 40-run multi-seed robustness
study, and the prospectively declared held-out candidate. It is a results companion to
[`PRTI_MO_OBAM_DOCUMENTATION.md`](PRTI_MO_OBAM_DOCUMENTATION.md), which remains
the authority for data preparation, equations, implementation decisions, and
known paper ambiguities.

The current machine-readable reporting package is
[`evidence/prti-results-v2/`](evidence/prti-results-v2/). It contains eight
aggregate CSV tables, five PNG/SVG figure pairs, an index, and a hash manifest;
it contains no patient rows or cluster centroids.

The results support a methodological write-up. They do **not** establish that
the generated patient-level file is approved for unrestricted public release.

The inclusive cluster-range protocol is implemented but not complete. Its
partial pilot checkpoints and dry-run plans are not incorporated into the
results below and must not be used to claim an optimal cluster count. The
historical results remain the current completed evidence until the new range,
selection, and held-out run finish. See
[`PRTI_CLUSTER_RANGE_STUDY.md`](PRTI_CLUSTER_RANGE_STUDY.md).

The documented scalar joint k/C/lambda server plan contains 2,145
configurations and 21,450 comparison runs. Its v3 server evidence is not
tracked in this checkout and is therefore not incorporated into the tables in
this document.

The newer additive multi-objective C=1 through C=30 protocol contains 150 k/C
configurations and 1,500 comparison outcomes. The reviewed server evidence
recorded 1,279 successful runs and 221 no-feasible-particle outcomes. Twenty
configurations passed every comparison seed, and the declared rule selected
k=20, C=5 with mean IL 0.4681464343. The aggregate files are not versioned in
this checkout and do not change the historical tables below. A separate
exploratory refinement is required before any held-out execution.

## 1. Analysis scope

The analysis evaluated:

1. information loss from replacing QIDs with cluster representatives;
2. hard $k$-anonymity and released equivalence-class sizes;
3. equivalence-class linkage-risk proxies at $\tau=0.05$, $0.075$, and $0.1$;
4. homogeneity exposure for each of the 77 sensitive attributes and their
   record-level union;
5. minimum cluster entropy for every sensitive attribute;
6. optimizer feasibility evidence; and
7. one-factor sensitivity to $k$, configured cluster count, and $\lambda$;
8. stochastic robustness of four predeclared configurations across seeds 0-9;
   and
9. a held-out seed-10 candidate assessed against criteria recorded before it
   was run.

No source value was imputed or cleaned. The original workbook was not modified.
All 29 QIDs were replaced in the released candidate, while all 77 SAs were
preserved row by row and used in the entropy objective.

## 2. Recorded baseline

The recorded baseline used $k=10$, 30 configured clusters, $\lambda=1$, 100
particles, 100 iterations, and seed 9.

| Measure | Prepared data | MO-OBAM result |
|---|---:|---:|
| Equivalence classes | 1,481 | 30 |
| Minimum class size | 1 | 16 |
| Maximum class size | 2 | 148 |
| Linkage-risk records, $\tau=0.05$ | 1,485 | 67 |
| Linkage-risk records, $\tau=0.075$ | 1,485 | 0 |
| Linkage-risk records, $\tau=0.1$ | 1,485 | 0 |
| Homogeneity-risk records for any SA | 1,485 | 1,249 |
| Information loss | Not applicable | 0.4366537386 |
| Sum of minimum SA entropies | Not applicable | 52.3112654127 |
| Objective | Not applicable | -51.8746116741 |

The baseline rerun inside the sensitivity workflow reproduced the information
loss and entropy sum to the stored numerical tolerance.

### 2.1 Information loss

The recorded information loss was 0.4366537386. This is the ratio defined by
Wei's Equations (13)-(14), combining squared numerical-QID deviation and
categorical-QID mismatch. It is not a literal statement that 43.67% of the
dataset was deleted or rendered unusable.

### 2.2 Equivalence classes and linkage risk

Every released class contained at least 16 records, exceeding the configured
$k=10$. The class-size distribution was:

| Statistic | Records per class |
|---|---:|
| Minimum | 16 |
| First quartile | 24 |
| Median | 35 |
| Mean | 49.5 |
| Third quartile | 58 |
| Maximum | 148 |

For a released equivalence class $E_g$, the linkage-risk proxy was
$1/|E_g|$. Records were counted when this proxy was strictly greater than the
reported threshold. The number of records above $\tau=0.05$ decreased from
1,485 to 67, a 95.49% reduction. No released record exceeded $\tau=0.075$ or
$0.1$. The highest class-level proxy was $1/16=0.0625$.

These figures measure QID equivalence-class exposure. They are not the result
of linking the release to a real external population database.

### 2.3 Homogeneity and sensitive-attribute disclosure

The baseline reduced the record-level union of homogeneity exposure from 1,485
to 1,249 records, a 15.89% reduction. Therefore, 84.11% of records remained in
a class that was homogeneous for at least one of the 77 SAs. This is the main
privacy limitation of the baseline.

Thirty SAs had a minimum cluster entropy of zero. The highest per-SA
homogeneity counts were:

| Sensitive attribute | Records | Percent of 1,485 |
|---|---:|---:|
| `ultrasound` | 902 | 60.74% |
| `intensiveCareUnitIcuAdmission` | 807 | 54.34% |
| `subarachnoidHemorrhage` | 654 | 44.04% |
| `amputation2` | 484 | 32.59% |
| `plastics` | 479 | 32.26% |
| `musculoskeletalPain` | 468 | 31.52% |
| `epiduralHemorrhage` | 458 | 30.84% |
| `abdomenInjuriesSuffered` | 453 | 30.51% |
| `aisAbdomenAndPelvic` | 453 | 30.51% |
| `neckInjuriesSuffered` | 430 | 28.96% |
| `mortality` | 275 | 18.52% |

Per-SA counts overlap and must not be added. The 1,249 value is the union of
records exposed for at least one SA.

## 3. One-factor sensitivity study

### 3.1 Design

The comparison was declared before the additional runs were inspected. It used
only the values printed in Wei et al. Appendix A and held all non-varied
settings at the baseline:

- $k\in\{5,10,15,20\}$ at $C=30$ and $\lambda=1$;
- $C\in\{4,10,20,30,40,50,60,70,80,90,100\}$ at $k=10$ and
  $\lambda=1$; and
- $\lambda\in\{0.0001,0.001,0.01,0.1,1\}$ at $k=10$ and $C=30$.

After removing the repeated reference configuration, this produced 18 unique
configurations. Nine completed. Nine returned `no_feasible_particle` because
none of the 100 seeded initial particles satisfied hard $k$. Such a result is
an algorithmic search failure, not proof that the mathematical configuration
has no feasible partition.

Across the full declared 220-point grid, 39 rows are currently recorded: 10
completed runs, 9 `no_feasible_particle` outcomes, and 20 settings identified
directly as mathematically infeasible because $k\times C>1{,}485$. The 18-point
one-factor subset is included in those 39 rows, not additional to them. One
supplementary completed grid point, $k=5$, $C=4$, $\lambda=0.0001$, lies outside
the one-factor subset. The remaining 181 grid points have not been evaluated.

Every attempted run used population 100, 100 iterations, seed 9, and the same
29/77 role assignment.

### 3.2 Sensitivity to $k$

| $k$ | Status | IL | Entropy sum | Linkage, $\tau=.05$ | Homogeneity union | Feasible evaluations |
|---:|---|---:|---:|---:|---:|---:|
| 5 | Completed | 0.431442 | 53.882236 | 16 | 1,485 | 2,117/10,000 |
| 10 | Completed | 0.436654 | 52.311265 | 67 | 1,249 | 160/10,000 |
| 15 | No feasible initial particle | — | — | — | — | — |
| 20 | No feasible initial particle | — | — | — | — | — |

Lowering $k$ from 10 to 5 slightly reduced IL and the 0.05 linkage count but
increased the homogeneity union to all 1,485 records. The comparison therefore
does not support treating either $k$ value as uniformly superior.

### 3.3 Sensitivity to the configured number of clusters

| Clusters | Status | IL | Entropy sum | Linkage, $\tau=.05$ | Homogeneity union | Zero-entropy SAs | Feasible rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | Completed | 0.588464 | 101.287490 | 0 | 0 | 0 | 99.82% |
| 10 | Completed | 0.481520 | 87.150485 | 0 | 736 | 6 | 90.38% |
| 20 | Completed | 0.450961 | 71.092415 | 0 | 1,485 | 21 | 25.41% |
| 30 | Completed | 0.436654 | 52.311265 | 67 | 1,249 | 30 | 1.60% |
| 40-100 | No feasible initial particle | — | — | — | — | — | — |

Four clusters produced the strongest recorded privacy result: no record
exceeded any linkage threshold, no record was homogeneous for any configured
SA, and no SA had zero minimum entropy. The cost was the highest one-factor IL,
0.588464. Increasing the number of clusters generally reduced IL but weakened
SA diversity and made hard-$k$ feasibility increasingly difficult. The
homogeneity union was not monotonic, so it should be reported directly rather
than inferred from cluster count alone.

### 3.4 Sensitivity to $\lambda$

| $\lambda$ | IL | Entropy sum | Linkage .05 / .075 / .1 | Homogeneity union | Zero-entropy SAs |
|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0.422613 | 46.924318 | 49 / 34 / 0 | 1,485 | 37 |
| 0.001 | 0.416892 | 45.414649 | 106 / 22 / 0 | 1,485 | 35 |
| 0.01 | 0.419260 | 52.405858 | 88 / 22 / 0 | 1,485 | 31 |
| 0.1 | 0.436654 | 52.311265 | 67 / 0 / 0 | 1,249 | 30 |
| 1 | 0.436654 | 52.311265 | 67 / 0 / 0 | 1,249 | 30 |

The lowest observed IL was 0.416892 at $\lambda=0.001$, but all 1,485 records
were homogeneous for at least one SA. The $\lambda=0.1$ and $\lambda=1$
runs produced identical recorded IL, entropy, class-size extrema, and attack
counts; their aggregate-only checkpoints do not establish row-level identity.
Only the weighted objective differed. Objective values should not be ranked
directly across $\lambda$ because the weight itself changes the objective scale.

## 4. Multi-seed robustness study

### 4.1 Design and completion

Four configurations were selected from the recorded baseline and completed
seed-9 one-factor results before the robustness runs were inspected:

- baseline: $k=10$, $C=30$, $\lambda=1$;
- privacy candidate: $k=10$, $C=4$, $\lambda=1$;
- lowest-IL candidate: $k=10$, $C=30$, $\lambda=0.001$; and
- compromise candidate: $k=10$, $C=10$, $\lambda=1$.

Every configuration used population 100, 100 iterations, the unchanged 29/77
role assignment, and deterministic seeds 0-9. This ten-seed protocol is a
project robustness design; Wei et al. do not prescribe it for PRTI-GH.

All 40 runs were recorded. Thirty-four completed, while six returned
`no_feasible_particle`. The baseline and lowest-IL configurations both failed
for seeds 1, 6, and 8. All 4-cluster and 10-cluster runs completed.

### 4.2 Results

Means and sample standard deviations below use completed runs only. Failed
initializations remain part of the reported completion count.

| Configuration | Completed | IL mean ± SD | Entropy mean ± SD | Linkage .05 mean | Homogeneity mean (% records) | Zero-entropy SAs mean | Feasible evaluations mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 7/10 | 0.442868 ± 0.009161 | 52.637562 ± 1.854143 | 51.86 | 1,300.43 (87.57%) | 31.57 | 1.02% |
| Privacy candidate | 10/10 | 0.552210 ± 0.032539 | 101.133721 ± 0.174107 | 0 | 57.50 (3.87%) | 0.20 | 99.78% |
| Lowest-IL candidate | 7/10 | 0.423191 ± 0.007301 | 46.719227 ± 3.160390 | 96.29 | 1,374.86 (92.58%) | 34.57 | 1.07% |
| Compromise candidate | 10/10 | 0.487063 ± 0.015204 | 86.550143 ± 1.224603 | 0 | 954.00 (64.24%) | 9.10 | 87.86% |

The privacy candidate produced zero linkage records at every reported threshold
for all ten seeds. Its homogeneity union ranged from 0 to 291 records, with a
mean of 57.5. Therefore, the seed-9 value of zero homogeneity records was a
valid result for that seed but was not stable across every initialization. Its
mean IL was 24.69% higher than the multi-seed baseline mean.

The compromise candidate also produced zero linkage records for all ten seeds.
Compared with the multi-seed baseline means, it reduced homogeneity exposure by
26.64% while increasing IL by 9.98%. However, its mean homogeneity union was
still 954 records and ranged from 736 to 1,290, so it cannot be called low-risk
without a predeclared acceptance threshold.

The lowest-IL candidate reduced mean IL by 4.44% relative to the baseline, but
increased mean linkage exposure by 85.67% and mean homogeneity exposure by
5.72%. It also shared the baseline's three initialization failures. It is not a
defensible privacy candidate merely because it has the lowest IL.

The robustness study supported selecting the four-cluster privacy candidate for
a prospectively declared held-out run because anonymization, rather than
machine-learning accuracy, is the current project objective. This selection did
not by itself approve a patient-level release.

### 4.3 Held-out final-candidate screening

Before seed 10 was executed, the project recorded the following rule and
criteria in `evidence/prti-final-selection-v1/prti_final_selection_plan.json`:

- keep $k=10$, $C=4$, $\lambda=1$, population 100, and 100 iterations;
- use seed 10, the smallest non-negative seed outside the seeds 0-9 comparison;
- require zero linkage-risk records at $\tau=0.05$, $0.075$, and $0.1$;
- require homogeneity exposure for any SA to cover no more than 20% of rows;
- allow at most one SA with minimum cluster entropy zero; and
- require information loss no greater than 0.60.

These are project screening criteria, not thresholds prescribed by Wei et al.
The 20% homogeneity and 0.60 IL limits round the observed ten-seed stability
envelope of the privacy candidate. They are not independent legal, regulatory,
or disclosure-control standards.

The held-out run completed with 9,994 hard-$k$-feasible and six infeasible
particle evaluations. Its four equivalence classes contained 471, 238, 339,
and 437 records.

| Measure | Held-out seed-10 result | Project criterion | Result |
|---|---:|---:|---|
| Information loss | 0.5342360914 | $\le 0.60$ | Pass |
| Entropy sum | 101.0762772244 | Descriptive | — |
| Objective | -100.5420411330 | Descriptive | — |
| Linkage records, $\tau=0.05$ | 0 | 0 | Pass |
| Linkage records, $\tau=0.075$ | 0 | 0 | Pass |
| Linkage records, $\tau=0.1$ | 0 | 0 | Pass |
| Homogeneity records for any SA | 0 (0.00%) | $\le 20\%$ | Pass |
| SAs with minimum entropy zero | 0 | $\le 1$ | Pass |
| Minimum equivalence-class size | 238 | $\ge k=10$ | Pass |
| Maximum equivalence-class size | 471 | Descriptive | — |

All predeclared project checks passed. The status is therefore
`passes_predeclared_project_screening_pending_independent_review`, not
"approved for public release."

All 29 QID columns were affected: 31,386 of 43,065 QID cells (72.88%) changed.
Each released QID had between one and four distinct representative values. The
1,001 missing source values in `presenceOfSupervisionsRoadGuideF` were treated
as an explicit categorical state during optimization, but no cluster selected
that state as its representative; consequently, that QID has zero missing
values in the candidate. This is QID replacement by the anonymization model,
not preprocessing imputation. Every one of the 77 SA columns remained unchanged
row by row, including its missing values.

## 5. Main findings for the write-up

1. The baseline satisfied hard $k=10$ and substantially reduced the measured
   QID linkage proxy, including a 95.49% reduction at $\tau=0.05$.
2. The baseline did not adequately eliminate attribute-disclosure exposure:
   84.11% of records remained homogeneous for at least one SA, and 30 SAs had
   zero minimum cluster entropy.
3. Stronger recorded SA diversity was achievable. Across ten seeds, the
   four-cluster setting had zero measured linkage exposure, mean homogeneity
   exposure of 57.5 records, and mean IL of 0.552210.
4. The lowest-IL setting did not have the lowest privacy exposure. This confirms
   a material privacy-utility trade-off and justifies reporting several metrics
   rather than the optimization objective alone.
5. Hard-$k$ feasibility became a major optimizer limitation as cluster count or
   $k$ increased. Nine of 18 one-factor configurations failed under seed 9, and
   both 30-cluster robustness configurations failed for three of ten seeds.
6. Four- and ten-cluster configurations completed for every robustness seed and
   were much easier for the hard-$k$ search than the 30-cluster settings.
7. The prospectively selected seed-10 privacy candidate passed every recorded
   project screening criterion, with IL 0.534236 and zero recorded linkage,
   homogeneity-union, and zero-minimum-entropy-SA counts.
8. Passing project screening does not establish safe unrestricted release. The
   QID/SA roles and thresholds still require custodian and independent
   disclosure-control review.

## 6. Copy-ready results paragraph

> The PRTI-GH dataset contained 1,485 records and 106 attributes, of which 29
> were treated as quasi-identifiers and 77 as sensitive attributes. MO-OBAM was
> applied with $k=10$, 30 clusters, $\lambda=1$, a population of 100, 100
> iterations, and seed 9. The recorded baseline release contained 30 equivalence
> classes ranging from 16 to 148 records and had an information-loss value of
> 0.4367. Records above the linkage-risk thresholds of 0.05, 0.075, and 0.1
> decreased from 1,485 at each threshold before anonymization to 67, 0, and 0,
> respectively. However, 1,249 records remained homogeneous for at least one
> sensitive attribute, and 30 sensitive attributes had a minimum cluster
> entropy of zero. A subsequent predeclared robustness study evaluated four
> configurations across seeds 0-9. The four-cluster privacy candidate completed
> all ten runs, produced zero linkage-risk records at every reported threshold,
> and had mean homogeneity exposure of 57.5 records and mean information loss of
> 0.5522. The ten-cluster compromise also completed every run and produced zero
> linkage-risk records, but its mean homogeneity exposure remained 954 records
> at a lower mean information loss of 0.4871. These outcomes demonstrate a
> material and initialization-sensitive privacy-utility trade-off. The
> four-cluster configuration was then selected prospectively for seed 10 using
> project screening limits recorded before execution. The held-out run had four
> classes of 238 to 471 records, information loss of 0.5342, zero linkage-risk
> records at all three thresholds, zero homogeneity-risk records across the
> 77-SA union, and no SA with minimum cluster entropy zero. It passed the
> recorded project screening criteria but remains subject to formal role and
> independent disclosure-control review before any external release.

This paragraph should be accompanied by the limitations in the next section.

## 7. Limitations and claim boundaries

- Linkage risk is the equivalence-class proxy $1/|E_g|$, not a real linkage
  attempt against an external database.
- Homogeneity is evaluated against the 77 configured SAs. The role assignment
  remains a project baseline pending formal custodian and disclosure-control
  review.
- The entropy sum gives all SAs one term, but high-cardinality SAs can have a
  larger numerical entropy range. Missingness is also retained as an SA state.
- `xray` and `xray1` are exact duplicate source columns and contribute separate
  entropy terms in the source-faithful baseline.
- The one-factor comparison uses seed 9, the robustness subset uses seeds 0-9,
  and the held-out candidate uses seed 10. None proves global optimality.
- Configurations marked `no_feasible_particle` may succeed with another
  initialization or a documented feasibility method.
- The sensitivity evidence records 39 of the planned 220 grid points; 181 remain
  unevaluated. The one-factor subset overlaps those 39 records.
- No downstream ML utility, external attack simulation, or independent
  disclosure-control review has been completed.
- The project thresholds were chosen from the observed robustness envelope;
  they are not paper-supplied or independently approved release standards.
- The candidate release must not be called safe for unrestricted publication
  solely from these results.
- The new inclusive range can identify only the best observed eligible
  configuration within its declared values, seeds, and PSO budget; it cannot
  prove a global mathematical optimum.
- The scalar joint-grid implementation compared declared k, C, and lambda
  values on the external server, but its v3 evidence is not tracked in this
  checkout and is therefore not integrated into these results.
- The additive multi-objective v1 result is aggregate comparison evidence only.
  Its selected k=20, C=5 configuration is not a held-out result, does not
  approve release, and remains subject to the documented exploratory
  refinement.

## 8. Reproduction

Recreate or resume the aggregate-only robustness study:

```bash
.venv/bin/python run_prti_experiments.py robustness \
  --output-directory evidence/prti-robustness-v1 \
  --population 100 \
  --iterations 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --workers 4
```

Existing matching checkpoints are reused, so this command does not repeat
completed runs unless their recorded source or configuration differs.

The held-out candidate was generated with the prospectively recorded default
policy:

```bash
.venv/bin/python run_prti_release.py
```

The command refuses to overwrite an existing candidate. Recheck the saved
candidate without rerunning PSO:

```bash
.venv/bin/python run_prti_release.py --evaluate-only
```

Regenerate the aggregate result tables without rerunning anonymization:

```bash
.venv/bin/python generate_prti_results.py
```

Continue the full Appendix A sensitivity grid one checkpoint at a time:

```bash
.venv/bin/python run_prti_experiments.py sensitivity --max-new-runs 1
```

Write the new inclusive k=10 range plan without running PSO:

```bash
./run_prti_cluster_range_study.sh 1 148 \
  --k 10 \
  --lambda-weight 1 \
  --population 100 \
  --iterations 100 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --held-out-seed 11 \
  --workers 4 \
  --output-directory evidence/prti-primary-range-k10-v1 \
  --dry-run
```

After inspection, remove `--dry-run`; optionally add `--max-new-runs N` and
rerun the identical command to complete the study in resumable batches. The
new results must not be added to this results document until the selection JSON
reports `selection_complete`.

Run validation:

```bash
.venv/bin/python -m unittest discover -v
```

The verification snapshot on 5 November 2026 contains 95 passing tests. The
results-v2 CSVs were also reloaded with the spreadsheet validation tooling,
all output hashes were checked, and each PNG figure was visually inspected at
1600 by 1000 pixels.

The machine-readable write-up tables are stored under
[`evidence/prti-results-v1/`](evidence/prti-results-v1/):

- `prti_baseline_privacy_utility.csv`;
- `prti_one_factor_sensitivity.csv`;
- `prti_multi_seed_robustness.csv`;
- `prti_sensitive_attribute_risk.csv`;
- `prti_cluster_risk.csv`; and
- `prti_results_summary.json`.

Patient-level source, before/after, and release files remain outside this
aggregate evidence directory.

The consolidated reporting package, including the held-out candidate, QID and
SA tables, figures, experiment coverage, and a provenance manifest, is stored
under [`evidence/prti-results-v2/`](evidence/prti-results-v2/). Regenerate it
without rerunning anonymization:

```bash
.venv/bin/python generate_prti_results_package.py
```

The aggregate-only selection plan and acceptance results are stored under
[`evidence/prti-final-selection-v1/`](evidence/prti-final-selection-v1/).
Patient-level files are stored under the ignored
`outputs/prti/privacy-candidate-heldout-seed10/` directory.
