# PRTI-GH MO-OBAM Anonymization: Data, Method, Implementation, and Results

## Document control

| Field | Value |
|---|---|
| Status | Living research document |
| Document version | 1.3.0 |
| Last updated | 5 November 2026 |
| Dataset | Pediatric Road Traffic Injury in Ghana (PRTI-GH) |
| Methodological authority | Wei et al., MO-OBAM, arXiv:2501.01002v2, 15 May 2025 |
| Current experiment | Baseline `baseline-v1-seed9`; held-out candidate `privacy-candidate-heldout-seed10`; 29 QIDs and 77 sensitive attributes |
| Sensitivity evidence | 220 configurations declared; 39 recorded: 10 completed, 9 found no feasible initial particle, 20 mathematically infeasible, and 181 not yet evaluated. The 18-point one-factor subset contains 9 completed and 9 no-feasible-particle outcomes |
| Robustness study | 40 planned seed runs recorded: 34 completed and 6 found no feasible initial particle |
| Held-out candidate | Four clusters, seed 10; passed all predeclared project screening checks; external release not approved |
| Follow-up range protocol | Code ready and primary execution in progress for a mathematically feasible inclusive cluster sweep with explicit comparison seeds and a prospectively reserved held-out seed; selection not yet complete |
| Scalar joint parameter protocol | Expanded v3 plan contains 2,145 configurations and 21,450 runs; server evidence is not tracked in this checkout and is not integrated into the recorded results |
| Additive multi-objective protocol | C=1 through C=30 across five k values and ten comparison seeds; 1,500 aggregate runs planned, no result recorded yet |
| Current reporting package | `evidence/prti-results-v2/`: 8 aggregate CSV tables, 5 PNG/SVG figure pairs, index, and provenance manifest; no patient rows or centroids |
| Release readiness | Automated 106-attribute triage complete; external release remains blocked pending custodian, clinical, privacy/governance, and independent disclosure-control review |
| Machine-learning phase | Paused; not evaluated in the current PRTI-GH result |

This is the official project record for the current PRTI-GH adaptation of the
Multi-Objective Optimization-Based Anonymization Model (MO-OBAM). It is a
living document: changes to the data roles, preprocessing policy, mathematical
implementation, optimizer settings, or reported results must be recorded here.

> **Publication and release warning:** This document may be published after its
> authorship, repository URL, and license are supplied. The generated patient-
> level CSV files must not be treated as approved public releases merely because
> the optimization completed. The current release preserves all 77 sensitive
> attributes at row level, and the QID/SA classification has not yet been
> documented as an approved decision of the original data custodian, hospitals,
> ethics committees, or a formal disclosure-control review.

## 1. Purpose and scope

The project asks whether the MO-OBAM formulation can be adapted to the PRTI-GH
structured health dataset while:

1. preserving the original workbook unchanged;
2. representing mixed numerical and categorical data reproducibly;
3. replacing quasi-identifier (QID) values with cluster representatives;
4. including multiple sensitive attributes (SAs) in the entropy objective;
5. enforcing hard $k$-anonymity on every configured cluster;
6. producing aligned before/after and complete final datasets; and
7. recording enough evidence to reproduce and audit the experiment.

This document covers data preparation, anonymization mathematics, software
behavior, the recorded PRTI-GH run, validation, and known limitations. It does
not claim that the current role assignment is the only correct threat model. It
does not claim a globally optimal solution. It does not include the paper's
machine-learning evaluation phase.

## 2. Sources and authority

### 2.1 PRTI-GH data source

The local source is the Excel 97-2003 workbook `PRTI-GH-study.xls`. The source
study is:

> Appiah, A. B., Wilson, M. L., Dambach, P., et al. (2025). Epidemiology of
> pediatric road traffic injuries: a multicenter hospital-based study in Ghana.
> *Injury Epidemiology, 12*, 83. https://doi.org/10.1186/s40621-025-00646-1

The study reports 1,485 pediatric road-traffic-injury cases collected across
three Ghanaian teaching hospitals from 2021 to 2024 and identifies the shared
data as anonymized open data. Project links:

- [Published study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12690968/)
- [OSF data project](https://osf.io/r5fy7/?view_only=235b689570094e7687bd76a00adf2cc9)

The installed workbook is treated as a read-only research input. Its reviewed
SHA-256 fingerprint is:

```text
0deb1fc54a3dccba3a4c0339696ed1fb4f124a0349afde9f06e4b79dbf89a37e
```

The pipeline verifies this fingerprint before and after preparation and after
anonymization. A different workbook is rejected by the default reviewed-data
configuration.

### 2.2 MO-OBAM authority

The mathematical authority is:

> Wei, Y., Benson, H., Agor, J. K., and Capan, M. Multi-Objective
> Optimization-Based Anonymization of Structured Data for Machine Learning
> Application. [arXiv:2501.01002v2](https://arxiv.org/abs/2501.01002v2),
> revised 15 May 2025.

The implementation uses version 2 because it contains the active formulation
and Algorithm 1 PSO pseudocode. The January 2025 version 1 objective and the
version 2 objective are not interchangeable. The authors' public repository is
[yw825/Multi-Objective-Optimization-based-Anonymization-Model](https://github.com/yw825/Multi-Objective-Optimization-based-Anonymization-Model).

When the paper, pseudocode, and public code disagree, this project follows the
printed active mathematical constraints and records the disagreement rather
than silently inventing a missing parameter.

## 3. Terminology and disclosure model

- A **direct identifier** explicitly identifies a person, such as a name or
  medical-record number. The current adapter has no separate direct-identifier
  role and does not automatically detect such columns.
- A **quasi-identifier (QID)** may help link a record to an individual when
  combined with other information. QIDs are the values replaced by cluster
  representatives.
- A **sensitive attribute (SA)** contains information whose inference is to be
  protected. SAs are not replaced by this implementation, but every configured
  SA contributes to the entropy objective and privacy evaluation.
- An **equivalence class** is a set of released records with the same QID
  combination. In the current implementation, every configured cluster is an
  equivalence class.

Attribute roles depend on the release context, intended recipient, available
external information, and research purpose. An attribute can be a QID in one
release and not in another. The 29/77 split below is therefore a project
baseline, not a universal property of PRTI-GH and not a role assignment made by
the MO-OBAM paper.

## 4. Dataset inventory and current roles

The reviewed workbook has 1,485 rows and 106 columns. The current baseline
contains 29 QIDs and 77 SAs.

| Role | Count | Current treatment |
|---|---:|---|
| Numerical QID | 2 | Kept numerical for the model; replaced by cluster centroid values in the release |
| Categorical QID | 27 | Reversibly encoded; compared using categorical mismatch; replaced by optimized cluster representatives in the release |
| Sensitive attribute | 77 | Reversibly represented; used in entropy; preserved row by row in the release |

### 4.1 Numerical QIDs

```text
childsAge
year
```

### 4.2 Categorical QIDs

```text
zonesGhana
region
district
remotenessCommunity
a3Hospital
sexOfTheChild
educationLevel
religiousAffiliation
months
weatherSeason
dayOfWeek
timeOfDay
modePrehospitalTransport
typeOfRoadUser
carSeatBeltUse
typeOfRoad
natureOfRoad
presenceOfSupervisionsRoadGuideF
crashType
circumstancesOfInjury
mechanismOfInjury
whatIsTheActualCauseTheAccident
overtaking
perceivedSpeeding
vehicleCrossLane
driverWasTiredSleeping
vehicleTyreBust
```

`religiousAffiliation` is a QID in the current threat model.
`temperatureDegreeCelsius` and `oxygenSaturation` remain SAs in this baseline.
A later sensitivity analysis may change those roles, but such a run must have a
new label and must not overwrite or silently relabel this baseline.

### 4.3 Sensitive attributes

<details>
<summary>Show all 77 sensitive attributes</summary>

```text
referredHospital
radiologicInvestigations
xray
xray1
ctScan
ultrasound
bodyPartsExaminedXray
bodyPartsExaminedCtScan
sufferedMultipleInjuries
commonInjuriesSuffered
headInjuriesSuffered
facialInjuriesSuffered
neckInjuriesSuffered
abdomenInjuriesSuffered
upperLimbsInjuriesSuffered
lowerLimbsInjuriesSuffered
superficialExternalInjuriesSuff
musculoskeletalPain
typeOfHeadInjuries
scalpLacerationHematoma
skullFracture
intracranialHemorrhage
epiduralHemorrhage
subduralHemorrhage
subarachnoidHemorrhage
cerebralContusion
cerebralEdema
diffuseAxonalInjury
concussion
unspecifiedHeadInjury
othersSpecifyHeadInjury
abbreviatedInjuryScaleAis
aisHeadNeckOrSpine
aisFace
aisChest
aisAbdomenAndPelvic
aisExtremities
aisExternalOrSuperficial
iss
catIss
binaryIss
gcs
totalGcs
triageAndMonitoring
heartRateBpm
systolicRange
diastolicRange
temperatureDegreeCelsius
oxygenSaturation
intensiveCareUnitIcuAdmission
mortality
majorTreatmentManagement
majorSurgical
minorSurgical
nonSurgical
nonSurgicalTreatment
tetanusInjection
catheterization
bloodTransfusion
ivFluids
antibiotics
analgesia
pop
admittingService
internalMedicine
generalSurgery
orthopaedics
ophthalmology
neurosurgery
emergencyServiceOnly
plastics
cardio
treatmentOutcome
lohDays
amputation2
catGlasgowOutcomeScale
ordinalGlasgowOutcomeScale
```

</details>

All non-QID columns are currently assigned to the SA set by complement. This is
reproducible, but it is also a limitation: the project does not yet contain an
approved, literature-supported, attribute-by-attribute rationale for all 106
roles. The full machine-readable baseline register is
[`outputs/prti/baseline-v1-seed9/prti_input_audit.csv`](outputs/prti/baseline-v1-seed9/prti_input_audit.csv). Exact
categorical states and their codes are in
[`outputs/prti/baseline-v1-seed9/prti_category_mappings.csv`](outputs/prti/baseline-v1-seed9/prti_category_mappings.csv).

## 5. Reproducible data preparation

### 5.1 Source preservation and column names

The pipeline never writes to `PRTI-GH-study.xls`. It reads the workbook, applies
reviewed canonical camel-case names in memory, and writes new artifacts to a
labelled run directory under `outputs/prti/`.

Known duplicate source columns are preserved as separate attributes. They are
documented and tested; they are not automatically merged, removed, or given new
values. This policy protects exact source fidelity while keeping any later
deduplication or reweighting decision separate from the recorded baseline.

Most names are transformed mechanically. Headers that cannot be reconstructed
reliably by a general regular expression use the reviewed
`MANUAL_COLUMN_RENAMES` dictionary. This changes headers only, not cell values.
The canonical names are necessary so that code, audit tables, and documentation
refer to the same variables consistently.

### 5.2 Categorical representation

Each exact non-missing categorical state receives a deterministic one-to-one
integer identifier. Missing categorical values receive code `0`.

This is not ordinalization and it is not the anonymization itself. For a
categorical QID, MO-OBAM uses only equality or inequality:

$$
\delta(x_{ij},q_{cj})=
\begin{cases}
1, & x_{ij}\ne q_{cj},\\
0, & x_{ij}=q_{cj}.
\end{cases}
$$

Consequently, a code of `6` is not treated as farther from `1` than a code of
`2`. One-hot encoding is unnecessary for the paper's categorical mismatch
distance and would change the number and weighting of model dimensions.

Reviewed semantic orders exist for a small set of variables such as education,
month, day, ISS category, and Glasgow outcome category. They make the exported
codebook stable and understandable. They do not turn those fields into numeric
distance variables in MO-OBAM.

### 5.3 Numerical values

Numerical input values are not normalized, binned, rounded, or imputed during
preparation. The two numerical QIDs, `childsAge` and `year`, enter the squared-
distance terms unchanged.

After anonymization, a numerical QID is replaced by its optimized cluster
centroid and may therefore be non-integral. A decimal `year` or age in the
released dataset is a mathematical representative, not a claim that the source
observation was fractional. Any later decision to round or format these values
would be an additional transformation and must be evaluated separately.

### 5.4 Missing values

The paper does not define missing-value treatment. This project therefore uses
an explicit, reversible policy and labels it as a project decision:

- no values are imputed;
- missing categorical values become the explicit category code `0`;
- missing non-QID numerical values become the explicit model state
  `<MISSING>`; and
- a missing SA state contributes to entropy like any other observed state.

The two numerical QIDs must not contain missing values. The adapter raises an
error if they do, because the paper defines neither numerical imputation nor a
missing-number distance.

### 5.5 Spelling and source categories

Cell values are preserved exactly. The pipeline does not correct spelling,
case, whitespace, or variant labels, and it does not split multi-response text.
For example, an exact combined injury response remains one categorical state.
This avoids changing frequencies before anonymization, but it can create
high-cardinality states and clinically equivalent categories that remain
separate. Any cleaning or ontology mapping must be a separately reviewed data-
quality phase with before/after evidence.

### 5.6 Distribution guarantee and its boundary

The preparation step decodes every represented value and compares every cell
and every exact value-state frequency with the source. The recorded audit
reports:

- shape preserved: `true`;
- column order preserved: `true`;
- lossless round trip: `true`;
- all 106 attribute distributions preserved: `true`; and
- largest frequency difference after round trip: `0`.

This guarantee applies to representation before optimization. Anonymization is
expected to change QID distributions because QIDs are replaced by cluster
representatives. The 77 SAs are preserved row by row, so their global source
distributions remain unchanged in the final release.

## 6. Mathematical formulation

This section follows the active equations in arXiv v2. Let:

- $i=1,\ldots,n$ index records;
- $j$ index attributes;
- $c=1,\ldots,n_C$ index clusters;
- $n_{NQI}$ and $n_{CQI}$ be the numbers of numerical and categorical QIDs;
- $n_{SA}$ be the number of SAs;
- $q_{cj}$ be the representative for QID $j$ in cluster $c$; and
- $w_{ic}\in\{0,1\}$ indicate whether record $i$ belongs to cluster $c$.

### 6.1 One cluster per record: Equation (9)

$$
\sum_{c=1}^{n_C} w_{ic}=1,
\qquad i=1,\ldots,n.
\tag{9}
$$

### 6.2 Numerical and categorical deviation: Equations (10)-(13)

For numerical QIDs in cluster $c$:

$$
\sum_{i=1}^{n}w_{ic}
\sum_{j=1}^{n_{NQI}}(x_{ij}-q_{cj})^2,
\qquad c=1,\ldots,n_C.
\tag{10}
$$

For categorical QIDs in cluster $c$:

$$
\sum_{i=1}^{n}w_{ic}
\sum_{j=n_{NQI}+1}^{n_{NQI}+n_{CQI}}
\delta(x_{ij},q_{cj}),
\qquad c=1,\ldots,n_C.
\tag{11}
$$

The record-to-centroid distance is:

$$
d(i,c)=
\sum_{j=1}^{n_{NQI}}(x_{ij}-q_{cj})^2
+
\sum_{j=n_{NQI}+1}^{n_{NQI}+n_{CQI}}
\delta(x_{ij},q_{cj}).
\tag{12}
$$

The total deviation introduced by anonymization is:

$$
D_{\text{anon}}=
\sum_{c=1}^{n_C}\sum_{i=1}^{n}w_{ic}
\left(
\sum_{j=1}^{n_{NQI}}(x_{ij}-q_{cj})^2
+
\sum_{j=n_{NQI}+1}^{n_{NQI}+n_{CQI}}
\delta(x_{ij},q_{cj})
\right).
\tag{13}
$$

### 6.3 Original-data deviation and information loss: Equation (14)

Let $\overline{X}_j$ be the dataset mean of numerical QID $j$ and
$\widehat{X}_j$ be the dataset mode of categorical QID $j$. The original-data
deviation is:

$$
D_{\text{original}}=
\sum_{j=1}^{n_{NQI}}\sum_{i=1}^{n}
(x_{ij}-\overline{X}_j)^2
+
\sum_{j=n_{NQI}+1}^{n_{NQI}+n_{CQI}}\sum_{i=1}^{n}
\delta(x_{ij},\widehat{X}_j).
\tag{14}
$$

Information loss is the normalized ratio:

$$
IL=\frac{D_{\text{anon}}}{D_{\text{original}}}.
$$

The implementation reports an error if the denominator is zero. It does not add
an undocumented smoothing constant.

### 6.4 Entropy for multiple sensitive attributes: Equations (15)-(19)

For sensitive attribute $SA_j$ and cluster $c$:

$$
\operatorname{entropy}(w_c,SA_j)=
\sum_{s\in SA_j}
-p(w_c,s)\log_2 p(w_c,s).
\tag{15}
$$

The observed proportion of sensitive value $s$ is:

$$
p(w_c,s)=
\frac{\sum_{i=1}^{n}w_{ic}\mathbb{I}(SA_{ij}=s)}
{\sum_{i=1}^{n}w_{ic}}.
\tag{16}
$$

The indicator is:

$$
\mathbb{I}(SA_{ij}=s)=
\begin{cases}
1, & SA_{ij}=s,\\
0, & \text{otherwise}.
\end{cases}
\tag{17}
$$

For each SA, MO-OBAM takes its worst cluster entropy:

$$
H_j=
\min_{c\in\{1,\ldots,n_C\}}
\left\{\operatorname{entropy}(w_c,SA_j)\right\}.
\tag{18}
$$

The privacy term sums those per-SA minima:

$$
H_{\text{sum}}=
\sum_{j=n_{NQI}+n_{CQI}+1}^{n_{NQI}+n_{CQI}+n_{SA}}
H_j.
\tag{19}
$$

This is how the current model handles 77 SAs. It does not concatenate them into
one joint SA and it does not ignore 76 of them. It calculates the minimum
cluster entropy separately for every SA and then sums the 77 minima. The paper's
printed equations support multiple SAs even though one of its Adult examples
uses only one SA.

### 6.5 Hard k-anonymity: Equation (20)

$$
\sum_{i=1}^{n}w_{ic}\ge k,
\qquad c=1,\ldots,n_C.
\tag{20}
$$

Every configured cluster, including a cluster that would otherwise be empty,
must contain at least $k$ records. Infeasible particles are rejected.

### 6.6 Active objective

The version 2 scalar objective minimized by the code is:

$$
\min_{q,w}\quad IL-\lambda H_{\text{sum}},
$$

subject to Equations (9) and (15)-(20), with distances and information loss
defined by Equations (10)-(14).

A negative objective is valid. It occurs when the weighted entropy term is
larger than information loss. Under the same dataset, role assignment, $k$,
$n_C$, $\lambda$, and preprocessing policy, a smaller objective is preferred.
The objective is not itself a probability, an accuracy percentage, or an
absolute privacy certificate. Objectives from different schemas or numbers of
SAs are not directly comparable because the entropy sum changes scale.

## 7. Optimization procedure

The scalable solver follows the structure of Algorithm 1:

1. represent each particle as a matrix of cluster centroids;
2. initialize every centroid entry from an observed QID value;
3. assign each record to the nearest centroid using Equation (12);
4. reject the particle if any cluster violates Equation (20);
5. calculate $IL$, every SA's minimum entropy, and the active objective;
6. update personal and global best feasible particles; and
7. update continuous and categorical particle positions using the authors'
   public PSO mechanics.

Duplicate QID tuples are evaluated once with record-count weights. Records with
identical QIDs have identical distances to all centroids and therefore receive
the same assignment. Tests compare this accelerated evaluation with the direct
record-level equations.

The optimizer is stochastic. A completed run is the best feasible solution
found for its recorded seed and search budget, not a proof of the global
optimum. Exhaustive enumeration is available only for small validation data.
For a fixed membership, the exact small-data solver can derive a mean for each
numerical QID and a mode for each categorical QID. The scalable PSO instead
searches the paper's $q_{cj}$ variables directly, so its categorical
representatives are not guaranteed to be the observed within-cluster modes.

## 8. Current PRTI-GH experiment

### 8.1 Recorded configuration

| Parameter | Value | Basis |
|---|---:|---|
| Records, $n$ | 1,485 | Reviewed workbook |
| QIDs | 29 | Current project threat-model baseline |
| SAs | 77 | All remaining attributes |
| Minimum cluster size, $k$ | 10 | First PRTI-GH baseline |
| Number of clusters, $n_C$ | 30 | Largest tested paper-grid count that produced hard-$k$ feasible particles for the recorded configuration |
| Entropy weight, $\lambda$ | 1 | High emphasis on SA entropy |
| PSO population | 100 | Authors' public implementation/notebook setting |
| PSO iterations | 100 | Authors' public implementation/notebook setting |
| Random coefficient maximum | 0.01 | Authors' public PSO mechanics |
| Seed | 9 | Selected after a 20-iteration screen of seeds 0-9 |

The paper did not use PRTI-GH and did not publish a PRTI-GH seed. Seed 0 is the
historical local baseline. Seed 9 improved the objective in the recorded
screen and was then run for the full 100 iterations. The original per-seed
screening table was not retained. A replacement screen was executed from clean
commit `b1edff87` and is retained in
[`evidence/prti-baseline-v1/prti_seed_screen.csv`](evidence/prti-baseline-v1/prti_seed_screen.csv).
Seven seeds completed. Seeds 1, 6, and 8 contained no hard-$k$-feasible particle
in their first evaluated populations and are recorded as failures rather than
being assigned values. Seed 9 again had the minimum objective among completed
runs and was selected by the predeclared rule. This is exploratory seed
selection, not a paper-defined estimator or evidence of global optimality.

### 8.2 Command

Run from the repository root:

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

Regenerate the complete exploratory seed screen without saving patient rows:

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

Compare the fresh labelled run with the legacy output using aggregate metrics:

```bash
.venv/bin/python run_prti_experiments.py compare \
  --legacy-directory outputs/prti/archive/legacy-seed9 \
  --fresh-directory outputs/prti/baseline-v1-seed9 \
  --evidence-directory evidence/prti-baseline-v1
```

Write the PRTI-GH sensitivity plan without starting optimization:

```bash
.venv/bin/python run_prti_experiments.py sensitivity --dry-run
```

Run the plan in resumable batches, retaining aggregate evidence only:

```bash
.venv/bin/python run_prti_experiments.py sensitivity --max-new-runs 1
```

This plan uses exactly the 220 combinations printed for Adult in Wei et al.
Appendix A: $k\in\{5,10,15,20\}$, 11 cluster counts from 4 through 100, and
$\lambda\in\{0.0001,0.001,0.01,0.1,1\}$. The paper did not apply this grid to
PRTI-GH, so the resulting experiment is explicitly a project sensitivity
adaptation rather than a paper reproduction. Checkpoints contain aggregate
metrics only; they do not contain patient rows, assignments, or centroids.

Recreate or resume the predeclared multi-seed robustness study:

```bash
.venv/bin/python run_prti_experiments.py robustness \
  --output-directory evidence/prti-robustness-v1 \
  --population 100 \
  --iterations 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --workers 4
```

The four configurations were declared from the baseline and seed-9 one-factor
results before their multi-seed outcomes were inspected. Seeds 0-9 are a
deterministic project robustness protocol, not a setting prescribed by Wei et
al. Every checkpoint is aggregate-only and can be resumed without repeating a
matching completed run.

Generate the prospectively declared held-out candidate:

```bash
.venv/bin/python run_prti_release.py
```

The command first writes or verifies the selection plan, refuses to overwrite
an existing candidate, runs seed 10, and saves an aggregate acceptance result.
Re-evaluate the existing candidate without rerunning PSO:

```bash
.venv/bin/python run_prti_release.py --evaluate-only
```

Regenerate the first aggregate table set from the baseline, sensitivity, and
robustness evidence:

```bash
.venv/bin/python generate_prti_results.py
```

Regenerate the current consolidated aggregate package, including the held-out
candidate, QID/SA tables, figures, experiment coverage, and provenance:

```bash
.venv/bin/python generate_prti_results_package.py
```

The second command requires GNUPlot for its PNG/SVG figures. Neither command
reruns PSO or copies patient rows into `evidence/`.

Generate the aggregate-only release-readiness register and decision record:

```bash
.venv/bin/python generate_prti_release_readiness.py
```

This command performs value-free identifier-pattern and potential-free-text
triage for all 106 attributes. It does not approve the current roles and does
not replace the required clinical, privacy, governance, custodian, or
independent disclosure-control reviews. See
[`PRTI_GH_RELEASE_READINESS.md`](PRTI_GH_RELEASE_READINESS.md).

Prepare and audit without anonymization:

```bash
.venv/bin/python run_prti.py --audit-only
```

Display previously saved tables without rerunning PSO:

```bash
.venv/bin/python run_prti.py --show-tables
```

Refresh hashes and code provenance for the saved run without rerunning PSO or
changing any research-data CSV:

```bash
.venv/bin/python run_prti.py --refresh-manifest
```

### 8.3 Recorded results

The fresh labelled run has ID
`prti-20261027T173235Z-baseline-v1-seed9-k10-c30-lambda1p0-seed9`. It ran from
17:32:35 to 17:34:57 UTC on 27 October 2026 in 142.2 seconds. Its generation
snapshot records clean commit `e8d86597d2915d7f0977884aeeb18a12935a9d0e`.

| Measure | Prepared baseline | MO-OBAM result |
|---|---:|---:|
| Equivalence classes | 1,481 | 30 |
| Minimum class size | 1 | 16 |
| Maximum class size | 2 | 148 |
| Linkage-risk records, $\tau=0.05$ | 1,485 | 67 |
| Linkage-risk records, $\tau=0.075$ | 1,485 | 0 |
| Linkage-risk records, $\tau=0.1$ | 1,485 | 0 |
| Homogeneity-risk records for any SA | 1,485 | 1,249 |
| Information loss | Not applicable | 0.4366537386 |
| Sum of minimum SA entropies | Not reported for baseline | 52.3112654127 |
| Objective | Not applicable | -51.8746116741 |

Additional optimization evidence:

- 160 of 10,000 particle evaluations were hard-$k$ feasible;
- 9,840 evaluations were rejected as infeasible;
- the stored zero-based best-found iteration index is 34;
- 30 of 77 SAs have a minimum cluster entropy of zero; and
- the fresh recorded runtime was approximately 142.2 seconds on the recorded
  Linux environment; the legacy run recorded approximately 176.2 seconds.

The result satisfies $k=10$ because the smallest released equivalence class has
16 records. It materially reduces the recorded linkage-risk counts. It does
not eliminate homogeneity risk across all 77 SAs. The value 1,249 is a record-
level union: a record is counted if its equivalence class is homogeneous for at
least one SA.

The seed-9 result improved upon the seed-0 local baseline:

| Measure | Seed 0 | Seed 9 |
|---|---:|---:|
| Information loss | 0.4540905901 | 0.4366537386 |
| Entropy sum | 52.2712468208 | 52.3112654127 |
| Objective | -51.8171562306 | -51.8746116741 |
| Linkage-risk records at $\tau=0.05$ | 51 | 67 |
| Homogeneity-risk records for any SA | 1,259 | 1,249 |
| SAs with zero minimum entropy | 33 | 30 |

Seed 9 improves the optimized objective, information loss, entropy sum, and two
homogeneity measures, but increases the strict $\tau=0.05$ linkage-risk count.
It is therefore a trade-off, not a universal improvement across every privacy
metric.

The fresh run exactly reproduced the legacy information loss, entropy sum,
objective, ordered cluster sizes, linkage counts, homogeneity union, zero-
entropy-SA count, feasible/infeasible evaluation counts, and best iteration.
The final anonymized data, aligned before/after data, cluster table, and
sensitive-entropy table are also value-for-value identical. The aggregate
comparison is retained in
[`evidence/prti-baseline-v1/prti_fresh_vs_legacy_comparison.csv`](evidence/prti-baseline-v1/prti_fresh_vs_legacy_comparison.csv).

### 8.4 Parameter-sensitivity progress

The full sensitivity plan contains 220 declared configurations. Equation (20)
immediately rules out 20 of them for this 1,485-record dataset because
$k\times n_C>1{,}485$. Those settings are retained as
`mathematically_infeasible`; they are not run and are not assigned fabricated
metrics.

The current aggregate results table records 39 of the 220 declared grid points:
10 completed PSO runs, 9 `no_feasible_particle` outcomes, and the 20 direct
mathematical-infeasibility records. Therefore, 181 declared grid points remain
unevaluated. The completed total includes one supplementary grid point outside
the one-factor subset: $k=5$, $C=4$, and $\lambda=0.0001$.

For the first write-up, 18 unique one-factor configurations were declared using
only Wei's Appendix A values while holding population 100, 100 iterations,
seed 9, and the 29/77 role assignment fixed. Nine completed and nine returned
`no_feasible_particle`. The latter means no seeded initial particle satisfied
hard $k$; it is not proof of mathematical infeasibility.

The recorded baseline was reproduced inside this workflow. Among completed
one-factor runs, $k=10$, $n_C=4$, and $\lambda=1$ eliminated measured linkage
and homogeneity counts across the configured SAs but increased information loss
to 0.5884643568. The lowest observed information loss, 0.4168919618, occurred
at $k=10$, $n_C=30$, and $\lambda=0.001$, but all 1,485 records remained
homogeneous for at least one SA. These results demonstrate a material
privacy-utility trade-off and do not identify a universally best setting.

The complete write-up tables, interpretation, copy-ready results paragraph,
and claim boundaries are maintained in
[`PRTI_MO_OBAM_RESULTS.md`](PRTI_MO_OBAM_RESULTS.md). Aggregate progress remains
stored in
[`evidence/prti-sensitivity-v1/prti_sensitivity_results.csv`](evidence/prti-sensitivity-v1/prti_sensitivity_results.csv).

### 8.5 Multi-seed robustness results

The robustness study evaluated the baseline ($k=10$, $C=30$, $\lambda=1$), a
four-cluster privacy candidate, a 30-cluster lowest-IL candidate with
$\lambda=0.001$, and a ten-cluster compromise candidate. Each used population
100, 100 iterations, and seeds 0-9.

All 40 seed outcomes were recorded. Thirty-four completed. The baseline and
lowest-IL configurations both returned `no_feasible_particle` for seeds 1, 6,
and 8. All four- and ten-cluster runs completed.

| Configuration | Completed | Mean IL | Mean linkage records at $\tau=.05$ | Mean homogeneity records | Mean feasible evaluations |
|---|---:|---:|---:|---:|---:|
| Baseline | 7/10 | 0.442868 | 51.86 | 1,300.43 | 1.02% |
| Privacy candidate | 10/10 | 0.552210 | 0 | 57.50 | 99.78% |
| Lowest-IL candidate | 7/10 | 0.423191 | 96.29 | 1,374.86 | 1.07% |
| Compromise candidate | 10/10 | 0.487063 | 0 | 954.00 | 87.86% |

The privacy candidate had the strongest measured privacy but the highest mean
IL. Its homogeneity union ranged from 0 to 291 records, showing that the seed-9
zero value did not hold for every seed. The compromise candidate had zero
measured linkage exposure for every seed and lower IL, but its mean homogeneity
union remained 954 records. The project therefore selected the privacy
candidate for a separate prospectively declared held-out run. Exact means,
sample standard deviations, ranges, and seed-level statuses are stored under
`evidence/prti-robustness-v1/`.

### 8.6 Held-out candidate and project screening

The final-selection plan was written at 20:06:01 UTC on 2 November 2026 before
seed 10 was executed. It kept $k=10$, $C=4$, $\lambda=1$, population 100, and
100 iterations. Seed 10 was selected by the recorded rule: use the smallest
non-negative integer seed outside the seeds 0-9 configuration comparison.

The project criteria required zero linkage records at all three reported
thresholds, no more than 20% of rows in the any-SA homogeneity union, at most
one SA with minimum entropy zero, IL no greater than 0.60, unchanged SAs, an
unchanged source fingerprint, and hard $k$. Wei et al. define the reported
metrics but do not provide these PRTI-GH thresholds. The homogeneity and IL
limits round the privacy candidate's ten-seed stability envelope and must not
be represented as independent standards.

The run ID is
`prti-20261102T200626Z-privacy-candidate-heldout-seed10-k10-c4-lambda1p0-seed10`.
It completed in 111.7 seconds with 9,994 feasible and six infeasible particle
evaluations.

| Measure | Result |
|---|---:|
| Information loss | 0.5342360914 |
| Entropy sum | 101.0762772244 |
| Objective | -100.5420411330 |
| Equivalence classes | 4 |
| Class sizes | 471, 238, 339, 437 |
| Linkage records at $\tau=.05/.075/.1$ | 0 / 0 / 0 |
| Homogeneity records for any SA | 0 |
| SAs with zero minimum entropy | 0 |

All recorded project checks passed. This means the candidate passed the
declared internal screen. It does not mean the candidate is approved for public
or unrestricted release.

Anonymization changed 31,386 of 43,065 QID cells (72.88%), and every QID column
changed at least once. Each QID had one to four distinct representatives after
replacement. The categorical QID `presenceOfSupervisionsRoadGuideF` contained
1,001 source missing values represented explicitly during optimization. None
of the four selected representatives was the missing state, so its candidate
column contains no missing values. This is a consequence of replacing QIDs by
cluster representatives, not preprocessing imputation. All 77 SAs, including
their missingness, remained unchanged row by row.

### 8.7 Follow-up inclusive cluster-range protocol

The earlier four-candidate robustness study did not evaluate every integer
cluster count. A separate follow-up protocol now supports an inclusive range,
for example C=1 through C=148 at k=10. This is new project work and does not
retroactively change the recorded seed-10 result.

The protocol enforces `C <= floor(1485/k)` before optimization, requires all
model and PSO settings plus comparison seeds at the terminal, and reserves a
held-out seed that cannot appear in the comparison set. Selection evidence is
aggregate-only. Every configured C must complete every comparison seed and
pass the predeclared hard-k, linkage, homogeneity, entropy and information-loss
rules. Only passing configurations are ranked by mean information loss.

The selected setting is described as the best observed eligible configuration
within the declared range, seeds and PSO budget. The stochastic PSO results do
not prove a global optimum. The complete protocol, commands, outputs and status
are in [`PRTI_CLUSTER_RANGE_STUDY.md`](PRTI_CLUSTER_RANGE_STUDY.md).

The selected patient-level candidate is generated only by
`run_prti_selected_candidate.py`, which reads the completed selection evidence.
It does not accept manual k, C, lambda, population, iteration, or seed
overrides. The primary range and its new held-out candidate have not yet been
completed, so no new range result is reported here.

### 8.8 Joint k/C/lambda server protocol

The inclusive range in Section 8.7 varies C while holding k and lambda fixed.
It therefore cannot identify a jointly preferred k, C, and lambda setting. A
separate `parameter-grid` mode now supports that question without changing the
existing one-factor protocol.

For each declared k, the runner includes only
`C <= floor(1485/k)`, crosses those C values with every declared lambda and
comparison seed, and writes aggregate-only resumable checkpoints. It requires
an unseen held-out seed and an explicit checkpoint-reuse decision. Selection
remains privacy-first and is withheld until every declared outcome is
recorded. The runner uses atomic file replacement and a single-writer lock so
interrupted server runs can resume without accepting partial checkpoints.

For analysis, the summary and selection evidence retain mean, standard
deviation, minimum, and maximum entropy sums across seeds. Among configurations
passing every predeclared privacy and feasibility rule, they also identify the
Pareto frontier that minimizes mean information loss and maximizes mean
sensitive-attribute entropy sum. This descriptive frontier does not replace the
predeclared automatic rule: select the lowest mean information loss among
eligible configurations, with the documented deterministic tie breakers.

The expanded k={10,15,20,25,30}, paper-derived
lambda={0.0001,0.001,0.01,0.1,1}, C=1 through its k-specific maximum,
ten-seed plan contains 2,145 configurations and 21,450 runs. Values k=25 and
k=30 extend the paper-derived sensitivity values as a declared PRTI-GH project
adaptation so the search includes stronger privacy settings. The v3 plan was
executed on the external server, but its complete evidence directory is not
tracked in this checkout. It is therefore not incorporated into the recorded
results in this document. Commands and server preflight checks are in
[`PRTI_CLUSTER_RANGE_STUDY.md`](PRTI_CLUSTER_RANGE_STUDY.md).

Runtime optimization does not reduce that plan. Equation 12 now uses compiled
SciPy squared-Euclidean and Hamming distance kernels, and the sensitive-state
count uses an equivalent compiled `bincount` aggregation. Linux server runs may
use forked process workers through `--worker-backend process`; threads remain
available as the portable default. These are execution changes only: k, C,
lambda, population, iterations, comparison seeds, held-out seed, and privacy
rules remain unchanged.

### 8.9 Additive multi-objective protocol

The additive multi-objective extension keeps the scalar PSO baseline unchanged.
For each k/C/seed run it searches information loss and sensitive-attribute
entropy jointly and retains a bounded non-dominated archive. The declared study
uses C=1 through C=30, k={10,15,20,25,30}, and comparison seeds 0 through 9:
150 k/C configurations and 1,500 optimizer runs. Held-out seed 12 is reserved
and is not executed during aggregate selection.

Lambda is not repeated as a separate optimizer dimension. For the discovered
privacy-eligible archive, the report evaluates `IL - lambda * entropy_sum` and
calculates candidate crossover intervals over the full lambda domain [0,1].
Population, iterations, archive size, initialization attempts, random-
coefficient bound, privacy thresholds, seeds, k values, cluster limits and
worker count are terminal parameters and are recorded in the immutable plan.

The workflow stores an immutable plan, aggregate selection JSON, an Excel
report, and one SQLite checkpoint database. SQLite contains aggregate run and
candidate metrics only and supports duplicate prevention and resumability. It
contains no patient rows, assignments, centroids, prepared records or
anonymized records. Selection is withheld until all declared comparison runs
are recorded. The workflow has not yet produced a result and does not currently
generate the final held-out anonymized dataset.

## 9. Output artifacts

Full baseline runs, saved-table display, and manifest refresh use
`outputs/prti/baseline-v1-seed9/` by default. Audit-only and smoke-test modes
use `outputs/prti/audit-only/` and `outputs/prti/smoke-test/`, respectively.
The original unlabeled seed-9 output is archived at
`outputs/prti/archive/legacy-seed9/`; it is retained only for the recorded
fresh-versus-legacy comparison and is not a third result.

| File | Meaning |
|---|---|
| `prti_prepared_input.csv` | Complete 1,485 by 106 model representation before anonymization. Codes are reversible and are not anonymized values. |
| `prti_input_audit.csv` | One row per attribute with position, role, data type, cardinality, missingness, treatment, review flags, and round-trip distribution checks. |
| `prti_category_mappings.csv` | Exact categorical value-to-code dictionary, including counts, proportions, roles, and explicit missing states. |
| `prti_preparation_summary.json` | Source fingerprint, role lists and counts, preparation policies, baseline equivalence classes, flags, and validations. |
| `prti_before_after_anonymization.csv` | Aligned record view with every QID's `Before` and `After` columns side by side, each SA once, record position, and cluster ID. |
| `prti_anonymized_final.csv` | Complete 1,485 by 106 candidate release in original column order. QIDs are replaced; SAs are unchanged. Internal cluster IDs are excluded. |
| `prti_cluster_details.csv` | One row per configured cluster with size, proportion, QID representatives, and entropy summaries: 30 rows for the baseline and 4 for the held-out candidate. |
| `prti_sensitive_entropy.csv` | One row per cluster-SA pair: $30\times77=2{,}310$ baseline rows and $4\times77=308$ held-out-candidate rows. |
| `prti_model_evaluation.csv` | Two-row baseline/result table for information loss and privacy-risk metrics. `configuredClusters` and `equivalenceClasses` are separate fields. Despite its name, it does not contain PRTI-GH machine-learning performance. |
| `prti_run_summary.json` | Authoritative machine-readable record of paper revision, environment, schema, configuration, metrics, cluster sizes, optimizer evidence, validation, runtime, and output names. |
| `prti_artifact_manifest.json` | Source and artifact SHA-256 evidence, run identity, configuration, timestamps, canonical command, and the generation-time Git/code snapshot. Artifact refreshes preserve an existing run-start snapshot. It never modifies research data. |

The JSON summaries are the authoritative source for exact configuration and
unrounded metrics. CSV files should be read with a conforming CSV parser because
source text values may contain commas or embedded line breaks.

Versioned aggregate-only evidence is stored under `evidence/prti-baseline-v1/`:

- the complete replacement seed-screen CSV and JSON;
- the fresh-versus-legacy comparison CSV and JSON; and
- the fresh artifact manifest containing source, artifact, executable-code,
  test, documentation, Git, command, and timestamp provenance.

These files contain configurations, metrics, and hashes, not patient rows.

The sensitivity plan and its resumable aggregate checkpoints are stored under
`evidence/prti-sensitivity-v1/`. That directory follows the same no-patient-row
rule.

The 40-run multi-seed plan, seed checkpoints, run-level table, and aggregate
summary are stored under `evidence/prti-robustness-v1/`. They contain no
assignments, centroids, before/after records, or patient-level release rows.

The prospective selection plan and aggregate acceptance CSV/JSON are stored
under `evidence/prti-final-selection-v1/`. The corresponding patient-level
before/after and candidate files remain only under the ignored
`outputs/prti/privacy-candidate-heldout-seed10/` directory.

Write-up-ready baseline, one-factor, multi-seed, per-SA, and per-cluster tables
are stored under `evidence/prti-results-v1/` and regenerated with
`generate_prti_results.py`. They also contain no patient rows.

The current consolidated package is stored under `evidence/prti-results-v2/`
and regenerated with `generate_prti_results_package.py`. It adds the held-out
candidate, aggregate transformation counts for all 29 QIDs, entropy and
homogeneity results for all 77 SAs, the four-class summary, ten screening
checks, experiment coverage, five figures in both PNG and SVG, and a manifest
covering every input and output hash. It contains no patient rows, assignments,
QID representatives, or cluster centroids.

## 10. Validation and software structure

### 10.1 Implementation map

| Component | Responsibility |
|---|---|
| [`mo_obam/prti_data.py`](mo_obam/prti_data.py) | Source fingerprinting, canonical headers, role assignment, reversible representation, codebook generation, and distribution validation |
| [`mo_obam/formulation.py`](mo_obam/formulation.py) | Direct reference implementation of Equations (9)-(20), anonymization, and the active objective |
| [`mo_obam/pso_solver.py`](mo_obam/pso_solver.py) | Scalable Algorithm 1 PSO search, hard-$k$ rejection, profile acceleration, and vectorized multi-SA entropy |
| [`mo_obam/privacy_metrics.py`](mo_obam/privacy_metrics.py) | Equivalence classes and record-level linkage/homogeneity risk counts |
| [`mo_obam/provenance.py`](mo_obam/provenance.py) | Refreshable run identity, source/code/artifact hashes, timestamps, and Git-state manifest |
| [`mo_obam/prti_runner.py`](mo_obam/prti_runner.py) | End-to-end preparation, optimization, decoding, validation, reporting, and output generation |
| [`mo_obam/prti_experiments.py`](mo_obam/prti_experiments.py) | Aggregate-only seed screening, Wei-grid sensitivity experiments, and fresh-versus-legacy reproduction comparison |
| [`mo_obam/prti_robustness.py`](mo_obam/prti_robustness.py) | Resumable multi-seed execution, per-seed aggregate checkpoints, stability statistics, and provenance |
| [`mo_obam/prti_dense_clusters.py`](mo_obam/prti_dense_clusters.py) | Inclusive C-only and joint scalar parameter-grid planning, aggregate checkpointing, selection and Pareto summaries |
| [`mo_obam/prti_dense_release.py`](mo_obam/prti_dense_release.py) | Validates completed scalar selection evidence and derives one prospectively held-out candidate without manual parameter overrides |
| [`mo_obam/prti_multiobjective_solver.py`](mo_obam/prti_multiobjective_solver.py) | Additive Pareto PSO extension that jointly minimizes information loss and maximizes sensitive-attribute entropy |
| [`mo_obam/prti_multiobjective.py`](mo_obam/prti_multiobjective.py) | Immutable multi-objective planning, process execution, SQLite checkpoints, aggregate selection and Excel reporting |
| [`mo_obam/prti_results.py`](mo_obam/prti_results.py) | Aggregate write-up tables, per-SA risk ranking, class-risk table, and results provenance |
| [`mo_obam/prti_results_package.py`](mo_obam/prti_results_package.py) | Validates recorded evidence and builds the aggregate-only results-v2 tables, GNUPlot figures, index, and hash manifest |
| [`mo_obam/prti_release.py`](mo_obam/prti_release.py) | Prospective selection-plan enforcement, held-out execution, project-screening assessment, and aggregate evidence |
| [`mo_obam/prti_release_readiness.py`](mo_obam/prti_release_readiness.py) | Value-free attribute triage, bounded experiment/utility decisions, and explicit external release gates |
| [`run_prti.py`](run_prti.py) | Reproducible command-line interface |
| [`run_prti_experiments.py`](run_prti_experiments.py) | Reproducible seed-screen, sensitivity, robustness, and comparison command-line interface |
| [`run_prti_multiobjective.py`](run_prti_multiobjective.py) | Reproducible aggregate-only C=1 through C=30 multi-objective command-line interface |
| [`run_prti_release.py`](run_prti_release.py) | Reproducible held-out candidate and evaluate-only command-line interface |
| [`run_prti_selected_candidate.py`](run_prti_selected_candidate.py) | Derives and executes a scalar study's prospectively reserved held-out candidate |
| [`generate_prti_release_readiness.py`](generate_prti_release_readiness.py) | Regenerates the aggregate-only attribute review and release-readiness decision record |
| [`generate_prti_results.py`](generate_prti_results.py) | Regenerates write-up-ready result tables without rerunning PSO |
| [`generate_prti_results_package.py`](generate_prti_results_package.py) | Regenerates the consolidated results-v2 package without rerunning PSO |
| [`tests/`](tests/) | Equation, solver, data-preservation, output, and regression checks |

The active implementation is the `mo_obam/` package and the entry points shown
above. The root `MO-OBAM.ipynb`, `constants.py`, `utils.py`, `model_train.py`,
and `particle_swarm.py` files are retained authors' reference material. The
active PRTI-GH workflow does not import them.

### 10.2 Recorded export checks

Before the run summary is saved, the pipeline reloads every generated CSV and
checks that:

- the source, prepared, paired, and final rows remain aligned;
- the final release has exactly 1,485 rows and 106 source columns;
- the before/after file has 1,485 rows and 137 columns;
- final QID values match the paired `After` values;
- all 77 SAs match their source values;
- cluster sizes sum to all source records;
- every cluster satisfies hard $k$-anonymity;
- the expected cluster-SA entropy rows are present (2,310 for the 30-cluster
  baseline and 308 for the four-cluster candidate); and
- the source workbook fingerprint remains unchanged.

### 10.3 Reproducible environment

The recorded run used:

```text
Python 3.12.3
NumPy 2.5.2
pandas 3.0.5
SciPy 1.18.1
xlrd 2.0.2
Linux x86_64
```

Exact Python package versions are pinned in [`requirements.txt`](requirements.txt).
Create the environment and run the tests with:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
```

The current pinned environment must pass the complete test suite. The results
package was also independently checked for numerical
agreement with the saved summaries, output hashes, 1600 by 1000 PNG dimensions,
CSV reloadability, and absence of patient-level dimensions and centroid fields.

GNUPlot 6.0 patchlevel 0 generated the reporting package's PNG/SVG figures. It
is a system dependency for `generate_prti_results_package.py`, not a Python
package in `requirements.txt`; its exact version is recorded in the package
manifest.

The current suite also pins the recorded PRTI-GH configuration and metrics,
confirms that known duplicate columns remain separate unchanged SAs, and proves
that refreshing the artifact manifest does not change its source or result
files. These regression expectations are intentionally updateable: when a new
baseline is formally adopted, update the run label, documentation, saved
summary, and tests together rather than silently relabeling the old result.

## 11. Paper ambiguities and implementation decisions

The following boundaries are material to interpretation and reproduction.

1. **k-violation sign:** Algorithm 1 prints a negative violation term in a
   minimization fitness, which would reward violations. The authors' Python
   adds a positive penalty, but the Adult penalty multiplier is not published.
   This implementation enforces Equation (20) as a hard constraint and does
   not use an invented multiplier.
2. **Objective normalization:** The paper states that normalization may be
   required but does not define a formula or publish bounds. The implementation
   uses the active objective literally.
3. **Multiple-SA entropy:** Equations (18)-(19) take a minimum over clusters for
   each SA and then sum the minima. The authors' public utility code instead
   sums SA entropies inside each cluster and then takes one minimum. This
   project follows the printed equations.
4. **Linkage boundary:** The paper's prose says risk "exceeds" $\tau$, while a
   displayed expression uses $\ge\tau$. The Adult Table 6 baseline is reproduced
   with strict $1/|E_g|>\tau$, so reporting uses the strict operator.
5. **Initialization:** Algorithm 1 samples centroid values from observed QIDs.
   The public code samples numeric integers within bounds. This implementation
   follows the printed algorithm.
6. **Empty clusters:** Equation (20) applies to every configured cluster. The
   public helper checks only cluster labels that appear. This implementation
   treats empty clusters as infeasible.
7. **Missing values:** MO-OBAM does not specify their treatment. The explicit
   missing-state policy is a documented PRTI-GH adaptation.
8. **Role assignment:** The paper does not specify PRTI-GH QIDs or SAs. The
   current 29/77 split is a project baseline requiring formal contextual review.
9. **Stochastic reproduction:** The paper does not publish every seed or a
   complete end-to-end experiment script. A single PSO run is a best-found
   result, not guaranteed equality with a reported table.

The detailed equation audit is maintained in
[`PAPER_REVIEW_V2.md`](PAPER_REVIEW_V2.md).

## 12. Current data concerns

The automated audit flags 34 attributes for at least one review concern. The
flags are warnings, not automatic errors.

- `district` and `whatIsTheActualCauseTheAccident` are high-cardinality QIDs.
- Several SAs are high-cardinality and may contribute much more entropy than
  low-cardinality or binary SAs.
- Numerical SAs such as `iss`, vital signs, temperature, oxygen saturation, and
  length of hospital stay are treated as exact observed discrete states for
  entropy. This may inflate diversity relative to a clinically justified
  categorization.
- Missingness is retained as an entropy state for several attributes. A model
  may therefore optimize diversity of missingness as part of privacy.
- Free-text-like or multi-response values remain whole exact categories, which
  can create sparse states and unstable modes.
- The unweighted sum in Equation (19) gives every SA one minimum-entropy term,
  but entropy magnitude still depends on cardinality. With 77 heterogeneous
  SAs, $\lambda$ has a dataset-specific scale.
- `xray` and `xray1` are separate columns in the reviewed workbook but contain
  exactly the same values in all 1,485 records. They are deliberately preserved
  as two SAs in this source-faithful baseline. Consequently, Equation (19)
  includes their equal entropy contributions separately. This is a documented
  property of the selected schema, not an accidental data-cleaning omission.
  Any deduplicated or reweighted comparison must be labeled as a new sensitivity
  scenario and must leave the source workbook unchanged.
- In the recorded 30-cluster seed-9 baseline, 30 SAs had zero minimum entropy,
  including `mortality`, and only 1.6% of particle evaluations were hard-$k$
  feasible. These are baseline-specific findings, not properties of the
  held-out candidate.
- In the four-cluster held-out candidate, no SA had zero minimum entropy and
  9,994 of 10,000 evaluations (99.94%) were hard-$k$ feasible. This improvement
  does not remove the need for independent disclosure-control review and shows
  that feasibility and SA diversity are configuration- and seed-dependent.

These concerns must not be "fixed" silently. Each proposed change needs a new
documented scenario so its effect on information loss, entropy, attack counts,
and clinical meaning can be compared with the baseline.

## 13. Reproduction boundaries

The following claims are supported by the current evidence:

- the original PRTI-GH workbook was not modified;
- preprocessing is reversible and preserves exact source distributions;
- all 29 QIDs are replaced by cluster representatives in the final output;
- all 77 SAs are preserved and used in the multiple-SA entropy objective;
- the recorded release satisfies hard $k=10$;
- the saved artifacts are internally cross-checked;
- a fresh run from a clean recorded commit exactly reproduces the legacy
  selected metrics and row-level result artifacts; and
- the implementation follows the active printed v2 equations where the paper
  and public code conflict.

The following claims are not yet supported:

- that the 29/77 classification is approved by the data custodian or ethics
  bodies for a specific release;
- that direct identifiers or residual free-text identifiers are absent;
- that the candidate release is safe for unrestricted public publication;
- that seed 9, seed 10, or any recorded candidate is globally optimal or
  universally best;
- that the result reproduces a paper table, because the paper did not use
  PRTI-GH;
- that downstream machine-learning utility is preserved; or
- that every attribute treatment has a completed literature-backed clinical
  rationale.

## 14. Required work before an external release or paper

The project decision recorded on 4 November 2026 is that the remaining 181
grid points are not required for the bounded current claims, because those
claims do not assert exhaustive grid coverage or globally optimal
hyperparameters. Downstream utility testing is also not required unless a
defined predictive or clinical task or a utility-preservation claim is added.
These decisions and their reopen conditions are machine-readable under
`evidence/prti-release-readiness-v1/`.

1. Obtain and document the data custodian's threat model and intended recipient.
2. Review all 106 roles attribute by attribute with clinical, privacy, and data-
   governance expertise.
3. Screen for direct identifiers and identifying free text outside the current
   QID list.
4. Decide whether numerical/high-cardinality SAs require clinically justified
   representation and run labeled sensitivity analyses.
5. Keep the remaining 181 unevaluated points out of the current protocol. Run
   them only if a later claim, reviewer, or preregistered protocol requires
   exhaustive grid coverage; do not present unattempted points as completed.
6. Have the data custodian and an independent disclosure-control specialist
   review or replace the exploratory project screening thresholds.
7. Keep machine-learning utility out of the current claims. If a downstream
   use is proposed, predeclare its task, outcome, protocol, metrics, and
   acceptable degradation before running a utility analysis.
8. Add repository authorship, a license, citation metadata, and a stable release
   tag or DOI. No license or `CITATION.cff` is currently recorded in this
   repository.
9. Conduct an independent disclosure-control review of the final candidate
   dataset.

## 15. Updating this document

When the project changes, update the document version and revision history and
record:

- the source fingerprint and dataset version;
- every role or representation change and its rationale;
- the exact command, code revision, environment, seed, and hyperparameters;
- the generated summary filename;
- baseline and result metrics;
- validation status;
- known deviations from the paper; and
- whether the change supersedes a prior baseline or creates a new scenario.

Refresh artifact hashes when saved artifacts change. For a new run, the
manifest records Git state and code hashes before data preparation begins. A
later artifact refresh preserves that run-start generation snapshot rather than
replacing it with newer code or documentation hashes. Legacy runs without a
run-start snapshot or execution timestamp are explicitly labelled as such;
neither provenance nor dates are reconstructed.

Do not manually copy thousands of category mappings into this document. Keep
the exact mapping dictionary machine-readable in
`prti_category_mappings.csv`, and use this document to explain its policy,
meaning, limitations, and provenance.

## Revision history

| Version | Date | Change |
|---|---|---|
| 1.3.0 | 5 November 2026 | Documented the 1,500-run additive multi-objective protocol, aggregate SQLite boundary, scalar v3 evidence boundary, active-versus-reference code layout, expanded implementation map, and 94-test verification |
| 1.2.0 | 5 November 2026 | Added compiled distance kernels, process-based configuration/seed parallelism, standalone repository paths, project-local environment support, portable aggregate provenance, and 85-test verification |
| 1.1.0 | 4 November 2026 | Added the separate joint k/C/lambda server protocol, k-specific feasibility bounds, explicit no-reuse mode, atomic checkpoints, periodic aggregate refresh, single-writer locking, held-out handoff, and 83-test verification |
| 1.0.0 | 4 November 2026 | Added the mathematically bounded inclusive cluster-range protocol, explicit comparison and held-out seed separation, privacy-first selection, selection-aware final-candidate command, a clear pending-results boundary, and 73-test verification |
| 0.9.1 | 4 November 2026 | Archived the unlabeled legacy seed-9 bundle without changing content, switched defaults and tests to labelled run directories, and marked results-v1 as a retained technical appendix |
| 0.9.0 | 4 November 2026 | Added the 106-attribute aggregate-only readiness screen, explicit human-review gates, the decision not to run the remaining 181 grid points for current claims, the downstream-utility claim boundary, and 57-test verification |
| 0.8.0 | 3 November 2026 | Added the consolidated results-v2 package, full 39-of-220 sensitivity coverage accounting, package generator and validation map, 54-test snapshot, GNUPlot provenance, and an explicit distinction between seed-9 baseline concerns and held-out-candidate results |
| 0.7.0 | 2 November 2026 | Added the prospective final-selection policy, held-out seed-10 privacy candidate, screening results, patient-level output locations, QID replacement effects, and release boundaries |
| 0.6.0 | 1 November 2026 | Added the predeclared four-configuration, ten-seed robustness workflow; recorded 34 completed and 6 no-feasible-particle outcomes; integrated the robustness table into the reproducible results package |
| 0.5.0 | 1 November 2026 | Completed the predeclared 18-configuration one-factor study; recorded nine completed and nine no-feasible-particle outcomes; added reproducible aggregate result tables and the write-up-ready results document |
| 0.4.0 | 31 October 2026 | Added the aggregate-only, resumable PRTI-GH sensitivity workflow derived from Wei Appendix A; recorded the 220-point plan, 20 mathematically infeasible settings, and first completed full checkpoint; updated verification coverage |
| 0.3.0 | 27 October 2026 | Corrected cluster/equivalence-class reporting; retained the complete replacement seed screen; added generation-time Git, command, timestamp, and hash provenance; and recorded exact fresh reproduction of the legacy seed-9 result |
| 0.2.0 | 27 October 2026 | Documented intentional duplicate preservation; added refreshable run provenance and artifact hashes; added duplicate-preservation, manifest-safety, and recorded-result regression coverage |
| 0.1.0 | 27 October 2026 | Initial consolidated documentation of the current PRTI-GH data preparation, MO-OBAM formulation, implementation, recorded run, outputs, validation, and limitations |
