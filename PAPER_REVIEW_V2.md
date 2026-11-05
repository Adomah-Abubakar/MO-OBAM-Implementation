# MO-OBAM arXiv v2 review

Reviewed source: Wei et al., arXiv:2501.01002v2, revised 15 May 2025.
The repository was previously based on the January 2025 v1 PDF. This review
records which v2 statements are implemented and which ambiguities remain.
Implementation cross-check refreshed: 5 November 2026.

## Formulation implemented

| Paper element | Implementation |
|---|---|
| One cluster per record, Equation 9 | Binary membership validation |
| Numeric and categorical deviation, Equations 10-14 | Squared numeric distance plus categorical mismatch; IL is divided by original-data deviation |
| Multiple sensitive attributes, Equations 15-19 | Entropy per cluster and SA; minimum over clusters is taken separately for each SA, then summed |
| k-anonymity, Equation 20 | Every configured cluster must contain at least `k` records |
| Active objective | Minimize `IL - lambda * entropy_sum` |
| Released QIDs | Replace QIDs with their assigned centroid; preserve SAs and other columns |

The v1 scaled expression using `MIN(IL)` and `MAX(entropy)` is commented out in
the v2 source and is not used. The former three-stage reference estimation has
therefore been removed.

## Algorithm 1 implemented

The scalable solver represents each particle as the centroid matrix `q`,
initializes centroid entries from observed QI values, assigns each record to
the nearest centroid using Equation 12, evaluates the direct objective, and
updates personal/global best particles. The continuous and categorical PSO
updates are based on the authors' public `particle_swarm.py`.

## Source conflicts and explicit boundaries

1. **k-violation sign.** Algorithm 1 prints
   `IL - lambda * entropy - l_multi * violation`. Because the algorithm
   minimizes fitness, a positive violation would reduce fitness and be
   rewarded. The authors' Python instead adds a positive normalized violation.
   Equation 20 is an explicit hard constraint. This implementation enforces
   Equation 20 and does not invent an Adult penalty multiplier.

2. **Optional normalization.** The paper says normalization may be required but
   provides no mathematical definition or reported Adult normalization bounds.
   The implementation evaluates the active objective literally. Reproducing a
   hidden normalization would require additional author-supplied settings.

3. **Entropy in the public code.** The paper takes the minimum entropy per SA
   and then sums those minima. The public utility code instead sums SAs inside
   each cluster and then takes one minimum. Those expressions differ for more
   than one SA. The implementation follows Equations 18-19.

4. **Linkage threshold.** Section II-B says risk "exceeds" a threshold but
   prints `1 / |E_g| >= tau`. Adult Table 6 is reproduced only with strict `>`:
   inclusive comparison adds 280 records at `tau=0.05` and 340 records at
   `tau=0.1`. Reporting retains strict `>` and documents why.

5. **Particle initialization.** Algorithm 1 says to initialize centroids by
   selecting values from the original QIs. The public code instead samples
   numeric values from the integer range between column bounds. The scalable
   solver follows the printed algorithm and samples observed values.

6. **Reproduction boundary.** Population size and iteration count are available,
   but the paper does not publish every random seed, k-penalty multiplier, or a
   complete Adult experiment script. Matching Table 6 exactly is therefore a
   reproduction target, not a guaranteed property of one stochastic run.

7. **Large-cluster feasibility.** Equation 20 requires every configured cluster,
   including an empty one, to contain at least `k` records. The authors' public
   `calculate_k_constraint` checks only cluster labels present in
   `value_counts()`, so empty configured clusters are omitted. The notebook's
   visible example also uses `l_multi_k = 0`, while arXiv v2 does not publish the
   Adult multiplier used for Table 6. Random initialization at `n_C = 100`
   produced no hard-k feasible particle in our controlled smoke test. The grid
   runner records this as blocked instead of using the code's empty-cluster
   omission or inventing a repair/penalty.

## Non-ML experimental coverage

`run_adult_experiments.py` now records the exact Adult Appendix A grid (220
configurations), the eight MO-OBAM boundary configurations highlighted in
Table 6, paper attack thresholds, published comparison values, resumable
checkpoints, and full before/after export for a selected completed run. Machine
learning is deliberately excluded from this phase.

The Adult baseline matches the published 45,222 records, 1,900 equivalence
classes, linkage counts 6,506/4,906/3,910, and 634 homogeneity-risk records. A
full `k=5`, `n_C=4`, `lambda=1`, population-100, iteration-100 run remains
bit-for-bit unchanged after lossless QID-profile acceleration: information loss
is `0.1136623462`, versus the paper's `0.1074`, with zero reported linkage and
homogeneity risk.

The completed Table 6 attempt gives the same observed result for all four
high-protection `k` values (5, 10, 15, and 20), matching the paper on four
equivalence classes and zero linkage/homogeneity-risk records, while retaining
the `0.0062623462` information-loss gap. All four low-protection `n_C=100`
settings are recorded as blocked for the documented initialization/penalty
reason; no values were substituted from the paper.

This is not yet the paper's entire empirical study. German Credit and Sepsis
source data are absent locally. The cited Domingo-Ferrer/Torra and Zheng et al.
comparison procedures are not implemented because the MO-OBAM paper and its
public repository do not provide enough implementation detail to reproduce
their exact data handling, distance choices, and tie-breaking without adding
assumptions.

## Current PRTI-GH adaptation boundary

The PRTI-GH implementation applies the same active v2 equations to 29 QIDs and
77 SAs. It preserves every SA row by row, replaces every QID with its assigned
cluster representative, and evaluates all 77 SAs through the printed
`sum(min(entropy))` formulation. The PRTI-GH role assignment, missing-state
policy, seeds, robustness design, and held-out screening thresholds are
documented project adaptations; they are not settings reported by Wei et al.

The prospectively declared held-out seed-10 candidate used four clusters and
passed all ten internal screening checks, but it is not approved for
unrestricted release. The current aggregate-only evidence package is
[`evidence/prti-results-v2/`](evidence/prti-results-v2/); its patient-level
inputs remain under the ignored `outputs/` directory.

## Verification

The exact solver checks the direct objective on small data by enumerating all
feasible unlabeled partitions. The scalable solver is tested for deterministic
results under a fixed seed, feasibility, metric agreement with the formulation,
and proximity to the exact small-data optimum. Adult preparation and baseline
privacy counts are also regression-tested against Table 6.

The current repository verification snapshot contains 94 passing tests,
including the held-out PRTI-GH policy, joint-grid feasibility and recovery,
aggregate results-package boundaries, and the additive multi-objective
extension's Pareto, hard-k, immutable-plan, SQLite, and Excel-report checks.

Sources:

- [arXiv abstract and revision history](https://arxiv.org/abs/2501.01002)
- [arXiv v2 HTML](https://arxiv.org/html/2501.01002v2)
- [Authors' public repository](https://github.com/yw825/Multi-Objective-Optimization-based-Anonymization-Model)
