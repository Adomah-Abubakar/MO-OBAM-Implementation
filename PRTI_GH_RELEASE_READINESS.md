# PRTI-GH release readiness

## Current decision

The held-out candidate remains **blocked from external or unrestricted
release**. Passing the project screening checks is technical evidence, not a
data-custodian approval or an independent disclosure-control decision.

## Experiment-scope decision

The earlier decision not to run the remaining 181 points of the paper-derived
220-point k/C/lambda grid still applies to the bounded historical claims. The
new user-authorized inclusive cluster-range study is a different, one-factor
follow-up: it varies C only while keeping k, lambda and PSO settings fixed. It
must be labelled as new project evidence, not completion of Wei et al.'s
Appendix A grid.

The follow-up can support a claim of the best observed eligible C within its
declared range, comparison seeds and search budget. It cannot support a claim
of global optimality. Its partial checkpoints must not be interpreted until
every planned cluster/seed result is recorded and the prospectively reserved
held-out seed is evaluated once.

A separate scalar joint k/C/lambda implementation is available for a broader
server study. The documented expanded plan contains 2,145 configurations and
21,450 comparison runs. Its v3 server evidence is not tracked in this checkout,
so it does not alter the historical or fixed-range claims above.

An additive multi-objective implementation is also available for C=1 through
C=30 across five k values and ten comparison seeds. It searches information
loss and sensitive-attribute entropy jointly and analyses lambda after the
Pareto search. No result is recorded in this repository. It must complete all
1,500 aggregate comparison runs before selection, and it does not itself
generate or approve patient-level output.

## Downstream-utility decision

No downstream machine-learning or clinical-task experiment is required for the
current anonymization and disclosure-metric claims. The project must continue
to state that downstream utility preservation has **not** been established.

Before starting a utility experiment, record:

1. the intended downstream user and task;
2. the target outcome and whether it is suitable for prediction;
3. the train/test or cross-validation protocol;
4. the baseline and anonymized-data metrics; and
5. an acceptance threshold declared before results are inspected.

## Automated attribute screen

Run:

```bash
.venv/bin/python generate_prti_release_readiness.py
```

The command reads the unchanged reviewed workbook and the held-out candidate's
input audit. It writes aggregate-only evidence under
`evidence/prti-release-readiness-v1/`:

- `prti_attribute_role_review.csv`: one row per source attribute, current role,
  cardinality/missingness, value-free identifier and free-text indicators, and
  empty fields for named human reviewers; and
- `prti_release_readiness.json`: the experiment-scope decision, utility-scope
  decision, technical-screen summary, and unresolved release gates.

The automated screen exports no patient values. It is triage evidence only and
cannot approve an attribute role.

## Required human and institutional reviews

The following gates remain open:

- Data custodian: document the intended recipient, purpose, linkage adversary,
  external data reasonably available to that adversary, access controls,
  retention period, and permitted onward sharing.
- Clinical reviewer: assess every current QID/SA role and any proposed
  representation or category grouping.
- Privacy/data-governance reviewer: assess direct identifiers, indirect
  identifiers, free text, rare categories, high-cardinality attributes, and
  whether row-level sensitive attributes may be released to the intended
  recipient.
- Independent disclosure-control specialist: review the actual candidate in a
  controlled environment and either approve a bounded release scenario or
  specify required changes.

Reviewers must complete the `PENDING` fields in the attribute register, record
their name and review date, and sign a separate institutional decision. The
repository must not convert pending entries to approved without that evidence.

## Metadata and snapshot boundary

Repository authorship, the public software licence, citation identity, and any
DOI remain owner decisions. A Git snapshot tag may preserve the research state,
but it must not be described as approval to publish the patient-level candidate.
