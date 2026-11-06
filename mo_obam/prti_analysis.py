"""Post-anonymization analysis for PRTI-GH releases.

This module measures saved results; it does not alter Wei's optimizer or data.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from .privacy_metrics import PAPER_LINKAGE_THRESHOLDS
from .schema import MOOBAMSchema


@dataclass(frozen=True)
class PRTIAnalysisBundle:
    """Return generated paths and headline metrics to the calling runner."""

    paths: dict[str, Path]
    metrics: dict[str, object]


def _class_information(data: pd.DataFrame, qids: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    class_id = data.groupby(list(qids), dropna=False, observed=True, sort=False).ngroup()
    sizes = class_id.value_counts(sort=False).sort_index()
    return class_id, sizes


def _risk_row(label: str, data: pd.DataFrame, schema: MOOBAMSchema) -> dict[str, object]:
    class_id, sizes = _class_information(data, schema.qids)
    record_sizes = class_id.map(sizes).to_numpy(dtype=float)
    risks = 1.0 / record_sizes
    row: dict[str, object] = {
        "dataset_version": label,
        "records": len(data),
        "equivalence_classes": len(sizes),
        "minimum_class_size": int(sizes.min()),
        "median_class_size": float(sizes.median()),
        "mean_class_size": float(sizes.mean()),
        "maximum_class_size": int(sizes.max()),
        "prosecutor_risk_maximum": float(risks.max()),
        "prosecutor_risk_mean": float(risks.mean()),
        "prosecutor_risk_median": float(np.median(risks)),
        # Average sample-record match probability; algebraically G / N.
        "marketer_risk": float(len(sizes) / len(data)),
        "marketer_expected_correct_matches": int(len(sizes)),
    }
    for threshold in PAPER_LINKAGE_THRESHOLDS:
        key = str(threshold).replace(".", "_")
        exposed = int(np.count_nonzero(risks > threshold))
        row[f"records_risk_above_{key}"] = exposed
        row[f"fraction_risk_above_{key}"] = exposed / len(data)
    return row


def risk_analysis(
    before: pd.DataFrame, after: pd.DataFrame, schema: MOOBAMSchema
) -> pd.DataFrame:
    """Calculate sample-based prosecutor and marketer risk before and after."""

    rows = [_risk_row("before", before, schema), _risk_row("after", after, schema)]
    frame = pd.DataFrame(rows)
    before_row, after_row = rows
    change: dict[str, object] = {"dataset_version": "change_after_minus_before"}
    for column in frame.columns:
        if column == "dataset_version":
            continue
        change[column] = float(after_row[column]) - float(before_row[column])
    return pd.concat([frame, pd.DataFrame([change])], ignore_index=True)


def diversity_analysis(data: pd.DataFrame, schema: MOOBAMSchema) -> pd.DataFrame:
    """Measure distinct and entropy-effective diversity in released QID classes."""

    class_id, sizes = _class_information(data, schema.qids)
    working = data.loc[:, schema.sensitive_attributes].copy()
    working.insert(0, "_class", class_id.to_numpy())
    grouped = working.groupby("_class", sort=False, observed=True)
    rows: list[dict[str, object]] = []
    for attribute in schema.sensitive_attributes:
        distinct = grouped[attribute].nunique(dropna=False)
        effective: list[float] = []
        for _, values in grouped[attribute]:
            probabilities = values.value_counts(dropna=False, normalize=True).to_numpy()
            entropy = -float(np.sum(probabilities * np.log(probabilities)))
            effective.append(float(np.exp(entropy)))
        below_two = distinct.index[distinct < 2]
        below_three = distinct.index[distinct < 3]
        rows.append(
            {
                "sensitive_attribute": attribute,
                "minimum_distinct_l": int(distinct.min()),
                "median_distinct_l": float(distinct.median()),
                "minimum_effective_l": float(min(effective)),
                "classes": len(distinct),
                "classes_l_at_least_2": int((distinct >= 2).sum()),
                "classes_l_at_least_3": int((distinct >= 3).sum()),
                "records_in_classes_below_l2": int(sizes.loc[below_two].sum()) if len(below_two) else 0,
                "records_in_classes_below_l3": int(sizes.loc[below_three].sum()) if len(below_three) else 0,
            }
        )
    return pd.DataFrame(rows)


def _categorical_distances(before: pd.Series, after: pd.Series) -> tuple[float, float]:
    left = before.astype("string").fillna("<MISSING>").value_counts(normalize=True)
    right = after.astype("string").fillna("<MISSING>").value_counts(normalize=True)
    labels = left.index.union(right.index)
    p = left.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    q = right.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    return float(0.5 * np.abs(p - q).sum()), float(jensenshannon(p, q, base=2.0))


def distribution_analysis(
    before: pd.DataFrame, after: pd.DataFrame, schema: MOOBAMSchema
) -> pd.DataFrame:
    """Measure per-attribute pre/post distribution changes with typed metrics."""

    rows: list[dict[str, object]] = []
    qids = set(schema.qids)
    for attribute in before.columns:
        left, right = before[attribute], after[attribute]
        equal = left.eq(right) | (left.isna() & right.isna())
        row: dict[str, object] = {
            "attribute": attribute,
            "role": "quasi-identifier" if attribute in qids else "sensitive attribute",
            "metric_family": "numeric" if pd.api.types.is_numeric_dtype(left) else "categorical",
            "records": len(left),
            "changed_records": int((~equal).sum()),
            "changed_fraction": float((~equal).mean()),
            "missing_before": int(left.isna().sum()),
            "missing_after": int(right.isna().sum()),
            "distinct_before": int(left.nunique(dropna=False)),
            "distinct_after": int(right.nunique(dropna=False)),
            "normalized_wasserstein": np.nan,
            "ks_statistic": np.nan,
            "mean_before": np.nan,
            "mean_after": np.nan,
            "median_before": np.nan,
            "median_after": np.nan,
            "total_variation_distance": np.nan,
            "jensen_shannon_distance": np.nan,
        }
        if pd.api.types.is_numeric_dtype(left):
            a = pd.to_numeric(left, errors="coerce").dropna().to_numpy(dtype=float)
            b = pd.to_numeric(right, errors="coerce").dropna().to_numpy(dtype=float)
            if len(a) and len(b):
                scale = float(np.max(a) - np.min(a))
                raw_wasserstein = float(wasserstein_distance(a, b))
                row.update(
                    {
                        "normalized_wasserstein": raw_wasserstein / scale if scale else (0.0 if raw_wasserstein == 0 else np.nan),
                        "ks_statistic": float(ks_2samp(a, b).statistic),
                        "mean_before": float(np.mean(a)),
                        "mean_after": float(np.mean(b)),
                        "median_before": float(np.median(a)),
                        "median_after": float(np.median(b)),
                    }
                )
        else:
            tvd, jsd = _categorical_distances(left, right)
            row["total_variation_distance"] = tvd
            row["jensen_shannon_distance"] = jsd
        rows.append(row)
    return pd.DataFrame(rows)


def data_dictionary(data: pd.DataFrame, schema: MOOBAMSchema) -> pd.DataFrame:
    """Create a value-free technical data dictionary for the release."""

    qids = set(schema.qids)
    return pd.DataFrame(
        [
            {
                "position": position,
                "attribute": attribute,
                "role": "quasi-identifier" if attribute in qids else "sensitive attribute",
                "source_dtype": str(data[attribute].dtype),
                "missing_values": int(data[attribute].isna().sum()),
                "distinct_values": int(data[attribute].nunique(dropna=False)),
                "wei_treatment": (
                    "replaced by released cluster representative"
                    if attribute in qids
                    else "preserved row by row; included in entropy analysis"
                ),
            }
            for position, attribute in enumerate(data.columns, start=1)
        ]
    )


def _bar_svg(title: str, labels: list[str], values: list[float], path: Path) -> None:
    width, height, margin = 760, 420, 70
    maximum = max(values) if values and max(values) > 0 else 1.0
    usable = width - 2 * margin
    bar_width = usable / max(len(values), 1) * 0.62
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{escape(title)}</text>',
        f'<line x1="{margin}" y1="350" x2="{width-margin}" y2="350" stroke="#334155"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = margin + (index + 0.5) * usable / len(values) - bar_width / 2
        bar_height = 260 * value / maximum
        y = 350 - bar_height
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#0f766e"/>',
                f'<text x="{x + bar_width/2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.4g}</text>',
                f'<text x="{x + bar_width/2:.1f}" y="370" text-anchor="middle" font-family="sans-serif" font-size="12">{escape(label)}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def generate_analysis_bundle(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    schema: MOOBAMSchema,
    information_loss: float,
    configured_k: int,
    configuration: dict[str, object],
    output_directory: Path,
) -> PRTIAnalysisBundle:
    """Generate tables, report, and figures from one completed release."""

    output_directory.mkdir(parents=True, exist_ok=True)
    figures = output_directory / "figures"
    figures.mkdir(exist_ok=True)
    risk = risk_analysis(before, after, schema)
    diversity = diversity_analysis(after, schema)
    distributions = distribution_analysis(before, after, schema)
    dictionary = data_dictionary(before, schema)
    after_risk = risk.loc[risk["dataset_version"].eq("after")].iloc[0]
    summary = pd.DataFrame(
        [
            {"metric": "records", "value": len(after), "unit": "records"},
            {"metric": "quasi_identifiers", "value": len(schema.qids), "unit": "attributes"},
            {"metric": "sensitive_attributes", "value": len(schema.sensitive_attributes), "unit": "attributes"},
            {"metric": "configured_k", "value": configured_k, "unit": "records"},
            {"metric": "achieved_k", "value": after_risk["minimum_class_size"], "unit": "records"},
            {"metric": "equivalence_classes", "value": after_risk["equivalence_classes"], "unit": "classes"},
            {"metric": "information_loss", "value": information_loss, "unit": "Wei Equation 14 ratio"},
            {"metric": "prosecutor_risk_maximum", "value": after_risk["prosecutor_risk_maximum"], "unit": "probability"},
            {"metric": "marketer_risk", "value": after_risk["marketer_risk"], "unit": "expected match fraction"},
            {"metric": "minimum_distinct_l", "value": diversity["minimum_distinct_l"].min(), "unit": "distinct SA values"},
            {"metric": "minimum_effective_l", "value": diversity["minimum_effective_l"].min(), "unit": "exp(entropy)"},
        ]
    )
    paths = {
        "distribution_analysis": output_directory / "prti_distribution_analysis.csv",
        "summary_statistics": output_directory / "prti_summary_statistics.csv",
        "reidentification_risk": output_directory / "prti_reidentification_risk.csv",
        "diversity_analysis": output_directory / "prti_l_diversity.csv",
        "data_dictionary": output_directory / "prti_data_dictionary.csv",
        "anonymization_report": output_directory / "prti_anonymization_report.md",
        "risk_figure": figures / "reidentification_risk_before_after.svg",
        "class_figure": figures / "equivalence_class_summary.svg",
        "distribution_figure": figures / "distribution_similarity.svg",
    }
    distributions.to_csv(paths["distribution_analysis"], index=False)
    summary.to_csv(paths["summary_statistics"], index=False)
    risk.to_csv(paths["reidentification_risk"], index=False)
    diversity.to_csv(paths["diversity_analysis"], index=False)
    dictionary.to_csv(paths["data_dictionary"], index=False)

    before_risk = risk.loc[risk["dataset_version"].eq("before")].iloc[0]
    report = f"""# PRTI-GH anonymization report

## Method and configuration

Wei/MO-OBAM was used without adding suppression, masking, differential privacy, or another anonymization transformation.

```json
{json.dumps(configuration, indent=2)}
```

## Headline results

| Metric | Before | After |
|---|---:|---:|
| Equivalence classes | {int(before_risk['equivalence_classes'])} | {int(after_risk['equivalence_classes'])} |
| Achieved k | {int(before_risk['minimum_class_size'])} | {int(after_risk['minimum_class_size'])} |
| Maximum prosecutor risk | {before_risk['prosecutor_risk_maximum']:.8f} | {after_risk['prosecutor_risk_maximum']:.8f} |
| Mean prosecutor risk | {before_risk['prosecutor_risk_mean']:.8f} | {after_risk['prosecutor_risk_mean']:.8f} |
| Marketer risk | {before_risk['marketer_risk']:.8f} | {after_risk['marketer_risk']:.8f} |
| Information loss | not applicable | {information_loss:.10f} |

The prosecutor and marketer values are sample-based equivalence-class risks. Journalist risk is not estimated because population QID frequencies and the sampling fraction are unavailable.

## Diversity

- Minimum distinct l-diversity across all released classes and sensitive attributes: **{int(diversity['minimum_distinct_l'].min())}**.
- Minimum entropy-effective diversity, `exp(H)`: **{diversity['minimum_effective_l'].min():.6f}**.
- This is measured evidence, not an added l-diversity constraint in Wei's optimizer.

## Distribution similarity

- QIDs with changed records: **{int((distributions.loc[distributions['role'].eq('quasi-identifier'), 'changed_records'] > 0).sum())}** of {len(schema.qids)}.
- Sensitive attributes with changed records: **{int((distributions.loc[distributions['role'].eq('sensitive attribute'), 'changed_records'] > 0).sum())}** of {len(schema.sensitive_attributes)}.
- Numeric attributes use normalized Wasserstein distance and the KS statistic.
- Categorical attributes use total variation and Jensen-Shannon distance.

## Release boundary

These measurements support technical review but do not replace custodian, governance, clinical, or independent disclosure-control approval.
"""
    paths["anonymization_report"].write_text(report, encoding="utf-8")

    _bar_svg(
        "Re-identification risk before and after",
        ["Prosecutor before", "Prosecutor after", "Marketer before", "Marketer after"],
        [
            float(before_risk["prosecutor_risk_maximum"]),
            float(after_risk["prosecutor_risk_maximum"]),
            float(before_risk["marketer_risk"]),
            float(after_risk["marketer_risk"]),
        ],
        paths["risk_figure"],
    )
    _bar_svg(
        "Released equivalence-class summary",
        ["Minimum", "Median", "Mean", "Maximum"],
        [float(after_risk[name]) for name in ("minimum_class_size", "median_class_size", "mean_class_size", "maximum_class_size")],
        paths["class_figure"],
    )
    qid_distances = distributions.loc[distributions["role"].eq("quasi-identifier")]
    _bar_svg(
        "Mean QID distribution distance",
        ["Wasserstein", "KS", "Total variation", "Jensen-Shannon"],
        [
            float(qid_distances[column].dropna().mean()) if qid_distances[column].notna().any() else 0.0
            for column in ("normalized_wasserstein", "ks_statistic", "total_variation_distance", "jensen_shannon_distance")
        ],
        paths["distribution_figure"],
    )
    metrics = {
        "configured_k": configured_k,
        "achieved_k": int(after_risk["minimum_class_size"]),
        "minimum_distinct_l": int(diversity["minimum_distinct_l"].min()),
        "minimum_effective_l": float(diversity["minimum_effective_l"].min()),
        "prosecutor_risk_before": float(before_risk["prosecutor_risk_maximum"]),
        "prosecutor_risk_after": float(after_risk["prosecutor_risk_maximum"]),
        "marketer_risk_before": float(before_risk["marketer_risk"]),
        "marketer_risk_after": float(after_risk["marketer_risk"]),
        "journalist_risk": None,
        "journalist_risk_status": "not estimable without population QID frequencies and sampling fraction",
        "information_loss": information_loss,
    }
    return PRTIAnalysisBundle(paths=paths, metrics=metrics)
