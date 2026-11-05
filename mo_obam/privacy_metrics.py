"""Privacy-risk measurements used in the MO-OBAM evaluation tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import MOOBAMSchema


PAPER_LINKAGE_THRESHOLDS = (0.05, 0.075, 0.1)


@dataclass(frozen=True)
class PrivacyRiskMetrics:
    """Summarize equivalence classes and the paper's two attack measures."""

    equivalence_classes: int
    minimum_equivalence_class_size: int
    maximum_equivalence_class_size: int
    linkage_attack_records: dict[str, int]
    homogeneity_attack_records: int
    homogeneity_attack_records_by_sensitive_attribute: dict[str, int]


class PrivacyRiskEvaluator:
    """Evaluate linkage and homogeneity risk from released QID combinations.

    The implementation uses strict ``1 / class_size > threshold`` because that
    exactly reproduces the published Adult baseline counts. Section II-B of
    arXiv v2 says "exceeds" but prints ``>=``; the inclusive operator would add
    the Adult classes of size exactly 20 and 10 and would not match Table 6.
    A record is homogeneity-vulnerable when its class has one SA value. For
    multiple SAs, per-SA counts and their record-level union are both reported;
    the Adult experiment has one SA, so these definitions are identical.
    """

    def __init__(self, data: pd.DataFrame, schema: MOOBAMSchema) -> None:
        self.data = data.copy(deep=False)
        self.schema = schema
        missing = sorted(set(schema.protected_columns) - set(data.columns))
        if missing:
            raise ValueError(f"Columns missing from privacy evaluation: {missing}")
        if data.empty:
            raise ValueError("Privacy evaluation requires at least one record.")

    def evaluate(
        self,
        thresholds: tuple[float, ...] = PAPER_LINKAGE_THRESHOLDS,
    ) -> PrivacyRiskMetrics:
        """Return the attack counts described in Section V-C of the paper."""

        for threshold in thresholds:
            if not 0.0 < threshold <= 1.0:
                raise ValueError("Linkage thresholds must be in the interval (0, 1].")

        class_ids = self.data.groupby(
            list(self.schema.qids),
            dropna=False,
            observed=True,
            sort=False,
        ).ngroup()
        class_sizes = class_ids.value_counts(sort=False).sort_index()
        record_class_sizes = class_ids.map(class_sizes).to_numpy(dtype=float)
        record_risk = 1.0 / record_class_sizes
        linkage_counts = {
            self.threshold_label(threshold): int(
                np.count_nonzero(record_risk > threshold)
            )
            for threshold in thresholds
        }

        sensitive_with_class = self.data.loc[
            :, self.schema.sensitive_attributes
        ].copy()
        sensitive_with_class.insert(0, "_equivalence_class", class_ids.to_numpy())
        distinct_by_class = sensitive_with_class.groupby(
            "_equivalence_class", sort=False
        )[list(self.schema.sensitive_attributes)].nunique(dropna=False)
        homogeneous_by_sa = distinct_by_class.eq(1)
        homogeneity_by_sa = {
            column: int(
                class_sizes.loc[
                    homogeneous_by_sa.index[homogeneous_by_sa[column]]
                ].sum()
            )
            for column in self.schema.sensitive_attributes
        }
        homogeneous_for_any_sa = homogeneous_by_sa.any(axis=1)
        homogeneity_union = int(
            class_sizes.loc[homogeneous_for_any_sa.index[homogeneous_for_any_sa]].sum()
        )

        return PrivacyRiskMetrics(
            equivalence_classes=int(len(class_sizes)),
            minimum_equivalence_class_size=int(class_sizes.min()),
            maximum_equivalence_class_size=int(class_sizes.max()),
            linkage_attack_records=linkage_counts,
            homogeneity_attack_records=homogeneity_union,
            homogeneity_attack_records_by_sensitive_attribute=homogeneity_by_sa,
        )

    @staticmethod
    def threshold_label(threshold: float) -> str:
        """Return a stable CSV/JSON label for a paper linkage threshold."""

        return format(threshold, "g")
