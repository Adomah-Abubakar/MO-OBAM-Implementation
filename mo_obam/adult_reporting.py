"""Create complete Adult release data and readable reproduction tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .adult_data import ADULT_COLUMNS
from .privacy_metrics import PrivacyRiskEvaluator, PrivacyRiskMetrics
from .schema import MOOBAMConfig, MOOBAMSchema


# Published values from the high-protection Adult rows in arXiv v2 Table 6(b).
PAPER_ADULT_TABLE_VB = {
    "prepared_records": 45_222,
    "baseline_equivalence_classes": 1_900,
    "baseline_linkage_0.05": 6_506,
    "baseline_linkage_0.075": 4_906,
    "baseline_linkage_0.1": 3_910,
    "baseline_homogeneity": 634,
    "mo_obam_equivalence_classes": 4,
    "mo_obam_information_loss": 0.1074,
    "mo_obam_linkage_0.05": 0,
    "mo_obam_linkage_0.075": 0,
    "mo_obam_linkage_0.1": 0,
    "mo_obam_homogeneity": 0,
}

# The two MO-OBAM boundary settings printed in Adult Table 6.  The paper
# reports the same values for k=5, 10, 15, and 20 at each boundary.
PAPER_ADULT_MO_OBAM_BOUNDARIES = {
    (100, 0.0001): {
        "section": "Table 6(a), lower homogeneity protection",
        "equivalence_classes": 100,
        "information_loss": 0.0116,
        "linkage_0.05": 103,
        "linkage_0.075": 69,
        "linkage_0.1": 47,
        "homogeneity": 3,
    },
    (4, 1.0): {
        "section": "Table 6(b), higher homogeneity protection",
        "equivalence_classes": 4,
        "information_loss": 0.1074,
        "linkage_0.05": 0,
        "linkage_0.075": 0,
        "linkage_0.1": 0,
        "homogeneity": 0,
    },
}


@dataclass(frozen=True)
class AdultReportArtifacts:
    """Hold final release data and the three human-readable result tables."""

    anonymized_final: pd.DataFrame
    model_evaluation: pd.DataFrame
    paper_comparison: pd.DataFrame
    cluster_details: pd.DataFrame
    baseline_risk: PrivacyRiskMetrics
    anonymized_risk: PrivacyRiskMetrics


class AdultOutputReporter:
    """Build, save, and display Adult artifacts without changing source data."""

    FINAL_FILE = "adult_anonymized_final.csv"
    EVALUATION_FILE = "adult_model_evaluation.csv"
    COMPARISON_FILE = "adult_paper_comparison.csv"
    CLUSTERS_FILE = "adult_cluster_details.csv"

    def __init__(self, schema: MOOBAMSchema) -> None:
        self.schema = schema

    def build(
        self,
        original: pd.DataFrame,
        anonymized_with_cluster: pd.DataFrame,
        model_config: MOOBAMConfig,
        information_loss: float,
        entropy_sum: float,
    ) -> AdultReportArtifacts:
        """Create the full release and paper-style evaluation tables."""

        self._validate_frames(original, anonymized_with_cluster)
        anonymized_final = anonymized_with_cluster.loc[:, ADULT_COLUMNS].copy()
        baseline_risk = PrivacyRiskEvaluator(original, self.schema).evaluate()
        anonymized_risk = PrivacyRiskEvaluator(
            anonymized_final, self.schema
        ).evaluate()

        model_evaluation = self._model_evaluation_table(
            baseline_risk,
            anonymized_risk,
            model_config,
            information_loss,
            entropy_sum,
        )
        paper_comparison = self._paper_comparison_table(
            len(original),
            baseline_risk,
            anonymized_risk,
            model_config,
            information_loss,
        )
        cluster_details = self._cluster_details(anonymized_with_cluster)
        return AdultReportArtifacts(
            anonymized_final=anonymized_final,
            model_evaluation=model_evaluation,
            paper_comparison=paper_comparison,
            cluster_details=cluster_details,
            baseline_risk=baseline_risk,
            anonymized_risk=anonymized_risk,
        )

    def write(
        self,
        output_directory: Path,
        artifacts: AdultReportArtifacts,
    ) -> dict[str, str]:
        """Write the release and tables, returning summary-friendly filenames."""

        output_directory.mkdir(parents=True, exist_ok=True)
        artifacts.anonymized_final.to_csv(
            output_directory / self.FINAL_FILE, index=False
        )
        artifacts.model_evaluation.to_csv(
            output_directory / self.EVALUATION_FILE, index=False
        )
        artifacts.paper_comparison.to_csv(
            output_directory / self.COMPARISON_FILE, index=False
        )
        artifacts.cluster_details.to_csv(
            output_directory / self.CLUSTERS_FILE, index=False
        )
        return {
            "anonymized_final": self.FINAL_FILE,
            "model_evaluation": self.EVALUATION_FILE,
            "paper_comparison": self.COMPARISON_FILE,
            "cluster_details": self.CLUSTERS_FILE,
        }

    @classmethod
    def print_saved_tables(cls, output_directory: Path) -> None:
        """Print all saved CSV tables without rerunning the optimizer."""

        tables = (
            ("Observed Adult model evaluation", cls.EVALUATION_FILE),
            ("Comparison with the paper's Adult Table 6", cls.COMPARISON_FILE),
            ("Final cluster details", cls.CLUSTERS_FILE),
        )
        for title, filename in tables:
            path = output_directory / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run the experiment or use --report-existing first."
                )
            table = pd.read_csv(path)
            print(f"\n{title}\n{'=' * len(title)}")
            print(
                table.to_string(
                    index=False,
                    na_rep="-",
                    float_format=lambda value: f"{value:.6f}",
                )
            )

    @staticmethod
    def risk_as_dict(metrics: PrivacyRiskMetrics) -> dict[str, object]:
        """Convert nested metrics to a JSON-serializable dictionary."""

        return asdict(metrics)

    def reconstruct_from_audit(
        self,
        original: pd.DataFrame,
        before_after: pd.DataFrame,
    ) -> pd.DataFrame:
        """Reconstruct the full anonymized frame from an earlier audit output."""

        if len(original) != len(before_after):
            raise ValueError("Prepared-original and audit row counts do not match.")
        expected_positions = np.arange(len(original))
        if "record_position" not in before_after or not np.array_equal(
            before_after["record_position"].to_numpy(), expected_positions
        ):
            raise ValueError("Audit record positions are missing or out of order.")
        if "cluster" not in before_after:
            raise ValueError("Audit output does not contain cluster assignments.")

        reconstructed = pd.DataFrame(index=original.index)
        for column in ADULT_COLUMNS:
            source = f"{column}_after" if column in self.schema.qids else column
            if source not in before_after:
                raise ValueError(f"Audit output is missing {source!r}.")
            reconstructed[column] = before_after[source].to_numpy()
        reconstructed["cluster"] = before_after["cluster"].to_numpy()
        return reconstructed

    def _validate_frames(
        self,
        original: pd.DataFrame,
        anonymized_with_cluster: pd.DataFrame,
    ) -> None:
        if tuple(original.columns) != ADULT_COLUMNS:
            raise ValueError("Prepared Adult columns are not in the expected original order.")
        expected = set(ADULT_COLUMNS) | {"cluster"}
        missing = sorted(expected - set(anonymized_with_cluster.columns))
        if missing:
            raise ValueError(f"Anonymized Adult data is missing columns: {missing}")
        if len(original) != len(anonymized_with_cluster):
            raise ValueError("Original and anonymized Adult row counts do not match.")
        non_qids = [column for column in ADULT_COLUMNS if column not in self.schema.qids]
        changed = [
            column
            for column in non_qids
            if not original[column].equals(anonymized_with_cluster[column])
        ]
        if changed:
            raise ValueError(f"Non-QID columns changed during anonymization: {changed}")

    @staticmethod
    def _risk_columns(metrics: PrivacyRiskMetrics) -> dict[str, int]:
        return {
            "linkage_attack_tau_0.05": metrics.linkage_attack_records["0.05"],
            "linkage_attack_tau_0.075": metrics.linkage_attack_records["0.075"],
            "linkage_attack_tau_0.1": metrics.linkage_attack_records["0.1"],
            "homogeneity_attack_records": metrics.homogeneity_attack_records,
        }

    def _model_evaluation_table(
        self,
        baseline: PrivacyRiskMetrics,
        anonymized: PrivacyRiskMetrics,
        config: MOOBAMConfig,
        information_loss: float,
        entropy_sum: float,
    ) -> pd.DataFrame:
        rows = [
            {
                "dataset_version": "Prepared baseline",
                "k": None,
                "number_of_clusters": baseline.equivalence_classes,
                "lambda": None,
                "information_loss": None,
                "entropy_sum": None,
                **self._risk_columns(baseline),
            },
            {
                "dataset_version": "MO-OBAM observed",
                "k": config.k,
                "number_of_clusters": anonymized.equivalence_classes,
                "lambda": config.lambda_weight,
                "information_loss": information_loss,
                "entropy_sum": entropy_sum,
                **self._risk_columns(anonymized),
            },
        ]
        return pd.DataFrame(rows)

    def _paper_comparison_table(
        self,
        prepared_records: int,
        baseline: PrivacyRiskMetrics,
        anonymized: PrivacyRiskMetrics,
        config: MOOBAMConfig,
        information_loss: float,
    ) -> pd.DataFrame:
        comparisons: list[tuple[str, float, float]] = [
            (
                "Prepared records",
                PAPER_ADULT_TABLE_VB["prepared_records"],
                prepared_records,
            ),
            (
                "Baseline equivalence classes",
                PAPER_ADULT_TABLE_VB["baseline_equivalence_classes"],
                baseline.equivalence_classes,
            ),
            (
                "Baseline linkage records (tau=0.05)",
                PAPER_ADULT_TABLE_VB["baseline_linkage_0.05"],
                baseline.linkage_attack_records["0.05"],
            ),
            (
                "Baseline linkage records (tau=0.075)",
                PAPER_ADULT_TABLE_VB["baseline_linkage_0.075"],
                baseline.linkage_attack_records["0.075"],
            ),
            (
                "Baseline linkage records (tau=0.1)",
                PAPER_ADULT_TABLE_VB["baseline_linkage_0.1"],
                baseline.linkage_attack_records["0.1"],
            ),
            (
                "Baseline homogeneity records",
                PAPER_ADULT_TABLE_VB["baseline_homogeneity"],
                baseline.homogeneity_attack_records,
            ),
        ]
        expected_boundary = next(
            (
                expected
                for (clusters, lambda_weight), expected in (
                    PAPER_ADULT_MO_OBAM_BOUNDARIES.items()
                )
                if config.n_clusters == clusters
                and np.isclose(config.lambda_weight, lambda_weight)
            ),
            None,
        )
        if config.k in (5, 10, 15, 20) and expected_boundary is not None:
            comparisons.extend(
                [
                    (
                        "MO-OBAM equivalence classes",
                        expected_boundary["equivalence_classes"],
                        anonymized.equivalence_classes,
                    ),
                    (
                        "MO-OBAM information loss",
                        expected_boundary["information_loss"],
                        information_loss,
                    ),
                    (
                        "MO-OBAM linkage records (tau=0.05)",
                        expected_boundary["linkage_0.05"],
                        anonymized.linkage_attack_records["0.05"],
                    ),
                    (
                        "MO-OBAM linkage records (tau=0.075)",
                        expected_boundary["linkage_0.075"],
                        anonymized.linkage_attack_records["0.075"],
                    ),
                    (
                        "MO-OBAM linkage records (tau=0.1)",
                        expected_boundary["linkage_0.1"],
                        anonymized.linkage_attack_records["0.1"],
                    ),
                    (
                        "MO-OBAM homogeneity records",
                        expected_boundary["homogeneity"],
                        anonymized.homogeneity_attack_records,
                    ),
                ]
            )

        rows = []
        for metric, expected, observed in comparisons:
            difference = float(observed) - float(expected)
            rows.append(
                {
                    "metric": metric,
                    "paper_expected": expected,
                    "observed": observed,
                    "difference": difference,
                    "exact_match": bool(
                        np.isclose(observed, expected, rtol=0.0, atol=1e-12)
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _cluster_details(self, anonymized: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for cluster, cluster_data in anonymized.groupby("cluster", sort=True):
            row: dict[str, object] = {
                "cluster": int(cluster),
                "records": int(len(cluster_data)),
                "proportion": len(cluster_data) / len(anonymized),
            }
            for column in self.schema.qids:
                row[f"centroid_{column}"] = cluster_data[column].iloc[0]
            for column in self.schema.sensitive_attributes:
                probabilities = cluster_data[column].value_counts(
                    normalize=True, dropna=False
                ).to_numpy(dtype=float)
                row[f"distinct_{column}"] = int(
                    cluster_data[column].nunique(dropna=False)
                )
                row[f"entropy_{column}"] = float(
                    -np.sum(probabilities * np.log2(probabilities))
                )
            rows.append(row)
        return pd.DataFrame(rows)
