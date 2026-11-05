"""Build aggregate, write-up-ready tables from recorded PRTI-GH results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adult_experiments import (
    PAPER_ADULT_CLUSTER_VALUES,
    PAPER_K_VALUES,
    PAPER_LAMBDA_VALUES,
)
from .provenance import ArtifactManifestWriter
from .prti_runner import PAPER_IDENTIFIER, PAPER_REVISION, PROJECT_ROOT


@dataclass(frozen=True)
class PRTIResultsConfig:
    """Identify recorded baseline, sensitivity, robustness, and report inputs."""

    baseline_directory: Path = Path("outputs/prti/baseline-v1-seed9")
    sensitivity_results: Path = Path(
        "evidence/prti-sensitivity-v1/prti_sensitivity_results.csv"
    )
    robustness_summary: Path = Path(
        "evidence/prti-robustness-v1/prti_robustness_summary.json"
    )
    output_directory: Path = Path("evidence/prti-results-v1")


class PRTIResultsReporter:
    """Create aggregate result tables without reading or exporting patient rows."""

    BASELINE_K = 10
    BASELINE_CLUSTERS = 30
    BASELINE_LAMBDA = 1.0
    N_RECORDS = 1485
    N_QIDS = 29
    N_SENSITIVE_ATTRIBUTES = 77
    THRESHOLDS = (0.05, 0.075, 0.1)

    def __init__(self, config: PRTIResultsConfig) -> None:
        self.config = config

    @classmethod
    def one_factor_run_ids(cls) -> set[str]:
        """Return the 18 unique configurations declared before the study ran."""

        ids = {
            cls._run_id(k, cls.BASELINE_CLUSTERS, cls.BASELINE_LAMBDA)
            for k in PAPER_K_VALUES
        }
        ids.update(
            cls._run_id(cls.BASELINE_K, clusters, cls.BASELINE_LAMBDA)
            for clusters in PAPER_ADULT_CLUSTER_VALUES
        )
        ids.update(
            cls._run_id(cls.BASELINE_K, cls.BASELINE_CLUSTERS, weight)
            for weight in PAPER_LAMBDA_VALUES
        )
        return ids

    def write(self) -> dict[str, object]:
        """Validate evidence and write five CSV tables plus JSON provenance."""

        summary_path = self.config.baseline_directory / "prti_run_summary.json"
        baseline = json.loads(summary_path.read_text(encoding="utf-8"))
        sensitivity = pd.read_csv(self.config.sensitivity_results)
        robustness = json.loads(
            self.config.robustness_summary.read_text(encoding="utf-8")
        )
        self._validate_inputs(baseline, sensitivity, robustness)

        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        tables = {
            "baseline_privacy_utility": self._baseline_table(baseline),
            "one_factor_sensitivity": self._one_factor_table(sensitivity),
            "multi_seed_robustness": self._robustness_table(robustness),
            "sensitive_attribute_risk": self._sensitive_attribute_table(baseline),
            "cluster_risk": self._cluster_table(baseline),
        }
        paths: dict[str, Path] = {}
        for name, table in tables.items():
            path = output / f"prti_{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = path

        report = self._summary(
            baseline,
            sensitivity,
            robustness,
            summary_path,
            paths,
        )
        summary_output = output / "prti_results_summary.json"
        summary_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths["results_summary"] = summary_output
        return {
            "summary": report,
            "tables": tables,
            "paths": paths,
        }

    def _validate_inputs(
        self,
        baseline: dict[str, object],
        sensitivity: pd.DataFrame,
        robustness: dict[str, object],
    ) -> None:
        """Reject mismatched data, roles, configuration, or incomplete study rows."""

        if baseline["validation"]["source_shape"] != [self.N_RECORDS, 106]:
            raise ValueError("Baseline source shape is not the reviewed PRTI-GH shape.")
        schema = baseline["schema"]
        qids = schema["numeric_qids"] + schema["categorical_qids"]
        if len(qids) != self.N_QIDS:
            raise ValueError("Baseline does not contain the reviewed 29 QIDs.")
        if len(schema["sensitive_attributes"]) != self.N_SENSITIVE_ATTRIBUTES:
            raise ValueError("Baseline does not contain the reviewed 77 SAs.")

        expected_model = {
            "k": self.BASELINE_K,
            "n_clusters": self.BASELINE_CLUSTERS,
            "lambda_weight": self.BASELINE_LAMBDA,
        }
        if baseline["model_configuration"] != expected_model:
            raise ValueError("Baseline model configuration has changed.")

        indexed = sensitivity.set_index("run_id", drop=False)
        missing = sorted(self.one_factor_run_ids() - set(indexed.index))
        if missing:
            raise ValueError(f"One-factor sensitivity rows are missing: {missing}")

        if robustness.get("paper_revision") != PAPER_REVISION:
            raise ValueError("Robustness evidence uses a different paper revision.")
        if robustness.get("source_sha256") != baseline["preparation"][
            "source_sha256_before"
        ]:
            raise ValueError("Robustness evidence uses a different PRTI-GH source.")
        if robustness.get("patient_level_outputs") is not False:
            raise ValueError("Robustness evidence must exclude patient-level rows.")
        if robustness.get("planned_runs") != 40 or robustness.get(
            "recorded_runs"
        ) != 40:
            raise ValueError("The predeclared 40-run robustness study is incomplete.")
        table = pd.DataFrame(robustness.get("configurations", []))
        expected_ids = {
            "baseline",
            "privacy_candidate",
            "lowest_il_candidate",
            "compromise_candidate",
        }
        if set(table.get("configuration_id", [])) != expected_ids:
            raise ValueError("Robustness evidence does not contain the four candidates.")
        if not (
            table["seeds_planned"].eq(10)
            & table["runs_recorded"].eq(10)
            & table["runs_completed"].add(
                table["runs_no_feasible_particle"]
            ).eq(10)
        ).all():
            raise ValueError("Robustness seed accounting is inconsistent.")

    def _baseline_table(self, baseline: dict[str, object]) -> pd.DataFrame:
        """Compare prepared and released privacy/utility measures."""

        privacy = baseline["privacy_evaluation"]
        before = privacy["prepared_baseline"]
        after = privacy["anonymized"]
        final = baseline["final_metrics"]
        rows = [
            self._comparison_row(
                "equivalence_classes",
                before["equivalence_classes"],
                after["equivalence_classes"],
            ),
            self._comparison_row(
                "minimum_equivalence_class_size",
                before["minimum_equivalence_class_size"],
                after["minimum_equivalence_class_size"],
            ),
            self._comparison_row(
                "maximum_equivalence_class_size",
                before["maximum_equivalence_class_size"],
                after["maximum_equivalence_class_size"],
            ),
        ]
        for threshold in self.THRESHOLDS:
            key = format(threshold, "g")
            rows.append(
                self._comparison_row(
                    f"linkage_attack_records_tau_{key}",
                    before["linkage_attack_records"][key],
                    after["linkage_attack_records"][key],
                )
            )
        rows.extend(
            [
                self._comparison_row(
                    "homogeneity_attack_records_any_sa",
                    before["homogeneity_attack_records"],
                    after["homogeneity_attack_records"],
                ),
                self._comparison_row(
                    "information_loss", None, final["information_loss"]
                ),
                self._comparison_row("entropy_sum", None, final["entropy_sum"]),
                self._comparison_row("objective", None, final["objective"]),
            ]
        )
        return pd.DataFrame(rows)

    def _one_factor_table(self, sensitivity: pd.DataFrame) -> pd.DataFrame:
        """Return the three one-factor axes, repeating the reference as needed."""

        indexed = sensitivity.set_index("run_id", drop=False)
        rows: list[dict[str, object]] = []
        axes = (
            (
                "k",
                PAPER_K_VALUES,
                lambda value: self._run_id(
                    int(value), self.BASELINE_CLUSTERS, self.BASELINE_LAMBDA
                ),
            ),
            (
                "configured_clusters",
                PAPER_ADULT_CLUSTER_VALUES,
                lambda value: self._run_id(
                    self.BASELINE_K, int(value), self.BASELINE_LAMBDA
                ),
            ),
            (
                "lambda",
                PAPER_LAMBDA_VALUES,
                lambda value: self._run_id(
                    self.BASELINE_K, self.BASELINE_CLUSTERS, float(value)
                ),
            ),
        )
        for axis, values, identifier in axes:
            for value in values:
                row = indexed.loc[identifier(value)].to_dict()
                row["analysis_axis"] = axis
                row["axis_value"] = value
                row["is_reference"] = bool(
                    int(row["k"]) == self.BASELINE_K
                    and int(row["configured_clusters"]) == self.BASELINE_CLUSTERS
                    and np.isclose(float(row["lambda"]), self.BASELINE_LAMBDA)
                )
                row["linkage_percent_tau_0_05"] = self._percentage(
                    row.get("linkage_records_tau_0_05")
                )
                row["homogeneity_percent_any_sa"] = self._percentage(
                    row.get("homogeneity_records_any_sa")
                )
                feasible = row.get("feasible_evaluations")
                infeasible = row.get("infeasible_evaluations")
                total = self._number(feasible) + self._number(infeasible)
                row["feasible_evaluation_percent"] = (
                    100.0 * self._number(feasible) / total if total else None
                )
                rows.append(row)
        leading = ["analysis_axis", "axis_value", "is_reference"]
        table = pd.DataFrame(rows)
        return table.loc[:, leading + [c for c in table if c not in leading]]

    def _sensitive_attribute_table(
        self, baseline: dict[str, object]
    ) -> pd.DataFrame:
        """Rank per-SA minimum entropy and homogeneity exposure."""

        entropy = baseline["final_metrics"]["entropy_by_sensitive_attribute"]
        homogeneity = baseline["privacy_evaluation"]["anonymized"][
            "homogeneity_attack_records_by_sensitive_attribute"
        ]
        rows = [
            {
                "sensitive_attribute": attribute,
                "minimum_cluster_entropy": float(value),
                "zero_minimum_entropy": bool(np.isclose(value, 0.0)),
                "homogeneity_attack_records": int(homogeneity[attribute]),
                "homogeneity_attack_percent": (
                    100.0 * int(homogeneity[attribute]) / self.N_RECORDS
                ),
            }
            for attribute, value in entropy.items()
        ]
        table = pd.DataFrame(rows).sort_values(
            ["homogeneity_attack_records", "sensitive_attribute"],
            ascending=[False, True],
        )
        table.insert(0, "homogeneity_risk_rank", range(1, len(table) + 1))
        return table.reset_index(drop=True)

    def _robustness_table(
        self,
        robustness: dict[str, object],
    ) -> pd.DataFrame:
        """Create a compact multi-seed privacy, utility, and stability table."""

        table = pd.DataFrame(robustness["configurations"]).copy()
        baseline_il = float(
            table.loc[
                table["configuration_id"].eq("baseline"),
                "information_loss_mean",
            ].iloc[0]
        )
        table["information_loss_change_from_baseline_percent"] = (
            100.0 * (table["information_loss_mean"] - baseline_il) / baseline_il
        )
        table["linkage_percent_tau_0_05_mean"] = (
            100.0 * table["linkage_records_tau_0_05_mean"] / self.N_RECORDS
        )
        table["homogeneity_percent_any_sa_mean"] = (
            100.0 * table["homogeneity_records_any_sa_mean"] / self.N_RECORDS
        )
        columns = [
            "configuration_id",
            "purpose",
            "k",
            "configured_clusters",
            "lambda",
            "seeds_planned",
            "runs_completed",
            "runs_no_feasible_particle",
            "completion_percent",
            "information_loss_mean",
            "information_loss_std",
            "information_loss_min",
            "information_loss_max",
            "information_loss_change_from_baseline_percent",
            "entropy_sum_mean",
            "entropy_sum_std",
            "linkage_records_tau_0_05_mean",
            "linkage_records_tau_0_05_max",
            "linkage_percent_tau_0_05_mean",
            "homogeneity_records_any_sa_mean",
            "homogeneity_records_any_sa_std",
            "homogeneity_records_any_sa_min",
            "homogeneity_records_any_sa_max",
            "homogeneity_percent_any_sa_mean",
            "zero_minimum_entropy_sensitive_attributes_mean",
            "zero_minimum_entropy_sensitive_attributes_min",
            "zero_minimum_entropy_sensitive_attributes_max",
            "minimum_cluster_size_min",
            "maximum_cluster_size_max",
            "feasible_evaluation_percent_mean",
        ]
        return table.loc[:, columns]

    def _cluster_table(self, baseline: dict[str, object]) -> pd.DataFrame:
        """Describe released class sizes and class-level linkage-risk proxies."""

        rows = []
        for cluster, size in enumerate(baseline["cluster_sizes"]):
            risk = 1.0 / int(size)
            row: dict[str, object] = {
                "cluster": cluster,
                "records": int(size),
                "record_percent": 100.0 * int(size) / self.N_RECORDS,
                "linkage_risk_proxy": risk,
            }
            for threshold in self.THRESHOLDS:
                row[f"risk_above_tau_{format(threshold, 'g')}"] = risk > threshold
            rows.append(row)
        return pd.DataFrame(rows).sort_values("records").reset_index(drop=True)

    def _summary(
        self,
        baseline: dict[str, object],
        sensitivity: pd.DataFrame,
        robustness: dict[str, object],
        baseline_summary_path: Path,
        output_paths: dict[str, Path],
    ) -> dict[str, object]:
        """Create machine-readable headline findings and provenance."""

        selected = sensitivity.loc[
            sensitivity["run_id"].isin(self.one_factor_run_ids())
        ].copy()
        completed = selected.loc[selected["status"].eq("completed")]
        after = baseline["privacy_evaluation"]["anonymized"]
        before = baseline["privacy_evaluation"]["prepared_baseline"]
        lowest_il = completed.loc[completed["information_loss"].idxmin()]
        lowest_homogeneity = completed.loc[
            completed["homogeneity_records_any_sa"].idxmin()
        ]
        baseline_sensitivity = selected.loc[
            selected["run_id"].eq(
                self._run_id(
                    self.BASELINE_K,
                    self.BASELINE_CLUSTERS,
                    self.BASELINE_LAMBDA,
                )
            )
        ].iloc[0]

        manifest_writer = ArtifactManifestWriter(PROJECT_ROOT)
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "scope": "Aggregate PRTI-GH MO-OBAM anonymization results; ML excluded",
            "dataset": {
                "records": self.N_RECORDS,
                "attributes": 106,
                "qids": self.N_QIDS,
                "sensitive_attributes": self.N_SENSITIVE_ATTRIBUTES,
                "source_sha256": baseline["preparation"]["source_sha256_before"],
            },
            "baseline": {
                "model_configuration": baseline["model_configuration"],
                "pso_configuration": baseline["pso_configuration"],
                "information_loss": baseline["final_metrics"]["information_loss"],
                "entropy_sum": baseline["final_metrics"]["entropy_sum"],
                "objective": baseline["final_metrics"]["objective"],
                "equivalence_classes": after["equivalence_classes"],
                "minimum_class_size": after["minimum_equivalence_class_size"],
                "maximum_class_size": after["maximum_equivalence_class_size"],
                "linkage_records": after["linkage_attack_records"],
                "homogeneity_records_any_sa": after[
                    "homogeneity_attack_records"
                ],
                "homogeneity_percent_any_sa": (
                    100.0 * after["homogeneity_attack_records"] / self.N_RECORDS
                ),
                "zero_minimum_entropy_sensitive_attributes": int(
                    sum(
                        np.isclose(value, 0.0)
                        for value in baseline["final_metrics"][
                            "entropy_by_sensitive_attribute"
                        ].values()
                    )
                ),
            },
            "risk_reduction": {
                "linkage_tau_0_05_percent": self._reduction_percent(
                    before["linkage_attack_records"]["0.05"],
                    after["linkage_attack_records"]["0.05"],
                ),
                "linkage_tau_0_075_percent": self._reduction_percent(
                    before["linkage_attack_records"]["0.075"],
                    after["linkage_attack_records"]["0.075"],
                ),
                "linkage_tau_0_1_percent": self._reduction_percent(
                    before["linkage_attack_records"]["0.1"],
                    after["linkage_attack_records"]["0.1"],
                ),
                "homogeneity_any_sa_percent": self._reduction_percent(
                    before["homogeneity_attack_records"],
                    after["homogeneity_attack_records"],
                ),
            },
            "one_factor_sensitivity": {
                "unique_configurations_declared": len(self.one_factor_run_ids()),
                "completed": int(selected["status"].eq("completed").sum()),
                "no_feasible_particle": int(
                    selected["status"].eq("no_feasible_particle").sum()
                ),
                "lowest_information_loss_completed": self._headline_row(lowest_il),
                "lowest_homogeneity_risk_completed": self._headline_row(
                    lowest_homogeneity
                ),
                "baseline_reproduced_in_sensitivity": bool(
                    np.isclose(
                        baseline_sensitivity["information_loss"],
                        baseline["final_metrics"]["information_loss"],
                        rtol=0.0,
                        atol=1e-12,
                    )
                    and np.isclose(
                        baseline_sensitivity["entropy_sum"],
                        baseline["final_metrics"]["entropy_sum"],
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
            },
            "multi_seed_robustness": {
                "planned_runs": robustness["planned_runs"],
                "recorded_runs": robustness["recorded_runs"],
                "completed_runs": robustness["completed_runs"],
                "no_feasible_particle_runs": robustness[
                    "no_feasible_particle_runs"
                ],
                "configurations": robustness["configurations"],
                "selection_status": (
                    "No final release configuration selected; acceptance "
                    "thresholds and disclosure-control review remain required."
                ),
            },
            "interpretation_boundaries": [
                "Linkage risk is the equivalence-class proxy 1/|E|, not an external-data re-identification attack.",
                "Homogeneity risk counts records homogeneous for at least one configured SA.",
                "Objective values must not be ranked across different lambda values without accounting for the changed weighting.",
                "No-feasible-particle means the seeded initial population did not satisfy hard k; it does not prove mathematical infeasibility.",
                "The baseline and one-factor study use seed 9; the selected robustness subset uses seeds 0-9.",
                "Neither the single-seed nor multi-seed study proves global optimality or external release safety.",
            ],
            "inputs": {
                "baseline_summary": {
                    "path": str(baseline_summary_path),
                    "sha256": manifest_writer.file_sha256(baseline_summary_path),
                },
                "sensitivity_results": {
                    "path": str(self.config.sensitivity_results),
                    "sha256": manifest_writer.file_sha256(
                        self.config.sensitivity_results
                    ),
                },
                "robustness_summary": {
                    "path": str(self.config.robustness_summary),
                    "sha256": manifest_writer.file_sha256(
                        self.config.robustness_summary
                    ),
                },
            },
            "outputs": {
                name: {
                    "path": str(path),
                    "sha256": manifest_writer.file_sha256(path),
                }
                for name, path in output_paths.items()
            },
            "report_generation_snapshot": manifest_writer.capture_snapshot(
                "captured_when_aggregate_results_report_was_generated"
            ),
        }

    @staticmethod
    def _comparison_row(
        metric: str,
        before: int | float | None,
        after: int | float,
    ) -> dict[str, object]:
        before_value = float(before) if before is not None else None
        after_value = float(after)
        return {
            "metric": metric,
            "prepared_baseline": before_value,
            "anonymized": after_value,
            "absolute_change": (
                after_value - before_value if before_value is not None else None
            ),
            "relative_reduction_percent": (
                100.0 * (before_value - after_value) / before_value
                if before_value not in (None, 0.0)
                and (
                    metric.startswith("linkage_attack_records")
                    or metric.startswith("homogeneity_attack_records")
                )
                else None
            ),
        }

    @classmethod
    def _run_id(cls, k: int, clusters: int, weight: float) -> str:
        lambda_label = format(weight, "g").replace(".", "p")
        return f"k{k:02d}_c{clusters:03d}_lambda{lambda_label}"

    @classmethod
    def _percentage(cls, value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return 100.0 * float(value) / cls.N_RECORDS

    @staticmethod
    def _number(value: object) -> float:
        return 0.0 if value is None or pd.isna(value) else float(value)

    @staticmethod
    def _reduction_percent(before: int, after: int) -> float:
        return 100.0 * (before - after) / before

    @staticmethod
    def _headline_row(row: pd.Series) -> dict[str, object]:
        return {
            "run_id": str(row["run_id"]),
            "k": int(row["k"]),
            "configured_clusters": int(row["configured_clusters"]),
            "lambda": float(row["lambda"]),
            "information_loss": float(row["information_loss"]),
            "entropy_sum": float(row["entropy_sum"]),
            "linkage_records_tau_0_05": int(row["linkage_records_tau_0_05"]),
            "homogeneity_records_any_sa": int(
                row["homogeneity_records_any_sa"]
            ),
        }
