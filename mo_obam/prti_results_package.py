"""Build the aggregate PRTI-GH results package used for reporting.

The builder reads saved patient-level artifacts only to calculate aggregates.
It never copies patient rows, QID representatives, or cluster centroids into
the versioned evidence package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from .provenance import ArtifactManifestWriter
from .prti_data import (
    DEFAULT_PRTI_SOURCE,
    EXPECTED_PRTI_SHA256,
    PRTI_CATEGORICAL_QIDS,
    PRTI_NUMERIC_QIDS,
    PRTI_QIDS,
)
from .prti_results import PRTIResultsReporter
from .prti_runner import PAPER_IDENTIFIER, PAPER_REVISION, PROJECT_ROOT


@dataclass(frozen=True)
class PRTIResultsPackageConfig:
    """Locate the recorded evidence used by the aggregate package."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    baseline_directory: Path = Path("outputs/prti/baseline-v1-seed9")
    final_directory: Path = Path(
        "outputs/prti/privacy-candidate-heldout-seed10"
    )
    final_acceptance: Path = Path(
        "evidence/prti-final-selection-v1/prti_final_candidate_acceptance.json"
    )
    sensitivity_plan: Path = Path(
        "evidence/prti-sensitivity-v1/prti_sensitivity_plan.json"
    )
    sensitivity_results: Path = Path(
        "evidence/prti-sensitivity-v1/prti_sensitivity_results.csv"
    )
    robustness_summary: Path = Path(
        "evidence/prti-robustness-v1/prti_robustness_summary.json"
    )
    output_directory: Path = Path("evidence/prti-results-v2")


class PRTIResultsPackageBuilder:
    """Create validated aggregate tables, figures, and provenance evidence."""

    N_RECORDS = 1_485
    N_ATTRIBUTES = 106
    N_QIDS = 29
    N_SENSITIVE_ATTRIBUTES = 77
    LINKAGE_THRESHOLDS = ("0.05", "0.075", "0.1")

    TABLE_FILENAMES = {
        "dataset_and_roles": "table_01_dataset_and_roles.csv",
        "baseline_vs_final": "table_02_baseline_vs_final_candidate.csv",
        "qid_transformation": "table_03_qid_transformation.csv",
        "sensitive_attribute_results": (
            "table_04_sensitive_attribute_results.csv"
        ),
        "final_cluster_summary": "table_05_final_cluster_summary.csv",
        "robustness_summary": "table_06_robustness_summary.csv",
        "final_screening_checks": "table_07_final_screening_checks.csv",
        "experiment_coverage": "table_08_experiment_coverage.csv",
    }

    def __init__(self, config: PRTIResultsPackageConfig) -> None:
        self.config = config
        self.manifest_writer = ArtifactManifestWriter(PROJECT_ROOT)

    def write(self) -> dict[str, object]:
        """Build and validate the complete aggregate results package."""

        evidence = self._load_evidence()
        self._validate_evidence(evidence)
        tables = self._build_tables(evidence)

        output = self.config.output_directory
        table_directory = output / "tables"
        figure_directory = output / "figures"
        manifest_directory = output / "manifests"
        for directory in (table_directory, figure_directory, manifest_directory):
            directory.mkdir(parents=True, exist_ok=True)

        table_paths: dict[str, Path] = {}
        for name, table in tables.items():
            path = table_directory / self.TABLE_FILENAMES[name]
            table.to_csv(path, index=False)
            table_paths[name] = path

        figure_paths = self._write_figures(tables, evidence, figure_directory)
        readme_path = output / "README.md"
        readme_path.write_text(self._readme(evidence), encoding="utf-8")

        self._validate_outputs(tables, table_paths, figure_paths)
        manifest_path = manifest_directory / "prti_results_v2_manifest.json"
        manifest_path.write_text(
            json.dumps(
                self._manifest(
                    evidence,
                    [*table_paths.values(), *figure_paths, readme_path],
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "tables": tables,
            "table_paths": table_paths,
            "figure_paths": figure_paths,
            "readme_path": readme_path,
            "manifest_path": manifest_path,
        }

    def _load_evidence(self) -> dict[str, object]:
        """Load the recorded run summaries and only the rows needed for aggregates."""

        baseline_summary = self._read_json(
            self.config.baseline_directory / "prti_run_summary.json"
        )
        final_summary = self._read_json(
            self.config.final_directory / "prti_run_summary.json"
        )
        return {
            "baseline": baseline_summary,
            "final": final_summary,
            "acceptance": self._read_json(self.config.final_acceptance),
            "sensitivity_plan": self._read_json(self.config.sensitivity_plan),
            "sensitivity": pd.read_csv(self.config.sensitivity_results),
            "robustness": self._read_json(self.config.robustness_summary),
            "input_audit": pd.read_csv(
                self.config.final_directory / "prti_input_audit.csv"
            ),
            "before_after": pd.read_csv(
                self.config.final_directory / "prti_before_after_anonymization.csv",
                dtype=str,
                keep_default_na=False,
            ),
            "baseline_entropy": pd.read_csv(
                self.config.baseline_directory / "prti_sensitive_entropy.csv"
            ),
            "final_entropy": pd.read_csv(
                self.config.final_directory / "prti_sensitive_entropy.csv"
            ),
            "final_clusters": pd.read_csv(
                self.config.final_directory / "prti_cluster_details.csv"
            ),
        }

    def _validate_evidence(self, evidence: dict[str, object]) -> None:
        """Reject altered roles, runs, source fingerprints, or incomplete evidence."""

        baseline = evidence["baseline"]
        final = evidence["final"]
        acceptance = evidence["acceptance"]
        robustness = evidence["robustness"]
        sensitivity_plan = evidence["sensitivity_plan"]
        sensitivity = evidence["sensitivity"]
        audit = evidence["input_audit"]
        paired = evidence["before_after"]

        if not isinstance(baseline, dict) or not isinstance(final, dict):
            raise TypeError("Run summaries must be JSON objects.")
        for label, summary in (("baseline", baseline), ("final", final)):
            if summary["validation"]["source_shape"] != [
                self.N_RECORDS,
                self.N_ATTRIBUTES,
            ]:
                raise ValueError(f"The {label} source shape has changed.")
            schema = summary["schema"]
            qids = schema["numeric_qids"] + schema["categorical_qids"]
            if tuple(qids) != (*PRTI_NUMERIC_QIDS, *PRTI_CATEGORICAL_QIDS):
                raise ValueError(f"The reviewed {label} QID roles have changed.")
            if len(schema["sensitive_attributes"]) != self.N_SENSITIVE_ATTRIBUTES:
                raise ValueError(f"The reviewed {label} SA roles have changed.")
            if not summary["validation"]["source_unchanged"]:
                raise ValueError(f"The {label} run did not preserve the source.")
            if not summary["validation"]["all_sensitive_attributes_unchanged"]:
                raise ValueError(f"The {label} run altered sensitive values.")

        if baseline["model_configuration"] != {
            "k": 10,
            "n_clusters": 30,
            "lambda_weight": 1.0,
        } or baseline["pso_configuration"]["seed"] != 9:
            raise ValueError("The recorded seed-9 baseline configuration changed.")
        if final["model_configuration"] != {
            "k": 10,
            "n_clusters": 4,
            "lambda_weight": 1.0,
        } or final["pso_configuration"]["seed"] != 10:
            raise ValueError("The held-out final-candidate configuration changed.")

        source_hashes = {
            baseline["preparation"]["source_sha256_before"],
            final["preparation"]["source_sha256_before"],
            robustness["source_sha256"],
        }
        if source_hashes != {EXPECTED_PRTI_SHA256}:
            raise ValueError("The evidence does not use one reviewed source workbook.")
        if (
            not self.config.source_path.is_file()
            or self.manifest_writer.file_sha256(self.config.source_path)
            != EXPECTED_PRTI_SHA256
        ):
            raise ValueError("The installed PRTI-GH source fingerprint has changed.")

        if not isinstance(acceptance, dict) or not acceptance["all_checks_passed"]:
            raise ValueError("The held-out candidate did not pass its recorded checks.")
        if acceptance["run_id"] != final["run_id"]:
            raise ValueError("The acceptance evidence refers to a different run.")
        if len(acceptance["checks"]) != 10:
            raise ValueError("The final screening evidence must contain ten checks.")

        if robustness["planned_runs"] != 40 or robustness["recorded_runs"] != 40:
            raise ValueError("The 40-run robustness study is incomplete.")
        if robustness["patient_level_outputs"] is not False:
            raise ValueError("Robustness evidence unexpectedly contains patient rows.")
        if sensitivity_plan["number_of_configurations"] != 220:
            raise ValueError("The declared sensitivity grid has changed.")
        planned_ids = {
            item["run_id"] for item in sensitivity_plan["configurations"]
        }
        if not set(sensitivity["run_id"]).issubset(planned_ids):
            raise ValueError("Sensitivity results contain an undeclared run.")

        audit_qids = audit.loc[audit["role"].str.startswith("QID"), "attribute"]
        audit_sensitive = audit.loc[
            audit["role"].eq("sensitive attribute"), "attribute"
        ]
        if set(audit_qids) != set(PRTI_QIDS) or set(audit_sensitive) != set(
            baseline["schema"]["sensitive_attributes"]
        ):
            raise ValueError("The input audit no longer follows the reviewed roles.")
        expected_paired_columns = {
            f"{attribute}{suffix}"
            for attribute in PRTI_QIDS
            for suffix in ("Before", "After")
        }
        if len(paired) != self.N_RECORDS or not expected_paired_columns.issubset(
            paired.columns
        ):
            raise ValueError("The held-out before/after file is incomplete.")

    def _build_tables(
        self, evidence: dict[str, object]
    ) -> dict[str, pd.DataFrame]:
        """Create the eight aggregate tables in their stable display order."""

        return {
            "dataset_and_roles": self._dataset_table(evidence),
            "baseline_vs_final": self._comparison_table(evidence),
            "qid_transformation": self._qid_table(evidence),
            "sensitive_attribute_results": self._sensitive_table(evidence),
            "final_cluster_summary": self._cluster_table(evidence),
            "robustness_summary": self._robustness_table(evidence),
            "final_screening_checks": self._screening_table(evidence),
            "experiment_coverage": self._coverage_table(evidence),
        }

    def _dataset_table(self, evidence: dict[str, object]) -> pd.DataFrame:
        final = evidence["final"]
        rows = (
            ("records", self.N_RECORDS, "records", "Rows in the reviewed workbook"),
            ("attributes", self.N_ATTRIBUTES, "attributes", "Columns in the reviewed workbook"),
            ("quasi_identifiers", self.N_QIDS, "attributes", "Attributes replaced by cluster representatives"),
            ("numeric_qids", len(PRTI_NUMERIC_QIDS), "attributes", "QIDs evaluated with numerical distance"),
            ("categorical_qids", len(PRTI_CATEGORICAL_QIDS), "attributes", "QIDs evaluated with categorical mismatch"),
            ("sensitive_attributes", self.N_SENSITIVE_ATTRIBUTES, "attributes", "Attributes preserved row by row and used in entropy"),
            ("source_sha256", EXPECTED_PRTI_SHA256, "SHA-256", "Fingerprint of the unchanged source workbook"),
            ("paper_identifier", PAPER_IDENTIFIER, "reference", "Implemented MO-OBAM paper"),
            ("paper_revision", PAPER_REVISION, "reference", "Paper revision used by the implementation"),
            ("final_run_id", final["run_id"], "run identifier", "Held-out candidate evaluated in this package"),
        )
        return pd.DataFrame(rows, columns=["item", "value", "unit", "definition"])

    def _comparison_table(self, evidence: dict[str, object]) -> pd.DataFrame:
        baseline = evidence["baseline"]
        final = evidence["final"]
        prepared = final["privacy_evaluation"]["prepared_baseline"]
        baseline_privacy = baseline["privacy_evaluation"]["anonymized"]
        final_privacy = final["privacy_evaluation"]["anonymized"]

        rows: list[dict[str, object]] = []

        def add(
            metric: str,
            prepared_value: object,
            baseline_value: object,
            final_value: object,
            unit: str,
            direction: str,
            definition: str,
        ) -> None:
            rows.append(
                {
                    "metric": metric,
                    "prepared_baseline": prepared_value,
                    "recorded_seed9_baseline": baseline_value,
                    "heldout_seed10_candidate": final_value,
                    "unit": unit,
                    "preferred_direction": direction,
                    "definition": definition,
                }
            )

        add("equivalence_classes", prepared["equivalence_classes"], baseline_privacy["equivalence_classes"], final_privacy["equivalence_classes"], "classes", "context dependent", "Distinct released QID patterns")
        add("minimum_equivalence_class_size", prepared["minimum_equivalence_class_size"], baseline_privacy["minimum_equivalence_class_size"], final_privacy["minimum_equivalence_class_size"], "records", "higher", "Smallest released equivalence class")
        add("maximum_equivalence_class_size", prepared["maximum_equivalence_class_size"], baseline_privacy["maximum_equivalence_class_size"], final_privacy["maximum_equivalence_class_size"], "records", "context dependent", "Largest released equivalence class")
        for threshold in self.LINKAGE_THRESHOLDS:
            add(
                f"linkage_attack_records_tau_{threshold.replace('.', '_')}",
                prepared["linkage_attack_records"][threshold],
                baseline_privacy["linkage_attack_records"][threshold],
                final_privacy["linkage_attack_records"][threshold],
                "records",
                "lower",
                f"Records in classes where 1/|E| is greater than {threshold}",
            )
        add("homogeneity_attack_records_any_sa", prepared["homogeneity_attack_records"], baseline_privacy["homogeneity_attack_records"], final_privacy["homogeneity_attack_records"], "records", "lower", "Records homogeneous for at least one configured SA")
        add("information_loss", np.nan, baseline["final_metrics"]["information_loss"], final["final_metrics"]["information_loss"], "ratio", "lower", "Wei Equation (14) QID information-loss ratio")
        add("entropy_sum", np.nan, baseline["final_metrics"]["entropy_sum"], final["final_metrics"]["entropy_sum"], "bits", "higher", "Sum of minimum cluster entropy across 77 SAs")
        add("objective", np.nan, baseline["final_metrics"]["objective"], final["final_metrics"]["objective"], "objective units", "lower", "Information loss minus lambda times entropy sum; lambda is 1 for both runs")
        add("zero_minimum_entropy_sensitive_attributes", np.nan, self._zero_entropy_count(baseline), self._zero_entropy_count(final), "attributes", "lower", "SAs with zero entropy in at least one released class")
        return pd.DataFrame(rows)

    def _qid_table(self, evidence: dict[str, object]) -> pd.DataFrame:
        audit = evidence["input_audit"].set_index("attribute")
        paired = evidence["before_after"]
        final = evidence["final"]
        rows = []
        changed_total = 0
        for attribute in PRTI_QIDS:
            before = paired[f"{attribute}Before"]
            after = paired[f"{attribute}After"]
            changed = int(before.ne(after).sum())
            changed_total += changed
            source_missing = int(audit.loc[attribute, "missingValues"])
            if int(before.eq("").sum()) != source_missing:
                raise ValueError(f"Missing-value count changed while reading {attribute}.")
            expected_changed = int(
                final["validation"]["changed_qid_cells"][attribute]
            )
            if changed != expected_changed:
                raise ValueError(f"Changed-cell count differs for {attribute}.")
            rows.append(
                {
                    "position": int(audit.loc[attribute, "position"]),
                    "attribute": attribute,
                    "qid_type": (
                        "numeric" if attribute in PRTI_NUMERIC_QIDS else "categorical"
                    ),
                    "source_distinct_including_missing": int(before.nunique()),
                    "final_distinct_including_missing": int(after.nunique()),
                    "source_missing_records": source_missing,
                    "final_missing_records": int(after.eq("").sum()),
                    "changed_records": changed,
                    "changed_percent": 100.0 * changed / self.N_RECORDS,
                }
            )
        if changed_total != sum(final["validation"]["changed_qid_cells"].values()):
            raise ValueError("Total changed QID cells does not match the run summary.")
        return pd.DataFrame(rows)

    def _sensitive_table(self, evidence: dict[str, object]) -> pd.DataFrame:
        audit = evidence["input_audit"].set_index("attribute")
        baseline = evidence["baseline"]
        final = evidence["final"]
        baseline_entropy = evidence["baseline_entropy"].groupby(
            "sensitiveAttribute", sort=False
        )["entropy"].agg(["min", "mean", "max"])
        final_entropy = evidence["final_entropy"].groupby(
            "sensitiveAttribute", sort=False
        )["entropy"].agg(["min", "mean", "max"])
        baseline_homogeneity = baseline["privacy_evaluation"]["anonymized"][
            "homogeneity_attack_records_by_sensitive_attribute"
        ]
        final_homogeneity = final["privacy_evaluation"]["anonymized"][
            "homogeneity_attack_records_by_sensitive_attribute"
        ]
        rows = []
        for attribute in final["schema"]["sensitive_attributes"]:
            row = audit.loc[attribute]
            rows.append(
                {
                    "position": int(row["position"]),
                    "sensitive_attribute": attribute,
                    "source_dtype": row["sourceDtype"],
                    "source_distinct_nonmissing": int(row["distinctNonMissingValues"]),
                    "source_missing_records": int(row["missingValues"]),
                    "values_preserved_row_by_row": True,
                    "baseline_minimum_cluster_entropy": float(baseline_entropy.loc[attribute, "min"]),
                    "baseline_mean_cluster_entropy": float(baseline_entropy.loc[attribute, "mean"]),
                    "baseline_maximum_cluster_entropy": float(baseline_entropy.loc[attribute, "max"]),
                    "baseline_homogeneity_records": int(baseline_homogeneity[attribute]),
                    "final_minimum_cluster_entropy": float(final_entropy.loc[attribute, "min"]),
                    "final_mean_cluster_entropy": float(final_entropy.loc[attribute, "mean"]),
                    "final_maximum_cluster_entropy": float(final_entropy.loc[attribute, "max"]),
                    "final_homogeneity_records": int(final_homogeneity[attribute]),
                }
            )
        return pd.DataFrame(rows)

    def _cluster_table(self, evidence: dict[str, object]) -> pd.DataFrame:
        clusters = evidence["final_clusters"]
        allowed = [
            "cluster",
            "records",
            "proportion",
            "minimumSensitiveEntropy",
            "meanSensitiveEntropy",
            "maximumSensitiveEntropy",
            "zeroEntropySensitiveAttributes",
        ]
        table = clusters.loc[:, allowed].copy()
        table.columns = [
            "cluster",
            "records",
            "record_proportion",
            "minimum_sensitive_entropy",
            "mean_sensitive_entropy",
            "maximum_sensitive_entropy",
            "zero_entropy_sensitive_attributes",
        ]
        table["record_percent"] = 100.0 * table["record_proportion"]
        table["linkage_risk_proxy"] = 1.0 / table["records"]
        for threshold in self.LINKAGE_THRESHOLDS:
            table[f"risk_above_tau_{threshold.replace('.', '_')}"] = (
                table["linkage_risk_proxy"] > float(threshold)
            )
        return table

    def _robustness_table(self, evidence: dict[str, object]) -> pd.DataFrame:
        table = pd.DataFrame(evidence["robustness"]["configurations"]).copy()
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

    def _screening_table(self, evidence: dict[str, object]) -> pd.DataFrame:
        return pd.DataFrame(evidence["acceptance"]["checks"])[
            [
                "category",
                "criterion",
                "operator",
                "threshold",
                "observed",
                "passed",
                "authority",
            ]
        ]

    def _coverage_table(self, evidence: dict[str, object]) -> pd.DataFrame:
        sensitivity = evidence["sensitivity"]
        robustness = evidence["robustness"]
        one_factor = sensitivity.loc[
            sensitivity["run_id"].isin(PRTIResultsReporter.one_factor_run_ids())
        ]

        def counts(frame: pd.DataFrame) -> tuple[int, int, int]:
            return (
                int(frame["status"].eq("completed").sum()),
                int(frame["status"].eq("no_feasible_particle").sum()),
                int(frame["status"].eq("mathematically_infeasible").sum()),
            )

        sensitivity_counts = counts(sensitivity)
        one_factor_counts = counts(one_factor)
        rows = [
            {
                "study_component": "full_sensitivity_grid",
                "scope_note": "Declared Appendix A grid; recorded rows are a partial execution",
                "planned_runs": 220,
                "recorded_runs": len(sensitivity),
                "completed_runs": sensitivity_counts[0],
                "no_feasible_particle_runs": sensitivity_counts[1],
                "mathematically_infeasible_runs": sensitivity_counts[2],
                "not_yet_evaluated_runs": 220 - len(sensitivity),
            },
            {
                "study_component": "one_factor_subset",
                "scope_note": "Overlapping subset of the sensitivity grid",
                "planned_runs": len(PRTIResultsReporter.one_factor_run_ids()),
                "recorded_runs": len(one_factor),
                "completed_runs": one_factor_counts[0],
                "no_feasible_particle_runs": one_factor_counts[1],
                "mathematically_infeasible_runs": one_factor_counts[2],
                "not_yet_evaluated_runs": 0,
            },
            {
                "study_component": "multi_seed_robustness",
                "scope_note": "Four configurations across seeds 0-9",
                "planned_runs": robustness["planned_runs"],
                "recorded_runs": robustness["recorded_runs"],
                "completed_runs": robustness["completed_runs"],
                "no_feasible_particle_runs": robustness["no_feasible_particle_runs"],
                "mathematically_infeasible_runs": 0,
                "not_yet_evaluated_runs": 0,
            },
            {
                "study_component": "heldout_final_candidate",
                "scope_note": "Prospectively declared seed-10 screening run",
                "planned_runs": 1,
                "recorded_runs": 1,
                "completed_runs": 1,
                "no_feasible_particle_runs": 0,
                "mathematically_infeasible_runs": 0,
                "not_yet_evaluated_runs": 0,
            },
        ]
        return pd.DataFrame(rows)

    def _write_figures(
        self,
        tables: dict[str, pd.DataFrame],
        evidence: dict[str, object],
        output: Path,
    ) -> list[Path]:
        """Render five neutral scientific figures as high-resolution PNG and SVG."""

        if shutil.which("gnuplot") is None:
            raise RuntimeError("GNUPlot is required to generate the results figures.")
        comparison = tables["baseline_vs_final"].set_index("metric")
        robustness = tables["robustness_summary"]
        clusters = tables["final_cluster_summary"]

        figure_specs = [
            (
                "figure_01_information_loss",
                self._information_loss_plot(comparison),
            ),
            (
                "figure_02_linkage_exposure",
                self._linkage_plot(comparison),
            ),
            (
                "figure_03_homogeneity_exposure",
                self._homogeneity_plot(comparison),
            ),
            (
                "figure_04_robustness_tradeoff",
                self._robustness_plot(robustness),
            ),
            (
                "figure_05_final_cluster_sizes",
                self._cluster_size_plot(clusters),
            ),
        ]
        paths = []
        for stem, plot_commands in figure_specs:
            paths.extend(self._render_plot(output / stem, plot_commands))
        return paths

    @staticmethod
    def _plot_preamble(title: str) -> str:
        return f"""
set title {PRTIResultsPackageBuilder._quote(title)} font ',18'
set border 3 back lc rgb '#4B5563'
set tics nomirror out
set grid ytics back lc rgb '#E5E7EB' lw 1
set key top center horizontal opaque
set style fill solid 0.86 border rgb '#374151'
set datafile separator whitespace
set lmargin 12
set rmargin 5
set bmargin 7
set tmargin 4
"""

    def _information_loss_plot(self, table: pd.DataFrame) -> str:
        baseline = float(table.loc["information_loss", "recorded_seed9_baseline"])
        final = float(table.loc["information_loss", "heldout_seed10_candidate"])
        return self._plot_preamble("Information loss for recorded single runs") + f"""
set ylabel 'Information loss ratio (Wei Equation 14)'
set yrange [0:0.65]
set format y '%.2f'
set boxwidth 0.62
unset key
$DATA << EOD
"Baseline seed 9" {baseline:.17g}
"Held-out seed 10" {final:.17g}
EOD
plot $DATA using 2:xtic(1) with boxes lc rgb '#0F766E', \\
     '' using 0:2:(sprintf('%.3f',$2)) with labels offset 0,1 textcolor rgb '#111827'
"""

    def _linkage_plot(self, table: pd.DataFrame) -> str:
        rows = []
        for label, column in (
            ("Prepared", "prepared_baseline"),
            ("Baseline seed 9", "recorded_seed9_baseline"),
            ("Held-out seed 10", "heldout_seed10_candidate"),
        ):
            values = [
                int(table.loc[f"linkage_attack_records_tau_{threshold.replace('.', '_')}", column])
                for threshold in self.LINKAGE_THRESHOLDS
            ]
            rows.append(f'{self._quote(label)} {values[0]} {values[1]} {values[2]}')
        return self._plot_preamble("Records above equivalence-class linkage thresholds") + f"""
set ylabel 'Records'
set yrange [0:{self.N_RECORDS * 1.12}]
set format y '%.0f'
set style data histograms
set style histogram clustered gap 1
$DATA << EOD
{'\n'.join(rows)}
EOD
plot $DATA using 2:xtic(1) title 'tau = 0.05' lc rgb '#0F766E', \\
     '' using 3 title 'tau = 0.075' lc rgb '#2563EB', \\
     '' using 4 title 'tau = 0.10' lc rgb '#D97706'
"""

    def _homogeneity_plot(self, table: pd.DataFrame) -> str:
        row = table.loc["homogeneity_attack_records_any_sa"]
        data = [
            ("Prepared", int(row["prepared_baseline"])),
            ("Baseline seed 9", int(row["recorded_seed9_baseline"])),
            ("Held-out seed 10", int(row["heldout_seed10_candidate"])),
        ]
        text = "\n".join(f"{self._quote(label)} {value}" for label, value in data)
        return self._plot_preamble("Homogeneity exposure for any sensitive attribute") + f"""
set ylabel 'Records'
set yrange [0:{self.N_RECORDS * 1.12}]
set format y '%.0f'
set boxwidth 0.62
unset key
$DATA << EOD
{text}
EOD
plot $DATA using 2:xtic(1) with boxes lc rgb '#2563EB', \\
     '' using 0:2:(sprintf('%.0f',$2)) with labels offset 0,1 textcolor rgb '#111827'
"""

    def _robustness_plot(self, table: pd.DataFrame) -> str:
        labels = {
            "baseline": "Baseline",
            "privacy_candidate": "Privacy candidate",
            "lowest_il_candidate": "Lowest IL",
            "compromise_candidate": "Compromise",
        }
        rows = []
        for row in table.itertuples(index=False):
            homogeneity_mean = 100.0 * row.homogeneity_records_any_sa_mean / self.N_RECORDS
            homogeneity_std = 100.0 * row.homogeneity_records_any_sa_std / self.N_RECORDS
            rows.append(
                f'{self._quote(labels[row.configuration_id])} '
                f'{row.information_loss_mean:.17g} {homogeneity_mean:.17g} '
                f'{row.information_loss_std:.17g} {homogeneity_std:.17g}'
            )
        return self._plot_preamble("Multi-seed privacy-utility robustness") + f"""
unset xlabel
set label 99 'Mean information loss ratio' at screen 0.52,0.06 center front
set ylabel 'Mean homogeneity-exposed records (%)'
set xrange [0.39:0.62]
set yrange [0:105]
set format x '%.2f'
set format y '%.0f%%'
unset key
$DATA << EOD
{'\n'.join(rows)}
EOD
plot $DATA using 2:3:4:5 with xyerrorbars pt 7 ps 1.6 lw 2 lc rgb '#7C3AED', \\
     '' using 2:3:1 with labels offset 0,1.4 textcolor rgb '#111827'
"""

    def _cluster_size_plot(self, table: pd.DataFrame) -> str:
        rows = "\n".join(
            f'{self._quote(f"Cluster {int(row.cluster)}")} {int(row.records)}'
            for row in table.itertuples(index=False)
        )
        return self._plot_preamble("Held-out candidate equivalence-class sizes") + f"""
set ylabel 'Records'
set yrange [0:550]
set format y '%.0f'
set boxwidth 0.62
unset key
$DATA << EOD
{rows}
EOD
plot $DATA using 2:xtic(1) with boxes lc rgb '#0F766E', \\
     '' using 0:2:(sprintf('%.0f',$2)) with labels offset 0,1 textcolor rgb '#111827'
"""

    def _render_plot(self, stem: Path, commands: str) -> list[Path]:
        paths = []
        terminals = {
            "png": "pngcairo size 1600,1000 enhanced font 'DejaVu Sans,15'",
            "svg": "svg size 1600,1000 dynamic enhanced font 'DejaVu Sans,15'",
        }
        for extension, terminal in terminals.items():
            path = stem.with_suffix(f".{extension}")
            script = (
                f"set terminal {terminal}\n"
                f"set output {self._quote(str(path.resolve()))}\n"
                f"{commands}\n"
                "unset output\n"
            )
            result = subprocess.run(
                ["gnuplot"],
                input=script,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"GNUPlot failed for {path.name}: {result.stderr.strip()}"
                )
            paths.append(path)
        return paths

    def _validate_outputs(
        self,
        tables: dict[str, pd.DataFrame],
        table_paths: dict[str, Path],
        figure_paths: list[Path],
    ) -> None:
        """Verify exact table coverage and the aggregate-only package boundary."""

        expected_rows = {
            "dataset_and_roles": 10,
            "baseline_vs_final": 11,
            "qid_transformation": self.N_QIDS,
            "sensitive_attribute_results": self.N_SENSITIVE_ATTRIBUTES,
            "final_cluster_summary": 4,
            "robustness_summary": 4,
            "final_screening_checks": 10,
            "experiment_coverage": 4,
        }
        for name, expected in expected_rows.items():
            if len(tables[name]) != expected:
                raise ValueError(f"{name} has {len(tables[name])} rows; expected {expected}.")
            reloaded = pd.read_csv(table_paths[name])
            if len(reloaded) != expected:
                raise ValueError(f"Saved {name} table could not be reloaded exactly.")
        if any(
            column.startswith("centroid_")
            for table in tables.values()
            for column in table.columns
        ):
            raise ValueError("Aggregate evidence must not contain cluster centroids.")
        if len(figure_paths) != 10 or not all(
            path.is_file() and path.stat().st_size > 1_000 for path in figure_paths
        ):
            raise ValueError("The expected five PNG/SVG figure pairs were not created.")

    def _manifest(
        self,
        evidence: dict[str, object],
        output_paths: list[Path],
    ) -> dict[str, object]:
        inputs = [
            self.config.source_path,
            self.config.baseline_directory / "prti_run_summary.json",
            self.config.final_directory / "prti_run_summary.json",
            self.config.final_directory / "prti_input_audit.csv",
            self.config.final_directory / "prti_before_after_anonymization.csv",
            self.config.baseline_directory / "prti_sensitive_entropy.csv",
            self.config.final_directory / "prti_sensitive_entropy.csv",
            self.config.final_directory / "prti_cluster_details.csv",
            self.config.final_acceptance,
            self.config.sensitivity_plan,
            self.config.sensitivity_results,
            self.config.robustness_summary,
        ]
        gnuplot_version = subprocess.run(
            ["gnuplot", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return {
            "manifest_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": "Aggregate PRTI-GH MO-OBAM results package; ML excluded",
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "patient_level_rows_in_package": False,
            "centroids_in_package": False,
            "source_sha256": EXPECTED_PRTI_SHA256,
            "final_run_id": evidence["final"]["run_id"],
            "final_screening_status": evidence["acceptance"]["status"],
            "release_boundary": evidence["acceptance"]["release_boundary"],
            "canonical_command": ".venv/bin/python generate_prti_results_package.py",
            "figure_renderer": gnuplot_version,
            "inputs": [self._file_record(path) for path in inputs],
            "outputs": [self._file_record(path) for path in output_paths],
            "manifest_policy": "The manifest does not hash itself.",
            "generation_snapshot": self.manifest_writer.capture_snapshot(
                "captured_when_results_v2_package_was_generated"
            ),
        }

    def _readme(self, evidence: dict[str, object]) -> str:
        final = evidence["final"]
        metrics = final["final_metrics"]
        privacy = final["privacy_evaluation"]["anonymized"]
        sensitivity = evidence["sensitivity"]
        sensitivity_completed = int(sensitivity["status"].eq("completed").sum())
        sensitivity_no_feasible = int(
            sensitivity["status"].eq("no_feasible_particle").sum()
        )
        sensitivity_infeasible = int(
            sensitivity["status"].eq("mathematically_infeasible").sum()
        )
        robustness = evidence["robustness"]
        return f"""# PRTI-GH aggregate results package v2

This directory contains reporting-ready aggregate evidence for the recorded
PRTI-GH MO-OBAM experiments. It contains no patient rows, QID representatives,
cluster centroids, prepared records, or anonymized records.

## Held-out candidate recorded here

- Configuration: k=10, 4 clusters, lambda=1, population=100, iterations=100.
- Held-out seed: 10.
- Information loss: {metrics['information_loss']}.
- Sum of minimum sensitive-attribute entropy: {metrics['entropy_sum']}.
- Released class sizes: {', '.join(str(value) for value in final['cluster_sizes'])}.
- Linkage records above tau=0.05, 0.075, and 0.1: {', '.join(str(privacy['linkage_attack_records'][value]) for value in self.LINKAGE_THRESHOLDS)}.
- Homogeneity-exposed records for any SA: {privacy['homogeneity_attack_records']}.
- Sensitive attributes with zero minimum cluster entropy: {self._zero_entropy_count(final)}.

All ten predeclared project checks passed. This is not approval for public or
unrestricted release; formal role review and independent disclosure-control
review are still required.

## Evidence coverage

- Sensitivity grid: {len(sensitivity)} of 220 declared points recorded:
  {sensitivity_completed} completed, {sensitivity_no_feasible} `no_feasible_particle`,
  {sensitivity_infeasible} mathematically infeasible, and
  {220 - len(sensitivity)} not yet evaluated.
- One-factor subset: 18 of those sensitivity points — 9 completed and 9
  `no_feasible_particle`; it is not an additional experiment total.
- Robustness study: {robustness['completed_runs']} of {robustness['planned_runs']}
  seed runs completed; the remaining {robustness['no_feasible_particle_runs']}
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
"""

    def _file_record(self, path: Path) -> dict[str, object]:
        resolved = path.resolve()
        try:
            display = str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            display = str(resolved)
        return {
            "path": display,
            "bytes": resolved.stat().st_size,
            "sha256": self.manifest_writer.file_sha256(resolved),
        }

    @staticmethod
    def _zero_entropy_count(summary: dict[str, object]) -> int:
        return int(
            sum(
                np.isclose(float(value), 0.0)
                for value in summary["final_metrics"][
                    "entropy_by_sensitive_attribute"
                ].values()
            )
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError(f"Required evidence file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
