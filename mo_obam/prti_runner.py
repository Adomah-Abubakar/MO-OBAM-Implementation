"""Audit and run the MO-OBAM formulation on the PRTI-GH dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import re
import time
from typing import Callable

import numpy as np
import pandas as pd

from .formulation import MOOBAMFormulation
from .privacy_metrics import PrivacyRiskEvaluator, PrivacyRiskMetrics
from .provenance import ArtifactManifestWriter, atomic_write_text
from .prti_data import (
    DEFAULT_PRTI_SOURCE,
    PRTI_QIDS,
    PRTIDataAdapter,
    PRTIPreparedData,
)
from .pso_solver import MOOBAMParticleSwarmOptimizer, PSOConfig, ScalableMOOBAMResult
from .schema import MOOBAMConfig


PAPER_IDENTIFIER = "arXiv:2501.01002v2"
PAPER_REVISION = "15 May 2025"
PAPER_URL = "https://arxiv.org/abs/2501.01002v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PRTIRunConfig:
    """Store source, output, model, and PSO settings for one PRTI-GH run."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    output_directory: Path = Path("outputs/prti/baseline-v1-seed9")
    # PRTI-GH baseline, not a value claimed by the paper. C=30 is the largest
    # tested paper-grid cluster count that produced hard-k-feasible particles
    # for this 1,485-row dataset with the recorded seed and population.
    model: MOOBAMConfig = MOOBAMConfig(k=10, n_clusters=30, lambda_weight=1.0)
    pso: PSOConfig = PSOConfig(population_size=100, iterations=100, seed=9)
    run_label: str = "baseline-v1-seed9"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_label):
            raise ValueError(
                "run_label must start with a letter or number and contain only "
                "letters, numbers, dots, underscores, or hyphens."
            )


class PRTIAnonymizationRunner:
    """Prepare, anonymize, validate, and export a complete PRTI-GH release."""

    def __init__(self, config: PRTIRunConfig) -> None:
        self.config = config
        self.adapter = PRTIDataAdapter(config.source_path)

    def audit_only(self) -> dict[str, object]:
        """Prepare and inspect the data without executing the optimizer."""

        prepared = self.adapter.prepare()
        outputs = self.adapter.write_audit_outputs(
            self.config.output_directory, prepared=prepared
        )
        summary = dict(prepared.summary)
        summary["mode"] = "audit-only; no anonymization executed"
        summary["paper_provenance"] = self._paper_provenance()
        summary["runtime_environment"] = self._runtime_environment()
        summary["outputs"] = {name: path.name for name, path in outputs.items()}
        summary_path = outputs["preparation_summary"]
        atomic_write_text(summary_path, json.dumps(summary, indent=2))
        return summary

    def run(
        self,
        progress: Callable[[str, int, int, float, int, int], None] | None = None,
    ) -> dict[str, object]:
        """Execute one labeled PSO run and save aligned research artifacts."""

        started = time.monotonic()
        started_at = datetime.now(timezone.utc)
        manifest_writer = ArtifactManifestWriter(PROJECT_ROOT)
        generation_snapshot = manifest_writer.capture_snapshot()
        prepared = self.adapter.prepare()
        audit_outputs = self.adapter.write_audit_outputs(
            self.config.output_directory, prepared=prepared
        )

        formulation = MOOBAMFormulation(
            prepared.model_input, prepared.schema, self.config.model
        )
        result = MOOBAMParticleSwarmOptimizer(formulation, self.config.pso).solve(
            progress=progress
        )
        decoded = self._decode_result(prepared, result)
        before_after = self._build_before_after(prepared.original, decoded)
        validation = self._validate(prepared, result, decoded, before_after)
        baseline_risk = PrivacyRiskEvaluator(
            prepared.model_input, prepared.schema
        ).evaluate()
        anonymized_risk = PrivacyRiskEvaluator(
            result.anonymized_data.drop(columns="cluster"), prepared.schema
        ).evaluate()
        model_evaluation = self._model_evaluation(
            baseline_risk, anonymized_risk, result
        )
        cluster_details, sensitive_entropy = self._cluster_tables(
            decoded, result, prepared
        )

        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "before_after_anonymization": output
            / "prti_before_after_anonymization.csv",
            "anonymized_final": output / "prti_anonymized_final.csv",
            "cluster_details": output / "prti_cluster_details.csv",
            "sensitive_entropy": output / "prti_sensitive_entropy.csv",
            "model_evaluation": output / "prti_model_evaluation.csv",
            "run_summary": output / "prti_run_summary.json",
            "artifact_manifest": output / "prti_artifact_manifest.json",
        }
        before_after.to_csv(paths["before_after_anonymization"], index=False)
        decoded.drop(columns="cluster").to_csv(paths["anonymized_final"], index=False)
        cluster_details.to_csv(paths["cluster_details"], index=False)
        sensitive_entropy.to_csv(paths["sensitive_entropy"], index=False)
        model_evaluation.to_csv(paths["model_evaluation"], index=False)
        validation["saved_csvs"] = self._validate_saved_exports(
            {**audit_outputs, **paths}, prepared
        )

        summary: dict[str, object] = {
            "run_id": self._run_id(started_at),
            "run_label": self.config.run_label,
            "run_started_at_utc": started_at.isoformat(timespec="seconds"),
            "run_completed_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "method": {
                "paper": "Wei et al., Multi-Objective Optimization-Based Anonymization",
                **self._paper_provenance(),
                "paper_equations": "Active formulation, Equations (9)-(20)",
                "objective": "information_loss - lambda * entropy_sum",
                "search": "Algorithm 1 PSO with the authors' public update mechanics",
                "seed_boundary": self._seed_provenance(),
                "boundary": (
                    "This is a best-found stochastic PSO solution, not proof of a "
                    "global optimum. Missing-value representation and PRTI-GH column "
                    "roles are explicit project policies, not rules supplied by the paper."
                ),
            },
            "execution": {
                "canonical_command": self._canonical_command(),
                "working_directory": ".",
            },
            "generation_snapshot": generation_snapshot,
            "preparation": prepared.summary,
            "runtime_environment": self._runtime_environment(),
            "schema": asdict(prepared.schema),
            "model_configuration": asdict(self.config.model),
            "pso_configuration": asdict(self.config.pso),
            "final_metrics": asdict(result.metrics),
            "cluster_sizes": list(result.cluster_sizes),
            "stages": [asdict(stage) for stage in result.stages],
            "privacy_evaluation": {
                "prepared_baseline": asdict(baseline_risk),
                "anonymized": asdict(anonymized_risk),
            },
            "validation": validation,
            "runtime_seconds": time.monotonic() - started,
            "outputs": {
                **{name: path.name for name, path in audit_outputs.items()},
                **{name: path.name for name, path in paths.items()},
            },
        }
        atomic_write_text(paths["run_summary"], json.dumps(summary, indent=2))
        manifest_writer.write(
            output,
            self.config.source_path,
            summary,
        )
        return summary

    def refresh_saved_manifest(self) -> Path:
        """Refresh hashes and code provenance for an existing saved run."""

        summary_path = self.config.output_directory / "prti_run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Saved run summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        outputs = summary.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("Saved run summary does not contain an outputs dictionary.")
        outputs["artifact_manifest"] = "prti_artifact_manifest.json"
        atomic_write_text(summary_path, json.dumps(summary, indent=2))
        return ArtifactManifestWriter(PROJECT_ROOT).write(
            self.config.output_directory,
            self.config.source_path,
            summary,
        )

    def _run_id(self, started_at: datetime) -> str:
        """Build a readable identifier from time and explicit configuration."""

        timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        weight = str(self.config.model.lambda_weight).replace(".", "p")
        return (
            f"prti-{timestamp}-{self.config.run_label}"
            f"-k{self.config.model.k}"
            f"-c{self.config.model.n_clusters}"
            f"-lambda{weight}"
            f"-seed{self.config.pso.seed}"
        )

    def _canonical_command(self) -> list[str]:
        """Return a complete command that reproduces this configuration."""

        source = os.path.relpath(Path(self.config.source_path).resolve(), PROJECT_ROOT)
        output = os.path.relpath(
            Path(self.config.output_directory).resolve(), PROJECT_ROOT
        )
        return [
            ".venv/bin/python",
            "run_prti.py",
            "--source",
            source,
            "--output-directory",
            output,
            "--run-label",
            self.config.run_label,
            "--k",
            str(self.config.model.k),
            "--clusters",
            str(self.config.model.n_clusters),
            "--lambda-weight",
            str(self.config.model.lambda_weight),
            "--population",
            str(self.config.pso.population_size),
            "--iterations",
            str(self.config.pso.iterations),
            "--seed",
            str(self.config.pso.seed),
        ]

    def _seed_provenance(self) -> str:
        """Explain whether the recorded seed is the evaluated PRTI default."""

        if self.config.pso.seed == 9:
            return (
                "Seed 9 was selected after a deterministic 20-iteration screen "
                "of seeds 0-9, followed by a complete 100-iteration run. The "
                "paper does not publish a PRTI-GH seed or use this dataset."
            )
        return (
            f"Seed {self.config.pso.seed} was explicitly configured. The paper "
            "does not publish a PRTI-GH seed or use this dataset."
        )

    @staticmethod
    def _paper_provenance() -> dict[str, str]:
        """Return the exact paper revision used by the implementation."""

        return {
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "paper_url": PAPER_URL,
        }

    @staticmethod
    def _runtime_environment() -> dict[str, object]:
        """Record the interpreter and packages needed to reproduce a run."""

        return {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
            "packages": {
                name: package_version(name)
                for name in ("numpy", "pandas", "scipy", "xlrd")
            },
        }

    def _decode_result(
        self,
        prepared: PRTIPreparedData,
        result: ScalableMOOBAMResult,
    ) -> pd.DataFrame:
        """Decode categorical identifiers while retaining QID representatives."""

        model_columns = result.anonymized_data.loc[:, prepared.original.columns]
        decoded = self.adapter.decode_model_data(
            model_columns,
            prepared.original,
            prepared.decoding_maps,
        )
        decoded["cluster"] = result.anonymized_data["cluster"].to_numpy()
        return decoded

    @staticmethod
    def _build_before_after(
        original: pd.DataFrame,
        anonymized: pd.DataFrame,
    ) -> pd.DataFrame:
        """Place each QID's source and representative values side by side."""

        paired = pd.DataFrame(
            {
                "recordPosition": range(len(original)),
                "cluster": anonymized["cluster"].to_numpy(),
            }
        )
        for column in original.columns:
            if column in PRTI_QIDS:
                paired[f"{column}Before"] = original[column].to_numpy()
                paired[f"{column}After"] = anonymized[column].to_numpy()
            else:
                paired[column] = original[column].to_numpy()
        return paired

    def _validate_saved_exports(
        self,
        paths: dict[str, Path],
        prepared: PRTIPreparedData,
    ) -> dict[str, object]:
        """Reload every CSV and verify that the saved artifacts agree."""

        prepared_csv = pd.read_csv(paths["prepared_input"], keep_default_na=False)
        audit = pd.read_csv(paths["input_audit"])
        mappings = pd.read_csv(paths["category_mappings"], keep_default_na=False)
        paired = pd.read_csv(paths["before_after_anonymization"])
        final = pd.read_csv(paths["anonymized_final"])
        clusters = pd.read_csv(paths["cluster_details"])
        entropy = pd.read_csv(paths["sensitive_entropy"])
        evaluation = pd.read_csv(paths["model_evaluation"])

        original_columns = list(prepared.original.columns)
        expected_paired_columns = ["recordPosition", "cluster"]
        for column in original_columns:
            if column in prepared.schema.qids:
                expected_paired_columns.extend(
                    [f"{column}Before", f"{column}After"]
                )
            else:
                expected_paired_columns.append(column)

        if prepared_csv.shape != prepared.model_input.shape:
            raise AssertionError("Saved prepared input has the wrong shape.")
        if list(prepared_csv.columns) != original_columns:
            raise AssertionError("Saved prepared input changed the column order.")
        if len(audit) != len(original_columns):
            raise AssertionError("Saved input audit does not contain every attribute.")
        if not audit["distributionStatus"].eq("PASS").all():
            raise AssertionError("Saved input audit contains a distribution failure.")
        if audit["largestFrequencyDifferenceAfterRoundTrip"].max() != 0:
            raise AssertionError("Saved input audit reports a changed distribution.")

        mapping_attributes = set(mappings["attribute"])
        if mapping_attributes != set(prepared.decoding_maps):
            raise AssertionError("Saved category mappings omit or add an attribute.")
        mapping_totals = mappings.groupby("attribute", sort=False)["count"].sum()
        if not mapping_totals.eq(len(prepared.original)).all():
            raise AssertionError("Saved category mapping counts do not cover all rows.")

        if list(final.columns) != original_columns:
            raise AssertionError("Saved final release changed the source columns.")
        if len(final) != len(prepared.original):
            raise AssertionError("Saved final release changed the number of rows.")
        if list(paired.columns) != expected_paired_columns:
            raise AssertionError("Saved paired output has the wrong columns or order.")
        if paired["recordPosition"].tolist() != list(range(len(paired))):
            raise AssertionError("Saved paired output changed record alignment.")

        for column in prepared.schema.qids:
            if not self._series_equal(paired[f"{column}After"], final[column]):
                raise AssertionError(
                    f"Saved after-values differ from final QID {column!r}."
                )
        for column in prepared.schema.sensitive_attributes:
            if not self._series_equal(paired[column], final[column]):
                raise AssertionError(
                    f"Saved sensitive values differ for {column!r}."
                )

        expected_clusters = set(range(self.config.model.n_clusters))
        if set(clusters["cluster"]) != expected_clusters:
            raise AssertionError("Saved cluster details omit or add a cluster.")
        if int(clusters["records"].sum()) != len(final):
            raise AssertionError("Saved cluster sizes do not cover every record.")
        if int(clusters["records"].min()) < self.config.model.k:
            raise AssertionError("Saved cluster details violate k-anonymity.")

        expected_entropy_rows = self.config.model.n_clusters * len(
            prepared.schema.sensitive_attributes
        )
        if len(entropy) != expected_entropy_rows:
            raise AssertionError("Saved entropy table has the wrong number of rows.")
        if entropy.duplicated(["cluster", "sensitiveAttribute"]).any():
            raise AssertionError("Saved entropy table contains duplicate rows.")
        if len(evaluation) != 2:
            raise AssertionError("Saved evaluation must contain baseline and result rows.")
        expected_evaluation_columns = [
            "datasetVersion",
            "k",
            "configuredClusters",
            "equivalenceClasses",
            "lambda",
            "informationLoss",
            "entropySum",
            "linkageAttackRecordsTau0.05",
            "linkageAttackRecordsTau0.075",
            "linkageAttackRecordsTau0.1",
            "homogeneityAttackRecordsAnySA",
            "minimumEquivalenceClassSize",
            "maximumEquivalenceClassSize",
        ]
        if list(evaluation.columns) != expected_evaluation_columns:
            raise AssertionError(
                "Saved evaluation columns do not match the documented report schema."
            )

        return {
            "all_csvs_reloaded": True,
            "prepared_input_shape": list(prepared_csv.shape),
            "before_after_shape": list(paired.shape),
            "final_release_shape": list(final.shape),
            "after_qids_match_final_release": True,
            "sensitive_attributes_match_final_release": True,
            "cluster_rows_sum_to_source_rows": True,
            "entropy_rows": len(entropy),
        }

    @staticmethod
    def _series_equal(left: pd.Series, right: pd.Series) -> bool:
        """Compare reloaded CSV columns while treating paired missing cells as equal."""

        equal = left.eq(right) | (left.isna() & right.isna())
        return bool(equal.all())

    def _validate(
        self,
        prepared: PRTIPreparedData,
        result: ScalableMOOBAMResult,
        decoded: pd.DataFrame,
        before_after: pd.DataFrame,
    ) -> dict[str, object]:
        """Verify alignment, unchanged SAs, source integrity, and released k."""

        original = prepared.original
        if len(original) != len(decoded) or len(original) != len(before_after):
            raise AssertionError("PRTI-GH source and anonymized rows are not aligned.")

        changed_sensitive: list[str] = []
        for column in prepared.schema.sensitive_attributes:
            before = original[column]
            after = decoded[column]
            equal = before.eq(after) | (before.isna() & after.isna())
            if not bool(equal.all()):
                changed_sensitive.append(column)
        if changed_sensitive:
            raise AssertionError(
                f"Sensitive attributes changed during anonymization: {changed_sensitive}"
            )

        changed_qid_cells = {
            column: int(
                (~(
                    original[column].eq(decoded[column])
                    | (original[column].isna() & decoded[column].isna())
                )).sum()
            )
            for column in PRTI_QIDS
        }
        equivalence_sizes = result.anonymized_data.groupby(
            list(prepared.schema.qids), dropna=False, observed=True
        ).size()
        minimum_size = int(equivalence_sizes.min())
        if minimum_size < self.config.model.k:
            raise AssertionError("Released QID combinations do not satisfy k-anonymity.")

        source_hash_after = self.adapter.file_sha256(self.config.source_path)
        source_hash_before = prepared.summary["source_sha256_before"]
        if source_hash_after != source_hash_before:
            raise AssertionError("The PRTI-GH source workbook changed during the run.")
        return {
            "rows_aligned": True,
            "source_shape": list(original.shape),
            "before_after_shape": list(before_after.shape),
            "all_sensitive_attributes_unchanged": True,
            "changed_qid_cells": changed_qid_cells,
            "released_equivalence_classes": int(len(equivalence_sizes)),
            "minimum_released_equivalence_class_size": minimum_size,
            "all_cluster_sizes_at_least_k": all(
                size >= self.config.model.k for size in result.cluster_sizes
            ),
            "source_sha256_after": source_hash_after,
            "source_unchanged": True,
        }

    def _model_evaluation(
        self,
        baseline: PrivacyRiskMetrics,
        anonymized: PrivacyRiskMetrics,
        result: ScalableMOOBAMResult,
    ) -> pd.DataFrame:
        """Create a compact observed-before/after privacy and utility table."""

        rows = []
        for label, risk in (
            ("Prepared baseline", baseline),
            (f"MO-OBAM {self.config.run_label}", anonymized),
        ):
            is_result = label.startswith("MO-OBAM")
            rows.append(
                {
                    "datasetVersion": label,
                    "k": self.config.model.k if is_result else None,
                    "configuredClusters": (
                        self.config.model.n_clusters if is_result else None
                    ),
                    "equivalenceClasses": risk.equivalence_classes,
                    "lambda": self.config.model.lambda_weight if is_result else None,
                    "informationLoss": (
                        result.metrics.information_loss if is_result else None
                    ),
                    "entropySum": result.metrics.entropy_sum if is_result else None,
                    "linkageAttackRecordsTau0.05": risk.linkage_attack_records["0.05"],
                    "linkageAttackRecordsTau0.075": risk.linkage_attack_records["0.075"],
                    "linkageAttackRecordsTau0.1": risk.linkage_attack_records["0.1"],
                    "homogeneityAttackRecordsAnySA": risk.homogeneity_attack_records,
                    "minimumEquivalenceClassSize": risk.minimum_equivalence_class_size,
                    "maximumEquivalenceClassSize": risk.maximum_equivalence_class_size,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _cluster_tables(
        decoded: pd.DataFrame,
        result: ScalableMOOBAMResult,
        prepared: PRTIPreparedData,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create compact cluster rows and a long per-SA entropy audit table."""

        entropy_frame = MOOBAMFormulation(
            prepared.model_input,
            prepared.schema,
            MOOBAMConfig(
                k=min(result.cluster_sizes),
                n_clusters=len(result.cluster_sizes),
                lambda_weight=0.0,
            ),
        ).cluster_entropies(result.membership)
        entropy_frame = entropy_frame.mask(np.isclose(entropy_frame, 0.0), 0.0)

        cluster_rows: list[dict[str, object]] = []
        entropy_rows: list[dict[str, object]] = []
        for cluster, cluster_data in decoded.groupby("cluster", sort=True):
            cluster_number = int(cluster)
            entropies = entropy_frame.iloc[cluster_number]
            row: dict[str, object] = {
                "cluster": cluster_number,
                "records": int(len(cluster_data)),
                "proportion": len(cluster_data) / len(decoded),
                "minimumSensitiveEntropy": float(entropies.min()),
                "meanSensitiveEntropy": float(entropies.mean()),
                "maximumSensitiveEntropy": float(entropies.max()),
                "zeroEntropySensitiveAttributes": int(np.isclose(entropies, 0.0).sum()),
            }
            for column in prepared.schema.qids:
                row[f"centroid_{column}"] = cluster_data[column].iloc[0]
            cluster_rows.append(row)

            model_cluster = result.anonymized_data["cluster"].eq(cluster_number)
            for column in prepared.schema.sensitive_attributes:
                entropy_rows.append(
                    {
                        "cluster": cluster_number,
                        "sensitiveAttribute": column,
                        "records": int(model_cluster.sum()),
                        "distinctValues": int(
                            prepared.model_input.loc[model_cluster, column].nunique(
                                dropna=False
                            )
                        ),
                        "entropy": float(entropies[column]),
                    }
                )
        return pd.DataFrame(cluster_rows), pd.DataFrame(entropy_rows)

    @staticmethod
    def print_saved_tables(output_directory: str | Path) -> None:
        """Print the compact saved tables without rerunning optimization."""

        output = Path(output_directory)
        for title, filename, columns in (
            ("PRTI-GH model evaluation", "prti_model_evaluation.csv", None),
            (
                "PRTI-GH cluster summary",
                "prti_cluster_details.csv",
                [
                    "cluster",
                    "records",
                    "proportion",
                    "minimumSensitiveEntropy",
                    "meanSensitiveEntropy",
                    "maximumSensitiveEntropy",
                    "zeroEntropySensitiveAttributes",
                ],
            ),
        ):
            path = output / filename
            if not path.is_file():
                raise FileNotFoundError(f"Saved result table not found: {path}")
            table = pd.read_csv(path)
            if columns is not None:
                table = table.loc[:, columns]
            print(f"\n{title}\n{'=' * len(title)}")
            print(table.to_string(index=False, na_rep="-"))
