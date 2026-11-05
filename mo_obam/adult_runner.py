"""End-to-end Adult MO-OBAM run with the requested consolidated outputs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .adult_data import ADULT_COLUMNS, AdultDataPreparer
from .adult_reporting import AdultOutputReporter
from .formulation import MOOBAMFormulation
from .pso_solver import MOOBAMParticleSwarmOptimizer, PSOConfig, ScalableMOOBAMResult
from .schema import MOOBAMConfig, MOOBAMSchema


ADULT_SCHEMA = MOOBAMSchema(
    numeric_qids=("age",),
    categorical_qids=("race", "sex", "marital_status"),
    sensitive_attributes=("occupation",),
)
PAPER_REVISION = "arXiv:2501.01002v2 (15 May 2025)"


@dataclass(frozen=True)
class AdultRunConfig:
    """Store paths plus paper and repository parameters for one Adult run."""

    data_directory: Path
    output_directory: Path
    model: MOOBAMConfig = MOOBAMConfig(k=5, n_clusters=4, lambda_weight=1.0)
    pso: PSOConfig = PSOConfig(population_size=100, iterations=100, seed=0)


class AdultAnonymizationRunner:
    """Prepare Adult data, run MO-OBAM, validate it, and write all outputs."""

    def __init__(self, config: AdultRunConfig) -> None:
        self.config = config

    def run(
        self,
        progress: Callable[[str, int, int, float, int, int], None] | None = None,
    ) -> dict[str, object]:
        """Execute one complete run without modifying either UCI source file."""

        started = time.monotonic()
        original, preparation = AdultDataPreparer().prepare(self.config.data_directory)
        formulation = MOOBAMFormulation(original, ADULT_SCHEMA, self.config.model)
        result = MOOBAMParticleSwarmOptimizer(formulation, self.config.pso).solve(
            progress=progress
        )
        before_after = self._build_before_after(original, result)
        validation = self._validate(original, result, before_after)

        output_directory = self.config.output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        original_path = output_directory / "adult_prepared_original.csv"
        before_after_path = output_directory / "adult_before_after_anonymization.csv"
        summary_path = output_directory / "adult_run_summary.json"
        original.to_csv(original_path, index=False)
        before_after.to_csv(before_after_path, index=False)

        reporter = AdultOutputReporter(ADULT_SCHEMA)
        artifacts = reporter.build(
            original,
            result.anonymized_data,
            self.config.model,
            result.metrics.information_loss,
            result.metrics.entropy_sum,
        )
        report_outputs = reporter.write(output_directory, artifacts)

        summary: dict[str, object] = {
            "method": {
                "paper": "Wei et al., Multi-Objective Optimization-Based Anonymization",
                "paper_revision": PAPER_REVISION,
                "paper_equations": "Active formulation, Equations (9)-(20)",
                "objective": "information_loss - lambda * entropy_sum",
                "search": "Algorithm 1 PSO with authors' public update mechanics",
                "search_boundary": (
                    "The final solution is a best-found PSO estimate, not a proven "
                    "global optimum. Equation (20) is enforced as a hard constraint "
                    "because Algorithm 1's printed negative violation penalty conflicts "
                    "with both minimization and the authors' positive-penalty code."
                ),
            },
            "preparation": asdict(preparation),
            "schema": asdict(ADULT_SCHEMA),
            "model_configuration": asdict(self.config.model),
            "pso_configuration": asdict(self.config.pso),
            "final_metrics": asdict(result.metrics),
            "cluster_sizes": list(result.cluster_sizes),
            "stages": [asdict(stage) for stage in result.stages],
            "privacy_evaluation": {
                "prepared_baseline": reporter.risk_as_dict(artifacts.baseline_risk),
                "anonymized": reporter.risk_as_dict(artifacts.anonymized_risk),
            },
            "validation": validation,
            "runtime_seconds": time.monotonic() - started,
            "outputs": {
                "prepared_original": original_path.name,
                "before_after_anonymization": before_after_path.name,
                **report_outputs,
                "summary": summary_path.name,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def report_existing(self) -> dict[str, object]:
        """Build the final release and tables from an already completed run.

        This does not execute PSO. It is useful for upgrading older outputs
        that already contain the prepared original, before/after audit, and
        run summary.
        """

        output_directory = self.config.output_directory
        original_path = output_directory / "adult_prepared_original.csv"
        before_after_path = output_directory / "adult_before_after_anonymization.csv"
        summary_path = output_directory / "adult_run_summary.json"
        for path in (original_path, before_after_path, summary_path):
            if not path.exists():
                raise FileNotFoundError(f"Existing run artifact not found: {path}")

        original = pd.read_csv(original_path)
        before_after = pd.read_csv(before_after_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved_revision = summary.get("method", {}).get("paper_revision")
        if saved_revision != PAPER_REVISION:
            raise ValueError(
                "Existing artifacts were not produced by the current arXiv v2 "
                "implementation. Rerun run_adult.py instead of relabeling them."
            )
        saved_model = summary.get("model_configuration", {})
        model = MOOBAMConfig(
            k=int(saved_model["k"]),
            n_clusters=int(saved_model["n_clusters"]),
            lambda_weight=float(saved_model["lambda_weight"]),
        )
        final_metrics = summary.get("final_metrics", {})

        reporter = AdultOutputReporter(ADULT_SCHEMA)
        anonymized = reporter.reconstruct_from_audit(original, before_after)
        artifacts = reporter.build(
            original,
            anonymized,
            model,
            float(final_metrics["information_loss"]),
            float(final_metrics["entropy_sum"]),
        )
        report_outputs = reporter.write(output_directory, artifacts)
        summary["privacy_evaluation"] = {
            "prepared_baseline": reporter.risk_as_dict(artifacts.baseline_risk),
            "anonymized": reporter.risk_as_dict(artifacts.anonymized_risk),
        }
        summary.setdefault("outputs", {}).update(report_outputs)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    @staticmethod
    def _build_before_after(
        original: pd.DataFrame,
        result: ScalableMOOBAMResult,
    ) -> pd.DataFrame:
        """Place every QID's before/after values together in one reviewable CSV."""

        paired = pd.DataFrame(
            {
                "record_position": range(len(original)),
                "cluster": result.anonymized_data["cluster"].to_numpy(),
            }
        )
        qids = set(ADULT_SCHEMA.qids)
        for column in ADULT_COLUMNS:
            if column in qids:
                paired[f"{column}_before"] = original[column].to_numpy()
                paired[f"{column}_after"] = result.anonymized_data[column].to_numpy()
            else:
                paired[column] = original[column].to_numpy()
        return paired

    def _validate(
        self,
        original: pd.DataFrame,
        result: ScalableMOOBAMResult,
        before_after: pd.DataFrame,
    ) -> dict[str, object]:
        """Verify row alignment, unchanged non-QIDs, and released k-anonymity."""

        anonymized = result.anonymized_data
        if len(original) != len(anonymized) or len(original) != len(before_after):
            raise AssertionError("Before/after row counts are not aligned.")

        non_qids = [column for column in ADULT_COLUMNS if column not in ADULT_SCHEMA.qids]
        unchanged_columns = {
            column: bool(original[column].equals(anonymized[column]))
            for column in non_qids
        }
        if not all(unchanged_columns.values()):
            changed = [column for column, unchanged in unchanged_columns.items() if not unchanged]
            raise AssertionError(f"Non-QID columns changed: {changed}")

        changed_qid_cells = {
            column: int((original[column] != anonymized[column]).sum())
            for column in ADULT_SCHEMA.qids
        }
        equivalence_sizes = anonymized.groupby(
            list(ADULT_SCHEMA.qids), dropna=False, observed=True
        ).size()
        minimum_equivalence_size = int(equivalence_sizes.min())
        if minimum_equivalence_size < self.config.model.k:
            raise AssertionError("Released QID combinations do not satisfy k-anonymity.")

        return {
            "rows_aligned": True,
            "prepared_shape": list(original.shape),
            "before_after_shape": list(before_after.shape),
            "non_qid_columns_unchanged": unchanged_columns,
            "all_non_qid_columns_unchanged": True,
            "changed_qid_cells": changed_qid_cells,
            "configured_clusters": self.config.model.n_clusters,
            "released_equivalence_classes": int(len(equivalence_sizes)),
            "minimum_released_equivalence_class_size": minimum_equivalence_size,
            "all_cluster_sizes_at_least_k": all(
                size >= self.config.model.k for size in result.cluster_sizes
            ),
        }
