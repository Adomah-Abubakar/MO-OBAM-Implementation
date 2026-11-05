"""Reproducible aggregate experiments for the PRTI-GH MO-OBAM baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable

import numpy as np
import pandas as pd

from .adult_experiments import (
    PAPER_ADULT_CLUSTER_VALUES,
    PAPER_K_VALUES,
    PAPER_LAMBDA_VALUES,
)
from .formulation import MOOBAMFormulation
from .privacy_metrics import PrivacyRiskEvaluator
from .provenance import ArtifactManifestWriter
from .prti_data import DEFAULT_PRTI_SOURCE, PRTIDataAdapter
from .prti_runner import PAPER_IDENTIFIER, PAPER_REVISION, PROJECT_ROOT
from .pso_solver import (
    MOOBAMParticleSwarmOptimizer,
    NoFeasibleParticleError,
    PSOConfig,
)
from .schema import MOOBAMConfig


SeedProgressCallback = Callable[[int, int, int, float, int, int], None]
SensitivityProgressCallback = Callable[
    ["PRTISensitivityExperiment", int, int, float, int, int], None
]


@dataclass(frozen=True)
class PRTISensitivityExperiment:
    """Identify one PRTI-GH adaptation of a Wei Appendix A grid point."""

    k: int
    n_clusters: int
    lambda_weight: float

    @property
    def run_id(self) -> str:
        """Return a stable identifier for checkpoints and result tables."""

        lambda_label = format(self.lambda_weight, "g").replace(".", "p")
        return f"k{self.k:02d}_c{self.n_clusters:03d}_lambda{lambda_label}"

    @property
    def model_config(self) -> MOOBAMConfig:
        """Build the paper-constrained model configuration."""

        return MOOBAMConfig(
            k=self.k,
            n_clusters=self.n_clusters,
            lambda_weight=self.lambda_weight,
        )

    def is_mathematically_feasible(self, n_records: int) -> bool:
        """Check the necessary Equation (20) condition before running PSO."""

        return self.k * self.n_clusters <= n_records


class PRTISensitivityPlan:
    """Reuse the 220 hyperparameter combinations printed in Wei Appendix A."""

    @staticmethod
    def build() -> tuple[PRTISensitivityExperiment, ...]:
        """Return the exact paper grid, labelled as a PRTI-GH adaptation."""

        return tuple(
            PRTISensitivityExperiment(k=k, n_clusters=clusters, lambda_weight=weight)
            for k, clusters, weight in product(
                PAPER_K_VALUES,
                PAPER_ADULT_CLUSTER_VALUES,
                PAPER_LAMBDA_VALUES,
            )
        )


@dataclass(frozen=True)
class PRTISensitivityConfig:
    """Configure aggregate-only, resumable PRTI-GH sensitivity experiments."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    output_directory: Path = Path("evidence/prti-sensitivity-v1")
    pso: PSOConfig = PSOConfig(population_size=100, iterations=100, seed=9)


class PRTISensitivityRunner:
    """Run Wei's parameter grid on PRTI-GH without exporting patient-level data."""

    PLAN_FILE = "prti_sensitivity_plan.json"
    RESULTS_FILE = "prti_sensitivity_results.csv"
    RUNS_DIRECTORY = "runs"

    def __init__(self, config: PRTISensitivityConfig) -> None:
        self.config = config
        self.experiments = PRTISensitivityPlan.build()

    def write_plan(self) -> Path:
        """Record the paper-derived grid and its PRTI-specific interpretation."""

        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.config.output_directory / self.PLAN_FILE
        payload = {
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "scope": "PRTI-GH parameter sensitivity; ML paused",
            "status": "planned experiments, not paper-reported PRTI-GH results",
            "grid_source": (
                "The k, cluster-count, and lambda values are exactly the 220 "
                "combinations printed for Adult in Wei et al. Appendix A. Their "
                "application to PRTI-GH is a labelled project adaptation."
            ),
            "number_of_configurations": len(self.experiments),
            "pso_configuration": asdict(self.config.pso),
            "patient_level_outputs": False,
            "interpretation_boundary": (
                "The paper did not use PRTI-GH and does not publish a PRTI-GH "
                "seed. Results are local sensitivity evidence, not paper reproduction "
                "or proof of global optimality."
            ),
            "configurations": [
                {"run_id": experiment.run_id, **asdict(experiment)}
                for experiment in self.experiments
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def run(
        self,
        progress: SensitivityProgressCallback | None = None,
        max_new_runs: int | None = None,
        run_ids: set[str] | None = None,
        retry_blocked: bool = False,
    ) -> pd.DataFrame:
        """Execute pending configurations and retain aggregate checkpoints only."""

        if max_new_runs is not None and max_new_runs < 1:
            raise ValueError("max_new_runs must be at least 1 when supplied.")

        self.write_plan()
        prepared = PRTIDataAdapter(self.config.source_path).prepare()
        source_hash = str(prepared.summary["source_sha256_before"])
        rows: list[dict[str, object]] = []
        new_runs = 0

        for experiment in self.experiments:
            if run_ids is not None and experiment.run_id not in run_ids:
                continue
            checkpoint = self._load_checkpoint(experiment, source_hash)
            if checkpoint is not None:
                status = checkpoint["result_row"]["status"]
                if status != "no_feasible_particle" or not retry_blocked:
                    rows.append(checkpoint["result_row"])
                    continue
            if not experiment.is_mathematically_feasible(len(prepared.model_input)):
                row = self._infeasible_row(experiment, len(prepared.model_input))
                self._write_checkpoint(experiment, source_hash, row)
                rows.append(row)
                continue
            if max_new_runs is not None and new_runs >= max_new_runs:
                continue

            started = time.monotonic()
            new_runs += 1
            formulation = MOOBAMFormulation(
                prepared.model_input,
                prepared.schema,
                experiment.model_config,
            )

            def report_stage(
                _stage: str,
                iteration: int,
                total: int,
                best_score: float,
                feasible: int,
                infeasible: int,
            ) -> None:
                if progress is not None:
                    progress(
                        experiment,
                        iteration,
                        total,
                        best_score,
                        feasible,
                        infeasible,
                    )

            try:
                result = MOOBAMParticleSwarmOptimizer(
                    formulation,
                    self.config.pso,
                ).solve(progress=report_stage)
            except NoFeasibleParticleError as error:
                row = self._blocked_row(
                    experiment,
                    str(error),
                    time.monotonic() - started,
                )
                self._write_checkpoint(experiment, source_hash, row)
                rows.append(row)
                self._write_results(source_hash)
                continue

            risk = PrivacyRiskEvaluator(
                result.anonymized_data.drop(columns="cluster"),
                prepared.schema,
            ).evaluate()
            stage = result.stages[0]
            row = {
                **self._configuration_columns(experiment),
                "status": "completed",
                "issue": None,
                "information_loss": result.metrics.information_loss,
                "entropy_sum": result.metrics.entropy_sum,
                "objective": result.metrics.objective,
                "equivalence_classes": risk.equivalence_classes,
                "minimum_cluster_size": min(result.cluster_sizes),
                "maximum_cluster_size": max(result.cluster_sizes),
                "linkage_records_tau_0_05": risk.linkage_attack_records["0.05"],
                "linkage_records_tau_0_075": risk.linkage_attack_records["0.075"],
                "linkage_records_tau_0_1": risk.linkage_attack_records["0.1"],
                "homogeneity_records_any_sa": risk.homogeneity_attack_records,
                "zero_minimum_entropy_sensitive_attributes": int(
                    sum(
                        np.isclose(value, 0.0)
                        for value in result.metrics.entropy_by_sensitive_attribute.values()
                    )
                ),
                "best_iteration_zero_based": stage.iteration_found,
                "feasible_evaluations": stage.feasible_evaluations,
                "infeasible_evaluations": stage.infeasible_evaluations,
                "runtime_seconds": time.monotonic() - started,
            }
            self._write_checkpoint(experiment, source_hash, row)
            rows.append(row)
            self._write_results(source_hash)

        return self._write_results(source_hash)

    def _write_checkpoint(
        self,
        experiment: PRTISensitivityExperiment,
        source_hash: str,
        result_row: dict[str, object],
    ) -> None:
        """Save one aggregate result without assignments, centroids, or patient rows."""

        run_directory = self._run_directory(experiment)
        run_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "source_sha256": source_hash,
            "model_configuration": asdict(experiment.model_config),
            "pso_configuration": asdict(self.config.pso),
            "patient_level_outputs": False,
            "result_row": result_row,
        }
        (run_directory / "summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _load_checkpoint(
        self,
        experiment: PRTISensitivityExperiment,
        source_hash: str,
    ) -> dict[str, object] | None:
        """Reuse a checkpoint only when data, model, and PSO settings match."""

        path = self._run_directory(experiment) / "summary.json"
        if not path.exists():
            return None
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("paper_revision") != PAPER_REVISION
            or summary.get("source_sha256") != source_hash
            or summary.get("model_configuration") != asdict(experiment.model_config)
            or summary.get("pso_configuration") != asdict(self.config.pso)
            or summary.get("patient_level_outputs") is not False
        ):
            return None
        return summary

    def _write_results(self, source_hash: str) -> pd.DataFrame:
        """Collect matching checkpoints in the declared paper-grid order."""

        rows = []
        for experiment in self.experiments:
            checkpoint = self._load_checkpoint(experiment, source_hash)
            if checkpoint is not None:
                rows.append(checkpoint["result_row"])
        table = pd.DataFrame(rows)
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.config.output_directory / self.RESULTS_FILE, index=False)
        return table

    def _run_directory(self, experiment: PRTISensitivityExperiment) -> Path:
        return self.config.output_directory / self.RUNS_DIRECTORY / experiment.run_id

    def _configuration_columns(
        self, experiment: PRTISensitivityExperiment
    ) -> dict[str, object]:
        return {
            "run_id": experiment.run_id,
            "k": experiment.k,
            "configured_clusters": experiment.n_clusters,
            "lambda": experiment.lambda_weight,
            "population": self.config.pso.population_size,
            "iterations": self.config.pso.iterations,
            "seed": self.config.pso.seed,
        }

    def _infeasible_row(
        self,
        experiment: PRTISensitivityExperiment,
        n_records: int,
    ) -> dict[str, object]:
        return {
            **self._configuration_columns(experiment),
            "status": "mathematically_infeasible",
            "issue": (
                f"Equation (20) cannot hold because k * configured_clusters = "
                f"{experiment.k * experiment.n_clusters} exceeds n = {n_records}."
            ),
        }

    def _blocked_row(
        self,
        experiment: PRTISensitivityExperiment,
        issue: str,
        runtime_seconds: float,
    ) -> dict[str, object]:
        return {
            **self._configuration_columns(experiment),
            "status": "no_feasible_particle",
            "issue": (
                f"{issue} The run is recorded without inventing a k-penalty, "
                "repair rule, or result value."
            ),
            "runtime_seconds": runtime_seconds,
        }


@dataclass(frozen=True)
class PRTISeedScreenConfig:
    """Configure a labelled, aggregate-only screen of deterministic PSO seeds."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    output_directory: Path = Path("evidence/prti-baseline-v1")
    model: MOOBAMConfig = MOOBAMConfig(k=10, n_clusters=30, lambda_weight=1.0)
    population_size: int = 100
    iterations: int = 20
    seeds: tuple[int, ...] = tuple(range(10))

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Seed values must be unique.")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Seed values must be non-negative.")


class PRTISeedScreenRunner:
    """Run short PSO searches and retain every aggregate result without patient rows."""

    def __init__(self, config: PRTISeedScreenConfig) -> None:
        self.config = config

    def run(self, progress: SeedProgressCallback | None = None) -> dict[str, object]:
        """Execute every configured seed and save CSV/JSON selection evidence."""

        started = time.monotonic()
        started_at = datetime.now(timezone.utc)
        snapshot = ArtifactManifestWriter(PROJECT_ROOT).capture_snapshot()
        prepared = PRTIDataAdapter(self.config.source_path).prepare()
        formulation = MOOBAMFormulation(
            prepared.model_input,
            prepared.schema,
            self.config.model,
        )

        records: list[dict[str, object]] = []
        for seed in self.config.seeds:
            seed_started = time.monotonic()

            def seed_progress(
                _stage: str,
                iteration: int,
                total: int,
                best_score: float,
                feasible: int,
                infeasible: int,
            ) -> None:
                if progress is not None:
                    progress(
                        seed,
                        iteration,
                        total,
                        best_score,
                        feasible,
                        infeasible,
                    )

            try:
                result = MOOBAMParticleSwarmOptimizer(
                    formulation,
                    PSOConfig(
                        population_size=self.config.population_size,
                        iterations=self.config.iterations,
                        seed=seed,
                    ),
                ).solve(progress=seed_progress)
            except NoFeasibleParticleError as error:
                records.append(
                    {
                        "seed": seed,
                        "status": "no_feasible_particle",
                        "error": str(error),
                        "runtime_seconds": time.monotonic() - seed_started,
                    }
                )
                continue

            risk = PrivacyRiskEvaluator(
                result.anonymized_data.drop(columns="cluster"),
                prepared.schema,
            ).evaluate()
            stage = result.stages[0]
            records.append(
                {
                    "seed": seed,
                    "status": "completed",
                    "information_loss": result.metrics.information_loss,
                    "entropy_sum": result.metrics.entropy_sum,
                    "objective": result.metrics.objective,
                    "minimum_cluster_size": min(result.cluster_sizes),
                    "maximum_cluster_size": max(result.cluster_sizes),
                    "linkage_records_tau_0_05": risk.linkage_attack_records["0.05"],
                    "linkage_records_tau_0_075": risk.linkage_attack_records["0.075"],
                    "linkage_records_tau_0_1": risk.linkage_attack_records["0.1"],
                    "homogeneity_records_any_sa": risk.homogeneity_attack_records,
                    "zero_minimum_entropy_sensitive_attributes": int(
                        sum(
                            np.isclose(value, 0.0)
                            for value in result.metrics.entropy_by_sensitive_attribute.values()
                        )
                    ),
                    "best_iteration_zero_based": stage.iteration_found,
                    "feasible_evaluations": stage.feasible_evaluations,
                    "infeasible_evaluations": stage.infeasible_evaluations,
                    "runtime_seconds": time.monotonic() - seed_started,
                }
            )

        completed = [record for record in records if record["status"] == "completed"]
        selected = min(
            completed,
            key=lambda record: (float(record["objective"]), int(record["seed"])),
        ) if completed else None

        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / "prti_seed_screen.csv"
        json_path = output / "prti_seed_screen.json"
        pd.DataFrame(records).to_csv(csv_path, index=False)

        summary: dict[str, object] = {
            "experiment": "PRTI-GH deterministic PSO seed screen",
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "started_at_utc": started_at.isoformat(timespec="seconds"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "runtime_seconds": time.monotonic() - started,
            "source": {
                "filename": Path(self.config.source_path).name,
                "sha256": prepared.summary["source_sha256_before"],
                "rows": len(prepared.original),
                "columns": len(prepared.original.columns),
            },
            "model_configuration": asdict(self.config.model),
            "screen_configuration": {
                "population_size": self.config.population_size,
                "iterations": self.config.iterations,
                "seeds": list(self.config.seeds),
            },
            "selection_rule": (
                "Minimum MO-OBAM objective among completed screen runs; ties are "
                "resolved by the smallest seed. This is an exploratory local-screen "
                "rule, not a paper-specified estimator or proof of global optimality."
            ),
            "selected_seed": int(selected["seed"]) if selected else None,
            "canonical_command": self._canonical_command(),
            "generation_snapshot": snapshot,
            "runs": records,
            "outputs": {
                "csv": csv_path.name,
                "json": json_path.name,
            },
        }
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _canonical_command(self) -> list[str]:
        source = os.path.relpath(Path(self.config.source_path).resolve(), PROJECT_ROOT)
        output = os.path.relpath(
            Path(self.config.output_directory).resolve(), PROJECT_ROOT
        )
        return [
            ".venv/bin/python",
            "run_prti_experiments.py",
            "seed-screen",
            "--source",
            source,
            "--output-directory",
            output,
            "--k",
            str(self.config.model.k),
            "--clusters",
            str(self.config.model.n_clusters),
            "--lambda-weight",
            str(self.config.model.lambda_weight),
            "--population",
            str(self.config.population_size),
            "--iterations",
            str(self.config.iterations),
            "--seeds",
            *[str(seed) for seed in self.config.seeds],
        ]


class PRTIBaselineComparator:
    """Compare legacy and fresh summaries and export aggregate reproduction evidence."""

    METRICS = {
        "information_loss": lambda summary: summary["final_metrics"][
            "information_loss"
        ],
        "entropy_sum": lambda summary: summary["final_metrics"]["entropy_sum"],
        "objective": lambda summary: summary["final_metrics"]["objective"],
        "minimum_cluster_size": lambda summary: min(summary["cluster_sizes"]),
        "maximum_cluster_size": lambda summary: max(summary["cluster_sizes"]),
        "linkage_records_tau_0_05": lambda summary: summary["privacy_evaluation"][
            "anonymized"
        ]["linkage_attack_records"]["0.05"],
        "linkage_records_tau_0_075": lambda summary: summary["privacy_evaluation"][
            "anonymized"
        ]["linkage_attack_records"]["0.075"],
        "linkage_records_tau_0_1": lambda summary: summary["privacy_evaluation"][
            "anonymized"
        ]["linkage_attack_records"]["0.1"],
        "homogeneity_records_any_sa": lambda summary: summary["privacy_evaluation"][
            "anonymized"
        ]["homogeneity_attack_records"],
        "zero_minimum_entropy_sensitive_attributes": lambda summary: int(
            sum(
                np.isclose(value, 0.0)
                for value in summary["final_metrics"][
                    "entropy_by_sensitive_attribute"
                ].values()
            )
        ),
        "feasible_evaluations": lambda summary: summary["stages"][0][
            "feasible_evaluations"
        ],
        "infeasible_evaluations": lambda summary: summary["stages"][0][
            "infeasible_evaluations"
        ],
        "best_iteration_zero_based": lambda summary: summary["stages"][0][
            "iteration_found"
        ],
    }

    def compare(
        self,
        legacy_directory: str | Path,
        fresh_directory: str | Path,
        evidence_directory: str | Path,
    ) -> dict[str, object]:
        """Compare saved summaries and retain only non-row-level evidence."""

        legacy_path = Path(legacy_directory) / "prti_run_summary.json"
        fresh_path = Path(fresh_directory) / "prti_run_summary.json"
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))

        rows: list[dict[str, object]] = []
        for metric, extractor in self.METRICS.items():
            legacy_value = extractor(legacy)
            fresh_value = extractor(fresh)
            rows.append(
                {
                    "metric": metric,
                    "legacy_value": legacy_value,
                    "fresh_value": fresh_value,
                    "difference": float(fresh_value) - float(legacy_value),
                    "exact_match": bool(fresh_value == legacy_value),
                    "numerically_close": bool(
                        np.isclose(fresh_value, legacy_value, rtol=0.0, atol=1e-12)
                    ),
                }
            )

        configuration_match = (
            legacy["model_configuration"] == fresh["model_configuration"]
            and legacy["pso_configuration"] == fresh["pso_configuration"]
        )
        cluster_sizes_match = legacy["cluster_sizes"] == fresh["cluster_sizes"]
        source_hash_match = (
            legacy["preparation"]["source_sha256_before"]
            == fresh["preparation"]["source_sha256_before"]
        )
        exact_metrics = all(bool(row["exact_match"]) for row in rows)

        output = Path(evidence_directory)
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / "prti_fresh_vs_legacy_comparison.csv"
        json_path = output / "prti_fresh_vs_legacy_comparison.json"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        comparison: dict[str, object] = {
            "comparison": "fresh labelled PRTI-GH run versus legacy seed-9 run",
            "legacy_run": {
                "run_id": legacy.get("run_id"),
                "run_label": legacy.get("run_label"),
                "timestamps": "not recorded" if not legacy.get("run_started_at_utc") else {
                    "started_at_utc": legacy.get("run_started_at_utc"),
                    "completed_at_utc": legacy.get("run_completed_at_utc"),
                },
            },
            "fresh_run": {
                "run_id": fresh.get("run_id"),
                "run_label": fresh.get("run_label"),
                "started_at_utc": fresh.get("run_started_at_utc"),
                "completed_at_utc": fresh.get("run_completed_at_utc"),
                "canonical_command": fresh.get("execution", {}).get(
                    "canonical_command"
                ),
                "generation_git": fresh.get("generation_snapshot", {}).get("git"),
            },
            "configuration_match": configuration_match,
            "cluster_sizes_exact_match": cluster_sizes_match,
            "source_hash_match": source_hash_match,
            "all_selected_metrics_exact_match": exact_metrics,
            "reproduced_exactly": bool(
                configuration_match
                and cluster_sizes_match
                and source_hash_match
                and exact_metrics
            ),
            "metrics": rows,
            "outputs": {
                "comparison_csv": csv_path.name,
                "comparison_json": json_path.name,
                "fresh_artifact_manifest": "prti_fresh_artifact_manifest.json",
            },
        }
        json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

        manifest_source = Path(fresh_directory) / "prti_artifact_manifest.json"
        shutil.copyfile(
            manifest_source,
            output / "prti_fresh_artifact_manifest.json",
        )
        return comparison
