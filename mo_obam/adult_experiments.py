"""Run the non-ML Adult MO-OBAM experiments specified by arXiv v2."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from .adult_data import ADULT_COLUMNS, AdultDataPreparer
from .adult_reporting import (
    PAPER_ADULT_MO_OBAM_BOUNDARIES,
    AdultOutputReporter,
)
from .adult_runner import ADULT_SCHEMA, PAPER_REVISION
from .formulation import MOOBAMFormulation, MOOBAMMetrics
from .privacy_metrics import PrivacyRiskEvaluator, PrivacyRiskMetrics
from .pso_solver import (
    MOOBAMParticleSwarmOptimizer,
    NoFeasibleParticleError,
    PSOConfig,
    ScalableMOOBAMResult,
)
from .schema import MOOBAMConfig


AdultPreset = Literal["table-6", "full-grid"]
ExperimentProgress = Callable[["AdultExperiment", int, int, float, int, int], None]

PAPER_K_VALUES = (5, 10, 15, 20)
PAPER_ADULT_CLUSTER_VALUES = (4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
PAPER_LAMBDA_VALUES = (0.0001, 0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class AdultExperiment:
    """Identify one point in the paper's Adult MO-OBAM parameter grid."""

    k: int
    n_clusters: int
    lambda_weight: float
    paper_table_section: str | None = None

    @property
    def run_id(self) -> str:
        """Return a stable, filesystem-safe identifier for this configuration."""

        lambda_label = format(self.lambda_weight, "g").replace(".", "p")
        return f"k{self.k:02d}_c{self.n_clusters:03d}_lambda{lambda_label}"

    @property
    def model_config(self) -> MOOBAMConfig:
        """Build the validated model configuration used by the solver."""

        return MOOBAMConfig(
            k=self.k,
            n_clusters=self.n_clusters,
            lambda_weight=self.lambda_weight,
        )


class AdultExperimentPlan:
    """Construct only parameter combinations printed in Appendix A."""

    @staticmethod
    def build(preset: AdultPreset) -> tuple[AdultExperiment, ...]:
        """Return Table 6 boundary runs or all 220 Adult grid points."""

        if preset == "table-6":
            lower = tuple(
                AdultExperiment(
                    k=k,
                    n_clusters=100,
                    lambda_weight=0.0001,
                    paper_table_section="Table 6(a)",
                )
                for k in PAPER_K_VALUES
            )
            higher = tuple(
                AdultExperiment(
                    k=k,
                    n_clusters=4,
                    lambda_weight=1.0,
                    paper_table_section="Table 6(b)",
                )
                for k in PAPER_K_VALUES
            )
            return lower + higher
        if preset == "full-grid":
            return tuple(
                AdultExperiment(k=k, n_clusters=clusters, lambda_weight=weight)
                for k, clusters, weight in product(
                    PAPER_K_VALUES,
                    PAPER_ADULT_CLUSTER_VALUES,
                    PAPER_LAMBDA_VALUES,
                )
            )
        raise ValueError(f"Unknown Adult experiment preset: {preset!r}")


@dataclass(frozen=True)
class AdultExperimentRunConfig:
    """Store paths and reproducible search controls for the experiment runner."""

    data_directory: Path
    output_directory: Path
    preset: AdultPreset = "table-6"
    pso: PSOConfig = PSOConfig(population_size=100, iterations=100, seed=0)


class AdultExperimentRunner:
    """Execute and checkpoint the paper's non-ML Adult MO-OBAM experiments."""

    RESULTS_FILE = "adult_mo_obam_grid_results.csv"
    PLAN_FILE = "adult_mo_obam_experiment_plan.json"
    RUNS_DIRECTORY = "runs"
    EXPORTS_DIRECTORY = "exports"

    def __init__(self, config: AdultExperimentRunConfig) -> None:
        self.config = config
        self.experiments = AdultExperimentPlan.build(config.preset)

    def write_plan(self) -> Path:
        """Write the exact planned configurations without running PSO."""

        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.config.output_directory / self.PLAN_FILE
        payload = {
            "paper_revision": PAPER_REVISION,
            "scope": "MO-OBAM anonymization and privacy/utility evaluation; ML paused",
            "preset": self.config.preset,
            "number_of_configurations": len(self.experiments),
            "pso_configuration": asdict(self.config.pso),
            "paper_seed_boundary": (
                "The paper does not publish a random seed; this run records the "
                "explicit local seed instead of claiming an exact stochastic replay."
            ),
            "configurations": [
                {
                    "run_id": experiment.run_id,
                    **asdict(experiment),
                }
                for experiment in self.experiments
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def run(
        self,
        progress: ExperimentProgress | None = None,
        max_new_runs: int | None = None,
        run_ids: set[str] | None = None,
        retry_blocked: bool = False,
    ) -> pd.DataFrame:
        """Run pending configurations, resuming matching checkpoints safely.

        ``max_new_runs`` is an operational batching control only; it does not
        alter the paper parameter grid recorded in the plan.
        """

        if max_new_runs is not None and max_new_runs < 1:
            raise ValueError("max_new_runs must be at least 1 when supplied.")

        self.write_plan()
        original, preparation = AdultDataPreparer().prepare(self.config.data_directory)
        baseline = PrivacyRiskEvaluator(original, ADULT_SCHEMA).evaluate()
        rows: list[dict[str, object]] = []
        new_runs = 0

        for experiment in self.experiments:
            if run_ids is not None and experiment.run_id not in run_ids:
                continue
            checkpoint = self._load_checkpoint(experiment)
            if checkpoint is not None:
                rows.append(checkpoint["result_row"])
                self._write_results(rows)
                continue
            blocked_checkpoint = self._load_blocked(experiment)
            if blocked_checkpoint is not None and not retry_blocked:
                rows.append(blocked_checkpoint["result_row"])
                self._write_results(rows)
                continue
            if max_new_runs is not None and new_runs >= max_new_runs:
                continue

            started = time.monotonic()
            new_runs += 1
            formulation = MOOBAMFormulation(
                original,
                ADULT_SCHEMA,
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
                blocked = self._blocked_row(
                    experiment,
                    str(error),
                    time.monotonic() - started,
                )
                self._write_blocked(experiment, blocked)
                rows.append(blocked)
                self._write_results(rows)
                continue
            risk = PrivacyRiskEvaluator(result.anonymized_data, ADULT_SCHEMA).evaluate()
            runtime_seconds = time.monotonic() - started
            row = self._result_row(experiment, result.metrics, risk, runtime_seconds)
            self._write_checkpoint(
                experiment,
                result,
                row,
                asdict(preparation),
                baseline,
                risk,
                runtime_seconds,
            )
            rows.append(row)
            self._write_results(rows)

        return self._write_results(rows)

    def export_run(self, run_id: str) -> Path:
        """Rebuild full before/after CSVs from one completed grid checkpoint."""

        experiment = next(
            (item for item in self.experiments if item.run_id == run_id),
            None,
        )
        if experiment is None:
            raise ValueError(f"Run {run_id!r} is not part of preset {self.config.preset!r}.")
        checkpoint = self._load_checkpoint(experiment)
        if checkpoint is None:
            raise FileNotFoundError(f"No completed checkpoint exists for {run_id!r}.")

        original, _ = AdultDataPreparer().prepare(self.config.data_directory)
        run_directory = self._run_directory(experiment)
        centroids = pd.read_csv(run_directory / "centroids.csv")
        assignment_table = pd.read_csv(run_directory / "assignments.csv")
        expected_positions = np.arange(len(original))
        if not np.array_equal(
            assignment_table["record_position"].to_numpy(), expected_positions
        ):
            raise ValueError("Checkpoint record positions are missing or out of order.")

        formulation = MOOBAMFormulation(original, ADULT_SCHEMA, experiment.model_config)
        assignments = assignment_table["cluster"].to_numpy(dtype=int)
        membership = formulation.membership_from_assignments(assignments)
        metrics = formulation.evaluate(membership, centroids)
        anonymized = formulation.anonymize(membership, centroids)

        export_directory = (
            self.config.output_directory / self.EXPORTS_DIRECTORY / experiment.run_id
        )
        export_directory.mkdir(parents=True, exist_ok=True)
        original.to_csv(export_directory / "adult_prepared_original.csv", index=False)
        self._before_after(original, anonymized).to_csv(
            export_directory / "adult_before_after_anonymization.csv", index=False
        )
        reporter = AdultOutputReporter(ADULT_SCHEMA)
        artifacts = reporter.build(
            original,
            anonymized,
            experiment.model_config,
            metrics.information_loss,
            metrics.entropy_sum,
        )
        report_files = reporter.write(export_directory, artifacts)
        export_summary = {
            "run_id": experiment.run_id,
            "paper_revision": PAPER_REVISION,
            "model_configuration": asdict(experiment.model_config),
            "pso_configuration": asdict(self.config.pso),
            "final_metrics": asdict(metrics),
            "outputs": {
                "prepared_original": "adult_prepared_original.csv",
                "before_after_anonymization": "adult_before_after_anonymization.csv",
                **report_files,
            },
        }
        (export_directory / "adult_export_summary.json").write_text(
            json.dumps(export_summary, indent=2), encoding="utf-8"
        )
        return export_directory

    def _write_checkpoint(
        self,
        experiment: AdultExperiment,
        result: ScalableMOOBAMResult,
        result_row: dict[str, object],
        preparation: dict[str, object],
        baseline: PrivacyRiskMetrics,
        anonymized_risk: PrivacyRiskMetrics,
        runtime_seconds: float,
    ) -> None:
        """Save compact data sufficient to audit and reconstruct the release."""

        run_directory = self._run_directory(experiment)
        run_directory.mkdir(parents=True, exist_ok=True)
        result.centroids.to_csv(run_directory / "centroids.csv", index=False)
        assignments = np.argmax(result.membership, axis=1)
        pd.DataFrame(
            {
                "record_position": np.arange(len(assignments)),
                "cluster": assignments,
            }
        ).to_csv(run_directory / "assignments.csv", index=False)
        summary = {
            "run_id": experiment.run_id,
            "paper_revision": PAPER_REVISION,
            "model_configuration": asdict(experiment.model_config),
            "pso_configuration": asdict(self.config.pso),
            "preparation": preparation,
            "baseline_privacy": AdultOutputReporter.risk_as_dict(baseline),
            "final_metrics": asdict(result.metrics),
            "cluster_sizes": list(result.cluster_sizes),
            "stages": [asdict(stage) for stage in result.stages],
            "privacy_evaluation": {
                "prepared_baseline": AdultOutputReporter.risk_as_dict(baseline),
                "anonymized": AdultOutputReporter.risk_as_dict(anonymized_risk),
            },
            "runtime_seconds": runtime_seconds,
            "result_row": result_row,
        }
        (run_directory / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        blocked_path = run_directory / "blocked.json"
        if blocked_path.exists():
            blocked_path.unlink()

    def _load_checkpoint(
        self, experiment: AdultExperiment
    ) -> dict[str, object] | None:
        """Return a checkpoint only when its paper and run settings match."""

        path = self._run_directory(experiment) / "summary.json"
        if not path.exists():
            return None
        summary = json.loads(path.read_text(encoding="utf-8"))
        expected_model = asdict(experiment.model_config)
        expected_pso = asdict(self.config.pso)
        if (
            summary.get("paper_revision") != PAPER_REVISION
            or summary.get("model_configuration") != expected_model
            or summary.get("pso_configuration") != expected_pso
        ):
            return None
        for filename in ("centroids.csv", "assignments.csv"):
            if not (path.parent / filename).exists():
                return None
        return summary

    def _result_row(
        self,
        experiment: AdultExperiment,
        metrics: MOOBAMMetrics,
        risk: PrivacyRiskMetrics,
        runtime_seconds: float,
    ) -> dict[str, object]:
        expected = self._paper_expectation(experiment)
        return {
            "status": "completed",
            "issue": None,
            "run_id": experiment.run_id,
            "paper_table_section": experiment.paper_table_section,
            "k": experiment.k,
            "number_of_clusters": experiment.n_clusters,
            "lambda": experiment.lambda_weight,
            "population": self.config.pso.population_size,
            "iterations": self.config.pso.iterations,
            "seed": self.config.pso.seed,
            "information_loss": float(metrics.information_loss),
            "paper_information_loss": expected.get("information_loss"),
            "information_loss_difference": self._difference(
                metrics.information_loss, expected.get("information_loss")
            ),
            "entropy_sum": float(metrics.entropy_sum),
            "objective": float(metrics.objective),
            "equivalence_classes": risk.equivalence_classes,
            "paper_equivalence_classes": expected.get("equivalence_classes"),
            "minimum_class_size": risk.minimum_equivalence_class_size,
            "maximum_class_size": risk.maximum_equivalence_class_size,
            "linkage_tau_0.05": risk.linkage_attack_records["0.05"],
            "paper_linkage_tau_0.05": expected.get("linkage_0.05"),
            "linkage_tau_0.075": risk.linkage_attack_records["0.075"],
            "paper_linkage_tau_0.075": expected.get("linkage_0.075"),
            "linkage_tau_0.1": risk.linkage_attack_records["0.1"],
            "paper_linkage_tau_0.1": expected.get("linkage_0.1"),
            "homogeneity_attack_records": risk.homogeneity_attack_records,
            "paper_homogeneity_attack_records": expected.get("homogeneity"),
            "runtime_seconds": runtime_seconds,
        }

    def _blocked_row(
        self,
        experiment: AdultExperiment,
        issue: str,
        runtime_seconds: float,
    ) -> dict[str, object]:
        """Represent a source-level reproducibility blocker without fake metrics."""

        return {
            "status": "blocked",
            "issue": (
                f"{issue} Algorithm 1 supplies a k-violation multiplier, but arXiv v2 "
                "does not publish its Adult value and prints a penalty sign that "
                "conflicts with minimization. No penalty or repair was invented."
            ),
            "run_id": experiment.run_id,
            "paper_table_section": experiment.paper_table_section,
            "k": experiment.k,
            "number_of_clusters": experiment.n_clusters,
            "lambda": experiment.lambda_weight,
            "population": self.config.pso.population_size,
            "iterations": self.config.pso.iterations,
            "seed": self.config.pso.seed,
            "runtime_seconds": runtime_seconds,
        }

    def _write_blocked(
        self,
        experiment: AdultExperiment,
        blocked_row: dict[str, object],
    ) -> None:
        """Write a diagnostic separately so it cannot be mistaken for a result."""

        run_directory = self._run_directory(experiment)
        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "blocked.json").write_text(
            json.dumps(
                {
                    "paper_revision": PAPER_REVISION,
                    "model_configuration": asdict(experiment.model_config),
                    "pso_configuration": asdict(self.config.pso),
                    "result_row": blocked_row,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_blocked(
        self, experiment: AdultExperiment
    ) -> dict[str, object] | None:
        """Reuse a matching blocker so a resumed batch can advance."""

        path = self._run_directory(experiment) / "blocked.json"
        if not path.exists():
            return None
        blocked = json.loads(path.read_text(encoding="utf-8"))
        if (
            blocked.get("paper_revision") != PAPER_REVISION
            or blocked.get("model_configuration") != asdict(experiment.model_config)
            or blocked.get("pso_configuration") != asdict(self.config.pso)
        ):
            return None
        return blocked

    def _write_results(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        """Write completed rows in paper-plan order after every configuration."""

        order = {experiment.run_id: index for index, experiment in enumerate(self.experiments)}
        saved_rows: list[dict[str, object]] = []
        for experiment in self.experiments:
            completed = self._load_checkpoint(experiment)
            if completed is not None:
                saved_rows.append(completed["result_row"])
                continue
            blocked = self._load_blocked(experiment)
            if blocked is not None:
                saved_rows.append(blocked["result_row"])
        deduplicated = {
            str(row["run_id"]): row
            for row in (*saved_rows, *rows)
        }
        sorted_rows = sorted(deduplicated.values(), key=lambda row: order[str(row["run_id"])])
        table = pd.DataFrame(sorted_rows)
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.config.output_directory / self.RESULTS_FILE, index=False)
        return table

    def _run_directory(self, experiment: AdultExperiment) -> Path:
        return self.config.output_directory / self.RUNS_DIRECTORY / experiment.run_id

    @staticmethod
    def _paper_expectation(experiment: AdultExperiment) -> dict[str, object]:
        for (clusters, weight), expected in PAPER_ADULT_MO_OBAM_BOUNDARIES.items():
            if experiment.n_clusters == clusters and np.isclose(
                experiment.lambda_weight, weight
            ):
                return expected
        return {}

    @staticmethod
    def _difference(observed: float, expected: object) -> float | None:
        if expected is None:
            return None
        return float(observed) - float(expected)

    @staticmethod
    def _before_after(
        original: pd.DataFrame,
        anonymized: pd.DataFrame,
    ) -> pd.DataFrame:
        """Place each QID's original and released values next to each other."""

        paired = pd.DataFrame(
            {
                "record_position": np.arange(len(original)),
                "cluster": anonymized["cluster"].to_numpy(),
            }
        )
        qids = set(ADULT_SCHEMA.qids)
        for column in ADULT_COLUMNS:
            if column in qids:
                paired[f"{column}_before"] = original[column].to_numpy()
                paired[f"{column}_after"] = anonymized[column].to_numpy()
            else:
                paired[column] = original[column].to_numpy()
        return paired
