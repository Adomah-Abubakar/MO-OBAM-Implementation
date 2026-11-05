"""Multi-seed robustness experiments for selected PRTI-GH MO-OBAM settings."""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd

from .formulation import MOOBAMFormulation
from .privacy_metrics import PrivacyRiskEvaluator
from .provenance import (
    ArtifactManifestWriter,
    atomic_write_text,
    exclusive_directory_lock,
)
from .prti_data import DEFAULT_PRTI_SOURCE, PRTIDataAdapter, PRTIPreparedData
from .prti_runner import PAPER_IDENTIFIER, PAPER_REVISION, PROJECT_ROOT
from .pso_solver import (
    MOOBAMParticleSwarmOptimizer,
    NoFeasibleParticleError,
    PSOConfig,
)
from .schema import MOOBAMConfig


RobustnessProgressCallback = Callable[
    ["PRTIRobustnessExperiment", int, int, int, float, int, int], None
]


_PROCESS_WORKER_STATE: tuple[
    object,
    PRTIPreparedData,
    str,
    RobustnessProgressCallback | None,
] | None = None


def _run_one_in_forked_process(
    task: tuple["PRTIRobustnessExperiment", int],
) -> None:
    """Execute one aggregate checkpoint using state inherited through ``fork``."""

    if _PROCESS_WORKER_STATE is None:
        raise RuntimeError("PRTI process worker state was not initialized.")
    runner, prepared, source_hash, progress = _PROCESS_WORKER_STATE
    experiment, seed = task
    if not isinstance(runner, PRTIRobustnessRunner):
        raise RuntimeError("PRTI process worker received an invalid runner.")
    runner._run_one(prepared, source_hash, experiment, seed, progress=progress)


@dataclass(frozen=True)
class PRTIRobustnessExperiment:
    """Describe one previously observed privacy-utility candidate."""

    configuration_id: str
    purpose: str
    selection_basis: str
    model: MOOBAMConfig

    @property
    def model_run_id(self) -> str:
        """Return the same stable model identifier used by sensitivity runs."""

        weight = format(self.model.lambda_weight, "g").replace(".", "p")
        return (
            f"k{self.model.k:02d}_c{self.model.n_clusters:03d}_lambda{weight}"
        )


class PRTIRobustnessPlan:
    """Declare the four settings selected before multi-seed execution."""

    @staticmethod
    def build() -> tuple[PRTIRobustnessExperiment, ...]:
        """Return the baseline and three seed-9 one-factor candidates."""

        return (
            PRTIRobustnessExperiment(
                configuration_id="baseline",
                purpose="Recorded PRTI-GH baseline",
                selection_basis=(
                    "The existing labelled baseline used k=10, 30 clusters, "
                    "and lambda=1."
                ),
                model=MOOBAMConfig(k=10, n_clusters=30, lambda_weight=1.0),
            ),
            PRTIRobustnessExperiment(
                configuration_id="privacy_candidate",
                purpose="Lowest seed-9 measured linkage and homogeneity exposure",
                selection_basis=(
                    "The completed seed-9 one-factor study observed zero linkage "
                    "and homogeneity records at k=10, 4 clusters, and lambda=1."
                ),
                model=MOOBAMConfig(k=10, n_clusters=4, lambda_weight=1.0),
            ),
            PRTIRobustnessExperiment(
                configuration_id="lowest_il_candidate",
                purpose="Lowest seed-9 information loss",
                selection_basis=(
                    "The completed seed-9 one-factor study observed its lowest "
                    "information loss at k=10, 30 clusters, and lambda=0.001."
                ),
                model=MOOBAMConfig(k=10, n_clusters=30, lambda_weight=0.001),
            ),
            PRTIRobustnessExperiment(
                configuration_id="compromise_candidate",
                purpose="Intermediate cluster-count candidate",
                selection_basis=(
                    "The completed seed-9 one-factor study placed the 10-cluster "
                    "setting between the 4- and 30-cluster settings in information "
                    "loss and sensitive-attribute diversity."
                ),
                model=MOOBAMConfig(k=10, n_clusters=10, lambda_weight=1.0),
            ),
        )


@dataclass(frozen=True)
class PRTIRobustnessConfig:
    """Configure an aggregate-only, resumable multi-seed study."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    output_directory: Path = Path("evidence/prti-robustness-v1")
    population_size: int = 100
    iterations: int = 100
    seeds: tuple[int, ...] = tuple(range(10))
    workers: int = 1
    worker_backend: str = "thread"

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
        if self.workers < 1:
            raise ValueError("workers must be at least 1.")
        if self.worker_backend not in {"thread", "process"}:
            raise ValueError("worker_backend must be 'thread' or 'process'.")
        if (
            self.worker_backend == "process"
            and "fork" not in multiprocessing.get_all_start_methods()
        ):
            raise ValueError("The process worker backend requires POSIX fork support.")


class PRTIRobustnessRunner:
    """Run selected configurations across seeds without saving patient rows."""

    PLAN_FILE = "prti_robustness_plan.json"
    RUNS_FILE = "prti_robustness_runs.csv"
    SUMMARY_FILE = "prti_robustness_summary.csv"
    SUMMARY_JSON_FILE = "prti_robustness_summary.json"
    RUNS_DIRECTORY = "runs"
    OUTPUT_REFRESH_INTERVAL = 25

    SUMMARY_METRICS = (
        "information_loss",
        "entropy_sum",
        "objective",
        "equivalence_classes",
        "minimum_cluster_size",
        "maximum_cluster_size",
        "linkage_records_tau_0_05",
        "linkage_records_tau_0_075",
        "linkage_records_tau_0_1",
        "homogeneity_records_any_sa",
        "zero_minimum_entropy_sensitive_attributes",
        "feasible_evaluation_percent",
        "runtime_seconds",
    )

    def __init__(self, config: PRTIRobustnessConfig) -> None:
        self.config = config
        self.experiments = PRTIRobustnessPlan.build()

    def write_plan(self, source_sha256: str | None = None) -> Path:
        """Write the predeclared settings, seeds, and interpretation boundary."""

        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.config.output_directory / self.PLAN_FILE
        payload = {
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "scope": "PRTI-GH multi-seed robustness; ML paused",
            "status": "project robustness protocol, not paper-reported PRTI-GH results",
            "source_sha256": source_sha256,
            "number_of_configurations": len(self.experiments),
            "number_of_seeds": len(self.config.seeds),
            "number_of_planned_runs": len(self.experiments) * len(self.config.seeds),
            "seeds": list(self.config.seeds),
            "seed_protocol": (
                "The configured seeds form a deterministic project robustness "
                "set. Wei et al. do not specify this PRTI-GH multi-seed protocol."
            ),
            "pso_configuration_except_seed": {
                "population_size": self.config.population_size,
                "iterations": self.config.iterations,
                "random_coefficient_max": PSOConfig().random_coefficient_max,
            },
            "patient_level_outputs": False,
            "selection_boundary": (
                "The study measures stochastic stability. It does not automatically "
                "approve a configuration or patient-level release."
            ),
            "configurations": [
                {
                    "configuration_id": experiment.configuration_id,
                    "model_run_id": experiment.model_run_id,
                    "purpose": experiment.purpose,
                    "selection_basis": experiment.selection_basis,
                    "model_configuration": asdict(experiment.model),
                }
                for experiment in self.experiments
            ],
        }
        atomic_write_text(path, json.dumps(payload, indent=2))
        return path

    def run(
        self,
        progress: RobustnessProgressCallback | None = None,
        max_new_runs: int | None = None,
        configuration_ids: set[str] | None = None,
        retry_blocked: bool = False,
    ) -> dict[str, object]:
        """Execute pending seed runs and refresh aggregate evidence."""

        with exclusive_directory_lock(self.config.output_directory):
            return self._run_locked(
                progress=progress,
                max_new_runs=max_new_runs,
                configuration_ids=configuration_ids,
                retry_blocked=retry_blocked,
            )

    def _run_locked(
        self,
        progress: RobustnessProgressCallback | None = None,
        max_new_runs: int | None = None,
        configuration_ids: set[str] | None = None,
        retry_blocked: bool = False,
    ) -> dict[str, object]:
        """Execute a study while holding its single-writer directory lock."""

        if max_new_runs is not None and max_new_runs < 1:
            raise ValueError("max_new_runs must be at least 1 when supplied.")
        known_ids = {experiment.configuration_id for experiment in self.experiments}
        if configuration_ids is not None:
            unknown = sorted(configuration_ids - known_ids)
            if unknown:
                raise ValueError(f"Unknown robustness configuration IDs: {unknown}")

        prepared = PRTIDataAdapter(self.config.source_path).prepare()
        source_hash = str(prepared.summary["source_sha256_before"])
        self.write_plan(source_hash)
        self._before_tasks(prepared, source_hash)
        tasks: list[tuple[PRTIRobustnessExperiment, int]] = []
        for experiment in self.experiments:
            if (
                configuration_ids is not None
                and experiment.configuration_id not in configuration_ids
            ):
                continue
            for seed in self.config.seeds:
                checkpoint = self._load_checkpoint(experiment, seed, source_hash)
                if checkpoint is not None:
                    status = checkpoint["result_row"]["status"]
                    if status != "no_feasible_particle" or not retry_blocked:
                        continue
                tasks.append((experiment, seed))

        if max_new_runs is not None:
            tasks = tasks[:max_new_runs]

        if tasks:
            self._write_outputs(source_hash)
        completed_tasks = 0
        if self.config.workers == 1:
            for experiment, seed in tasks:
                self._run_one(prepared, source_hash, experiment, seed, progress)
                completed_tasks += 1
                if (
                    completed_tasks % self.OUTPUT_REFRESH_INTERVAL == 0
                    and completed_tasks < len(tasks)
                ):
                    self._write_outputs(source_hash)
        else:
            use_processes = self.config.worker_backend == "process"
            global _PROCESS_WORKER_STATE
            if use_processes:
                _PROCESS_WORKER_STATE = (self, prepared, source_hash, progress)
                executor_context = ProcessPoolExecutor(
                    max_workers=self.config.workers,
                    mp_context=multiprocessing.get_context("fork"),
                )
            else:
                executor_context = ThreadPoolExecutor(
                    max_workers=self.config.workers
                )
            try:
                with executor_context as executor:
                    task_iterator = iter(tasks)
                    pending: dict[
                        Future[None], tuple[PRTIRobustnessExperiment, int]
                    ] = {}

                    def submit_next() -> bool:
                        try:
                            experiment, seed = next(task_iterator)
                        except StopIteration:
                            return False
                        if use_processes:
                            future = executor.submit(
                                _run_one_in_forked_process,
                                (experiment, seed),
                            )
                        else:
                            future = executor.submit(
                                self._run_one,
                                prepared,
                                source_hash,
                                experiment,
                                seed,
                                progress,
                            )
                        pending[future] = (experiment, seed)
                        return True

                    for _ in range(min(self.config.workers, len(tasks))):
                        submit_next()
                    try:
                        while pending:
                            completed, _ = wait(
                                tuple(pending),
                                return_when=FIRST_COMPLETED,
                            )
                            for future in completed:
                                pending.pop(future)
                                future.result()
                                completed_tasks += 1
                                if (
                                    completed_tasks
                                    % self.OUTPUT_REFRESH_INTERVAL
                                    == 0
                                    and completed_tasks < len(tasks)
                                ):
                                    self._write_outputs(source_hash)
                                submit_next()
                    except BaseException:
                        for future in pending:
                            future.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise
            finally:
                if use_processes:
                    _PROCESS_WORKER_STATE = None

        return self._write_outputs(source_hash)

    def _before_tasks(
        self,
        prepared: PRTIPreparedData,
        source_hash: str,
    ) -> None:
        """Run subclass setup after source validation and before task discovery."""

    def summarize_runs(self, runs: pd.DataFrame) -> pd.DataFrame:
        """Summarize completion and metric variation for each configuration."""

        rows: list[dict[str, object]] = []
        for experiment in self.experiments:
            group = runs.loc[
                runs.get("configuration_id", pd.Series(dtype=object)).eq(
                    experiment.configuration_id
                )
            ].copy()
            completed = group.loc[group.get("status", pd.Series(dtype=object)).eq("completed")]
            row: dict[str, object] = {
                "configuration_id": experiment.configuration_id,
                "purpose": experiment.purpose,
                "k": experiment.model.k,
                "configured_clusters": experiment.model.n_clusters,
                "lambda": experiment.model.lambda_weight,
                "seeds_planned": len(self.config.seeds),
                "runs_recorded": len(group),
                "runs_completed": len(completed),
                "runs_no_feasible_particle": int(
                    group.get("status", pd.Series(dtype=object))
                    .eq("no_feasible_particle")
                    .sum()
                ),
                "completion_percent": (
                    100.0 * len(completed) / len(self.config.seeds)
                ),
            }
            for metric in self.SUMMARY_METRICS:
                source = (
                    completed[metric]
                    if metric in completed
                    else pd.Series(dtype=float)
                )
                values = pd.to_numeric(source, errors="coerce").dropna()
                row[f"{metric}_mean"] = values.mean() if len(values) else None
                row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else None
                row[f"{metric}_min"] = values.min() if len(values) else None
                row[f"{metric}_max"] = values.max() if len(values) else None
            rows.append(row)
        return pd.DataFrame(rows)

    def _run_one(
        self,
        prepared: PRTIPreparedData,
        source_hash: str,
        experiment: PRTIRobustnessExperiment,
        seed: int,
        progress: RobustnessProgressCallback | None,
    ) -> None:
        """Run one independent seed and save one aggregate checkpoint."""

        started = time.monotonic()
        formulation = MOOBAMFormulation(
            prepared.model_input,
            prepared.schema,
            experiment.model,
        )
        pso = PSOConfig(
            population_size=self.config.population_size,
            iterations=self.config.iterations,
            seed=seed,
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
                    seed,
                    iteration,
                    total,
                    best_score,
                    feasible,
                    infeasible,
                )

        try:
            result = MOOBAMParticleSwarmOptimizer(formulation, pso).solve(
                progress=report_stage
            )
        except NoFeasibleParticleError as error:
            row = {
                **self._configuration_columns(experiment, seed),
                "status": "no_feasible_particle",
                "issue": (
                    f"{error} No penalty, repair rule, or result value was invented."
                ),
                "runtime_seconds": time.monotonic() - started,
            }
            self._write_checkpoint(experiment, seed, source_hash, pso, row)
            return

        risk = PrivacyRiskEvaluator(
            result.anonymized_data.drop(columns="cluster"),
            prepared.schema,
        ).evaluate()
        stage = result.stages[0]
        total_evaluations = stage.feasible_evaluations + stage.infeasible_evaluations
        row = {
            **self._configuration_columns(experiment, seed),
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
            "feasible_evaluation_percent": (
                100.0 * stage.feasible_evaluations / total_evaluations
            ),
            "runtime_seconds": time.monotonic() - started,
        }
        self._validate_completed_row(row)
        self._write_checkpoint(experiment, seed, source_hash, pso, row)

    def _validate_completed_row(self, row: dict[str, object]) -> None:
        """Cross-check the paper objective and hard-k result before saving."""

        finite_metrics = (
            "information_loss",
            "entropy_sum",
            "objective",
            "runtime_seconds",
            "feasible_evaluation_percent",
        )
        if any(not np.isfinite(float(row[name])) for name in finite_metrics):
            raise RuntimeError("Completed robustness row contains a non-finite metric.")
        if float(row["information_loss"]) < 0.0:
            raise RuntimeError("Completed robustness row has negative information loss.")
        if float(row["entropy_sum"]) < 0.0:
            raise RuntimeError("Completed robustness row has negative entropy.")
        if float(row["runtime_seconds"]) < 0.0:
            raise RuntimeError("Completed robustness row has negative runtime.")
        expected_objective = float(row["information_loss"]) - float(
            row["lambda"]
        ) * float(row["entropy_sum"])
        if not np.isclose(
            float(row["objective"]), expected_objective, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("Completed robustness row violates the v2 objective.")
        if int(row["minimum_cluster_size"]) < int(row["k"]):
            raise RuntimeError("Completed robustness row violates hard k-anonymity.")
        if not 1 <= int(row["equivalence_classes"]) <= int(
            row["configured_clusters"]
        ):
            raise RuntimeError("Released equivalence-class count is inconsistent.")
        if not int(row["minimum_cluster_size"]) <= int(
            row["maximum_cluster_size"]
        ) <= 1485:
            raise RuntimeError("Completed robustness row has invalid cluster sizes.")
        count_metrics = (
            "linkage_records_tau_0_05",
            "linkage_records_tau_0_075",
            "linkage_records_tau_0_1",
            "homogeneity_records_any_sa",
        )
        if any(not 0 <= int(row[name]) <= 1485 for name in count_metrics):
            raise RuntimeError("Completed robustness row has an invalid risk count.")
        if int(row["zero_minimum_entropy_sensitive_attributes"]) < 0:
            raise RuntimeError("Completed robustness row has a negative SA count.")
        feasible = int(row["feasible_evaluations"])
        infeasible = int(row["infeasible_evaluations"])
        expected_evaluations = int(row["population"]) * int(row["iterations"])
        if feasible < 1 or infeasible < 0 or feasible + infeasible != expected_evaluations:
            raise RuntimeError("Completed robustness row has inconsistent PSO counts.")
        if not 0 <= int(row["best_iteration_zero_based"]) < int(row["iterations"]):
            raise RuntimeError("Completed robustness row has an invalid best iteration.")

    def _write_checkpoint(
        self,
        experiment: PRTIRobustnessExperiment,
        seed: int,
        source_hash: str,
        pso: PSOConfig,
        result_row: dict[str, object],
    ) -> None:
        """Save one seed result without assignments, centroids, or patient rows."""

        directory = self._run_directory(experiment)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "source_sha256": source_hash,
            "configuration_id": experiment.configuration_id,
            "model_configuration": asdict(experiment.model),
            "pso_configuration": asdict(pso),
            "patient_level_outputs": False,
            "result_row": result_row,
        }
        atomic_write_text(
            self._checkpoint_path(experiment, seed),
            json.dumps(payload, indent=2),
        )

    def _load_checkpoint(
        self,
        experiment: PRTIRobustnessExperiment,
        seed: int,
        source_hash: str,
    ) -> dict[str, object] | None:
        """Reuse a checkpoint only when source, model, and PSO settings match."""

        path = self._checkpoint_path(experiment, seed)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected_pso = asdict(
            PSOConfig(
                population_size=self.config.population_size,
                iterations=self.config.iterations,
                seed=seed,
            )
        )
        if (
            payload.get("paper_revision") != PAPER_REVISION
            or payload.get("source_sha256") != source_hash
            or payload.get("configuration_id") != experiment.configuration_id
            or payload.get("model_configuration") != asdict(experiment.model)
            or payload.get("pso_configuration") != expected_pso
            or payload.get("patient_level_outputs") is not False
        ):
            return None
        row = payload.get("result_row")
        if not isinstance(row, dict) or not self._checkpoint_row_is_valid(
            experiment, seed, row
        ):
            return None
        return payload

    def _checkpoint_row_is_valid(
        self,
        experiment: PRTIRobustnessExperiment,
        seed: int,
        row: dict[str, object],
    ) -> bool:
        """Reject incomplete or internally inconsistent aggregate checkpoints."""

        expected = self._configuration_columns(experiment, seed)
        if any(row.get(key) != value for key, value in expected.items()):
            return False
        status = row.get("status")
        if status == "no_feasible_particle":
            try:
                return (
                    isinstance(row.get("issue"), str)
                    and bool(row["issue"])
                    and np.isfinite(float(row["runtime_seconds"]))
                    and float(row["runtime_seconds"]) >= 0.0
                )
            except (KeyError, TypeError, ValueError):
                return False
        if status != "completed":
            return False
        try:
            self._validate_completed_row(row)
        except (KeyError, TypeError, ValueError, RuntimeError):
            return False
        return True

    def _write_outputs(self, source_hash: str) -> dict[str, object]:
        """Refresh run-level, summary, and provenance evidence from checkpoints."""

        rows: list[dict[str, object]] = []
        for experiment in self.experiments:
            for seed in self.config.seeds:
                checkpoint = self._load_checkpoint(experiment, seed, source_hash)
                if checkpoint is not None:
                    rows.append(checkpoint["result_row"])
        runs = pd.DataFrame(rows)
        summary = self.summarize_runs(runs)
        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        runs_path = output / self.RUNS_FILE
        summary_path = output / self.SUMMARY_FILE
        atomic_write_text(runs_path, runs.to_csv(index=False))
        atomic_write_text(summary_path, summary.to_csv(index=False))

        manifest = ArtifactManifestWriter(PROJECT_ROOT)
        plan_path = output / self.PLAN_FILE
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "scope": "Aggregate PRTI-GH multi-seed robustness; ML excluded",
            "source_sha256": source_hash,
            "patient_level_outputs": False,
            "planned_runs": len(self.experiments) * len(self.config.seeds),
            "recorded_runs": len(runs),
            "completed_runs": int(
                runs.get("status", pd.Series(dtype=object)).eq("completed").sum()
            ),
            "no_feasible_particle_runs": int(
                runs.get("status", pd.Series(dtype=object))
                .eq("no_feasible_particle")
                .sum()
            ),
            "configurations": json.loads(
                summary.to_json(orient="records", double_precision=15)
            ),
            "interpretation_boundaries": [
                "These are local stochastic PSO results, not global optima.",
                "The ten-seed protocol is a project robustness design, not a Wei paper requirement.",
                "Linkage risk is the equivalence-class proxy 1/|E|, not an external linkage attack.",
                "Homogeneity counts records homogeneous for at least one configured sensitive attribute.",
                "Objective values are not directly comparable across different lambda weights.",
                "No configuration is automatically approved for patient-level release.",
            ],
            "canonical_command": self._canonical_command(),
            "inputs": {
                "plan": {
                    "path": str(plan_path),
                    "sha256": manifest.file_sha256(plan_path),
                }
            },
            "outputs": {
                "runs": {
                    "path": str(runs_path),
                    "sha256": manifest.file_sha256(runs_path),
                },
                "summary": {
                    "path": str(summary_path),
                    "sha256": manifest.file_sha256(summary_path),
                },
            },
            "generation_snapshot": manifest.capture_snapshot(
                "captured_when_robustness_summary_was_generated"
            ),
        }
        json_path = output / self.SUMMARY_JSON_FILE
        atomic_write_text(json_path, json.dumps(report, indent=2))
        return {
            "runs": runs,
            "summary": summary,
            "report": report,
            "paths": {
                "plan": plan_path,
                "runs": runs_path,
                "summary": summary_path,
                "summary_json": json_path,
            },
        }

    def _configuration_columns(
        self,
        experiment: PRTIRobustnessExperiment,
        seed: int,
    ) -> dict[str, object]:
        return {
            "configuration_id": experiment.configuration_id,
            "model_run_id": experiment.model_run_id,
            "purpose": experiment.purpose,
            "k": experiment.model.k,
            "configured_clusters": experiment.model.n_clusters,
            "lambda": experiment.model.lambda_weight,
            "population": self.config.population_size,
            "iterations": self.config.iterations,
            "seed": seed,
        }

    def _run_directory(self, experiment: PRTIRobustnessExperiment) -> Path:
        return (
            self.config.output_directory
            / self.RUNS_DIRECTORY
            / experiment.configuration_id
        )

    def _checkpoint_path(
        self,
        experiment: PRTIRobustnessExperiment,
        seed: int,
    ) -> Path:
        return self._run_directory(experiment) / f"seed_{seed:04d}.json"

    def _canonical_command(self) -> list[str]:
        source = os.path.relpath(Path(self.config.source_path).resolve(), PROJECT_ROOT)
        output = os.path.relpath(
            Path(self.config.output_directory).resolve(), PROJECT_ROOT
        )
        return [
            ".venv/bin/python",
            "run_prti_experiments.py",
            "robustness",
            "--source",
            source,
            "--output-directory",
            output,
            "--population",
            str(self.config.population_size),
            "--iterations",
            str(self.config.iterations),
            "--seeds",
            *[str(seed) for seed in self.config.seeds],
            "--workers",
            str(self.config.workers),
            "--worker-backend",
            self.config.worker_backend,
        ]
