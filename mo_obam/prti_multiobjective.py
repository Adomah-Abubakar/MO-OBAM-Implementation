"""Resumable aggregate-only PRTI-GH multi-objective study workflow."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Callable

import numpy as np
import pandas as pd

from .formulation import MOOBAMFormulation
from .privacy_metrics import PrivacyRiskEvaluator
from .provenance import atomic_write_text, exclusive_directory_lock
from .prti_data import DEFAULT_PRTI_SOURCE, EXPECTED_PRTI_SHAPE, PRTIDataAdapter, PRTIPreparedData
from .prti_multiobjective_solver import PRTIMultiObjectiveParticleSwarm
from .prti_runner import PAPER_IDENTIFIER, PAPER_REVISION
from .pso_solver import NoFeasibleParticleError, PSOConfig
from .schema import MOOBAMConfig


RunProgress = Callable[[int, int, int, int, int, str], None]
_WORKER_STATE: tuple[PRTIPreparedData, "PRTIMultiObjectiveConfig"] | None = None


@dataclass(frozen=True)
class PRTIMultiObjectiveConfig:
    """Declare the full k/C/seed study; lambda is analysed after optimization."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    output_directory: Path = Path("evidence/prti-multiobjective-v1")
    minimum_clusters: int = 1
    maximum_clusters: int = 30
    k_values: tuple[int, ...] = (10, 15, 20, 25, 30)
    seeds: tuple[int, ...] = tuple(range(10))
    held_out_seed: int = 12
    population_size: int = 100
    iterations: int = 100
    archive_size: int = 100
    initialization_attempts: int = 20
    random_coefficient_max: float = 0.01
    workers: int = 1
    maximum_information_loss: float = 0.60
    maximum_homogeneity_fraction: float = 0.20
    maximum_zero_minimum_entropy_sensitive_attributes: int = 1

    def __post_init__(self) -> None:
        if self.minimum_clusters < 1:
            raise ValueError("minimum_clusters must be at least 1.")
        if self.maximum_clusters < self.minimum_clusters:
            raise ValueError("maximum_clusters must be at least minimum_clusters.")
        if not self.k_values or len(set(self.k_values)) != len(self.k_values):
            raise ValueError("k_values must be a non-empty set of distinct values.")
        if any(k < 2 for k in self.k_values):
            raise ValueError("Every k value must be at least 2.")
        if any(self.minimum_clusters * k > EXPECTED_PRTI_SHAPE[0] for k in self.k_values):
            raise ValueError("minimum_clusters is infeasible for at least one k value.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty set of distinct values.")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Every comparison seed must be non-negative.")
        if self.held_out_seed < 0 or self.held_out_seed in self.seeds:
            raise ValueError("held_out_seed must be non-negative and absent from seeds.")
        if self.population_size < 2 or self.iterations < 1:
            raise ValueError("population_size must be at least 2 and iterations positive.")
        if self.archive_size < 2:
            raise ValueError("archive_size must be at least 2.")
        if self.initialization_attempts < 1:
            raise ValueError("initialization_attempts must be at least 1.")
        if self.random_coefficient_max <= 0.0:
            raise ValueError("random_coefficient_max must be positive.")
        if 2 * self.random_coefficient_max >= 1.0:
            raise ValueError(
                "random_coefficient_max must leave a positive PSO inertia term."
            )
        if self.workers < 1:
            raise ValueError("workers must be at least 1.")
        if self.workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
            raise ValueError("Parallel execution requires POSIX fork support.")
        if not 0.0 <= self.maximum_information_loss <= 1.0:
            raise ValueError("maximum_information_loss must be in [0, 1].")
        if not 0.0 <= self.maximum_homogeneity_fraction <= 1.0:
            raise ValueError("maximum_homogeneity_fraction must be in [0, 1].")
        if self.maximum_zero_minimum_entropy_sensitive_attributes < 0:
            raise ValueError("maximum zero-entropy SA count cannot be negative.")

    @property
    def configurations(self) -> tuple[tuple[int, int], ...]:
        values: list[tuple[int, int]] = []
        for k in self.k_values:
            endpoint = min(self.maximum_clusters, EXPECTED_PRTI_SHAPE[0] // k)
            values.extend(
                (k, clusters)
                for clusters in range(self.minimum_clusters, endpoint + 1)
            )
        return tuple(values)

    @property
    def planned_runs(self) -> int:
        return len(self.configurations) * len(self.seeds)


def _run_in_worker(task: tuple[int, int, int]) -> dict[str, object]:
    if _WORKER_STATE is None:
        raise RuntimeError("Multi-objective process worker was not initialized.")
    prepared, config = _WORKER_STATE
    return _run_one(prepared, config, *task)


def _run_one(
    prepared: PRTIPreparedData,
    config: PRTIMultiObjectiveConfig,
    k: int,
    clusters: int,
    seed: int,
) -> dict[str, object]:
    started = time.monotonic()
    formulation = MOOBAMFormulation(
        prepared.model_input,
        prepared.schema,
        MOOBAMConfig(k=k, n_clusters=clusters, lambda_weight=0.0),
    )
    optimizer = PRTIMultiObjectiveParticleSwarm(
        formulation,
        PSOConfig(
            population_size=config.population_size,
            iterations=config.iterations,
            seed=seed,
            random_coefficient_max=config.random_coefficient_max,
        ),
        archive_size=config.archive_size,
        initialization_attempts=config.initialization_attempts,
    )
    try:
        result = optimizer.solve_multiobjective()
    except NoFeasibleParticleError as error:
        return {
            "k": k,
            "clusters": clusters,
            "seed": seed,
            "status": "no_feasible_particle",
            "issue": str(error),
            "runtime_seconds": time.monotonic() - started,
            "feasible_evaluations": 0,
            "infeasible_evaluations": (
                config.population_size * config.initialization_attempts
            ),
            "candidates": [],
        }

    rows: list[dict[str, object]] = []
    for candidate in result.candidates:
        anonymized = formulation.anonymize(
            candidate.membership, candidate.centroids
        ).drop(columns="cluster")
        risk = PrivacyRiskEvaluator(anonymized, prepared.schema).evaluate()
        rows.append(
            {
                "candidate_index": candidate.candidate_index,
                "information_loss": candidate.information_loss,
                "entropy_sum": candidate.entropy_sum,
                "entropy_by_sa": candidate.entropy_by_sensitive_attribute,
                "minimum_cluster_size": min(candidate.cluster_sizes),
                "maximum_cluster_size": max(candidate.cluster_sizes),
                "equivalence_classes": risk.equivalence_classes,
                "linkage_records_tau_0_05": risk.linkage_attack_records["0.05"],
                "linkage_records_tau_0_075": risk.linkage_attack_records["0.075"],
                "linkage_records_tau_0_1": risk.linkage_attack_records["0.1"],
                "homogeneity_records_any_sa": risk.homogeneity_attack_records,
                "zero_minimum_entropy_sensitive_attributes": int(
                    sum(
                        np.isclose(value, 0.0)
                        for value in candidate.entropy_by_sensitive_attribute.values()
                    )
                ),
            }
        )
    return {
        "k": k,
        "clusters": clusters,
        "seed": seed,
        "status": "completed",
        "issue": None,
        "runtime_seconds": time.monotonic() - started,
        "feasible_evaluations": result.feasible_evaluations,
        "infeasible_evaluations": result.infeasible_evaluations,
        "candidates": rows,
    }


class PRTIMultiObjectiveStudy:
    """Plan, resume, summarize, and report the aggregate-only extension."""

    PLAN_FILE = "prti_multiobjective_plan.json"
    SELECTION_FILE = "prti_multiobjective_selection.json"
    REPORT_FILE = "prti_multiobjective_report.xlsx"
    DATABASE_FILE = "prti_multiobjective_checkpoints.sqlite3"

    def __init__(self, config: PRTIMultiObjectiveConfig) -> None:
        self.config = config

    def write_plan(self, source_sha256: str | None = None) -> Path:
        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        path = output / self.PLAN_FILE
        payload = {
            "declared_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "method": "additive multi-objective PSO extension; scalar baseline unchanged",
            "source_sha256": source_sha256,
            "cluster_interval": [
                self.config.minimum_clusters,
                self.config.maximum_clusters,
            ],
            "k_values": list(self.config.k_values),
            "comparison_seeds": list(self.config.seeds),
            "held_out_seed": self.config.held_out_seed,
            "population_size": self.config.population_size,
            "iterations": self.config.iterations,
            "archive_size": self.config.archive_size,
            "initialization_attempts": self.config.initialization_attempts,
            "random_coefficient_max": self.config.random_coefficient_max,
            "workers": self.config.workers,
            "number_of_configurations": len(self.config.configurations),
            "number_of_planned_runs": self.config.planned_runs,
            "lambda_analysis": {
                "domain": [0.0, 1.0],
                "method": (
                    "Calculate exact scalar-objective crossover intervals over the "
                    "privacy-eligible candidates discovered in each Pareto archive."
                ),
                "separate_optimizer_run_per_lambda": False,
            },
            "privacy_pass_rule": {
                "all_comparison_seeds_must_have_an_eligible_candidate": True,
                "minimum_cluster_size_at_least_k": True,
                "maximum_linkage_records_at_each_tau": 0,
                "maximum_homogeneity_fraction_any_sa": self.config.maximum_homogeneity_fraction,
                "maximum_zero_minimum_entropy_sensitive_attributes": self.config.maximum_zero_minimum_entropy_sensitive_attributes,
                "maximum_information_loss": self.config.maximum_information_loss,
            },
            "patient_level_outputs": False,
            "selection_rule": (
                "For each k/C/seed archive choose the privacy-eligible candidate with "
                "the lowest IL. A k/C configuration is eligible only when every "
                "comparison seed has such a candidate. Select the eligible k/C with "
                "the lowest mean IL; break ties by lower maximum homogeneity fraction, "
                "higher k, then smaller C. Lambda intervals are descriptive because "
                "the extension optimizes the two objectives directly."
            ),
            "optimality_boundary": (
                "The result is the best observed eligible point in a bounded stochastic "
                "Pareto archive, not proof of a mathematical global optimum."
            ),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            ignored = {"declared_at_utc", "source_sha256"}
            stable_existing = {k: v for k, v in existing.items() if k not in ignored}
            stable_payload = {k: v for k, v in payload.items() if k not in ignored}
            if stable_existing != stable_payload:
                raise ValueError(
                    "The existing multi-objective plan differs. Use a new output directory."
                )
            if source_sha256 is not None:
                old_hash = existing.get("source_sha256")
                if old_hash not in {None, source_sha256}:
                    raise ValueError("The existing plan uses a different source SHA-256.")
                if old_hash is None:
                    existing["source_sha256"] = source_sha256
                    atomic_write_text(path, json.dumps(existing, indent=2))
            return path
        atomic_write_text(path, json.dumps(payload, indent=2))
        return path

    def run(
        self,
        max_new_runs: int | None = None,
        retry_blocked: bool = False,
        progress: RunProgress | None = None,
    ) -> dict[str, object]:
        """Execute or resume the declared study; never run the held-out seed."""

        if max_new_runs is not None and max_new_runs < 1:
            raise ValueError("max_new_runs must be at least 1 when supplied.")
        with exclusive_directory_lock(self.config.output_directory):
            prepared = PRTIDataAdapter(self.config.source_path).prepare()
            source_hash = str(prepared.summary["source_sha256_before"])
            self.write_plan(source_hash)
            database = self.config.output_directory / self.DATABASE_FILE
            with closing(self._connect(database)) as connection:
                self._initialize_database(connection, source_hash)
                completed = self._completed_keys(connection, retry_blocked)
                tasks = [
                    (k, clusters, seed)
                    for k, clusters in self.config.configurations
                    for seed in self.config.seeds
                    if (k, clusters, seed) not in completed
                ]
                if max_new_runs is not None:
                    tasks = tasks[:max_new_runs]
                self._execute_tasks(connection, prepared, tasks, progress)
                return self._write_reports(connection, source_hash)

    def refresh_reports(self) -> dict[str, object]:
        """Regenerate selection JSON and Excel from existing SQLite evidence."""

        with exclusive_directory_lock(self.config.output_directory):
            plan = json.loads(
                (self.config.output_directory / self.PLAN_FILE).read_text(
                    encoding="utf-8"
                )
            )
            source_hash = str(plan.get("source_sha256"))
            database = self.config.output_directory / self.DATABASE_FILE
            with closing(self._connect(database)) as connection:
                self._initialize_database(connection, source_hash)
                return self._write_reports(connection, source_hash)

    def _execute_tasks(
        self,
        connection: sqlite3.Connection,
        prepared: PRTIPreparedData,
        tasks: list[tuple[int, int, int]],
        progress: RunProgress | None,
    ) -> None:
        if not tasks:
            return
        total = len(tasks)
        completed = 0
        if self.config.workers == 1:
            for task in tasks:
                result = _run_one(prepared, self.config, *task)
                self._store_result(connection, result)
                completed += 1
                if progress is not None:
                    progress(completed, total, *task, str(result["status"]))
            return

        global _WORKER_STATE
        _WORKER_STATE = (prepared, self.config)
        try:
            with ProcessPoolExecutor(
                max_workers=self.config.workers,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                iterator = iter(tasks)
                pending: dict[Future[dict[str, object]], tuple[int, int, int]] = {}

                def submit_next() -> None:
                    try:
                        task = next(iterator)
                    except StopIteration:
                        return
                    pending[executor.submit(_run_in_worker, task)] = task

                for _ in range(min(self.config.workers, total)):
                    submit_next()
                while pending:
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        task = pending.pop(future)
                        result = future.result()
                        self._store_result(connection, result)
                        completed += 1
                        if progress is not None:
                            progress(completed, total, *task, str(result["status"]))
                        submit_next()
        finally:
            _WORKER_STATE = None

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize_database(
        self, connection: sqlite3.Connection, source_hash: str
    ) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS study_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                k INTEGER NOT NULL,
                clusters INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                issue TEXT,
                runtime_seconds REAL NOT NULL,
                feasible_evaluations INTEGER NOT NULL,
                infeasible_evaluations INTEGER NOT NULL,
                completed_at_utc TEXT NOT NULL,
                PRIMARY KEY (k, clusters, seed)
            );
            CREATE TABLE IF NOT EXISTS candidates (
                k INTEGER NOT NULL,
                clusters INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                candidate_index INTEGER NOT NULL,
                information_loss REAL NOT NULL,
                entropy_sum REAL NOT NULL,
                entropy_by_sa_json TEXT NOT NULL,
                minimum_cluster_size INTEGER NOT NULL,
                maximum_cluster_size INTEGER NOT NULL,
                equivalence_classes INTEGER NOT NULL,
                linkage_records_tau_0_05 INTEGER NOT NULL,
                linkage_records_tau_0_075 INTEGER NOT NULL,
                linkage_records_tau_0_1 INTEGER NOT NULL,
                homogeneity_records_any_sa INTEGER NOT NULL,
                zero_minimum_entropy_sensitive_attributes INTEGER NOT NULL,
                PRIMARY KEY (k, clusters, seed, candidate_index),
                FOREIGN KEY (k, clusters, seed) REFERENCES runs(k, clusters, seed)
            );
            """
        )
        metadata = {
            "source_sha256": source_hash,
            "paper_revision": PAPER_REVISION,
            "configuration_json": json.dumps(asdict(self.config), default=str, sort_keys=True),
        }
        for key, value in metadata.items():
            existing = connection.execute(
                "SELECT value FROM study_metadata WHERE key = ?", (key,)
            ).fetchone()
            if existing is not None and existing["value"] != value:
                raise ValueError(
                    f"SQLite checkpoint metadata differs for {key}; use a new directory."
                )
            connection.execute(
                "INSERT OR IGNORE INTO study_metadata(key, value) VALUES (?, ?)",
                (key, value),
            )
        connection.commit()

    @staticmethod
    def _completed_keys(
        connection: sqlite3.Connection, retry_blocked: bool
    ) -> set[tuple[int, int, int]]:
        if retry_blocked:
            rows = connection.execute(
                "SELECT k, clusters, seed FROM runs WHERE status = 'completed'"
            )
        else:
            rows = connection.execute("SELECT k, clusters, seed FROM runs")
        return {(int(row[0]), int(row[1]), int(row[2])) for row in rows}

    @staticmethod
    def _store_result(
        connection: sqlite3.Connection, result: dict[str, object]
    ) -> None:
        k = int(result["k"])
        clusters = int(result["clusters"])
        seed = int(result["seed"])
        with connection:
            connection.execute(
                "DELETE FROM candidates WHERE k=? AND clusters=? AND seed=?",
                (k, clusters, seed),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    k,
                    clusters,
                    seed,
                    result["status"],
                    result["issue"],
                    float(result["runtime_seconds"]),
                    int(result["feasible_evaluations"]),
                    int(result["infeasible_evaluations"]),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            for candidate in result["candidates"]:
                connection.execute(
                    """
                    INSERT INTO candidates VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        k,
                        clusters,
                        seed,
                        int(candidate["candidate_index"]),
                        float(candidate["information_loss"]),
                        float(candidate["entropy_sum"]),
                        json.dumps(candidate["entropy_by_sa"], sort_keys=True),
                        int(candidate["minimum_cluster_size"]),
                        int(candidate["maximum_cluster_size"]),
                        int(candidate["equivalence_classes"]),
                        int(candidate["linkage_records_tau_0_05"]),
                        int(candidate["linkage_records_tau_0_075"]),
                        int(candidate["linkage_records_tau_0_1"]),
                        int(candidate["homogeneity_records_any_sa"]),
                        int(candidate["zero_minimum_entropy_sensitive_attributes"]),
                    ),
                )

    def _write_reports(
        self, connection: sqlite3.Connection, source_hash: str
    ) -> dict[str, object]:
        runs = pd.read_sql_query(
            "SELECT * FROM runs ORDER BY k, clusters, seed", connection
        )
        candidates = pd.read_sql_query(
            "SELECT * FROM candidates ORDER BY k, clusters, seed, candidate_index",
            connection,
        )
        if not candidates.empty:
            candidates["homogeneity_fraction_any_sa"] = (
                candidates["homogeneity_records_any_sa"] / EXPECTED_PRTI_SHAPE[0]
            )
            candidates["privacy_eligible"] = (
                candidates["minimum_cluster_size"].ge(candidates["k"])
                & candidates["linkage_records_tau_0_05"].eq(0)
                & candidates["linkage_records_tau_0_075"].eq(0)
                & candidates["linkage_records_tau_0_1"].eq(0)
                & candidates["homogeneity_fraction_any_sa"].le(
                    self.config.maximum_homogeneity_fraction
                )
                & candidates["zero_minimum_entropy_sensitive_attributes"].le(
                    self.config.maximum_zero_minimum_entropy_sensitive_attributes
                )
                & candidates["information_loss"].le(
                    self.config.maximum_information_loss
                )
            )
        selected_by_seed = self._best_eligible_by_seed(candidates)
        summary = self._configuration_summary(runs, selected_by_seed)
        complete = len(runs) == self.config.planned_runs
        eligible = summary.loc[summary.get("passes_all_seeds", False)].copy()
        if not eligible.empty:
            eligible = eligible.sort_values(
                [
                    "information_loss_mean",
                    "homogeneity_fraction_max",
                    "k",
                    "clusters",
                ],
                ascending=[True, True, False, True],
                kind="stable",
            )
        selected = eligible.iloc[0].to_dict() if complete and not eligible.empty else None
        lambda_intervals = self._lambda_intervals(candidates, selected)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "source_sha256": source_hash,
            "method": "multi-objective PSO extension",
            "status": "selection_complete" if complete else "selection_incomplete",
            "planned_runs": self.config.planned_runs,
            "recorded_runs": len(runs),
            "completed_runs": int(runs.get("status", pd.Series(dtype=str)).eq("completed").sum()),
            "selected_configuration": selected,
            "lambda_intervals_for_selected_configuration": lambda_intervals.to_dict(
                orient="records"
            ),
            "selection_boundary": (
                "No candidate is selected until every declared comparison run is "
                "recorded. The held-out seed is not executed by this command."
            ),
            "patient_level_outputs": False,
        }
        selection_path = self.config.output_directory / self.SELECTION_FILE
        atomic_write_text(selection_path, json.dumps(payload, indent=2, default=self._json_default))
        report_path = self.config.output_directory / self.REPORT_FILE
        self._write_excel(report_path, payload, runs, summary, lambda_intervals)
        return {
            "status": payload["status"],
            "planned_runs": self.config.planned_runs,
            "recorded_runs": len(runs),
            "selected_configuration": selected,
            "paths": {
                "plan": self.config.output_directory / self.PLAN_FILE,
                "selection": selection_path,
                "report": report_path,
                "checkpoints": self.config.output_directory / self.DATABASE_FILE,
            },
        }

    @staticmethod
    def _best_eligible_by_seed(candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates.copy()
        eligible = candidates.loc[candidates["privacy_eligible"]].copy()
        if eligible.empty:
            return eligible
        eligible = eligible.sort_values(
            ["k", "clusters", "seed", "information_loss", "entropy_sum"],
            ascending=[True, True, True, True, False],
            kind="stable",
        )
        return eligible.groupby(["k", "clusters", "seed"], as_index=False).first()

    def _configuration_summary(
        self, runs: pd.DataFrame, selected_by_seed: pd.DataFrame
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for k, clusters in self.config.configurations:
            run_group = runs.loc[(runs.get("k") == k) & (runs.get("clusters") == clusters)]
            selected_group = selected_by_seed.loc[
                (selected_by_seed.get("k") == k)
                & (selected_by_seed.get("clusters") == clusters)
            ]
            completed_runs = int(run_group.get("status", pd.Series(dtype=str)).eq("completed").sum())
            eligible_seeds = int(selected_group["seed"].nunique()) if not selected_group.empty else 0
            row: dict[str, object] = {
                "k": k,
                "clusters": clusters,
                "runs_recorded": len(run_group),
                "runs_completed": completed_runs,
                "seeds_planned": len(self.config.seeds),
                "privacy_eligible_seeds": eligible_seeds,
                "passes_all_seeds": (
                    completed_runs == len(self.config.seeds)
                    and eligible_seeds == len(self.config.seeds)
                ),
                "information_loss_mean": None,
                "information_loss_max": None,
                "entropy_sum_mean": None,
                "homogeneity_fraction_max": None,
                "minimum_cluster_size_min": None,
                "runtime_seconds_sum": float(run_group.get("runtime_seconds", pd.Series(dtype=float)).sum()),
            }
            if not selected_group.empty:
                row.update(
                    {
                        "information_loss_mean": float(selected_group["information_loss"].mean()),
                        "information_loss_max": float(selected_group["information_loss"].max()),
                        "entropy_sum_mean": float(selected_group["entropy_sum"].mean()),
                        "homogeneity_fraction_max": float(selected_group["homogeneity_fraction_any_sa"].max()),
                        "minimum_cluster_size_min": int(selected_group["minimum_cluster_size"].min()),
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _lambda_intervals(
        candidates: pd.DataFrame, selected: dict[str, object] | None
    ) -> pd.DataFrame:
        columns = [
            "lambda_start",
            "lambda_end",
            "information_loss_mean",
            "entropy_sum_mean",
            "objective_mean_at_midpoint",
            "seed_candidate_signature",
        ]
        if candidates.empty or selected is None:
            return pd.DataFrame(columns=columns)
        subset = candidates.loc[
            candidates["k"].eq(int(selected["k"]))
            & candidates["clusters"].eq(int(selected["clusters"]))
            & candidates["privacy_eligible"]
        ].copy()
        if subset.empty:
            return pd.DataFrame(columns=columns)
        groups = {int(seed): group for seed, group in subset.groupby("seed")}
        boundaries = {0.0, 1.0}
        for group in groups.values():
            values = group[["information_loss", "entropy_sum"]].to_numpy(float)
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    entropy_delta = values[left, 1] - values[right, 1]
                    if np.isclose(entropy_delta, 0.0):
                        continue
                    crossing = (values[left, 0] - values[right, 0]) / entropy_delta
                    if 0.0 < crossing < 1.0:
                        boundaries.add(round(float(crossing), 15))
        points = sorted(boundaries)
        rows: list[dict[str, object]] = []
        for start, end in zip(points[:-1], points[1:], strict=True):
            midpoint = (start + end) / 2.0
            choices: list[pd.Series] = []
            signature: list[str] = []
            for seed, group in sorted(groups.items()):
                scores = group["information_loss"] - midpoint * group["entropy_sum"]
                choice = group.loc[scores.idxmin()]
                choices.append(choice)
                signature.append(f"{seed}:{int(choice['candidate_index'])}")
            il_mean = float(np.mean([choice["information_loss"] for choice in choices]))
            entropy_mean = float(np.mean([choice["entropy_sum"] for choice in choices]))
            rows.append(
                {
                    "lambda_start": start,
                    "lambda_end": end,
                    "information_loss_mean": il_mean,
                    "entropy_sum_mean": entropy_mean,
                    "objective_mean_at_midpoint": il_mean - midpoint * entropy_mean,
                    "seed_candidate_signature": ",".join(signature),
                }
            )
        merged: list[dict[str, object]] = []
        for row in rows:
            if merged and merged[-1]["seed_candidate_signature"] == row["seed_candidate_signature"]:
                merged[-1]["lambda_end"] = row["lambda_end"]
            else:
                merged.append(dict(row))
        return pd.DataFrame(merged, columns=columns)

    @staticmethod
    def _write_excel(
        path: Path,
        payload: dict[str, object],
        runs: pd.DataFrame,
        summary: pd.DataFrame,
        lambda_intervals: pd.DataFrame,
    ) -> None:
        try:
            from openpyxl.styles import Alignment, Font, PatternFill
        except ImportError as error:
            raise RuntimeError(
                "Excel reporting requires openpyxl; install the pinned requirements."
            ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.xlsx"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            overview = pd.DataFrame(
                [
                    ("Status", payload["status"]),
                    ("Method", payload["method"]),
                    ("Planned runs", payload["planned_runs"]),
                    ("Recorded runs", payload["recorded_runs"]),
                    ("Completed runs", payload["completed_runs"]),
                    ("Patient-level outputs", payload["patient_level_outputs"]),
                ],
                columns=["Measure", "Value"],
            )
            run_status = (
                runs.groupby("status", as_index=False).size()
                if not runs.empty
                else pd.DataFrame(columns=["status", "size"])
            )
            with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
                overview.to_excel(writer, sheet_name="Summary", index=False, startrow=2)
                run_status.to_excel(writer, sheet_name="Summary", index=False, startrow=12)
                summary.to_excel(writer, sheet_name="Configurations", index=False)
                lambda_intervals.to_excel(writer, sheet_name="Lambda intervals", index=False)
                workbook = writer.book
                dark = "1F4E78"
                light = "D9EAF7"
                for worksheet in workbook.worksheets:
                    worksheet.sheet_view.showGridLines = False
                    worksheet.freeze_panes = "A2" if worksheet.title != "Summary" else None
                    for column_cells in worksheet.columns:
                        width = min(
                            45,
                            max(
                                10,
                                max(len(str(cell.value or "")) for cell in column_cells) + 2,
                            ),
                        )
                        worksheet.column_dimensions[column_cells[0].column_letter].width = width
                    for row in worksheet.iter_rows():
                        for cell in row:
                            cell.font = Font(name="Arial", size=10, bold=cell.font.bold)
                            cell.alignment = Alignment(vertical="center")
                summary_sheet = workbook["Summary"]
                summary_sheet["A1"] = "PRTI-GH multi-objective study"
                summary_sheet["A1"].font = Font(name="Arial", size=14, bold=True, color="1F1F1F")
                header_rows = {
                    "Summary": (3, 13),
                    "Configurations": (1,),
                    "Lambda intervals": (1,),
                }
                for sheet_name, row_numbers in header_rows.items():
                    worksheet = workbook[sheet_name]
                    for row_number in row_numbers:
                        for cell in worksheet[row_number]:
                            if cell.value is None:
                                continue
                            cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
                            cell.fill = PatternFill("solid", fgColor=dark)
                            cell.alignment = Alignment(horizontal="center", vertical="center")
                summary_sheet["A12"] = "Run status"
                summary_sheet["A12"].font = Font(name="Arial", size=10, bold=True)
                summary_sheet["A12"].fill = PatternFill("solid", fgColor=light)
                for worksheet in (workbook["Configurations"], workbook["Lambda intervals"]):
                    worksheet.auto_filter.ref = worksheet.dimensions
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if pd.isna(value):
            return None
        raise TypeError(f"Cannot serialize {type(value).__name__}")
