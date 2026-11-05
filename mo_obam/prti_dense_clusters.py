"""Configurable, aggregate-only PRTI-GH cluster-count selection study.

The protocol evaluates every integer cluster count in one prospectively
declared, mathematically feasible interval. It reuses only byte-verifiable
aggregate checkpoints and never writes patient-level data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import pandas as pd

from .provenance import atomic_write_text
from .prti_data import (
    DEFAULT_PRTI_SOURCE,
    EXPECTED_PRTI_SHAPE,
    PRTIPreparedData,
)
from .prti_robustness import (
    PRTIRobustnessConfig,
    PRTIRobustnessExperiment,
    PRTIRobustnessRunner,
)
from .prti_runner import PAPER_IDENTIFIER, PAPER_REVISION, PROJECT_ROOT
from .pso_solver import PSOConfig
from .schema import MOOBAMConfig


@dataclass(frozen=True)
class PRTIDenseClusterConfig(PRTIRobustnessConfig):
    """Configure a prospective inclusive cluster-count comparison."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    output_directory: Path = Path("evidence/prti-dense-clusters-v1")
    population_size: int = 100
    iterations: int = 100
    seeds: tuple[int, ...] = tuple(range(10))
    workers: int = 1
    minimum_clusters: int = 4
    maximum_clusters: int = 30
    k: int = 10
    lambda_weight: float = 1.0
    maximum_information_loss: float = 0.60
    maximum_homogeneity_fraction: float = 0.20
    maximum_zero_minimum_entropy_sensitive_attributes: int = 1
    reuse_directory: Path | None = Path("evidence/prti-robustness-v1")
    held_out_seed: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.minimum_clusters < 1:
            raise ValueError("minimum_clusters must be positive.")
        if self.maximum_clusters < self.minimum_clusters:
            raise ValueError("maximum_clusters must be at least minimum_clusters.")
        if self.k < 2:
            raise ValueError("k must be at least 2.")
        maximum_feasible_clusters = EXPECTED_PRTI_SHAPE[0] // self.k
        if self.maximum_clusters > maximum_feasible_clusters:
            raise ValueError(
                f"maximum_clusters={self.maximum_clusters} is mathematically "
                f"infeasible for {EXPECTED_PRTI_SHAPE[0]} rows with k={self.k}; "
                f"the maximum is {maximum_feasible_clusters}."
            )
        if self.held_out_seed is not None:
            if self.held_out_seed < 0:
                raise ValueError("held_out_seed must be non-negative.")
            if self.held_out_seed in self.seeds:
                raise ValueError(
                    "held_out_seed must not appear in the comparison seeds."
                )
        if not 0.0 <= self.maximum_information_loss <= 1.0:
            raise ValueError("maximum_information_loss must be in [0, 1].")
        if not 0.0 <= self.maximum_homogeneity_fraction <= 1.0:
            raise ValueError("maximum_homogeneity_fraction must be in [0, 1].")
        if self.maximum_zero_minimum_entropy_sensitive_attributes < 0:
            raise ValueError(
                "maximum_zero_minimum_entropy_sensitive_attributes cannot be negative."
            )


class PRTIDenseClusterPlan:
    """Build every integer cluster count in the declared interval."""

    @staticmethod
    def build(config: PRTIDenseClusterConfig) -> tuple[PRTIRobustnessExperiment, ...]:
        return tuple(
            PRTIRobustnessExperiment(
                configuration_id=f"clusters_{clusters:02d}",
                purpose=f"Dense cluster-count comparison at C={clusters}",
                selection_basis=(
                    "Prospectively included because every integer C from "
                    f"{config.minimum_clusters} through {config.maximum_clusters} "
                    "is evaluated under identical settings."
                ),
                model=MOOBAMConfig(
                    k=config.k,
                    n_clusters=clusters,
                    lambda_weight=config.lambda_weight,
                ),
            )
            for clusters in range(config.minimum_clusters, config.maximum_clusters + 1)
        )


@dataclass(frozen=True)
class PRTIParameterGridConfig(PRTIRobustnessConfig):
    """Configure a joint k, cluster-count, and lambda comparison."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    output_directory: Path = Path("evidence/prti-parameter-grid-v1")
    population_size: int = 100
    iterations: int = 100
    seeds: tuple[int, ...] = tuple(range(10))
    workers: int = 1
    minimum_clusters: int = 1
    maximum_clusters: int = 148
    k_values: tuple[int, ...] = (10,)
    lambda_weights: tuple[float, ...] = (1.0,)
    maximum_information_loss: float = 0.60
    maximum_homogeneity_fraction: float = 0.20
    maximum_zero_minimum_entropy_sensitive_attributes: int = 1
    reuse_directory: Path | None = None
    held_out_seed: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.minimum_clusters < 1:
            raise ValueError("minimum_clusters must be positive.")
        if self.maximum_clusters < self.minimum_clusters:
            raise ValueError("maximum_clusters must be at least minimum_clusters.")
        if not self.k_values:
            raise ValueError("At least one k value is required.")
        if len(self.k_values) != len(set(self.k_values)):
            raise ValueError("k values must be unique.")
        if any(k < 2 for k in self.k_values):
            raise ValueError("Every k value must be at least 2.")
        infeasible_k = [
            k
            for k in self.k_values
            if self.minimum_clusters > EXPECTED_PRTI_SHAPE[0] // k
        ]
        if infeasible_k:
            raise ValueError(
                "minimum_clusters leaves no feasible cluster count for k values: "
                f"{infeasible_k}."
            )
        if not self.lambda_weights:
            raise ValueError("At least one lambda weight is required.")
        if len(self.lambda_weights) != len(set(self.lambda_weights)):
            raise ValueError("Lambda weights must be unique.")
        if any(not 0.0 <= weight <= 1.0 for weight in self.lambda_weights):
            raise ValueError("Every lambda weight must be in [0, 1].")
        if self.held_out_seed is None:
            raise ValueError("A prospectively held-out seed is required.")
        if self.held_out_seed < 0:
            raise ValueError("held_out_seed must be non-negative.")
        if self.held_out_seed in self.seeds:
            raise ValueError("held_out_seed must not appear in comparison seeds.")
        if not 0.0 <= self.maximum_information_loss <= 1.0:
            raise ValueError("maximum_information_loss must be in [0, 1].")
        if not 0.0 <= self.maximum_homogeneity_fraction <= 1.0:
            raise ValueError("maximum_homogeneity_fraction must be in [0, 1].")
        if self.maximum_zero_minimum_entropy_sensitive_attributes < 0:
            raise ValueError(
                "maximum_zero_minimum_entropy_sensitive_attributes cannot be negative."
            )


class PRTIParameterGridPlan:
    """Build every feasible point in a declared k, C, and lambda grid."""

    @staticmethod
    def build(config: PRTIParameterGridConfig) -> tuple[PRTIRobustnessExperiment, ...]:
        experiments: list[PRTIRobustnessExperiment] = []
        for k in config.k_values:
            endpoint = min(
                config.maximum_clusters,
                EXPECTED_PRTI_SHAPE[0] // k,
            )
            for clusters in range(config.minimum_clusters, endpoint + 1):
                for weight in config.lambda_weights:
                    model = MOOBAMConfig(
                        k=k,
                        n_clusters=clusters,
                        lambda_weight=weight,
                    )
                    weight_text = str(float(weight))
                    if weight_text.endswith(".0"):
                        weight_text = weight_text[:-2]
                    weight_label = weight_text.replace(".", "p")
                    experiments.append(
                        PRTIRobustnessExperiment(
                            configuration_id=(
                                f"k{k:02d}_c{clusters:03d}_lambda{weight_label}"
                            ),
                            purpose=(
                                "Joint parameter comparison at "
                                f"k={k}, C={clusters}, lambda={weight:g}"
                            ),
                            selection_basis=(
                                "Prospectively included in the declared feasible "
                                "k, cluster-count, and lambda grid."
                            ),
                            model=model,
                        )
                    )
        return tuple(experiments)


class PRTIDenseClusterRunner(PRTIRobustnessRunner):
    """Execute, resume, screen, and rank the dense cluster-count study."""

    PLAN_FILE = "prti_dense_cluster_plan.json"
    RUNS_FILE = "prti_dense_cluster_runs.csv"
    SUMMARY_FILE = "prti_dense_cluster_summary.csv"
    SUMMARY_JSON_FILE = "prti_dense_cluster_summary.json"
    SELECTION_FILE = "prti_dense_cluster_selection.csv"
    SELECTION_JSON_FILE = "prti_dense_cluster_selection.json"

    _REUSE_CONFIGURATION_BY_CLUSTERS = {
        4: "privacy_candidate",
        10: "compromise_candidate",
        30: "baseline",
    }

    def __init__(self, config: PRTIDenseClusterConfig) -> None:
        super().__init__(config)
        self.config = config
        self.experiments = PRTIDenseClusterPlan.build(config)

    def write_plan(self, source_sha256: str | None = None) -> Path:
        """Create an immutable prospective plan including the selection rule."""

        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.config.output_directory / self.PLAN_FILE
        payload = {
            "declared_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "scope": "Aggregate-only PRTI-GH dense cluster-count study; ML excluded",
            "status": "prospective project protocol, not a paper-reported experiment",
            "source_sha256": source_sha256,
            "cluster_values": [
                experiment.model.n_clusters for experiment in self.experiments
            ],
            "number_of_configurations": len(self.experiments),
            "number_of_seeds": len(self.config.seeds),
            "number_of_planned_runs": len(self.experiments) * len(self.config.seeds),
            "seeds": list(self.config.seeds),
            "held_out_seed": self.config.held_out_seed,
            "seed_protocol": {
                "comparison_seeds": list(self.config.seeds),
                "held_out_seed": self.config.held_out_seed,
                "rule": (
                    "Use every comparison seed for configuration selection. "
                    "If a held-out seed is declared, do not inspect it until the "
                    "aggregate selection is complete and do not replace it after "
                    "seeing its result."
                ),
            },
            "fixed_model_configuration": {
                "k": self.config.k,
                "lambda_weight": self.config.lambda_weight,
            },
            "pso_configuration_except_seed": {
                "population_size": self.config.population_size,
                "iterations": self.config.iterations,
                "random_coefficient_max": PSOConfig().random_coefficient_max,
            },
            "patient_level_outputs": False,
            "exact_checkpoint_reuse": {
                "permitted": self.config.reuse_directory is not None,
                "source_directory": (
                    os.path.relpath(
                        Path(self.config.reuse_directory).resolve(), PROJECT_ROOT
                    )
                    if self.config.reuse_directory is not None
                    else None
                ),
                "requirements": [
                    "same source SHA-256",
                    "same paper revision",
                    "same k, C, lambda, population, iterations, seed and PSO coefficient",
                    "aggregate-only checkpoint",
                ],
            },
            "privacy_pass_rule": {
                "all_planned_seeds_must_complete": True,
                "minimum_cluster_size_at_least_k_for_every_seed": True,
                "maximum_linkage_records_at_each_tau": 0,
                "maximum_homogeneity_fraction_any_sa": (
                    self.config.maximum_homogeneity_fraction
                ),
                "maximum_zero_minimum_entropy_sensitive_attributes": (
                    self.config.maximum_zero_minimum_entropy_sensitive_attributes
                ),
                "maximum_information_loss_for_every_seed": (
                    self.config.maximum_information_loss
                ),
            },
            "selection_rule": (
                "Among configurations passing every privacy and feasibility rule "
                "across every comparison seed, select the lowest mean information "
                "loss. Break exact ties by lower maximum homogeneity fraction, then "
                "by the smaller cluster count."
            ),
            "analysis_rule": self._analysis_rule(),
            "release_boundary": (
                "Selection produces an aggregate research candidate only. It does "
                "not approve patient-level or unrestricted release; direct-identifier, "
                "free-text, custodian, and independent disclosure-control review remain."
            ),
        }
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            stable = lambda item: {
                key: value
                for key, value in item.items()
                if key not in {"declared_at_utc", "source_sha256"}
            }
            if stable(existing) != stable(payload):
                raise ValueError(
                    "The existing dense-study plan differs from the requested policy. "
                    "Use a new evidence directory instead of overwriting it."
                )
            if existing.get("source_sha256") is None and source_sha256 is not None:
                existing["source_sha256"] = source_sha256
                atomic_write_text(path, json.dumps(existing, indent=2))
            elif (
                source_sha256 is not None
                and existing.get("source_sha256") != source_sha256
            ):
                raise ValueError("The predeclared plan uses a different source SHA-256.")
            return path
        atomic_write_text(path, json.dumps(payload, indent=2))
        return path

    def _before_tasks(
        self,
        prepared: PRTIPreparedData,
        source_hash: str,
    ) -> None:
        """Import exact aggregate evidence before discovering pending tasks."""

        del prepared
        self.import_exact_checkpoints(source_hash)

    def import_exact_checkpoints(self, source_hash: str) -> int:
        """Reuse matching aggregate robustness checkpoints with provenance."""

        if self.config.reuse_directory is None:
            return 0
        imported = 0
        by_clusters = {
            experiment.model.n_clusters: experiment for experiment in self.experiments
        }
        for clusters, old_id in self._REUSE_CONFIGURATION_BY_CLUSTERS.items():
            experiment = by_clusters.get(clusters)
            if experiment is None:
                continue
            for seed in self.config.seeds:
                target = self._checkpoint_path(experiment, seed)
                if target.exists():
                    continue
                source = (
                    self.config.reuse_directory
                    / self.RUNS_DIRECTORY
                    / old_id
                    / f"seed_{seed:04d}.json"
                )
                if not source.is_file():
                    continue
                payload = json.loads(source.read_text(encoding="utf-8"))
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
                    or payload.get("configuration_id") != old_id
                    or payload.get("model_configuration") != asdict(experiment.model)
                    or payload.get("pso_configuration") != expected_pso
                    or payload.get("patient_level_outputs") is not False
                ):
                    continue
                row = dict(payload["result_row"])
                row.update(
                    {
                        "configuration_id": experiment.configuration_id,
                        "model_run_id": experiment.model_run_id,
                        "purpose": experiment.purpose,
                    }
                )
                source_digest = sha256(source.read_bytes()).hexdigest()
                imported_payload = {
                    **payload,
                    "configuration_id": experiment.configuration_id,
                    "result_row": row,
                    "reused_aggregate_checkpoint": {
                        "path": os.path.relpath(source.resolve(), PROJECT_ROOT),
                        "sha256": source_digest,
                        "original_configuration_id": old_id,
                    },
                }
                target.parent.mkdir(parents=True, exist_ok=True)
                if not self._checkpoint_row_is_valid(experiment, seed, row):
                    continue
                atomic_write_text(target, json.dumps(imported_payload, indent=2))
                imported += 1
        return imported

    def assess_summary(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Apply the predeclared pass rule and deterministic utility ranking."""

        rows: list[dict[str, object]] = []
        dataset_rows = EXPECTED_PRTI_SHAPE[0]
        for item in summary.to_dict(orient="records"):
            completed = int(item["runs_completed"])
            planned = int(item["seeds_planned"])
            homogeneity_max = self._number(item, "homogeneity_records_any_sa_max")
            homogeneity_fraction = (
                homogeneity_max / dataset_rows if homogeneity_max is not None else None
            )
            checks = {
                "all_seeds_completed": completed == planned,
                "no_infeasible_runs": int(item["runs_no_feasible_particle"]) == 0,
                "hard_k_every_seed": self._at_least(
                    item, "minimum_cluster_size_min", int(item["k"])
                ),
                "zero_linkage_tau_0_05": self._at_most(
                    item, "linkage_records_tau_0_05_max", 0
                ),
                "zero_linkage_tau_0_075": self._at_most(
                    item, "linkage_records_tau_0_075_max", 0
                ),
                "zero_linkage_tau_0_1": self._at_most(
                    item, "linkage_records_tau_0_1_max", 0
                ),
                "homogeneity_within_limit": (
                    homogeneity_fraction is not None
                    and homogeneity_fraction
                    <= self.config.maximum_homogeneity_fraction
                ),
                "zero_entropy_sa_within_limit": self._at_most(
                    item,
                    "zero_minimum_entropy_sensitive_attributes_max",
                    self.config.maximum_zero_minimum_entropy_sensitive_attributes,
                ),
                "information_loss_within_limit": self._at_most(
                    item,
                    "information_loss_max",
                    self.config.maximum_information_loss,
                ),
            }
            rows.append(
                {
                    "configuration_id": item["configuration_id"],
                    "k": int(item["k"]),
                    "configured_clusters": int(item["configured_clusters"]),
                    "lambda": float(item["lambda"]),
                    "runs_completed": completed,
                    "runs_planned": planned,
                    "information_loss_mean": self._number(
                        item, "information_loss_mean"
                    ),
                    "information_loss_max": self._number(item, "information_loss_max"),
                    "entropy_sum_mean": self._number(item, "entropy_sum_mean"),
                    "entropy_sum_std": self._number(item, "entropy_sum_std"),
                    "entropy_sum_min": self._number(item, "entropy_sum_min"),
                    "entropy_sum_max": self._number(item, "entropy_sum_max"),
                    "homogeneity_records_any_sa_max": homogeneity_max,
                    "homogeneity_fraction_any_sa_max": homogeneity_fraction,
                    "minimum_cluster_size_min": self._number(
                        item, "minimum_cluster_size_min"
                    ),
                    "zero_minimum_entropy_sensitive_attributes_max": self._number(
                        item, "zero_minimum_entropy_sensitive_attributes_max"
                    ),
                    **checks,
                    "passes_all_predeclared_rules": all(checks.values()),
                    "pareto_efficient_il_entropy": False,
                    "selection_rank": None,
                    "selected": False,
                }
            )
        assessed = pd.DataFrame(rows)
        if assessed.empty:
            return assessed
        eligible = assessed.loc[assessed["passes_all_predeclared_rules"]].copy()
        pareto_indices = self._pareto_efficient_indices(eligible)
        assessed.loc[
            list(pareto_indices), "pareto_efficient_il_entropy"
        ] = True
        eligible = eligible.sort_values(
            [
                "information_loss_mean",
                "homogeneity_fraction_any_sa_max",
                "configured_clusters",
            ],
            kind="stable",
        )
        for rank, index in enumerate(eligible.index, start=1):
            assessed.loc[index, "selection_rank"] = rank
            assessed.loc[index, "selected"] = rank == 1
        return assessed.sort_values("configured_clusters").reset_index(drop=True)

    def _write_outputs(self, source_hash: str) -> dict[str, object]:
        result = super()._write_outputs(source_hash)
        assessed = self.assess_summary(result["summary"])
        planned_runs = len(self.experiments) * len(self.config.seeds)
        selection_complete = len(result["runs"]) == planned_runs
        if not selection_complete:
            assessed = self.withhold_partial_selection(assessed)
        output = self.config.output_directory
        csv_path = output / self.SELECTION_FILE
        json_path = output / self.SELECTION_JSON_FILE
        atomic_write_text(csv_path, assessed.to_csv(index=False))
        selected_rows = assessed.loc[assessed.get("selected", False)].to_dict(
            orient="records"
        )
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "source_sha256": source_hash,
            "patient_level_outputs": False,
            "study_configuration": self._selection_study_configuration(
                planned_runs
            ),
            "privacy_pass_rule": {
                "all_planned_seeds_must_complete": True,
                "minimum_cluster_size_at_least_k_for_every_seed": True,
                "maximum_linkage_records_at_each_tau": 0,
                "maximum_homogeneity_fraction_any_sa": (
                    self.config.maximum_homogeneity_fraction
                ),
                "maximum_zero_minimum_entropy_sensitive_attributes": (
                    self.config.maximum_zero_minimum_entropy_sensitive_attributes
                ),
                "maximum_information_loss_for_every_seed": (
                    self.config.maximum_information_loss
                ),
            },
            "status": (
                "selection_complete" if selection_complete else "selection_incomplete"
            ),
            "selection_rule": self._selection_rule(),
            "analysis_rule": self._analysis_rule(),
            "selected_configuration": selected_rows[0] if selected_rows else None,
            "assessments": json.loads(
                assessed.to_json(orient="records", double_precision=15)
            ),
            "release_boundary": (
                "Aggregate selection is not patient-level release approval. The held-"
                "out seed and external disclosure-control gates remain required."
            ),
            "optimality_boundary": self._optimality_boundary(),
        }
        atomic_write_text(json_path, json.dumps(payload, indent=2))
        result["selection"] = assessed
        result["selection_report"] = payload
        result["paths"].update({"selection": csv_path, "selection_json": json_path})
        return result

    def _selection_study_configuration(
        self,
        planned_runs: int,
    ) -> dict[str, object]:
        """Describe the fixed-k, fixed-lambda cluster interval."""

        return {
            "study_type": "fixed_k_lambda_cluster_range",
            "minimum_clusters": self.config.minimum_clusters,
            "maximum_clusters": self.config.maximum_clusters,
            "k": self.config.k,
            "lambda_weight": self.config.lambda_weight,
            "population_size": self.config.population_size,
            "iterations": self.config.iterations,
            "comparison_seeds": list(self.config.seeds),
            "held_out_seed": self.config.held_out_seed,
            "planned_runs": planned_runs,
        }

    @staticmethod
    def _selection_rule() -> str:
        return (
            "Among configurations passing every predeclared rule across all "
            "comparison seeds, select the lowest mean information loss. Break "
            "exact ties by lower maximum homogeneity fraction and then smaller C."
        )

    @staticmethod
    def _analysis_rule() -> str:
        return (
            "Among configurations passing every predeclared privacy and feasibility "
            "rule, report the Pareto frontier that minimizes mean information loss "
            "and maximizes mean sensitive-attribute entropy sum. Pareto status is "
            "descriptive and does not replace the predeclared selection rule."
        )

    @staticmethod
    def _pareto_efficient_indices(eligible: pd.DataFrame) -> set[object]:
        """Return eligible rows not dominated on mean IL and mean entropy."""

        required = {"information_loss_mean", "entropy_sum_mean"}
        if eligible.empty or not required.issubset(eligible.columns):
            return set()
        valid = eligible.dropna(subset=list(required)).copy()
        efficient: set[object] = set()
        best_entropy_at_lower_il: float | None = None
        for _information_loss, group in valid.groupby(
            "information_loss_mean", sort=True
        ):
            group_max_entropy = float(group["entropy_sum_mean"].max())
            if (
                best_entropy_at_lower_il is None
                or group_max_entropy > best_entropy_at_lower_il
            ):
                efficient.update(
                    group.index[
                        group["entropy_sum_mean"].eq(group_max_entropy)
                    ].tolist()
                )
            if (
                best_entropy_at_lower_il is None
                or group_max_entropy > best_entropy_at_lower_il
            ):
                best_entropy_at_lower_il = group_max_entropy
        return efficient

    @staticmethod
    def _optimality_boundary() -> str:
        return (
            "The selected configuration is the best observed eligible setting "
            "within this declared cluster interval, fixed k/lambda parameterization, "
            "seed set, and stochastic search budget. It is not proof of a global "
            "optimum."
        )

    @staticmethod
    def withhold_partial_selection(assessed: pd.DataFrame) -> pd.DataFrame:
        """Keep diagnostics but never rank or select from an incomplete study."""

        withheld = assessed.copy()
        if not withheld.empty:
            if "pareto_efficient_il_entropy" in withheld:
                withheld["pareto_efficient_il_entropy"] = None
            withheld["selection_rank"] = None
            withheld["selected"] = False
        return withheld

    def _canonical_command(self) -> list[str]:
        source = os.path.relpath(Path(self.config.source_path).resolve(), PROJECT_ROOT)
        output = os.path.relpath(
            Path(self.config.output_directory).resolve(), PROJECT_ROOT
        )
        command = [
            ".venv/bin/python",
            "run_prti_experiments.py",
            "dense-clusters",
            "--source",
            source,
            "--output-directory",
            output,
            "--minimum-clusters",
            str(self.config.minimum_clusters),
            "--maximum-clusters",
            str(self.config.maximum_clusters),
            "--k",
            str(self.config.k),
            "--lambda-weight",
            str(self.config.lambda_weight),
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
        if self.config.held_out_seed is not None:
            command.extend(["--held-out-seed", str(self.config.held_out_seed)])
        if self.config.reuse_directory is not None:
            reuse = os.path.relpath(
                Path(self.config.reuse_directory).resolve(), PROJECT_ROOT
            )
            command.extend(["--reuse-directory", reuse])
        return command

    @staticmethod
    def _number(item: dict[str, object], key: str) -> float | None:
        value = item.get(key)
        if value is None or pd.isna(value):
            return None
        return float(value)

    @classmethod
    def _at_most(cls, item: dict[str, object], key: str, limit: float) -> bool:
        value = cls._number(item, key)
        return value is not None and value <= limit

    @classmethod
    def _at_least(cls, item: dict[str, object], key: str, limit: float) -> bool:
        value = cls._number(item, key)
        return value is not None and value >= limit


class PRTIParameterGridRunner(PRTIDenseClusterRunner):
    """Execute and rank a joint k, cluster-count, and lambda study."""

    PLAN_FILE = "prti_parameter_grid_plan.json"
    RUNS_FILE = "prti_parameter_grid_runs.csv"
    SUMMARY_FILE = "prti_parameter_grid_summary.csv"
    SUMMARY_JSON_FILE = "prti_parameter_grid_summary.json"
    SELECTION_FILE = "prti_parameter_grid_selection.csv"
    SELECTION_JSON_FILE = "prti_parameter_grid_selection.json"

    def __init__(self, config: PRTIParameterGridConfig) -> None:
        PRTIRobustnessRunner.__init__(self, config)
        self.config = config
        self.experiments = PRTIParameterGridPlan.build(config)

    def write_plan(self, source_sha256: str | None = None) -> Path:
        """Create an immutable prospective joint-parameter plan."""

        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.config.output_directory / self.PLAN_FILE
        feasible_endpoints = {
            str(k): min(
                self.config.maximum_clusters,
                EXPECTED_PRTI_SHAPE[0] // k,
            )
            for k in self.config.k_values
        }
        payload = {
            "declared_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "scope": (
                "Aggregate-only PRTI-GH joint k, cluster-count, and lambda study; "
                "ML excluded"
            ),
            "status": "prospective project protocol, not a paper-reported experiment",
            "source_sha256": source_sha256,
            "k_values": list(self.config.k_values),
            "lambda_weights": list(self.config.lambda_weights),
            "requested_cluster_interval": [
                self.config.minimum_clusters,
                self.config.maximum_clusters,
            ],
            "feasible_maximum_clusters_by_k": feasible_endpoints,
            "number_of_configurations": len(self.experiments),
            "number_of_seeds": len(self.config.seeds),
            "number_of_planned_runs": len(self.experiments)
            * len(self.config.seeds),
            "seeds": list(self.config.seeds),
            "held_out_seed": self.config.held_out_seed,
            "seed_protocol": {
                "comparison_seeds": list(self.config.seeds),
                "held_out_seed": self.config.held_out_seed,
                "rule": (
                    "Use every comparison seed for configuration selection. Do not "
                    "inspect or substitute the held-out seed before selection."
                ),
            },
            "pso_configuration_except_seed": {
                "population_size": self.config.population_size,
                "iterations": self.config.iterations,
                "random_coefficient_max": PSOConfig().random_coefficient_max,
            },
            "configurations": [
                {
                    "configuration_id": experiment.configuration_id,
                    "model_configuration": asdict(experiment.model),
                }
                for experiment in self.experiments
            ],
            "patient_level_outputs": False,
            "exact_checkpoint_reuse": {
                "permitted": self.config.reuse_directory is not None,
                "source_directory": (
                    os.path.relpath(
                        Path(self.config.reuse_directory).resolve(), PROJECT_ROOT
                    )
                    if self.config.reuse_directory is not None
                    else None
                ),
                "requirements": [
                    "same source SHA-256",
                    "same paper revision",
                    "same k, C, lambda, population, iterations, seed and PSO coefficient",
                    "aggregate-only checkpoint",
                ],
            },
            "privacy_pass_rule": {
                "all_planned_seeds_must_complete": True,
                "minimum_cluster_size_at_least_k_for_every_seed": True,
                "maximum_linkage_records_at_each_tau": 0,
                "maximum_homogeneity_fraction_any_sa": (
                    self.config.maximum_homogeneity_fraction
                ),
                "maximum_zero_minimum_entropy_sensitive_attributes": (
                    self.config.maximum_zero_minimum_entropy_sensitive_attributes
                ),
                "maximum_information_loss_for_every_seed": (
                    self.config.maximum_information_loss
                ),
            },
            "selection_rule": (
                "Among configurations passing every privacy and feasibility rule "
                "across every comparison seed, select the lowest mean information "
                "loss. Break exact ties by lower maximum homogeneity fraction, then "
                "higher k, smaller cluster count, and higher lambda."
            ),
            "analysis_rule": self._analysis_rule(),
            "optimality_boundary": (
                "Selection is the best observed eligible point in this declared "
                "finite grid and PSO budget, not a mathematical global optimum."
            ),
            "release_boundary": (
                "Selection produces an aggregate research candidate only. Patient-"
                "level release still requires the held-out run and external reviews."
            ),
        }
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))

            def stable(item: dict[str, object]) -> dict[str, object]:
                return {
                    key: value
                    for key, value in item.items()
                    if key not in {"declared_at_utc", "source_sha256"}
                }

            if stable(existing) != stable(payload):
                raise ValueError(
                    "The existing parameter-grid plan differs from the requested "
                    "policy. Use a new evidence directory instead of overwriting it."
                )
            if existing.get("source_sha256") is None and source_sha256 is not None:
                existing["source_sha256"] = source_sha256
                atomic_write_text(path, json.dumps(existing, indent=2))
            elif (
                source_sha256 is not None
                and existing.get("source_sha256") != source_sha256
            ):
                raise ValueError("The predeclared plan uses a different source SHA-256.")
            return path
        atomic_write_text(path, json.dumps(payload, indent=2))
        return path

    def import_exact_checkpoints(self, source_hash: str) -> int:
        """Import only exact k=10, lambda=1 robustness checkpoints."""

        if self.config.reuse_directory is None:
            return 0
        imported = 0
        for experiment in self.experiments:
            if experiment.model.k != 10 or experiment.model.lambda_weight != 1.0:
                continue
            old_id = self._REUSE_CONFIGURATION_BY_CLUSTERS.get(
                experiment.model.n_clusters
            )
            if old_id is None:
                continue
            for seed in self.config.seeds:
                target = self._checkpoint_path(experiment, seed)
                if target.exists():
                    continue
                source = (
                    self.config.reuse_directory
                    / self.RUNS_DIRECTORY
                    / old_id
                    / f"seed_{seed:04d}.json"
                )
                if not source.is_file():
                    continue
                try:
                    payload = json.loads(source.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
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
                    or payload.get("configuration_id") != old_id
                    or payload.get("model_configuration")
                    != asdict(experiment.model)
                    or payload.get("pso_configuration") != expected_pso
                    or payload.get("patient_level_outputs") is not False
                    or not isinstance(payload.get("result_row"), dict)
                ):
                    continue
                row = dict(payload["result_row"])
                row.update(
                    {
                        "configuration_id": experiment.configuration_id,
                        "model_run_id": experiment.model_run_id,
                        "purpose": experiment.purpose,
                    }
                )
                if not self._checkpoint_row_is_valid(experiment, seed, row):
                    continue
                imported_payload = {
                    **payload,
                    "configuration_id": experiment.configuration_id,
                    "result_row": row,
                    "reused_aggregate_checkpoint": {
                        "path": os.path.relpath(source.resolve(), PROJECT_ROOT),
                        "sha256": sha256(source.read_bytes()).hexdigest(),
                        "original_configuration_id": old_id,
                    },
                }
                atomic_write_text(target, json.dumps(imported_payload, indent=2))
                imported += 1
        return imported

    def assess_summary(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Apply the shared privacy rules and joint-grid tie breakers."""

        assessed = super().assess_summary(summary)
        if assessed.empty:
            return assessed
        assessed["selection_rank"] = None
        assessed["selected"] = False
        eligible = assessed.loc[assessed["passes_all_predeclared_rules"]].copy()
        eligible = eligible.sort_values(
            [
                "information_loss_mean",
                "homogeneity_fraction_any_sa_max",
                "k",
                "configured_clusters",
                "lambda",
            ],
            ascending=[True, True, False, True, False],
            kind="stable",
        )
        for rank, index in enumerate(eligible.index, start=1):
            assessed.loc[index, "selection_rank"] = rank
            assessed.loc[index, "selected"] = rank == 1
        return assessed.sort_values(
            ["k", "configured_clusters", "lambda"], kind="stable"
        ).reset_index(drop=True)

    def _selection_study_configuration(
        self,
        planned_runs: int,
    ) -> dict[str, object]:
        return {
            "study_type": "joint_k_cluster_lambda_grid",
            "minimum_clusters": self.config.minimum_clusters,
            "maximum_clusters": self.config.maximum_clusters,
            "k_values": list(self.config.k_values),
            "lambda_weights": list(self.config.lambda_weights),
            "population_size": self.config.population_size,
            "iterations": self.config.iterations,
            "comparison_seeds": list(self.config.seeds),
            "held_out_seed": self.config.held_out_seed,
            "planned_configurations": len(self.experiments),
            "planned_runs": planned_runs,
        }

    @staticmethod
    def _selection_rule() -> str:
        return (
            "Among configurations passing every predeclared rule across all "
            "comparison seeds, select the lowest mean information loss. Break "
            "exact ties by lower maximum homogeneity fraction, higher k, smaller C, "
            "and then higher lambda."
        )

    @staticmethod
    def _optimality_boundary() -> str:
        return (
            "The selected configuration is the best observed eligible point within "
            "this declared finite k/C/lambda grid, seed set, and stochastic search "
            "budget. It is not proof of a continuous or mathematical global optimum."
        )

    def _canonical_command(self) -> list[str]:
        source = os.path.relpath(Path(self.config.source_path).resolve(), PROJECT_ROOT)
        output = os.path.relpath(
            Path(self.config.output_directory).resolve(), PROJECT_ROOT
        )
        command = [
            ".venv/bin/python",
            "run_prti_experiments.py",
            "parameter-grid",
            "--source",
            source,
            "--output-directory",
            output,
            "--minimum-clusters",
            str(self.config.minimum_clusters),
            "--maximum-clusters",
            str(self.config.maximum_clusters),
            "--k-values",
            *[str(value) for value in self.config.k_values],
            "--lambda-weights",
            *[str(value) for value in self.config.lambda_weights],
            "--population",
            str(self.config.population_size),
            "--iterations",
            str(self.config.iterations),
            "--seeds",
            *[str(seed) for seed in self.config.seeds],
            "--held-out-seed",
            str(self.config.held_out_seed),
            "--workers",
            str(self.config.workers),
            "--worker-backend",
            self.config.worker_backend,
        ]
        if self.config.reuse_directory is None:
            command.append("--no-reuse")
        else:
            reuse = os.path.relpath(
                Path(self.config.reuse_directory).resolve(), PROJECT_ROOT
            )
            command.extend(["--reuse-directory", reuse])
        return command
