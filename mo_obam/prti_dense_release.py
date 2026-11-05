"""Generate one held-out PRTI-GH candidate from a completed dense study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .provenance import atomic_write_text, exclusive_directory_lock
from .prti_data import DEFAULT_PRTI_SOURCE, EXPECTED_PRTI_SHA256
from .prti_release import PRTIFinalCandidateRunner, PRTIReleasePolicy
from .prti_runner import (
    PAPER_IDENTIFIER,
    PAPER_REVISION,
    PRTIAnonymizationRunner,
    PRTIRunConfig,
)
from .pso_solver import PSOConfig
from .schema import MOOBAMConfig


@dataclass(frozen=True)
class PRTIDenseSelectedPolicy:
    """Paths and optional label for the held-out selected candidate."""

    study_directory: Path
    output_directory: Path
    evidence_directory: Path
    source_path: Path = DEFAULT_PRTI_SOURCE
    run_label: str | None = None


@dataclass(frozen=True)
class PRTIDenseSelectedContext:
    """Validated configuration recovered from aggregate selection evidence."""

    model: MOOBAMConfig
    pso: PSOConfig
    comparison_seeds: tuple[int, ...]
    run_label: str
    maximum_information_loss: float
    maximum_homogeneity_fraction: float
    maximum_zero_entropy_sensitive_attributes: int
    plan: dict[str, object]
    selection: dict[str, object]


class PRTIDenseSelectedCandidateRunner:
    """Refuse post-hoc settings and run the prospectively held-out seed once."""

    PLAN_FILE = "prti_dense_final_selection_plan.json"
    ACCEPTANCE_CSV = "prti_dense_final_candidate_acceptance.csv"
    ACCEPTANCE_JSON = "prti_dense_final_candidate_acceptance.json"
    STUDY_PLAN_FILE = "prti_dense_cluster_plan.json"
    STUDY_SELECTION_FILE = "prti_dense_cluster_selection.json"
    GRID_PLAN_FILE = "prti_parameter_grid_plan.json"
    GRID_SELECTION_FILE = "prti_parameter_grid_selection.json"

    def __init__(self, policy: PRTIDenseSelectedPolicy) -> None:
        self.policy = policy

    def predeclare(self) -> Path:
        """Validate completed selection evidence and immutably record final settings."""

        context = self._load_context()
        study_plan_path, study_selection_path = self._study_evidence_paths()
        payload = {
            "declared_at_utc": self._utc_now(),
            "status": "prospective held-out candidate from completed aggregate selection",
            "source_sha256": EXPECTED_PRTI_SHA256,
            "study_directory": str(self.policy.study_directory),
            "study_plan_sha256": self._file_sha256(study_plan_path),
            "study_selection_sha256": self._file_sha256(study_selection_path),
            "selection_rule": context.selection["selection_rule"],
            "selected_configuration": context.selection["selected_configuration"],
            "model_configuration": asdict(context.model),
            "pso_configuration": asdict(context.pso),
            "comparison_seeds": list(context.comparison_seeds),
            "held_out_seed": context.pso.seed,
            "seed_rule": (
                "The held-out seed was declared before the selection study, was not "
                "used for selection, and must not be replaced after inspection."
            ),
            "run_label": context.run_label,
            "project_screening_thresholds": {
                "linkage_attack_records_at_each_reported_threshold": 0,
                "maximum_homogeneity_fraction_any_sa": (
                    context.maximum_homogeneity_fraction
                ),
                "maximum_zero_minimum_entropy_sensitive_attributes": (
                    context.maximum_zero_entropy_sensitive_attributes
                ),
                "maximum_information_loss": context.maximum_information_loss,
            },
            "patient_level_outputs": True,
            "patient_level_output_directory": str(self.policy.output_directory),
            "release_boundary": (
                "A passing held-out candidate remains blocked from unrestricted "
                "release pending direct-identifier/free-text, custodian, governance, "
                "and independent disclosure-control review."
            ),
        }
        self.policy.evidence_directory.mkdir(parents=True, exist_ok=True)
        path = self.policy.evidence_directory / self.PLAN_FILE
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if self._stable(existing) != self._stable(payload):
                raise ValueError(
                    "The existing held-out plan differs from the completed selection. "
                    "Use a new evidence and output directory; do not overwrite it."
                )
            return path
        atomic_write_text(path, json.dumps(payload, indent=2))
        return path

    def run(
        self,
        progress: Callable[[str, int, int, float, int, int], None] | None = None,
    ) -> dict[str, object]:
        """Run the recorded held-out configuration exactly once and assess it."""

        with exclusive_directory_lock(
            self.policy.output_directory,
            lock_name=".prti-heldout.lock",
        ):
            return self._run_locked(progress=progress)

    def _run_locked(
        self,
        progress: Callable[[str, int, int, float, int, int], None] | None = None,
    ) -> dict[str, object]:
        """Generate the held-out candidate while holding its output lock."""

        plan_path = self.predeclare()
        context = self._load_context()
        summary_path = self.policy.output_directory / "prti_run_summary.json"
        if summary_path.exists():
            raise FileExistsError(
                f"Candidate output already exists: {summary_path}. Use "
                "--evaluate-only or choose a new output and evidence directory."
            )
        summary = self._candidate_runner(context).run(progress=progress)
        assessment = self._evaluate(summary, context)
        assessment["selection_plan"] = str(plan_path)
        assessment["candidate_output_directory"] = str(self.policy.output_directory)
        return self._write_assessment(assessment)

    def evaluate_existing(self) -> dict[str, object]:
        """Reassess an existing matching held-out candidate without rerunning PSO."""

        with exclusive_directory_lock(
            self.policy.output_directory,
            lock_name=".prti-heldout.lock",
        ):
            return self._evaluate_existing_locked()

    def _evaluate_existing_locked(self) -> dict[str, object]:
        """Evaluate a saved candidate while holding its output lock."""

        plan_path = self.predeclare()
        context = self._load_context()
        summary_path = self.policy.output_directory / "prti_run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Candidate summary not found: {summary_path}")
        manifest_path = self.policy.output_directory / "prti_artifact_manifest.json"
        if not manifest_path.is_file():
            self._candidate_runner(context).refresh_saved_manifest()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assessment = self._evaluate(summary, context)
        assessment["selection_plan"] = str(plan_path)
        assessment["candidate_output_directory"] = str(self.policy.output_directory)
        return self._write_assessment(assessment)

    def _load_context(self) -> PRTIDenseSelectedContext:
        plan_path, selection_path = self._study_evidence_paths()
        if not plan_path.is_file():
            raise FileNotFoundError(f"Dense-study plan not found: {plan_path}")
        if not selection_path.is_file():
            raise FileNotFoundError(f"Dense-study selection not found: {selection_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            plan.get("paper_identifier") != PAPER_IDENTIFIER
            or plan.get("paper_revision") != PAPER_REVISION
        ):
            raise ValueError("Study plan uses a different paper revision.")
        if (
            selection.get("paper_identifier") != PAPER_IDENTIFIER
            or selection.get("paper_revision") != PAPER_REVISION
        ):
            raise ValueError("Study selection uses a different paper revision.")
        if selection.get("status") != "selection_complete":
            raise ValueError("Dense selection is incomplete; finish every planned run.")
        if selection.get("patient_level_outputs") is not False:
            raise ValueError("Dense selection evidence must be aggregate-only.")
        if selection.get("source_sha256") != EXPECTED_PRTI_SHA256:
            raise ValueError("Dense selection uses a different PRTI-GH source.")
        if plan.get("source_sha256") != EXPECTED_PRTI_SHA256:
            raise ValueError("Dense plan does not contain the reviewed source hash.")
        selected = selection.get("selected_configuration")
        if not isinstance(selected, dict):
            raise ValueError(
                "No configuration passed every predeclared rule; no final candidate "
                "may be generated."
            )
        if (
            selected.get("selected") is not True
            or selected.get("passes_all_predeclared_rules") is not True
            or int(selected.get("selection_rank", 0)) != 1
        ):
            raise ValueError("Recorded selected configuration is not eligible rank 1.")

        study = selection.get("study_configuration", {})
        comparison_seeds = tuple(int(seed) for seed in study["comparison_seeds"])
        held_out_seed = study.get("held_out_seed")
        if held_out_seed is None:
            raise ValueError("The dense study did not prospectively declare a held-out seed.")
        held_out_seed = int(held_out_seed)
        if held_out_seed in comparison_seeds:
            raise ValueError("Held-out seed was used in configuration comparison.")
        if int(selected["runs_completed"]) != len(comparison_seeds):
            raise ValueError("Selected configuration did not complete every comparison seed.")
        if int(selected.get("runs_planned", -1)) != len(comparison_seeds):
            raise ValueError("Selected configuration has a different planned seed count.")
        if plan.get("seed_protocol", {}).get("held_out_seed") != held_out_seed:
            raise ValueError("Plan and selection disagree on the held-out seed.")
        if plan.get("seed_protocol", {}).get("comparison_seeds") != list(
            comparison_seeds
        ):
            raise ValueError("Plan and selection disagree on comparison seeds.")
        plan_pso = plan.get("pso_configuration_except_seed", {})
        if (
            int(plan_pso.get("population_size", -1))
            != int(study["population_size"])
            or int(plan_pso.get("iterations", -1)) != int(study["iterations"])
            or float(plan_pso.get("random_coefficient_max", -1.0))
            != PSOConfig().random_coefficient_max
        ):
            raise ValueError("Plan and selection disagree on the PSO search budget.")
        if int(study["planned_runs"]) != int(plan["number_of_planned_runs"]):
            raise ValueError("Plan and selection disagree on the number of runs.")
        if int(plan["number_of_planned_runs"]) != int(
            plan["number_of_configurations"]
        ) * len(comparison_seeds):
            raise ValueError("Plan run count does not match configurations times seeds.")
        if selection.get("privacy_pass_rule") != plan.get("privacy_pass_rule"):
            raise ValueError("Plan and selection disagree on privacy thresholds.")

        study_type = study.get(
            "study_type",
            "fixed_k_lambda_cluster_range",
        )
        if study_type == "fixed_k_lambda_cluster_range":
            expected_fixed = {
                "k": int(study["k"]),
                "lambda_weight": float(study["lambda_weight"]),
            }
            if plan.get("fixed_model_configuration") != expected_fixed:
                raise ValueError("Plan and selection disagree on k or lambda.")
            if plan.get("cluster_values") != list(
                range(
                    int(study["minimum_clusters"]),
                    int(study["maximum_clusters"]) + 1,
                )
            ):
                raise ValueError("Plan and selection disagree on the cluster interval.")
            if int(plan["number_of_configurations"]) != len(
                plan["cluster_values"]
            ):
                raise ValueError("Plan cluster count does not match its configuration count.")
            if int(selected["configured_clusters"]) not in plan["cluster_values"]:
                raise ValueError("Selected cluster count was not declared in the plan.")
            model = MOOBAMConfig(
                k=int(study["k"]),
                n_clusters=int(selected["configured_clusters"]),
                lambda_weight=float(study["lambda_weight"]),
            )
        elif study_type == "joint_k_cluster_lambda_grid":
            if plan.get("k_values") != study.get("k_values"):
                raise ValueError("Plan and selection disagree on the k grid.")
            if plan.get("lambda_weights") != study.get("lambda_weights"):
                raise ValueError("Plan and selection disagree on the lambda grid.")
            requested_interval = plan.get("requested_cluster_interval")
            if requested_interval != [
                int(study["minimum_clusters"]),
                int(study["maximum_clusters"]),
            ]:
                raise ValueError("Plan and selection disagree on the cluster interval.")
            selected_model = {
                "k": int(selected["k"]),
                "n_clusters": int(selected["configured_clusters"]),
                "lambda_weight": float(selected["lambda"]),
            }
            declared_models = [
                item.get("model_configuration")
                for item in plan.get("configurations", [])
                if isinstance(item, dict)
            ]
            if selected_model not in declared_models:
                raise ValueError("Selected model was not declared in the parameter grid.")
            if len(declared_models) != int(plan["number_of_configurations"]):
                raise ValueError("Plan grid does not match its configuration count.")
            if int(study["planned_configurations"]) != int(
                plan["number_of_configurations"]
            ):
                raise ValueError("Plan and selection disagree on configuration count.")
            model = MOOBAMConfig(**selected_model)
        else:
            raise ValueError(f"Unsupported selection study type: {study_type!r}.")
        pso = PSOConfig(
            population_size=int(study["population_size"]),
            iterations=int(study["iterations"]),
            seed=held_out_seed,
        )
        thresholds = selection["privacy_pass_rule"]
        run_label = self.policy.run_label or (
            f"dense-selected-c{model.n_clusters:03d}-k{model.k}-heldout-seed"
            f"{held_out_seed}"
        )
        return PRTIDenseSelectedContext(
            model=model,
            pso=pso,
            comparison_seeds=comparison_seeds,
            run_label=run_label,
            maximum_information_loss=float(
                thresholds["maximum_information_loss_for_every_seed"]
            ),
            maximum_homogeneity_fraction=float(
                thresholds["maximum_homogeneity_fraction_any_sa"]
            ),
            maximum_zero_entropy_sensitive_attributes=int(
                thresholds["maximum_zero_minimum_entropy_sensitive_attributes"]
            ),
            plan=plan,
            selection=selection,
        )

    def _study_evidence_paths(self) -> tuple[Path, Path]:
        """Resolve either a fixed-range or joint-grid study pair."""

        dense = (
            self.policy.study_directory / self.STUDY_PLAN_FILE,
            self.policy.study_directory / self.STUDY_SELECTION_FILE,
        )
        grid = (
            self.policy.study_directory / self.GRID_PLAN_FILE,
            self.policy.study_directory / self.GRID_SELECTION_FILE,
        )
        if all(path.is_file() for path in dense):
            return dense
        if all(path.is_file() for path in grid):
            return grid
        if any(path.exists() for path in (*dense, *grid)):
            raise FileNotFoundError(
                "The study directory contains an incomplete or mixed plan/selection pair."
            )
        return dense

    def _evaluate(
        self,
        summary: dict[str, object],
        context: PRTIDenseSelectedContext,
    ) -> dict[str, object]:
        evaluator = PRTIFinalCandidateRunner(
            PRTIReleasePolicy(
                source_path=self.policy.source_path,
                evidence_directory=self.policy.evidence_directory,
                output_directory=self.policy.output_directory,
                model=context.model,
                pso=context.pso,
                run_label=context.run_label,
                comparison_seeds=context.comparison_seeds,
                maximum_information_loss=context.maximum_information_loss,
                maximum_homogeneity_fraction=(
                    context.maximum_homogeneity_fraction
                ),
                maximum_zero_minimum_entropy_sensitive_attributes=(
                    context.maximum_zero_entropy_sensitive_attributes
                ),
            )
        )
        return evaluator.evaluate_summary(summary)

    def _candidate_runner(
        self,
        context: PRTIDenseSelectedContext,
    ) -> PRTIAnonymizationRunner:
        """Build the exact held-out runner recovered from selection evidence."""

        return PRTIAnonymizationRunner(
            PRTIRunConfig(
                source_path=self.policy.source_path,
                output_directory=self.policy.output_directory,
                model=context.model,
                pso=context.pso,
                run_label=context.run_label,
            )
        )

    def _write_assessment(self, assessment: dict[str, object]) -> dict[str, object]:
        self.policy.evidence_directory.mkdir(parents=True, exist_ok=True)
        csv_path = self.policy.evidence_directory / self.ACCEPTANCE_CSV
        json_path = self.policy.evidence_directory / self.ACCEPTANCE_JSON
        atomic_write_text(
            csv_path,
            pd.DataFrame(assessment["checks"]).to_csv(index=False),
        )
        assessment["outputs"] = {
            "acceptance_csv": csv_path.name,
            "acceptance_json": json_path.name,
        }
        atomic_write_text(json_path, json.dumps(assessment, indent=2))
        return assessment

    @staticmethod
    def _stable(payload: dict[str, object]) -> dict[str, object]:
        return {
            key: value for key, value in payload.items() if key != "declared_at_utc"
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
