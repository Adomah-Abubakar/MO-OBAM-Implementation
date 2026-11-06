"""Generate the held-out patient-level release from a completed Pareto study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd

from .formulation import MOOBAMFormulation
from .privacy_metrics import PrivacyRiskEvaluator, PrivacyRiskMetrics
from .provenance import atomic_write_text, exclusive_directory_lock
from .prti_data import DEFAULT_PRTI_SOURCE, EXPECTED_PRTI_SHAPE, PRTIDataAdapter
from .prti_multiobjective import PRTIMultiObjectiveStudy
from .prti_multiobjective_solver import (
    MultiObjectiveProgress,
    PRTIMultiObjectiveParticleSwarm,
    PRTIParetoCandidate,
)
from .prti_runner import PAPER_IDENTIFIER, PAPER_REVISION
from .pso_solver import PSOConfig
from .schema import MOOBAMConfig


FINAL_PLAN_FILE = "prti_multiobjective_final_selection_plan.json"
ACCEPTANCE_FILE = "prti_multiobjective_final_candidate_acceptance.json"
FINAL_CSV_FILE = "prti_anonymized_final.csv"
SUMMARY_FILE = "prti_run_summary.json"


@dataclass(frozen=True)
class PRTIMultiObjectiveSelectedPolicy:
    """Locate the completed comparison evidence and the new release outputs."""

    study_directory: Path
    output_directory: Path
    evidence_directory: Path
    source_path: Path = DEFAULT_PRTI_SOURCE
    run_label: str = "multiobjective-selected-heldout"


@dataclass(frozen=True)
class _SelectedContext:
    k: int
    clusters: int
    held_out_seed: int
    comparison_seeds: tuple[int, ...]
    population_size: int
    iterations: int
    archive_size: int
    initialization_attempts: int
    random_coefficient_max: float
    maximum_information_loss: float
    maximum_homogeneity_fraction: float
    maximum_zero_entropy_sensitive_attributes: int
    maximum_linkage_records_at_each_tau: int
    source_sha256: str


class PRTIMultiObjectiveSelectedCandidateRunner:
    """Run only the prospectively reserved seed for the selected k/C pair."""

    def __init__(self, policy: PRTIMultiObjectiveSelectedPolicy) -> None:
        self.policy = policy
        self.adapter = PRTIDataAdapter(policy.source_path)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load_context(self) -> tuple[_SelectedContext, Path, Path]:
        plan_path = self.policy.study_directory / PRTIMultiObjectiveStudy.PLAN_FILE
        selection_path = (
            self.policy.study_directory / PRTIMultiObjectiveStudy.SELECTION_FILE
        )
        if not plan_path.is_file() or not selection_path.is_file():
            raise FileNotFoundError(
                "The study directory must contain the multi-objective plan and "
                "selection JSON files."
            )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if plan.get("paper_identifier") != PAPER_IDENTIFIER:
            raise ValueError("The study plan uses an unexpected paper identifier.")
        if plan.get("paper_revision") != PAPER_REVISION:
            raise ValueError("The study plan uses an unexpected paper revision.")
        if selection.get("status") != "selection_complete":
            raise ValueError("The multi-objective comparison study is not complete.")
        if selection.get("patient_level_outputs") is not False:
            raise ValueError("The selection is not marked aggregate-only as expected.")
        planned = int(selection.get("planned_runs", -1))
        if planned < 1 or int(selection.get("recorded_runs", -2)) != planned:
            raise ValueError("Not every planned comparison run was recorded.")
        if int(selection.get("completed_runs", -2)) != planned:
            raise ValueError("Not every planned comparison run completed.")

        chosen = selection.get("selected_configuration")
        if not isinstance(chosen, dict) or not bool(chosen.get("passes_all_seeds")):
            raise ValueError("No all-seed-passing configuration was selected.")
        comparison_seeds = tuple(int(value) for value in plan["comparison_seeds"])
        if int(chosen.get("privacy_eligible_seeds", -1)) != len(comparison_seeds):
            raise ValueError("The selected configuration did not pass every seed.")
        held_out_seed = int(plan["held_out_seed"])
        if held_out_seed in comparison_seeds:
            raise ValueError("The held-out seed appears among the comparison seeds.")

        k = int(chosen["k"])
        clusters = int(chosen["clusters"])
        if "configuration_pairs" in plan:
            pairs = {tuple(map(int, pair)) for pair in plan["configuration_pairs"]}
            if (k, clusters) not in pairs:
                raise ValueError("The selected k/C pair was not declared in the plan.")

        thresholds = selection.get("privacy_thresholds")
        if not isinstance(thresholds, dict):
            raise ValueError("The selection has no privacy thresholds.")
        source_hash = str(selection.get("source_sha256"))
        if source_hash != str(plan.get("source_sha256")):
            raise ValueError("The plan and selection source hashes differ.")
        observed_hash = self.adapter.file_sha256(self.policy.source_path)
        if observed_hash != source_hash:
            raise ValueError("The current PRTI source differs from the completed study.")

        context = _SelectedContext(
            k=k,
            clusters=clusters,
            held_out_seed=held_out_seed,
            comparison_seeds=comparison_seeds,
            population_size=int(plan["population_size"]),
            iterations=int(plan["iterations"]),
            archive_size=int(plan["archive_size"]),
            initialization_attempts=int(plan["initialization_attempts"]),
            random_coefficient_max=float(plan["random_coefficient_max"]),
            maximum_information_loss=float(thresholds["maximum_information_loss"]),
            maximum_homogeneity_fraction=float(
                thresholds["maximum_homogeneity_fraction_any_sa"]
            ),
            maximum_zero_entropy_sensitive_attributes=int(
                thresholds["maximum_zero_minimum_entropy_sensitive_attributes"]
            ),
            maximum_linkage_records_at_each_tau=int(
                thresholds["maximum_linkage_records_at_each_tau"]
            ),
            source_sha256=source_hash,
        )
        return context, plan_path, selection_path

    def predeclare(self) -> Path:
        """Write the immutable final plan without reading patient rows."""

        context, study_plan, study_selection = self._load_context()
        path = self.policy.evidence_directory / FINAL_PLAN_FILE
        payload = {
            "declared_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "source_sha256": context.source_sha256,
            "study_plan_sha256": self._file_sha256(study_plan),
            "study_selection_sha256": self._file_sha256(study_selection),
            "selected_configuration": {"k": context.k, "clusters": context.clusters},
            "held_out_seed": context.held_out_seed,
            "comparison_seeds": list(context.comparison_seeds),
            "optimizer": {
                "population_size": context.population_size,
                "iterations": context.iterations,
                "archive_size": context.archive_size,
                "initialization_attempts": context.initialization_attempts,
                "random_coefficient_max": context.random_coefficient_max,
            },
            "privacy_thresholds": {
                "maximum_information_loss": context.maximum_information_loss,
                "maximum_homogeneity_fraction_any_sa": context.maximum_homogeneity_fraction,
                "maximum_zero_minimum_entropy_sensitive_attributes": (
                    context.maximum_zero_entropy_sensitive_attributes
                ),
                "maximum_linkage_records_at_each_tau": (
                    context.maximum_linkage_records_at_each_tau
                ),
            },
            "candidate_rule": (
                "From the held-out Pareto archive, retain candidates passing every "
                "declared privacy threshold; select lowest information loss, then "
                "highest entropy sum, then lowest archive index."
            ),
            "patient_level_outputs": True,
            "output_directory": str(self.policy.output_directory),
            "release_boundary": (
                "Internal research candidate only; external governance approval is "
                "still required before unrestricted release."
            ),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            ignored = {"declared_at_utc"}
            if ({k: v for k, v in existing.items() if k not in ignored} !=
                    {k: v for k, v in payload.items() if k not in ignored}):
                raise ValueError("The existing final plan differs; use new directories.")
            return path
        atomic_write_text(path, json.dumps(payload, indent=2))
        return path

    @staticmethod
    def _candidate_assessment(
        candidate: PRTIParetoCandidate,
        risk: PrivacyRiskMetrics,
        context: _SelectedContext,
    ) -> dict[str, object]:
        zero_entropy = int(
            sum(
                np.isclose(value, 0.0)
                for value in candidate.entropy_by_sensitive_attribute.values()
            )
        )
        homogeneity_fraction = risk.homogeneity_attack_records / EXPECTED_PRTI_SHAPE[0]
        checks = {
            "minimum_cluster_size_at_least_k": min(candidate.cluster_sizes) >= context.k,
            "information_loss_within_limit": (
                candidate.information_loss <= context.maximum_information_loss
            ),
            "homogeneity_fraction_within_limit": (
                homogeneity_fraction <= context.maximum_homogeneity_fraction
            ),
            "zero_entropy_sensitive_attributes_within_limit": (
                zero_entropy <= context.maximum_zero_entropy_sensitive_attributes
            ),
            "linkage_records_within_limit": all(
                value <= context.maximum_linkage_records_at_each_tau
                for value in risk.linkage_attack_records.values()
            ),
        }
        return {
            "candidate_index": candidate.candidate_index,
            "information_loss": candidate.information_loss,
            "entropy_sum": candidate.entropy_sum,
            "minimum_cluster_size": min(candidate.cluster_sizes),
            "maximum_cluster_size": max(candidate.cluster_sizes),
            "homogeneity_records_any_sa": risk.homogeneity_attack_records,
            "homogeneity_fraction_any_sa": homogeneity_fraction,
            "zero_minimum_entropy_sensitive_attributes": zero_entropy,
            "linkage_attack_records": risk.linkage_attack_records,
            "checks": checks,
            "privacy_eligible": all(checks.values()),
        }

    @staticmethod
    def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.to_csv(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def run(
        self,
        progress: MultiObjectiveProgress | None = None,
    ) -> dict[str, object]:
        """Execute held-out seed once and save the selected anonymized dataset."""

        with exclusive_directory_lock(self.policy.evidence_directory):
            plan_path = self.predeclare()
            final_path = self.policy.output_directory / FINAL_CSV_FILE
            summary_path = self.policy.output_directory / SUMMARY_FILE
            acceptance_path = self.policy.evidence_directory / ACCEPTANCE_FILE
            if final_path.exists() or summary_path.exists() or acceptance_path.exists():
                raise FileExistsError(
                    "Held-out output already exists. Refusing to overwrite or rerun it."
                )
            context, study_plan, study_selection = self._load_context()
            started = time.monotonic()
            started_at = datetime.now(timezone.utc)
            prepared = self.adapter.prepare()
            formulation = MOOBAMFormulation(
                prepared.model_input,
                prepared.schema,
                MOOBAMConfig(k=context.k, n_clusters=context.clusters, lambda_weight=0.0),
            )
            optimizer = PRTIMultiObjectiveParticleSwarm(
                formulation,
                PSOConfig(
                    population_size=context.population_size,
                    iterations=context.iterations,
                    seed=context.held_out_seed,
                    random_coefficient_max=context.random_coefficient_max,
                ),
                archive_size=context.archive_size,
                initialization_attempts=context.initialization_attempts,
            )
            result = optimizer.solve_multiobjective(progress=progress)

            assessed: list[tuple[PRTIParetoCandidate, pd.DataFrame, dict[str, object]]] = []
            for candidate in result.candidates:
                model_release = formulation.anonymize(
                    candidate.membership, candidate.centroids
                )
                risk = PrivacyRiskEvaluator(
                    model_release.drop(columns="cluster"), prepared.schema
                ).evaluate()
                assessment = self._candidate_assessment(candidate, risk, context)
                assessed.append((candidate, model_release, assessment))
            eligible = [item for item in assessed if bool(item[2]["privacy_eligible"])]
            if not eligible:
                failure = {
                    "status": "held_out_failed_privacy_rules",
                    "all_checks_passed": False,
                    "selected_configuration": {"k": context.k, "clusters": context.clusters},
                    "held_out_seed": context.held_out_seed,
                    "archive_candidates": [item[2] for item in assessed],
                    "release_boundary": "Do not substitute another seed or candidate.",
                }
                atomic_write_text(acceptance_path, json.dumps(failure, indent=2))
                raise RuntimeError(
                    "No held-out Pareto candidate passed every privacy rule. "
                    "Do not substitute another seed."
                )
            candidate, model_release, observed = min(
                eligible,
                key=lambda item: (
                    float(item[2]["information_loss"]),
                    -float(item[2]["entropy_sum"]),
                    int(item[2]["candidate_index"]),
                ),
            )

            decoded = self.adapter.decode_model_data(
                model_release.loc[:, prepared.original.columns],
                prepared.original,
                prepared.decoding_maps,
            )
            changed_sensitive: list[str] = []
            for column in prepared.schema.sensitive_attributes:
                before = prepared.original[column]
                after = decoded[column]
                if not bool((before.eq(after) | (before.isna() & after.isna())).all()):
                    changed_sensitive.append(column)
            if changed_sensitive:
                raise AssertionError(f"Sensitive attributes changed: {changed_sensitive}")
            if decoded.shape != prepared.original.shape:
                raise AssertionError("The anonymized data changed the source shape.")
            if list(decoded.columns) != list(prepared.original.columns):
                raise AssertionError("The anonymized data changed the source column order.")
            if self.adapter.file_sha256(self.policy.source_path) != context.source_sha256:
                raise AssertionError("The source workbook changed during the held-out run.")

            self._atomic_write_csv(decoded, final_path)
            saved = pd.read_csv(final_path)
            if saved.shape != EXPECTED_PRTI_SHAPE or list(saved.columns) != list(decoded.columns):
                raise AssertionError("The saved anonymized CSV failed shape/column validation.")
            final_hash = self._file_sha256(final_path)
            run_id = (
                f"prti-{started_at.strftime('%Y%m%dT%H%M%SZ')}-"
                f"k{context.k}-c{context.clusters}-seed{context.held_out_seed}"
            )
            summary = {
                "status": "held_out_candidate_generated",
                "all_checks_passed": True,
                "run_id": run_id,
                "run_label": self.policy.run_label,
                "started_at_utc": started_at.isoformat(timespec="seconds"),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "runtime_seconds": time.monotonic() - started,
                "source_sha256": context.source_sha256,
                "study_plan_sha256": self._file_sha256(study_plan),
                "study_selection_sha256": self._file_sha256(study_selection),
                "final_plan_sha256": self._file_sha256(plan_path),
                "selected_configuration": {"k": context.k, "clusters": context.clusters},
                "held_out_seed": context.held_out_seed,
                "optimizer": {
                    "population_size": context.population_size,
                    "iterations": context.iterations,
                    "archive_size": context.archive_size,
                    "initialization_attempts": context.initialization_attempts,
                    "random_coefficient_max": context.random_coefficient_max,
                    "feasible_evaluations": result.feasible_evaluations,
                    "infeasible_evaluations": result.infeasible_evaluations,
                    "returned_archive_candidates": len(result.candidates),
                },
                "observed": observed,
                "cluster_sizes": list(candidate.cluster_sizes),
                "entropy_by_sensitive_attribute": candidate.entropy_by_sensitive_attribute,
                "validation": {
                    "source_shape": list(prepared.original.shape),
                    "final_shape": list(decoded.shape),
                    "source_unchanged": True,
                    "all_sensitive_attributes_unchanged": True,
                    "column_order_preserved": True,
                    "saved_csv_reloaded": True,
                },
                "outputs": {
                    "anonymized_final": str(final_path),
                    "anonymized_final_sha256": final_hash,
                    "acceptance": str(acceptance_path),
                },
                "release_boundary": (
                    "Internal research candidate only; external governance approval is "
                    "still required before unrestricted release."
                ),
            }
            atomic_write_text(summary_path, json.dumps(summary, indent=2))
            acceptance = {
                "status": summary["status"],
                "all_checks_passed": True,
                "run_id": run_id,
                "selected_configuration": summary["selected_configuration"],
                "held_out_seed": context.held_out_seed,
                "observed": observed,
                "validation": summary["validation"],
                "outputs": summary["outputs"],
                "release_boundary": summary["release_boundary"],
            }
            atomic_write_text(acceptance_path, json.dumps(acceptance, indent=2))
            return acceptance
