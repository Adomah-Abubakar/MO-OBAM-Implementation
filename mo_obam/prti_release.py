"""Predeclare, run, and assess one held-out PRTI-GH release candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .prti_data import DEFAULT_PRTI_SOURCE, EXPECTED_PRTI_SHA256
from .prti_runner import (
    PAPER_IDENTIFIER,
    PAPER_REVISION,
    PRTIAnonymizationRunner,
    PRTIRunConfig,
)
from .pso_solver import PSOConfig
from .schema import MOOBAMConfig


@dataclass(frozen=True)
class PRTIReleasePolicy:
    """Store the prospectively declared candidate and screening criteria."""

    source_path: Path = DEFAULT_PRTI_SOURCE
    robustness_summary_path: Path = Path(
        "evidence/prti-robustness-v1/prti_robustness_summary.json"
    )
    evidence_directory: Path = Path("evidence/prti-final-selection-v1")
    output_directory: Path = Path(
        "outputs/prti/privacy-candidate-heldout-seed10"
    )
    model: MOOBAMConfig = MOOBAMConfig(
        k=10, n_clusters=4, lambda_weight=1.0
    )
    pso: PSOConfig = PSOConfig(
        population_size=100, iterations=100, seed=10
    )
    run_label: str = "privacy-candidate-heldout-seed10"
    comparison_seeds: tuple[int, ...] = tuple(range(10))
    maximum_information_loss: float = 0.60
    maximum_homogeneity_fraction: float = 0.20
    maximum_zero_minimum_entropy_sensitive_attributes: int = 1

    def __post_init__(self) -> None:
        if self.pso.seed in self.comparison_seeds:
            raise ValueError(
                "The final-candidate seed must not have been used in the "
                "configuration-comparison seed set."
            )
        if not 0.0 <= self.maximum_information_loss <= 1.0:
            raise ValueError("maximum_information_loss must be in [0, 1].")
        if not 0.0 <= self.maximum_homogeneity_fraction <= 1.0:
            raise ValueError("maximum_homogeneity_fraction must be in [0, 1].")
        if self.maximum_zero_minimum_entropy_sensitive_attributes < 0:
            raise ValueError(
                "maximum_zero_minimum_entropy_sensitive_attributes cannot be negative."
            )


class PRTIFinalCandidateRunner:
    """Create a prospective plan, run it once, and assess saved aggregates."""

    PLAN_FILE = "prti_final_selection_plan.json"
    ACCEPTANCE_CSV = "prti_final_candidate_acceptance.csv"
    ACCEPTANCE_JSON = "prti_final_candidate_acceptance.json"
    PRIVACY_CONFIGURATION_ID = "privacy_candidate"

    def __init__(self, policy: PRTIReleasePolicy) -> None:
        self.policy = policy

    def predeclare(self) -> Path:
        """Write or verify the plan before any patient-level candidate is run."""

        robustness = self._load_robustness_summary()
        selected = self._selected_robustness_configuration(robustness)
        self._validate_robustness_evidence(robustness, selected)
        payload = self._plan_payload(robustness, selected)

        self.policy.evidence_directory.mkdir(parents=True, exist_ok=True)
        path = self.policy.evidence_directory / self.PLAN_FILE
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if self._stable_plan(existing) != self._stable_plan(payload):
                raise ValueError(
                    "The existing final-selection plan differs from the requested "
                    "policy. Use a new evidence directory instead of overwriting it."
                )
            return path
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def run(
        self,
        progress: Callable[[str, int, int, float, int, int], None] | None = None,
    ) -> dict[str, object]:
        """Predeclare, execute the held-out run, and save its screening result."""

        plan_path = self.predeclare()
        summary_path = self.policy.output_directory / "prti_run_summary.json"
        if summary_path.exists():
            raise FileExistsError(
                f"Candidate output already exists: {summary_path}. Use "
                "evaluate_existing() or choose a new output directory."
            )
        config = PRTIRunConfig(
            source_path=self.policy.source_path,
            output_directory=self.policy.output_directory,
            model=self.policy.model,
            pso=self.policy.pso,
            run_label=self.policy.run_label,
        )
        summary = PRTIAnonymizationRunner(config).run(progress=progress)
        assessment = self.evaluate_summary(summary)
        assessment["selection_plan"] = str(plan_path)
        assessment["candidate_output_directory"] = str(
            self.policy.output_directory
        )
        return self._write_assessment(assessment)

    def evaluate_existing(self) -> dict[str, object]:
        """Reassess an already saved matching candidate without rerunning PSO."""

        plan_path = self.predeclare()
        summary_path = self.policy.output_directory / "prti_run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Candidate summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assessment = self.evaluate_summary(summary)
        assessment["selection_plan"] = str(plan_path)
        assessment["candidate_output_directory"] = str(
            self.policy.output_directory
        )
        return self._write_assessment(assessment)

    def evaluate_summary(self, summary: dict[str, object]) -> dict[str, object]:
        """Apply only the prospectively declared checks to one run summary."""

        self._validate_candidate_identity(summary)
        metrics = summary["final_metrics"]
        privacy = summary["privacy_evaluation"]["anonymized"]
        validation = summary["validation"]
        saved = validation["saved_csvs"]
        entropy = metrics["entropy_by_sensitive_attribute"]
        zero_entropy_count = sum(
            bool(np.isclose(float(value), 0.0)) for value in entropy.values()
        )
        rows = int(summary["preparation"]["rows"])
        homogeneity_records = int(privacy["homogeneity_attack_records"])
        homogeneity_fraction = homogeneity_records / rows

        checks = [
            self._check(
                "technical",
                "source workbook unchanged",
                "=",
                True,
                validation["source_unchanged"],
                bool(validation["source_unchanged"]),
                "project invariant",
            ),
            self._check(
                "technical",
                "all sensitive attributes unchanged",
                "=",
                True,
                validation["all_sensitive_attributes_unchanged"],
                bool(validation["all_sensitive_attributes_unchanged"]),
                "project invariant",
            ),
            self._check(
                "technical",
                "all clusters satisfy hard k",
                "=",
                True,
                validation["all_cluster_sizes_at_least_k"],
                bool(validation["all_cluster_sizes_at_least_k"]),
                "Wei Equation (20)",
            ),
            self._check(
                "technical",
                "final release shape",
                "=",
                "1485 x 106",
                " x ".join(str(value) for value in saved["final_release_shape"]),
                saved["final_release_shape"] == [1485, 106],
                "project invariant",
            ),
        ]
        for threshold in ("0.05", "0.075", "0.1"):
            observed = int(privacy["linkage_attack_records"][threshold])
            checks.append(
                self._check(
                    "privacy screening",
                    f"linkage-risk records above tau={threshold}",
                    "=",
                    0,
                    observed,
                    observed == 0,
                    "project threshold using Wei Section V-C metric",
                )
            )
        checks.extend(
            [
                self._check(
                    "privacy screening",
                    "homogeneity-risk record fraction for any SA",
                    "<=",
                    self.policy.maximum_homogeneity_fraction,
                    homogeneity_fraction,
                    homogeneity_fraction
                    <= self.policy.maximum_homogeneity_fraction,
                    "project threshold using Wei Section V-C metric",
                ),
                self._check(
                    "privacy screening",
                    "SAs with zero minimum cluster entropy",
                    "<=",
                    self.policy.maximum_zero_minimum_entropy_sensitive_attributes,
                    zero_entropy_count,
                    zero_entropy_count
                    <= self.policy.maximum_zero_minimum_entropy_sensitive_attributes,
                    "project threshold derived from Equations (15)-(19)",
                ),
                self._check(
                    "utility screening",
                    "information loss",
                    "<=",
                    self.policy.maximum_information_loss,
                    float(metrics["information_loss"]),
                    float(metrics["information_loss"])
                    <= self.policy.maximum_information_loss,
                    "project threshold using Wei Equation (14)",
                ),
            ]
        )
        passed = all(bool(check["passed"]) for check in checks)
        return {
            "assessed_at_utc": self._utc_now(),
            "status": (
                "passes_predeclared_project_screening_pending_independent_review"
                if passed
                else "does_not_pass_predeclared_project_screening"
            ),
            "all_checks_passed": passed,
            "release_boundary": (
                "Passing these project screening checks does not approve public or "
                "unrestricted release. Formal role review and independent disclosure-"
                "control review remain required."
            ),
            "run_id": summary["run_id"],
            "run_summary_sha256": self._file_sha256(
                self.policy.output_directory / "prti_run_summary.json"
            ),
            "artifact_manifest_sha256": self._file_sha256(
                self.policy.output_directory / "prti_artifact_manifest.json"
            ),
            "observed": {
                "information_loss": float(metrics["information_loss"]),
                "entropy_sum": float(metrics["entropy_sum"]),
                "objective": float(metrics["objective"]),
                "equivalence_classes": int(privacy["equivalence_classes"]),
                "minimum_equivalence_class_size": int(
                    privacy["minimum_equivalence_class_size"]
                ),
                "maximum_equivalence_class_size": int(
                    privacy["maximum_equivalence_class_size"]
                ),
                "linkage_attack_records": privacy["linkage_attack_records"],
                "homogeneity_attack_records_any_sa": homogeneity_records,
                "homogeneity_attack_fraction_any_sa": homogeneity_fraction,
                "zero_minimum_entropy_sensitive_attributes": zero_entropy_count,
            },
            "checks": checks,
        }

    def _plan_payload(
        self,
        robustness: dict[str, object],
        selected: dict[str, object],
    ) -> dict[str, object]:
        return {
            "declared_at_utc": self._utc_now(),
            "status": "prospective held-out candidate plan",
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "source_sha256": EXPECTED_PRTI_SHA256,
            "selection_basis": (
                "The project goal is anonymization. The four-cluster configuration "
                "had the strongest measured privacy results among the four robustness "
                "candidates and completed all ten comparison seeds."
            ),
            "selected_robustness_configuration": selected,
            "robustness_summary_sha256": self._file_sha256(
                self.policy.robustness_summary_path
            ),
            "model_configuration": asdict(self.policy.model),
            "pso_configuration": asdict(self.policy.pso),
            "run_label": self.policy.run_label,
            "seed_rule": {
                "comparison_seeds": list(self.policy.comparison_seeds),
                "held_out_seed": self.policy.pso.seed,
                "rule": (
                    "Use the smallest non-negative integer seed not present in the "
                    "configuration-comparison seed set."
                ),
            },
            "project_screening_thresholds": {
                "linkage_attack_records_at_each_reported_threshold": 0,
                "maximum_homogeneity_fraction_any_sa": (
                    self.policy.maximum_homogeneity_fraction
                ),
                "maximum_zero_minimum_entropy_sensitive_attributes": (
                    self.policy.maximum_zero_minimum_entropy_sensitive_attributes
                ),
                "maximum_information_loss": self.policy.maximum_information_loss,
            },
            "threshold_boundary": (
                "These are project screening criteria declared before seed 10 was "
                "run. Wei et al. define the metrics but do not prescribe these PRTI-"
                "GH release thresholds. The homogeneity and IL limits round the "
                "observed ten-seed privacy-candidate stability envelope; they are not "
                "independent regulatory standards."
            ),
            "patient_level_outputs": True,
            "patient_level_output_directory": str(self.policy.output_directory),
            "release_boundary": (
                "The result remains a candidate even if every check passes. Formal "
                "role review and independent disclosure-control review are required."
            ),
            "robustness_evidence": {
                "planned_runs": robustness["planned_runs"],
                "recorded_runs": robustness["recorded_runs"],
                "completed_runs": robustness["completed_runs"],
            },
        }

    def _load_robustness_summary(self) -> dict[str, object]:
        path = self.policy.robustness_summary_path
        if not path.is_file():
            raise FileNotFoundError(f"Robustness summary not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _selected_robustness_configuration(
        self, robustness: dict[str, object]
    ) -> dict[str, object]:
        matches = [
            row
            for row in robustness["configurations"]
            if row["configuration_id"] == self.PRIVACY_CONFIGURATION_ID
        ]
        if len(matches) != 1:
            raise ValueError("Robustness evidence must contain one privacy candidate.")
        return matches[0]

    def _validate_robustness_evidence(
        self,
        robustness: dict[str, object],
        selected: dict[str, object],
    ) -> None:
        if robustness["source_sha256"] != EXPECTED_PRTI_SHA256:
            raise ValueError("Robustness evidence uses a different PRTI-GH source.")
        expected = (
            self.policy.model.k,
            self.policy.model.n_clusters,
            self.policy.model.lambda_weight,
        )
        recorded = (
            int(selected["k"]),
            int(selected["configured_clusters"]),
            float(selected["lambda"]),
        )
        if recorded != expected:
            raise ValueError(
                f"Selected robustness configuration {recorded} does not match {expected}."
            )
        planned = len(self.policy.comparison_seeds)
        if int(selected["seeds_planned"]) != planned:
            raise ValueError("Robustness evidence uses a different comparison seed set.")
        if int(selected["runs_completed"]) != planned:
            raise ValueError("The selected robustness candidate is not complete.")
        if int(selected["runs_no_feasible_particle"]) != 0:
            raise ValueError("The selected robustness candidate contains failed seeds.")

    def _validate_candidate_identity(self, summary: dict[str, object]) -> None:
        if summary["model_configuration"] != asdict(self.policy.model):
            raise ValueError("Saved candidate model configuration differs from the plan.")
        if summary["pso_configuration"] != asdict(self.policy.pso):
            raise ValueError("Saved candidate PSO configuration differs from the plan.")
        if summary["run_label"] != self.policy.run_label:
            raise ValueError("Saved candidate run label differs from the plan.")
        if summary["preparation"]["source_sha256_before"] != EXPECTED_PRTI_SHA256:
            raise ValueError("Saved candidate source fingerprint differs from the plan.")

    def _write_assessment(
        self, assessment: dict[str, object]
    ) -> dict[str, object]:
        self.policy.evidence_directory.mkdir(parents=True, exist_ok=True)
        csv_path = self.policy.evidence_directory / self.ACCEPTANCE_CSV
        json_path = self.policy.evidence_directory / self.ACCEPTANCE_JSON
        pd.DataFrame(assessment["checks"]).to_csv(csv_path, index=False)
        assessment["outputs"] = {
            "acceptance_csv": csv_path.name,
            "acceptance_json": json_path.name,
        }
        json_path.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
        return assessment

    @staticmethod
    def _check(
        category: str,
        criterion: str,
        operator: str,
        threshold: object,
        observed: object,
        passed: bool,
        authority: str,
    ) -> dict[str, object]:
        return {
            "category": category,
            "criterion": criterion,
            "operator": operator,
            "threshold": threshold,
            "observed": observed,
            "passed": passed,
            "authority": authority,
        }

    @staticmethod
    def _stable_plan(plan: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in plan.items() if key != "declared_at_utc"}

    @staticmethod
    def _file_sha256(path: str | Path) -> str:
        digest = sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
