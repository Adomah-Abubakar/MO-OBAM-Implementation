from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mo_obam.prti_data import EXPECTED_PRTI_SHA256
from mo_obam.prti_dense_release import (
    PRTIDenseSelectedCandidateRunner,
    PRTIDenseSelectedPolicy,
)
from mo_obam.prti_runner import PAPER_IDENTIFIER, PAPER_REVISION


class PRTIDenseSelectedCandidateTests(unittest.TestCase):
    def _write_study(self, root: Path, selected: bool = True) -> Path:
        study = root / "study"
        study.mkdir()
        plan = {
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "source_sha256": EXPECTED_PRTI_SHA256,
            "cluster_values": [1, 2, 3, 4, 5],
            "number_of_configurations": 5,
            "number_of_planned_runs": 15,
            "seed_protocol": {
                "comparison_seeds": [0, 1, 2],
                "held_out_seed": 11,
            },
            "fixed_model_configuration": {
                "k": 10,
                "lambda_weight": 1.0,
            },
            "pso_configuration_except_seed": {
                "population_size": 100,
                "iterations": 100,
                "random_coefficient_max": 0.01,
            },
            "privacy_pass_rule": {
                "all_planned_seeds_must_complete": True,
                "minimum_cluster_size_at_least_k_for_every_seed": True,
                "maximum_linkage_records_at_each_tau": 0,
                "maximum_homogeneity_fraction_any_sa": 0.2,
                "maximum_zero_minimum_entropy_sensitive_attributes": 1,
                "maximum_information_loss_for_every_seed": 0.6,
            },
        }
        selected_configuration = (
            {
                "configuration_id": "clusters_05",
                "configured_clusters": 5,
                "runs_completed": 3,
                "runs_planned": 3,
                "passes_all_predeclared_rules": True,
                "selection_rank": 1,
                "selected": True,
            }
            if selected
            else None
        )
        selection = {
            "paper_identifier": PAPER_IDENTIFIER,
            "paper_revision": PAPER_REVISION,
            "status": "selection_complete",
            "source_sha256": EXPECTED_PRTI_SHA256,
            "patient_level_outputs": False,
            "selection_rule": "test rule",
            "study_configuration": {
                "minimum_clusters": 1,
                "maximum_clusters": 5,
                "k": 10,
                "lambda_weight": 1.0,
                "population_size": 100,
                "iterations": 100,
                "comparison_seeds": [0, 1, 2],
                "held_out_seed": 11,
                "planned_runs": 15,
            },
            "privacy_pass_rule": {
                "all_planned_seeds_must_complete": True,
                "minimum_cluster_size_at_least_k_for_every_seed": True,
                "maximum_linkage_records_at_each_tau": 0,
                "maximum_homogeneity_fraction_any_sa": 0.2,
                "maximum_zero_minimum_entropy_sensitive_attributes": 1,
                "maximum_information_loss_for_every_seed": 0.6,
            },
            "selected_configuration": selected_configuration,
        }
        (study / "prti_dense_cluster_plan.json").write_text(
            json.dumps(plan), encoding="utf-8"
        )
        (study / "prti_dense_cluster_selection.json").write_text(
            json.dumps(selection), encoding="utf-8"
        )
        return study

    def test_predeclare_uses_selected_configuration_and_unseen_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study = self._write_study(root)
            runner = PRTIDenseSelectedCandidateRunner(
                PRTIDenseSelectedPolicy(
                    study_directory=study,
                    output_directory=root / "candidate",
                    evidence_directory=root / "evidence",
                )
            )

            plan_path = runner.predeclare()
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

            self.assertEqual(plan["model_configuration"]["n_clusters"], 5)
            self.assertEqual(plan["model_configuration"]["k"], 10)
            self.assertEqual(plan["pso_configuration"]["seed"], 11)
            self.assertEqual(plan["comparison_seeds"], [0, 1, 2])
            self.assertTrue(plan["patient_level_outputs"])
            self.assertIn("must not be replaced", plan["seed_rule"])

    def test_no_candidate_is_generated_when_nothing_passed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study = self._write_study(root, selected=False)
            runner = PRTIDenseSelectedCandidateRunner(
                PRTIDenseSelectedPolicy(
                    study_directory=study,
                    output_directory=root / "candidate",
                    evidence_directory=root / "evidence",
                )
            )

            with self.assertRaisesRegex(ValueError, "No configuration passed"):
                runner.predeclare()

    def test_incomplete_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study = self._write_study(root)
            selection_path = study / "prti_dense_cluster_selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["status"] = "selection_incomplete"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            runner = PRTIDenseSelectedCandidateRunner(
                PRTIDenseSelectedPolicy(
                    study_directory=study,
                    output_directory=root / "candidate",
                    evidence_directory=root / "evidence",
                )
            )

            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.predeclare()

    def test_predeclare_accepts_a_completed_joint_parameter_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study = root / "grid"
            study.mkdir()
            privacy_rule = {
                "all_planned_seeds_must_complete": True,
                "minimum_cluster_size_at_least_k_for_every_seed": True,
                "maximum_linkage_records_at_each_tau": 0,
                "maximum_homogeneity_fraction_any_sa": 0.2,
                "maximum_zero_minimum_entropy_sensitive_attributes": 1,
                "maximum_information_loss_for_every_seed": 0.6,
            }
            model = {"k": 20, "n_clusters": 5, "lambda_weight": 0.5}
            plan = {
                "paper_identifier": PAPER_IDENTIFIER,
                "paper_revision": PAPER_REVISION,
                "source_sha256": EXPECTED_PRTI_SHA256,
                "k_values": [10, 20],
                "lambda_weights": [0.5, 1.0],
                "requested_cluster_interval": [1, 5],
                "number_of_configurations": 1,
                "number_of_planned_runs": 3,
                "seed_protocol": {
                    "comparison_seeds": [0, 1, 2],
                    "held_out_seed": 11,
                },
                "pso_configuration_except_seed": {
                    "population_size": 100,
                    "iterations": 100,
                    "random_coefficient_max": 0.01,
                },
                "configurations": [
                    {
                        "configuration_id": "k20_c005_lambda0p5",
                        "model_configuration": model,
                    }
                ],
                "privacy_pass_rule": privacy_rule,
            }
            selection = {
                "paper_identifier": PAPER_IDENTIFIER,
                "paper_revision": PAPER_REVISION,
                "status": "selection_complete",
                "source_sha256": EXPECTED_PRTI_SHA256,
                "patient_level_outputs": False,
                "selection_rule": "lowest mean information loss among eligible points",
                "study_configuration": {
                    "study_type": "joint_k_cluster_lambda_grid",
                    "minimum_clusters": 1,
                    "maximum_clusters": 5,
                    "k_values": [10, 20],
                    "lambda_weights": [0.5, 1.0],
                    "population_size": 100,
                    "iterations": 100,
                    "comparison_seeds": [0, 1, 2],
                    "held_out_seed": 11,
                    "planned_configurations": 1,
                    "planned_runs": 3,
                },
                "privacy_pass_rule": privacy_rule,
                "selected_configuration": {
                    "configuration_id": "k20_c005_lambda0p5",
                    "k": 20,
                    "configured_clusters": 5,
                    "lambda": 0.5,
                    "runs_completed": 3,
                    "runs_planned": 3,
                    "passes_all_predeclared_rules": True,
                    "selection_rank": 1,
                    "selected": True,
                },
            }
            (study / "prti_parameter_grid_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            (study / "prti_parameter_grid_selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            runner = PRTIDenseSelectedCandidateRunner(
                PRTIDenseSelectedPolicy(
                    study_directory=study,
                    output_directory=root / "candidate",
                    evidence_directory=root / "evidence",
                )
            )

            final_plan = json.loads(
                runner.predeclare().read_text(encoding="utf-8")
            )

            self.assertEqual(final_plan["model_configuration"], model)
            self.assertEqual(final_plan["pso_configuration"]["seed"], 11)


if __name__ == "__main__":
    unittest.main()
