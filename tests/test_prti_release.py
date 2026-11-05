from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mo_obam.prti_data import PRTI_QIDS, PRTIDataAdapter
from mo_obam.prti_release import PRTIFinalCandidateRunner, PRTIReleasePolicy
from mo_obam.provenance import ArtifactManifestWriter
from mo_obam.pso_solver import PSOConfig


class PRTIFinalCandidateTests(unittest.TestCase):
    project_root = Path(__file__).resolve().parents[1]
    saved_candidate = (
        project_root / "outputs/prti/privacy-candidate-heldout-seed10"
    )
    saved_evidence = project_root / "evidence/prti-final-selection-v1"

    def test_policy_requires_a_seed_held_out_from_comparison(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not have been used"):
            PRTIReleasePolicy(pso=PSOConfig(seed=9))

    def test_plan_is_prospective_and_uses_complete_privacy_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            robustness_path = root / "robustness.json"
            robustness = {
                "source_sha256": (
                    "0deb1fc54a3dccba3a4c0339696ed1fb4f124a0349afde9f06e4b79dbf89a37e"
                ),
                "planned_runs": 40,
                "recorded_runs": 40,
                "completed_runs": 34,
                "configurations": [
                    {
                        "configuration_id": "privacy_candidate",
                        "k": 10,
                        "configured_clusters": 4,
                        "lambda": 1.0,
                        "seeds_planned": 10,
                        "runs_completed": 10,
                        "runs_no_feasible_particle": 0,
                    }
                ],
            }
            robustness_path.write_text(json.dumps(robustness), encoding="utf-8")
            policy = PRTIReleasePolicy(
                robustness_summary_path=robustness_path,
                evidence_directory=root / "evidence",
                output_directory=root / "patient-level-output",
            )

            plan_path = PRTIFinalCandidateRunner(policy).predeclare()
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

            self.assertEqual(plan["seed_rule"]["held_out_seed"], 10)
            self.assertEqual(plan["model_configuration"]["n_clusters"], 4)
            self.assertTrue(plan["patient_level_outputs"])
            self.assertIn("not prescribe", plan["threshold_boundary"])

    def test_existing_plan_cannot_be_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / PRTIFinalCandidateRunner.PLAN_FILE).write_text(
                json.dumps({"status": "different policy"}), encoding="utf-8"
            )
            robustness_path = root / "robustness.json"
            robustness_path.write_text(
                json.dumps(
                    {
                        "source_sha256": (
                            "0deb1fc54a3dccba3a4c0339696ed1fb4f124a0349afde9f06e4b79dbf89a37e"
                        ),
                        "planned_runs": 40,
                        "recorded_runs": 40,
                        "completed_runs": 34,
                        "configurations": [
                            {
                                "configuration_id": "privacy_candidate",
                                "k": 10,
                                "configured_clusters": 4,
                                "lambda": 1.0,
                                "seeds_planned": 10,
                                "runs_completed": 10,
                                "runs_no_feasible_particle": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runner = PRTIFinalCandidateRunner(
                PRTIReleasePolicy(
                    robustness_summary_path=robustness_path,
                    evidence_directory=evidence,
                )
            )

            with self.assertRaisesRegex(ValueError, "differs"):
                runner.predeclare()

    @unittest.skipUnless(
        (saved_candidate / "prti_run_summary.json").is_file(),
        "Held-out PRTI-GH candidate is not installed",
    )
    def test_saved_candidate_matches_the_plan_and_preserves_sensitive_values(
        self,
    ) -> None:
        summary_path = self.saved_candidate / "prti_run_summary.json"
        manifest_path = self.saved_candidate / "prti_artifact_manifest.json"
        assessment_path = self.saved_evidence / (
            PRTIFinalCandidateRunner.ACCEPTANCE_JSON
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
        prepared = PRTIDataAdapter().prepare()
        final = pd.read_csv(self.saved_candidate / "prti_anonymized_final.csv")
        paired = pd.read_csv(
            self.saved_candidate / "prti_before_after_anonymization.csv"
        )

        self.assertEqual(summary["model_configuration"]["n_clusters"], 4)
        self.assertEqual(summary["pso_configuration"]["seed"], 10)
        self.assertEqual(final.shape, (1485, 106))
        self.assertEqual(paired.shape, (1485, 137))
        self.assertEqual(list(final.columns), list(prepared.original.columns))
        self.assertNotIn("cluster", final.columns)
        for column in prepared.schema.sensitive_attributes:
            pd.testing.assert_series_equal(
                final[column],
                prepared.original[column].reset_index(drop=True),
                check_dtype=False,
                check_names=False,
            )
        for column in PRTI_QIDS:
            pd.testing.assert_series_equal(
                paired[f"{column}After"],
                final[column],
                check_dtype=False,
                check_names=False,
            )

        self.assertTrue(assessment["all_checks_passed"])
        self.assertEqual(
            assessment["run_summary_sha256"],
            ArtifactManifestWriter.file_sha256(summary_path),
        )
        self.assertEqual(
            assessment["artifact_manifest_sha256"],
            ArtifactManifestWriter.file_sha256(manifest_path),
        )


if __name__ == "__main__":
    unittest.main()
