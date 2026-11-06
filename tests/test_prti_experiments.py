from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mo_obam.prti_experiments import (
    PRTIBaselineComparator,
    PRTISeedScreenConfig,
    PRTISensitivityConfig,
    PRTISensitivityExperiment,
    PRTISensitivityPlan,
    PRTISensitivityRunner,
    one_factor_run_ids,
)


class PRTIExperimentTests(unittest.TestCase):
    def test_seed_screen_configuration_requires_unique_nonnegative_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            PRTISeedScreenConfig(seeds=(1, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PRTISeedScreenConfig(seeds=(-1,))

    def test_sensitivity_plan_exactly_reuses_wei_appendix_a_grid(self) -> None:
        plan = PRTISensitivityPlan.build()

        self.assertEqual(len(plan), 220)
        self.assertEqual({item.k for item in plan}, {5, 10, 15, 20})
        self.assertEqual(
            {item.n_clusters for item in plan},
            {4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
        )
        self.assertEqual(
            {item.lambda_weight for item in plan},
            {0.0001, 0.001, 0.01, 0.1, 1.0},
        )

    def test_sensitivity_detects_equation_twenty_infeasibility(self) -> None:
        experiment = PRTISensitivityExperiment(k=20, n_clusters=100, lambda_weight=1)

        self.assertFalse(experiment.is_mathematically_feasible(1485))
        self.assertTrue(experiment.is_mathematically_feasible(2000))

    def test_sensitivity_plan_explicitly_excludes_patient_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = PRTISensitivityRunner(
                PRTISensitivityConfig(
                    source_path=Path("unused-for-dry-run.xls"),
                    output_directory=Path(temporary_directory),
                )
            )

            plan = json.loads(runner.write_plan().read_text(encoding="utf-8"))

            self.assertEqual(plan["number_of_configurations"], 220)
            self.assertFalse(plan["patient_level_outputs"])
            self.assertIn("project adaptation", plan["grid_source"])

    def test_writeup_one_factor_design_contains_18_unique_grid_points(self) -> None:
        run_ids = one_factor_run_ids()

        self.assertEqual(len(run_ids), 18)
        self.assertIn("k10_c030_lambda1", run_ids)
        self.assertIn("k05_c030_lambda1", run_ids)
        self.assertIn("k10_c100_lambda1", run_ids)
        self.assertIn("k10_c030_lambda0p0001", run_ids)

    def test_comparison_exports_aggregate_exact_reproduction_evidence(self) -> None:
        summary = {
            "run_id": "fresh-run",
            "run_label": "baseline-v1-seed9",
            "run_started_at_utc": "2026-10-27T10:00:00+00:00",
            "run_completed_at_utc": "2026-10-27T10:03:00+00:00",
            "execution": {"canonical_command": ["python", "run_prti.py"]},
            "generation_snapshot": {"git": {"commit": "abc", "working_tree_clean": True}},
            "model_configuration": {"k": 10, "n_clusters": 2, "lambda_weight": 1.0},
            "pso_configuration": {
                "population_size": 4,
                "iterations": 3,
                "seed": 9,
                "random_coefficient_max": 0.01,
            },
            "preparation": {"source_sha256_before": "source-hash"},
            "final_metrics": {
                "information_loss": 0.25,
                "entropy_sum": 2.0,
                "objective": -1.75,
                "entropy_by_sensitive_attribute": {"sa": 2.0},
            },
            "cluster_sizes": [4, 6],
            "stages": [
                {
                    "feasible_evaluations": 10,
                    "infeasible_evaluations": 2,
                    "iteration_found": 1,
                }
            ],
            "privacy_evaluation": {
                "anonymized": {
                    "linkage_attack_records": {"0.05": 0, "0.075": 0, "0.1": 0},
                    "homogeneity_attack_records": 0,
                }
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = root / "legacy"
            fresh = root / "fresh"
            evidence = root / "evidence"
            legacy.mkdir()
            fresh.mkdir()
            legacy_summary = dict(summary)
            legacy_summary["run_id"] = None
            legacy_summary["run_started_at_utc"] = None
            legacy_summary["run_completed_at_utc"] = None
            (legacy / "prti_run_summary.json").write_text(
                json.dumps(legacy_summary), encoding="utf-8"
            )
            (fresh / "prti_run_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (fresh / "prti_artifact_manifest.json").write_text(
                json.dumps({"manifest_version": 2}), encoding="utf-8"
            )

            comparison = PRTIBaselineComparator().compare(
                legacy, fresh, evidence
            )

            self.assertTrue(comparison["reproduced_exactly"])
            self.assertTrue((evidence / "prti_fresh_vs_legacy_comparison.csv").is_file())
            self.assertTrue((evidence / "prti_fresh_vs_legacy_comparison.json").is_file())
            self.assertTrue((evidence / "prti_fresh_artifact_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
