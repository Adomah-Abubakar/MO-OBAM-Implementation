from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from mo_obam.provenance import atomic_write_text, exclusive_directory_lock
from mo_obam.prti_robustness import (
    PRTIRobustnessConfig,
    PRTIRobustnessPlan,
    PRTIRobustnessRunner,
)
from mo_obam.prti_results_package import PRTIResultsPackageBuilder


class PRTIRobustnessTests(unittest.TestCase):
    def test_plan_contains_the_four_predeclared_candidates(self) -> None:
        plan = PRTIRobustnessPlan.build()

        self.assertEqual(
            [experiment.configuration_id for experiment in plan],
            [
                "baseline",
                "privacy_candidate",
                "lowest_il_candidate",
                "compromise_candidate",
            ],
        )
        self.assertEqual(
            {
                (
                    experiment.model.k,
                    experiment.model.n_clusters,
                    experiment.model.lambda_weight,
                )
                for experiment in plan
            },
            {
                (10, 30, 1.0),
                (10, 4, 1.0),
                (10, 30, 0.001),
                (10, 10, 1.0),
            },
        )

    def test_configuration_rejects_invalid_seed_and_worker_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            PRTIRobustnessConfig(seeds=(1, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PRTIRobustnessConfig(seeds=(-1,))
        with self.assertRaisesRegex(ValueError, "workers"):
            PRTIRobustnessConfig(workers=0)
        with self.assertRaisesRegex(ValueError, "worker_backend"):
            PRTIRobustnessConfig(worker_backend="invalid")

    def test_plan_labels_seed_protocol_and_excludes_patient_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = PRTIRobustnessRunner(
                PRTIRobustnessConfig(
                    source_path=Path("unused-for-dry-run.xls"),
                    output_directory=Path(temporary_directory),
                )
            )

            plan = json.loads(runner.write_plan().read_text(encoding="utf-8"))

            self.assertEqual(plan["number_of_configurations"], 4)
            self.assertEqual(plan["number_of_planned_runs"], 40)
            self.assertEqual(plan["seeds"], list(range(10)))
            self.assertFalse(plan["patient_level_outputs"])
            self.assertIn("not specify", plan["seed_protocol"])

    def test_atomic_write_never_leaves_a_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "checkpoint.json"

            atomic_write_text(destination, '{"status": "complete"}')

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"status": "complete"},
            )
            self.assertEqual(list(root.glob("*.tmp")), [])
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_study_directory_rejects_a_second_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            with exclusive_directory_lock(root):
                with self.assertRaisesRegex(RuntimeError, "already using"):
                    with exclusive_directory_lock(root):
                        self.fail("A second writer acquired the same study lock.")

    def test_truncated_checkpoint_is_treated_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = PRTIRobustnessRunner(
                PRTIRobustnessConfig(
                    output_directory=Path(temporary_directory),
                    seeds=(0,),
                )
            )
            experiment = runner.experiments[0]
            path = runner._checkpoint_path(experiment, 0)
            path.parent.mkdir(parents=True)
            path.write_text('{"paper_revision":', encoding="utf-8")

            self.assertIsNone(
                runner._load_checkpoint(experiment, 0, "reviewed-source")
            )

    def test_large_batches_refresh_reports_periodically_not_per_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = PRTIRobustnessRunner(
                PRTIRobustnessConfig(
                    source_path=Path("reviewed.xls"),
                    output_directory=Path(temporary_directory),
                    seeds=tuple(range(10)),
                    workers=1,
                )
            )
            runner.write_plan = MagicMock()
            runner._load_checkpoint = MagicMock(return_value=None)
            runner._run_one = MagicMock()
            runner._write_outputs = MagicMock(return_value={"status": "ok"})

            with patch("mo_obam.prti_robustness.PRTIDataAdapter") as adapter:
                adapter.return_value.prepare.return_value = SimpleNamespace(
                    summary={"source_sha256_before": "reviewed-source"}
                )
                result = runner.run(max_new_runs=30)

            self.assertEqual(runner._run_one.call_count, 30)
            self.assertEqual(runner._write_outputs.call_count, 3)
            self.assertEqual(result, {"status": "ok"})

    def test_process_backend_executes_independent_child_tasks(self) -> None:
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("The optimized process backend requires POSIX fork.")

        class MarkerRunner(PRTIRobustnessRunner):
            def write_plan(self, source_sha256: str | None = None) -> Path:
                del source_sha256
                return self.config.output_directory / self.PLAN_FILE

            def _load_checkpoint(self, experiment, seed, source_hash):
                del experiment, seed, source_hash
                return None

            def _run_one(
                self,
                prepared,
                source_hash,
                experiment,
                seed,
                progress,
            ) -> None:
                del prepared, source_hash, experiment, progress
                atomic_write_text(
                    self.config.output_directory / f"marker-{seed}.txt",
                    str(seed),
                )

            def _write_outputs(self, source_hash: str) -> dict[str, object]:
                del source_hash
                return {
                    "markers": len(
                        list(self.config.output_directory.glob("marker-*.txt"))
                    )
                }

        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = MarkerRunner(
                PRTIRobustnessConfig(
                    source_path=Path("reviewed.xls"),
                    output_directory=Path(temporary_directory),
                    seeds=(0, 1),
                    workers=2,
                    worker_backend="process",
                )
            )
            runner.experiments = runner.experiments[:1]
            with patch("mo_obam.prti_robustness.PRTIDataAdapter") as adapter:
                adapter.return_value.prepare.return_value = SimpleNamespace(
                    summary={"source_sha256_before": "reviewed-source"}
                )
                result = runner.run(max_new_runs=2)

            self.assertEqual(result, {"markers": 2})

    def test_summary_uses_sample_variation_and_keeps_failures_explicit(self) -> None:
        runner = PRTIRobustnessRunner(
            PRTIRobustnessConfig(seeds=(0, 1), workers=1)
        )
        runs = pd.DataFrame(
            [
                {
                    "configuration_id": "baseline",
                    "status": "completed",
                    "information_loss": 0.4,
                    "entropy_sum": 50.0,
                },
                {
                    "configuration_id": "baseline",
                    "status": "completed",
                    "information_loss": 0.6,
                    "entropy_sum": 54.0,
                },
                {
                    "configuration_id": "privacy_candidate",
                    "status": "no_feasible_particle",
                },
            ]
        )

        summary = runner.summarize_runs(runs).set_index("configuration_id")

        self.assertAlmostEqual(summary.loc["baseline", "information_loss_mean"], 0.5)
        self.assertAlmostEqual(
            summary.loc["baseline", "information_loss_std"],
            0.02 ** 0.5,
        )
        self.assertEqual(summary.loc["baseline", "runs_completed"], 2)
        self.assertEqual(
            summary.loc["privacy_candidate", "runs_no_feasible_particle"], 1
        )
        self.assertEqual(summary.loc["privacy_candidate", "runs_completed"], 0)

    def test_results_table_adds_percentages_without_changing_means(self) -> None:
        robustness = {
            "configurations": [
                {
                    "configuration_id": configuration_id,
                    "purpose": configuration_id,
                    "k": 10,
                    "configured_clusters": clusters,
                    "lambda": weight,
                    "seeds_planned": 10,
                    "runs_completed": completed,
                    "runs_no_feasible_particle": 10 - completed,
                    "completion_percent": 10.0 * completed,
                    "information_loss_mean": information_loss,
                    "information_loss_std": 0.01,
                    "information_loss_min": information_loss - 0.01,
                    "information_loss_max": information_loss + 0.01,
                    "entropy_sum_mean": 50.0,
                    "entropy_sum_std": 1.0,
                    "linkage_records_tau_0_05_mean": linkage,
                    "linkage_records_tau_0_05_max": linkage,
                    "homogeneity_records_any_sa_mean": homogeneity,
                    "homogeneity_records_any_sa_std": 1.0,
                    "homogeneity_records_any_sa_min": homogeneity - 1,
                    "homogeneity_records_any_sa_max": homogeneity + 1,
                    "zero_minimum_entropy_sensitive_attributes_mean": 1.0,
                    "zero_minimum_entropy_sensitive_attributes_min": 0.0,
                    "zero_minimum_entropy_sensitive_attributes_max": 2.0,
                    "minimum_cluster_size_min": 10.0,
                    "maximum_cluster_size_max": 200.0,
                    "feasible_evaluation_percent_mean": 50.0,
                }
                for configuration_id, clusters, weight, completed,
                    information_loss, linkage, homogeneity in (
                        ("baseline", 30, 1.0, 7, 0.4, 10.0, 1000.0),
                        ("privacy_candidate", 4, 1.0, 10, 0.6, 0.0, 0.0),
                        ("lowest_il_candidate", 30, 0.001, 7, 0.3, 20.0, 1200.0),
                        ("compromise_candidate", 10, 1.0, 10, 0.5, 0.0, 800.0),
                    )
            ]
        }

        table = PRTIResultsPackageBuilder._robustness_table(
            PRTIResultsPackageBuilder.__new__(PRTIResultsPackageBuilder),
            {"robustness": robustness},
        ).set_index("configuration_id")

        self.assertEqual(table.loc["baseline", "information_loss_mean"], 0.4)
        self.assertAlmostEqual(
            table.loc["compromise_candidate", "homogeneity_percent_any_sa_mean"],
            100.0 * 800.0 / 1485,
        )


if __name__ == "__main__":
    unittest.main()
