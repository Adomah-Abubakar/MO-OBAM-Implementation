from __future__ import annotations

from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from openpyxl import load_workbook
import numpy as np
import pandas as pd

from mo_obam.formulation import MOOBAMFormulation
from mo_obam.prti_multiobjective import (
    PRTIMultiObjectiveConfig,
    PRTIMultiObjectiveStudy,
)
from mo_obam.prti_multiobjective_solver import PRTIMultiObjectiveParticleSwarm
from mo_obam.pso_solver import PSOConfig
from mo_obam.schema import MOOBAMConfig, MOOBAMSchema


class PRTIMultiObjectiveTests(unittest.TestCase):
    def test_full_declared_range_contains_1500_runs_without_lambda_repetition(self) -> None:
        config = PRTIMultiObjectiveConfig(
            minimum_clusters=1,
            maximum_clusters=30,
            k_values=(10, 15, 20, 25, 30),
            seeds=tuple(range(10)),
            held_out_seed=12,
        )
        self.assertEqual(len(config.configurations), 150)
        self.assertEqual(config.planned_runs, 1500)

    def test_pareto_dominance_minimizes_il_and_maximizes_entropy(self) -> None:
        better = SimpleNamespace(information_loss=0.4, entropy_sum=10.0)
        worse = SimpleNamespace(information_loss=0.5, entropy_sum=9.0)
        tradeoff = SimpleNamespace(information_loss=0.3, entropy_sum=8.0)
        self.assertTrue(PRTIMultiObjectiveParticleSwarm.dominates(better, worse))
        self.assertFalse(PRTIMultiObjectiveParticleSwarm.dominates(worse, better))
        self.assertFalse(PRTIMultiObjectiveParticleSwarm.dominates(better, tradeoff))

    def test_small_multiobjective_run_is_deterministic_and_hard_k_feasible(self) -> None:
        data = pd.DataFrame(
            {
                "numeric_qid": [0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
                "categorical_qid": ["X", "X", "Y", "Y", "X", "X"],
                "sa": ["A", "B", "A", "B", "A", "B"],
            }
        )
        formulation = MOOBAMFormulation(
            data,
            MOOBAMSchema(("numeric_qid",), ("categorical_qid",), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=0.0),
        )
        pso = PSOConfig(
            population_size=20,
            iterations=10,
            seed=7,
            random_coefficient_max=0.01,
        )
        first = PRTIMultiObjectiveParticleSwarm(
            formulation, pso, archive_size=10, initialization_attempts=5
        ).solve_multiobjective()
        second = PRTIMultiObjectiveParticleSwarm(
            formulation, pso, archive_size=10, initialization_attempts=5
        ).solve_multiobjective()

        self.assertGreater(len(first.candidates), 0)
        self.assertEqual(len(first.candidates), len(second.candidates))
        for left, right in zip(first.candidates, second.candidates, strict=True):
            np.testing.assert_array_equal(left.membership, right.membership)
            self.assertTrue(all(size >= 2 for size in left.cluster_sizes))
            self.assertAlmostEqual(left.information_loss, right.information_loss)
            self.assertAlmostEqual(left.entropy_sum, right.entropy_sum)

    def test_lambda_intervals_cover_zero_through_one(self) -> None:
        candidates = pd.DataFrame(
            [
                {"k": 30, "clusters": 4, "seed": seed, "candidate_index": 0,
                 "information_loss": 0.4, "entropy_sum": 10.0, "privacy_eligible": True}
                for seed in (0, 1)
            ]
            + [
                {"k": 30, "clusters": 4, "seed": seed, "candidate_index": 1,
                 "information_loss": 0.5, "entropy_sum": 20.0, "privacy_eligible": True}
                for seed in (0, 1)
            ]
        )
        intervals = PRTIMultiObjectiveStudy._lambda_intervals(
            candidates, {"k": 30, "clusters": 4}
        )
        self.assertEqual(float(intervals.iloc[0]["lambda_start"]), 0.0)
        self.assertEqual(float(intervals.iloc[-1]["lambda_end"]), 1.0)
        self.assertAlmostEqual(float(intervals.iloc[0]["lambda_end"]), 0.01)

    def test_merged_lambda_intervals_recalculate_their_midpoint_objective(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "k": 20,
                    "clusters": 5,
                    "seed": seed,
                    "candidate_index": candidate_index,
                    "information_loss": information_loss,
                    "entropy_sum": entropy_sum,
                    "privacy_eligible": True,
                }
                for seed in (0, 1)
                for candidate_index, information_loss, entropy_sum in (
                    (0, 0.40, 10.0),
                    (1, 0.50, 20.0),
                    # This point creates extra crossover boundaries without ever
                    # becoming optimal. Adjacent equal-signature ranges are merged.
                    (2, 0.45, 11.0),
                )
            ]
        )
        intervals = PRTIMultiObjectiveStudy._lambda_intervals(
            candidates, {"k": 20, "clusters": 5}
        )

        self.assertEqual(len(intervals), 2)
        for row in intervals.to_dict(orient="records"):
            midpoint = (row["lambda_start"] + row["lambda_end"]) / 2.0
            expected = (
                row["information_loss_mean"]
                - midpoint * row["entropy_sum_mean"]
            )
            self.assertAlmostEqual(row["objective_mean_at_midpoint"], expected)

    def test_plan_is_immutable_and_records_no_lambda_run_dimension(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            config = PRTIMultiObjectiveConfig(
                output_directory=output,
                minimum_clusters=1,
                maximum_clusters=30,
                k_values=(10, 15, 20, 25, 30),
                seeds=tuple(range(10)),
                held_out_seed=12,
            )
            study = PRTIMultiObjectiveStudy(config)
            first = study.write_plan("abc")
            second = study.write_plan("abc")
            self.assertEqual(first, second)
            self.assertIn(
                '"separate_optimizer_run_per_lambda": false',
                first.read_text(encoding="utf-8"),
            )

    def test_random_coefficient_is_explicitly_recorded_in_plan(self) -> None:
        with TemporaryDirectory() as directory:
            config = PRTIMultiObjectiveConfig(
                output_directory=Path(directory),
                random_coefficient_max=0.02,
            )
            plan = PRTIMultiObjectiveStudy(config).write_plan("abc")
            self.assertIn(
                '"random_coefficient_max": 0.02',
                plan.read_text(encoding="utf-8"),
            )

    def test_report_only_rejects_changed_scientific_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            original = PRTIMultiObjectiveConfig(output_directory=output)
            study = PRTIMultiObjectiveStudy(original)
            study.write_plan("abc")
            database = output / study.DATABASE_FILE
            with closing(study._connect(database)) as connection:
                study._initialize_database(connection, "abc")

            changed = PRTIMultiObjectiveStudy(
                PRTIMultiObjectiveConfig(
                    output_directory=output,
                    random_coefficient_max=0.02,
                )
            )
            with self.assertRaisesRegex(ValueError, "configuration_json"):
                changed.refresh_reports()

    def test_excel_report_has_the_three_documented_sheets(self) -> None:
        with TemporaryDirectory() as directory:
            report = Path(directory) / "report.xlsx"
            payload = {
                "status": "selection_incomplete",
                "method": "multi-objective PSO extension",
                "planned_runs": 10,
                "recorded_runs": 1,
                "completed_runs": 1,
                "patient_level_outputs": False,
                "selected_configuration": {
                    "k": 20,
                    "clusters": 5,
                    "privacy_eligible_seeds": 10,
                    "information_loss_mean": 0.468,
                    "information_loss_max": 0.485,
                    "entropy_sum_mean": 96.36,
                    "homogeneity_fraction_max": 0.17,
                    "minimum_cluster_size_min": 130,
                },
            }
            PRTIMultiObjectiveStudy._write_excel(
                report,
                payload,
                pd.DataFrame([{"status": "completed"}]),
                pd.DataFrame([{"k": 10, "clusters": 1}]),
                pd.DataFrame(
                    [{"lambda_start": 0.0, "lambda_end": 1.0}]
                ),
            )
            workbook = load_workbook(report, read_only=True)
            try:
                self.assertEqual(
                    workbook.sheetnames,
                    ["Summary", "Configurations", "Lambda intervals"],
                )
                summary = workbook["Summary"]
                summary_values = {
                    summary.cell(row, 1).value: summary.cell(row, 2).value
                    for row in range(1, summary.max_row + 1)
                }
                self.assertEqual(summary_values["Patient-level outputs"], "No")
                self.assertEqual(summary_values["Selected k"], 20)
                self.assertEqual(summary_values["Selected clusters"], 5)
            finally:
                workbook.close()


if __name__ == "__main__":
    unittest.main()
