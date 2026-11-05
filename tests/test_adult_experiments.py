from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mo_obam.adult_experiments import (
    PAPER_ADULT_CLUSTER_VALUES,
    PAPER_K_VALUES,
    PAPER_LAMBDA_VALUES,
    AdultExperimentPlan,
    AdultExperimentRunConfig,
    AdultExperimentRunner,
)


class AdultExperimentPlanTests(unittest.TestCase):
    def test_table_six_plan_contains_the_eight_reported_mo_obam_rows(self) -> None:
        plan = AdultExperimentPlan.build("table-6")

        self.assertEqual(len(plan), 8)
        self.assertEqual(len({item.run_id for item in plan}), 8)
        self.assertEqual(
            {(item.k, item.n_clusters, item.lambda_weight) for item in plan},
            {
                *((k, 100, 0.0001) for k in PAPER_K_VALUES),
                *((k, 4, 1.0) for k in PAPER_K_VALUES),
            },
        )

    def test_full_grid_exactly_matches_appendix_a_values(self) -> None:
        plan = AdultExperimentPlan.build("full-grid")

        self.assertEqual(len(plan), 220)
        self.assertEqual({item.k for item in plan}, set(PAPER_K_VALUES))
        self.assertEqual(
            {item.n_clusters for item in plan}, set(PAPER_ADULT_CLUSTER_VALUES)
        )
        self.assertEqual(
            {item.lambda_weight for item in plan}, set(PAPER_LAMBDA_VALUES)
        )

    def test_dry_plan_records_paper_boundary_and_every_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            runner = AdultExperimentRunner(
                AdultExperimentRunConfig(
                    data_directory=Path("unused-for-dry-run"),
                    output_directory=output,
                    preset="table-6",
                )
            )

            plan_path = runner.write_plan()
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

            self.assertEqual(plan["number_of_configurations"], 8)
            self.assertEqual(len(plan["configurations"]), 8)
            self.assertIn("does not publish a random seed", plan["paper_seed_boundary"])


if __name__ == "__main__":
    unittest.main()
