from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mo_obam import (
    ExactMOOBAMSolver,
    ExactSearchLimitError,
    MOOBAMConfig,
    MOOBAMFormulation,
    MOOBAMSchema,
)


class ExactSolverTests(unittest.TestCase):
    def make_model(self) -> MOOBAMFormulation:
        data = pd.DataFrame(
            {
                "qid": [0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
                "sa": ["A", "B", "A", "B", "A", "B"],
                "untouched": list("abcdef"),
            }
        )
        return MOOBAMFormulation(
            data,
            MOOBAMSchema(("qid",), (), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=0.5),
        )

    def test_solver_checks_every_feasible_unlabeled_partition(self) -> None:
        result = ExactMOOBAMSolver(self.make_model()).solve()

        # S(6, 2)=31 total two-block partitions; six have a singleton.
        self.assertEqual(result.feasible_partitions_evaluated, 25)
        self.assertTrue((result.membership.sum(axis=0) >= 2).all())
        self.assertAlmostEqual(result.metrics.information_loss, 0.25280199252801994)
        self.assertAlmostEqual(result.metrics.entropy_sum, 1.0)
        self.assertAlmostEqual(result.metrics.objective, -0.24719800747198006)
        self.assertEqual(result.membership.argmax(axis=1).tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(result.anonymized_data["untouched"].tolist(), list("abcdef"))

    def test_result_exports_are_complete(self) -> None:
        result = ExactMOOBAMSolver(self.make_model()).solve()
        with tempfile.TemporaryDirectory() as directory:
            result.save(directory, prefix="test")
            names = {path.name for path in Path(directory).iterdir()}
        self.assertEqual(
            names,
            {
                "test_anonymized.csv",
                "test_membership.csv",
                "test_centroids.csv",
                "test_summary.json",
            },
        )

    def test_large_search_is_rejected_before_work_starts(self) -> None:
        with self.assertRaises(ExactSearchLimitError):
            ExactMOOBAMSolver(self.make_model(), max_records=5).solve()


if __name__ == "__main__":
    unittest.main()
