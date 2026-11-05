from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from mo_obam import AdultDataPreparer, MOOBAMSchema, PrivacyRiskEvaluator
from mo_obam.adult_runner import ADULT_SCHEMA


class PrivacyRiskTests(unittest.TestCase):
    def test_linkage_and_homogeneity_counts_records_not_classes(self) -> None:
        data = pd.DataFrame(
            {
                "qid": ["A", "A", "B", "B", "B", "C"],
                "sa": [0, 0, 0, 1, 1, 1],
            }
        )
        schema = MOOBAMSchema((), ("qid",), ("sa",))
        metrics = PrivacyRiskEvaluator(data, schema).evaluate((0.4, 0.5))

        self.assertEqual(metrics.equivalence_classes, 3)
        self.assertEqual(metrics.minimum_equivalence_class_size, 1)
        self.assertEqual(metrics.maximum_equivalence_class_size, 3)
        self.assertEqual(metrics.linkage_attack_records, {"0.4": 3, "0.5": 1})
        self.assertEqual(metrics.homogeneity_attack_records, 3)
        self.assertEqual(metrics.homogeneity_attack_records_by_sensitive_attribute, {"sa": 3})

    def test_prepared_adult_baseline_exactly_matches_arxiv_v2_table_six(self) -> None:
        data_directory = Path(__file__).parents[1] / "data" / "adult"
        prepared, _ = AdultDataPreparer().prepare(data_directory)
        metrics = PrivacyRiskEvaluator(prepared, ADULT_SCHEMA).evaluate()

        self.assertEqual(metrics.equivalence_classes, 1_900)
        self.assertEqual(
            metrics.linkage_attack_records,
            {"0.05": 6_506, "0.075": 4_906, "0.1": 3_910},
        )
        self.assertEqual(metrics.homogeneity_attack_records, 634)


if __name__ == "__main__":
    unittest.main()
