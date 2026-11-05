from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mo_obam import ADULT_COLUMNS, MOOBAMConfig
from mo_obam.adult_reporting import AdultOutputReporter
from mo_obam.adult_runner import ADULT_SCHEMA


class AdultReportingTests(unittest.TestCase):
    def make_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        original = pd.DataFrame(
            {
                "age": [20, 22, 50, 52],
                "workclass": ["Private"] * 4,
                "fnlwgt": [1, 2, 3, 4],
                "education": ["HS-grad"] * 4,
                "education_num": [9] * 4,
                "marital_status": ["Never", "Never", "Married", "Married"],
                "occupation": ["A", "B", "A", "B"],
                "relationship": ["Not-in-family"] * 4,
                "race": ["White", "White", "Black", "Black"],
                "sex": ["Female", "Male", "Female", "Male"],
                "capital_gain": [0] * 4,
                "capital_loss": [0] * 4,
                "hours_per_week": [40] * 4,
                "native_country": ["United-States"] * 4,
                "income": ["<=50K", "<=50K", ">50K", ">50K"],
            },
            columns=ADULT_COLUMNS,
        )
        anonymized = original.copy()
        anonymized["age"] = [21.0, 21.0, 51.0, 51.0]
        anonymized["race"] = ["White", "White", "Black", "Black"]
        anonymized["sex"] = ["Female", "Female", "Male", "Male"]
        anonymized["marital_status"] = ["Never", "Never", "Married", "Married"]
        anonymized["cluster"] = [0, 0, 1, 1]
        return original, anonymized

    def test_final_release_is_full_and_contains_no_internal_cluster_column(self) -> None:
        original, anonymized = self.make_frames()
        reporter = AdultOutputReporter(ADULT_SCHEMA)
        artifacts = reporter.build(
            original,
            anonymized,
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
            information_loss=0.1,
            entropy_sum=2.0,
        )

        self.assertEqual(tuple(artifacts.anonymized_final.columns), ADULT_COLUMNS)
        self.assertEqual(artifacts.anonymized_final.shape, original.shape)
        self.assertNotIn("cluster", artifacts.anonymized_final)
        non_qids = [column for column in ADULT_COLUMNS if column not in ADULT_SCHEMA.qids]
        pd.testing.assert_frame_equal(
            artifacts.anonymized_final.loc[:, non_qids],
            original.loc[:, non_qids],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = reporter.write(Path(temporary_directory), artifacts)
            self.assertEqual(
                set(paths),
                {
                    "anonymized_final",
                    "model_evaluation",
                    "paper_comparison",
                    "cluster_details",
                },
            )
            self.assertTrue(
                (Path(temporary_directory) / paths["anonymized_final"]).exists()
            )

    def test_both_table_six_mo_obam_boundary_settings_are_compared(self) -> None:
        original, anonymized = self.make_frames()
        reporter = AdultOutputReporter(ADULT_SCHEMA)
        artifacts = reporter.build(
            original,
            anonymized,
            MOOBAMConfig(k=20, n_clusters=100, lambda_weight=0.0001),
            information_loss=0.02,
            entropy_sum=2.0,
        )

        comparison = artifacts.paper_comparison.set_index("metric")
        self.assertEqual(
            comparison.loc["MO-OBAM equivalence classes", "paper_expected"],
            100,
        )
        self.assertEqual(
            comparison.loc["MO-OBAM information loss", "paper_expected"],
            0.0116,
        )


if __name__ == "__main__":
    unittest.main()
