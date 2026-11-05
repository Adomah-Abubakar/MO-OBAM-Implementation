from __future__ import annotations

import unittest
from pathlib import Path

from mo_obam import ADULT_COLUMNS, AdultDataPreparer


class AdultDataTests(unittest.TestCase):
    def test_preparation_reproduces_the_paper_shape_and_roles(self) -> None:
        data_directory = Path(__file__).parents[1] / "data" / "adult"
        prepared, report = AdultDataPreparer().prepare(data_directory)

        self.assertEqual(prepared.shape, (45_222, 15))
        self.assertEqual(tuple(prepared.columns), ADULT_COLUMNS)
        self.assertEqual(report.raw_training_rows, 32_561)
        self.assertEqual(report.raw_test_rows, 16_281)
        self.assertEqual(report.removed_training_rows, 2_399)
        self.assertEqual(report.removed_test_rows, 1_221)
        self.assertFalse(prepared.isna().any().any())
        self.assertFalse(prepared.astype(str).eq("?").any().any())
        self.assertEqual(set(prepared["income"]), {"<=50K", ">50K"})
        self.assertIn("occupation", prepared.columns)
        self.assertTrue({"age", "race", "sex", "marital_status"}.issubset(prepared.columns))


if __name__ == "__main__":
    unittest.main()
