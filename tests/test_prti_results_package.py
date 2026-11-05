from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mo_obam.prti_results_package import (
    PRTIResultsPackageBuilder,
    PRTIResultsPackageConfig,
)


class PRTIResultsPackageTests(unittest.TestCase):
    project_root = Path(__file__).resolve().parents[1]
    heldout_directory = (
        project_root / "outputs/prti/privacy-candidate-heldout-seed10"
    )

    @unittest.skipUnless(
        (heldout_directory / "prti_run_summary.json").is_file(),
        "Held-out PRTI-GH candidate is not installed",
    )
    def test_package_is_complete_and_aggregate_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = PRTIResultsPackageBuilder(
                PRTIResultsPackageConfig(
                    output_directory=Path(temporary_directory) / "results"
                )
            ).write()

            tables = result["tables"]
            self.assertEqual(len(tables["qid_transformation"]), 29)
            self.assertEqual(len(tables["sensitive_attribute_results"]), 77)
            self.assertEqual(len(tables["final_cluster_summary"]), 4)
            self.assertEqual(len(tables["final_screening_checks"]), 10)
            self.assertEqual(
                int(tables["qid_transformation"]["changed_records"].sum()),
                31_386,
            )
            self.assertTrue(
                tables["sensitive_attribute_results"][
                    "values_preserved_row_by_row"
                ].all()
            )
            self.assertFalse(
                any(
                    column.startswith("centroid_")
                    for table in tables.values()
                    for column in table.columns
                )
            )
            self.assertEqual(len(result["figure_paths"]), 10)

    def test_qid_table_preserves_missing_as_an_observed_state(self) -> None:
        if not (self.heldout_directory / "prti_run_summary.json").is_file():
            self.skipTest("Held-out PRTI-GH candidate is not installed")
        builder = PRTIResultsPackageBuilder(PRTIResultsPackageConfig())
        evidence = builder._load_evidence()
        builder._validate_evidence(evidence)

        table = builder._qid_table(evidence).set_index("attribute")

        row = table.loc["presenceOfSupervisionsRoadGuideF"]
        self.assertEqual(row["source_missing_records"], 1_001)
        self.assertEqual(row["final_missing_records"], 0)
        self.assertEqual(row["source_distinct_including_missing"], 3)
        self.assertEqual(row["final_distinct_including_missing"], 2)

    def test_saved_tables_reload_without_patient_level_dimensions(self) -> None:
        if not (self.heldout_directory / "prti_run_summary.json").is_file():
            self.skipTest("Held-out PRTI-GH candidate is not installed")
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = PRTIResultsPackageBuilder(
                PRTIResultsPackageConfig(
                    output_directory=Path(temporary_directory) / "results"
                )
            ).write()
            for path in result["table_paths"].values():
                table = pd.read_csv(path)
                self.assertLess(len(table), 1_485)


if __name__ == "__main__":
    unittest.main()
