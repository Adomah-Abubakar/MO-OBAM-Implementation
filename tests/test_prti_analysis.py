from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mo_obam.prti_analysis import (
    diversity_analysis,
    generate_analysis_bundle,
    risk_analysis,
)
from mo_obam.prti_runner import PRTIAnonymizationRunner
from mo_obam.schema import MOOBAMSchema


class PRTIAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = MOOBAMSchema(
            numeric_qids=("age",),
            categorical_qids=("area",),
            sensitive_attributes=("diagnosis",),
        )
        self.before = pd.DataFrame(
            {
                "age": [20, 21, 30, 31],
                "area": ["A", "B", "C", "D"],
                "diagnosis": ["x", "y", "x", "x"],
            }
        )
        self.after = pd.DataFrame(
            {
                "age": [20.5, 20.5, 30.5, 30.5],
                "area": ["AB", "AB", "CD", "CD"],
                "diagnosis": ["x", "y", "x", "x"],
            }
        )

    def test_risk_reports_real_before_after_values(self) -> None:
        report = risk_analysis(self.before, self.after, self.schema)
        self.assertEqual(report["dataset_version"].tolist(), [
            "before", "after", "change_after_minus_before"
        ])
        self.assertEqual(report.loc[0, "prosecutor_risk_maximum"], 1.0)
        self.assertEqual(report.loc[1, "prosecutor_risk_maximum"], 0.5)
        self.assertEqual(report.loc[0, "marketer_risk"], 1.0)
        self.assertEqual(report.loc[1, "marketer_risk"], 0.5)

    def test_diversity_is_measured_per_sensitive_attribute(self) -> None:
        report = diversity_analysis(self.after, self.schema)
        self.assertEqual(report.loc[0, "minimum_distinct_l"], 1)
        self.assertEqual(report.loc[0, "classes_l_at_least_2"], 1)
        self.assertEqual(report.loc[0, "records_in_classes_below_l2"], 2)

    def test_bundle_writes_and_reloads_tables_report_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            bundle = generate_analysis_bundle(
                before=self.before,
                after=self.after,
                schema=self.schema,
                information_loss=0.25,
                configured_k=2,
                configuration={"k": 2, "clusters": 2},
                output_directory=output,
            )
            validation = PRTIAnonymizationRunner._validate_analysis_outputs(
                bundle.paths, records=4, attributes=3
            )
            self.assertTrue(validation["all_analysis_outputs_reloaded"])
            self.assertEqual(bundle.metrics["achieved_k"], 2)
            self.assertEqual(bundle.metrics["minimum_distinct_l"], 1)
            self.assertIn("Journalist risk", bundle.paths["anonymization_report"].read_text())
            self.assertTrue(bundle.paths["risk_figure"].read_text().startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
