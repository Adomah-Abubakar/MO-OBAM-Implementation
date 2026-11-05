from __future__ import annotations

import unittest

import pandas as pd

from mo_obam.prti_release_readiness import (
    build_attribute_review,
    build_readiness_record,
)


class PRTIReleaseReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = pd.DataFrame(
            {
                "patientEmail": ["person@example.org", None],
                "clinicalNarrative": ["word " * 10, "brief"],
                "district": ["A", "B"],
            }
        )
        self.audit = pd.DataFrame(
            {
                "position": [1, 2, 3],
                "attribute": list(self.original.columns),
                "role": [
                    "sensitive attribute",
                    "sensitive attribute",
                    "QID - categorical",
                ],
                "sourceDtype": ["str", "str", "str"],
                "distinctNonMissingValues": [1, 2, 2],
                "missingValues": [1, 0, 0],
                "cardinalityPercentage": [0.5, 1.0, 1.0],
                "reviewFlag": ["", "", "high-cardinality QID"],
            }
        )

    def test_review_flags_identifiers_and_free_text_without_exporting_values(self) -> None:
        review = build_attribute_review(self.original, self.audit)

        self.assertEqual(len(review), 3)
        self.assertTrue(review.loc[0, "potential_direct_identifier"])
        self.assertTrue(review.loc[1, "potential_free_text"])
        self.assertNotIn("person@example.org", review.to_csv(index=False))
        self.assertTrue(review["clinical_review_status"].eq("PENDING").all())

    def test_review_rejects_audit_with_different_attribute_order(self) -> None:
        audit = self.audit.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "source order"):
            build_attribute_review(self.original, audit)

    def test_scope_decisions_do_not_overclaim_release_or_utility(self) -> None:
        review = build_attribute_review(self.original, self.audit)

        class Prepared:
            original = self.original
            summary = {"source_sha256_before": "test-hash"}

        record = build_readiness_record(Prepared(), review)  # type: ignore[arg-type]

        self.assertFalse(
            record["experiment_scope_decision"]["required_for_current_claims"]
        )
        self.assertFalse(
            record["downstream_utility_decision"]["required_for_current_claims"]
        )
        self.assertIn("BLOCKED", record["external_release_status"])


if __name__ == "__main__":
    unittest.main()
