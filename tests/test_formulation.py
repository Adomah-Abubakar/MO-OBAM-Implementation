from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mo_obam import (
    MOOBAMConfig,
    MOOBAMFormulation,
    MOOBAMSchema,
    UndefinedPaperMetricError,
)


class FormulationTests(unittest.TestCase):
    def test_paper_table_two_entropy_and_information_loss(self) -> None:
        data = pd.DataFrame(
            {
                "age": [18, 19, 18, 38, 36, 39, 87, 88, 82],
                "visits": [1, 1, 2, 5, 8, 7, 13, 15, 12],
                "gender": ["Female"] * 3 + ["Male"] * 3 + ["Female"] * 3,
                "race": ["White"] * 3 + ["Black"] * 3 + ["White"] * 3,
                "disease": [
                    "Flu",
                    "Flu",
                    "Obesity",
                    "Hypertension",
                    "Hypertension",
                    "Hypertension",
                    "Depression",
                    "Diabetes",
                    "Cancer",
                ],
            }
        )
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(
                numeric_qids=("age", "visits"),
                categorical_qids=("gender", "race"),
                sensitive_attributes=("disease",),
            ),
            MOOBAMConfig(k=3, n_clusters=3, lambda_weight=1.0),
        )
        membership = model.membership_from_assignments(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        )
        centroids = model.optimal_centroids(membership)
        entropies = model.cluster_entropies(membership)["disease"].to_numpy()

        np.testing.assert_allclose(
            entropies,
            [0.9182958340544896, 0.0, 1.584962500721156],
        )
        self.assertAlmostEqual(model.deviation_after_anonymization(membership, centroids), 36.0)
        self.assertEqual(model.entropy_sum(membership), 0.0)

    def test_multiple_sa_minimum_is_taken_before_sum(self) -> None:
        data = pd.DataFrame(
            {
                "qid": [0.0, 1.0, 2.0, 3.0],
                "sa_one": ["A", "B", "A", "A"],
                "sa_two": ["X", "X", "X", "Y"],
            }
        )
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(("qid",), (), ("sa_one", "sa_two")),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
        )
        membership = model.membership_from_assignments(np.array([0, 0, 1, 1]))

        by_sa = model.entropy_by_sensitive_attribute(membership)
        self.assertEqual(by_sa, {"sa_one": 0.0, "sa_two": 0.0})
        self.assertEqual(model.entropy_sum(membership), 0.0)
        self.assertEqual(float(model.cluster_entropies(membership).sum(axis=1).min()), 1.0)

    def test_numeric_and_categorical_deviations_match_equations_ten_to_fourteen(self) -> None:
        data = pd.DataFrame(
            {
                "numeric_qid": [0.0, 2.0, 8.0, 10.0],
                "categorical_qid": ["A", "B", "A", "B"],
                "sa": [0, 1, 0, 1],
            }
        )
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(("numeric_qid",), ("categorical_qid",), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
        )
        membership = model.membership_from_assignments(np.array([0, 0, 1, 1]))
        centroids = model.optimal_centroids(membership)

        # Equation (13) numerator: numeric deviation is 4 and mismatch is 2.
        self.assertAlmostEqual(
            model.deviation_after_anonymization(membership, centroids), 6.0
        )
        # Equation (14) denominator: numeric deviation is 68 and mismatch is 2.
        self.assertAlmostEqual(model.information_loss(membership, centroids), 6.0 / 70.0)

    def test_empty_configured_cluster_violates_equation_twenty(self) -> None:
        data = pd.DataFrame({"qid": range(6), "sa": [0, 1, 0, 1, 0, 1]})
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(("qid",), (), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
        )
        membership = np.zeros((6, 2), dtype=int)
        membership[:, 0] = 1

        with self.assertRaisesRegex(ValueError, "including empty clusters"):
            model.validate_membership(membership)

    def test_multiple_memberships_violate_equation_nine(self) -> None:
        data = pd.DataFrame({"qid": range(4), "sa": [0, 1, 0, 1]})
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(("qid",), (), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
        )
        membership = np.array([[1, 1], [1, 0], [0, 1], [0, 1]])

        with self.assertRaisesRegex(ValueError, "each record to belong to one cluster"):
            model.validate_membership(membership)

    def test_arxiv_v2_direct_objective_is_applied_literally(self) -> None:
        data = pd.DataFrame({"qid": [0.0, 1.0, 2.0, 3.0], "sa": [0, 1, 0, 1]})
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(("qid",), (), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=0.2),
        )
        objective = model.objective(information_loss=1.0, entropy_sum=1.0)
        self.assertAlmostEqual(objective, 0.8)

    def test_zero_denominators_are_reported_not_smoothed(self) -> None:
        data = pd.DataFrame({"qid": [1.0, 1.0, 1.0, 1.0], "sa": [0, 1, 0, 1]})
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(("qid",), (), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
        )
        membership = model.membership_from_assignments(np.array([0, 0, 1, 1]))
        centroids = model.optimal_centroids(membership)

        with self.assertRaises(UndefinedPaperMetricError):
            model.information_loss(membership, centroids)

    def test_anonymization_changes_only_qids_and_adds_cluster(self) -> None:
        data = pd.DataFrame(
            {
                "qid": [0.0, 2.0, 8.0, 10.0],
                "sa": ["A", "B", "A", "B"],
                "other": [10, 20, 30, 40],
            }
        )
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(("qid",), (), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
        )
        membership = model.membership_from_assignments(np.array([0, 0, 1, 1]))
        anonymized = model.anonymize(membership, model.optimal_centroids(membership))

        self.assertEqual(anonymized["qid"].tolist(), [1.0, 1.0, 9.0, 9.0])
        self.assertEqual(anonymized["sa"].tolist(), data["sa"].tolist())
        self.assertEqual(anonymized["other"].tolist(), data["other"].tolist())
        self.assertEqual(data["qid"].tolist(), [0.0, 2.0, 8.0, 10.0])

    def test_existing_cluster_column_is_not_overwritten(self) -> None:
        data = pd.DataFrame(
            {"qid": [0.0, 2.0, 8.0, 10.0], "sa": [0, 1, 0, 1], "cluster": [9] * 4}
        )
        model = MOOBAMFormulation(
            data,
            MOOBAMSchema(("qid",), (), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
        )
        membership = model.membership_from_assignments(np.array([0, 0, 1, 1]))

        with self.assertRaisesRegex(ValueError, "already exists"):
            model.anonymize(membership, model.optimal_centroids(membership))


if __name__ == "__main__":
    unittest.main()
