from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mo_obam import (
    ExactMOOBAMSolver,
    MOOBAMConfig,
    MOOBAMFormulation,
    MOOBAMParticleSwarmOptimizer,
    MOOBAMSchema,
    PSOConfig,
)


class ParticleSwarmTests(unittest.TestCase):
    def make_model(self) -> MOOBAMFormulation:
        data = pd.DataFrame(
            {
                "numeric_qid": [0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
                "categorical_qid": ["X", "X", "Y", "Y", "X", "X"],
                "sa": ["A", "B", "A", "B", "A", "B"],
                "untouched": list("abcdef"),
            }
        )
        return MOOBAMFormulation(
            data,
            MOOBAMSchema(("numeric_qid",), ("categorical_qid",), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=0.5),
        )

    def test_pso_is_feasible_deterministic_and_close_to_small_exact_reference(self) -> None:
        model = self.make_model()
        config = PSOConfig(
            population_size=20,
            iterations=50,
            seed=7,
        )
        first = MOOBAMParticleSwarmOptimizer(model, config).solve()
        second = MOOBAMParticleSwarmOptimizer(model, config).solve()
        exact = ExactMOOBAMSolver(model).solve()
        reference_metrics = model.evaluate(first.membership, first.centroids)

        np.testing.assert_array_equal(first.membership, second.membership)
        pd.testing.assert_frame_equal(first.centroids, second.centroids)
        self.assertTrue(all(size >= model.config.k for size in first.cluster_sizes))
        self.assertLess(abs(first.metrics.objective - exact.metrics.objective), 0.001)
        self.assertEqual(first.metrics, reference_metrics)
        self.assertEqual(len(first.stages), 1)
        self.assertEqual(first.stages[0].stage, "paper_objective")
        self.assertEqual(
            first.anonymized_data["untouched"].tolist(),
            model.data["untouched"].tolist(),
        )
        self.assertEqual(
            first.anonymized_data["sa"].tolist(), model.data["sa"].tolist()
        )

    def test_duplicate_qid_profiles_are_grouped_without_changing_equations(self) -> None:
        data = pd.DataFrame(
            {
                "numeric_qid": [0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0],
                "categorical_qid": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
                "sa": ["A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        formulation = MOOBAMFormulation(
            data,
            MOOBAMSchema(("numeric_qid",), ("categorical_qid",), ("sa",)),
            MOOBAMConfig(k=2, n_clusters=2, lambda_weight=1.0),
        )
        optimizer = MOOBAMParticleSwarmOptimizer(
            formulation,
            PSOConfig(population_size=10, iterations=10, seed=3),
        )

        result = optimizer.solve()

        self.assertEqual(optimizer._n_profiles, 4)
        self.assertEqual(len(result.anonymized_data), len(data))
        self.assertTrue(all(size >= 2 for size in result.cluster_sizes))

    def test_vectorized_entropy_exactly_matches_reference_formulation(self) -> None:
        model = self.make_model()
        optimizer = MOOBAMParticleSwarmOptimizer(
            model,
            PSOConfig(population_size=10, iterations=5, seed=2),
        )
        assignments = np.array([0, 0, 0, 1, 1, 1])
        membership = model.membership_from_assignments(assignments)
        profile_assignments = assignments[
            np.unique(optimizer._record_profile_ids, return_index=True)[1]
        ]
        cluster_sizes = membership.sum(axis=0)

        optimized = optimizer._calculate_entropy(
            profile_assignments,
            cluster_sizes,
        )
        reference = model.entropy_by_sensitive_attribute(membership)

        self.assertEqual(optimized.keys(), reference.keys())
        for column in optimized:
            self.assertAlmostEqual(optimized[column], reference[column])

    def test_compiled_distances_exactly_match_equation_12_arrays(self) -> None:
        model = self.make_model()
        optimizer = MOOBAMParticleSwarmOptimizer(
            model,
            PSOConfig(population_size=10, iterations=5, seed=2),
        )
        rng = np.random.default_rng(np.random.SeedSequence(2))
        numeric, categorical = optimizer._initialize_particles(rng)
        numeric_centroids = numeric[0]
        categorical_centroids = categorical[0]

        expected = np.zeros(
            (optimizer._n_profiles, model.config.n_clusters),
            dtype=float,
        )
        numeric_difference = (
            optimizer._numeric_profiles[:, None, :]
            - numeric_centroids[None, :, :]
        )
        expected += np.sum(numeric_difference**2, axis=2)
        expected += np.sum(
            optimizer._categorical_profiles[:, None, :]
            != categorical_centroids[None, :, :],
            axis=2,
        )

        actual = optimizer._distances_to_centroids(
            numeric_centroids,
            categorical_centroids,
        )

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
