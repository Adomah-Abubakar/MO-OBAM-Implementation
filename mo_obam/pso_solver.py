"""Scalable PSO search using the paper's MO-OBAM objective and constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .formulation import MOOBAMFormulation, MOOBAMMetrics


StageName = Literal["paper_objective"]
ProgressCallback = Callable[[StageName, int, int, float, int, int], None]


class NoFeasibleParticleError(RuntimeError):
    """Raised when random initialization contains no hard-k feasible particle."""


@dataclass(frozen=True)
class PSOConfig:
    """Configure the PSO mechanics published in the authors' repository.

    Population 100, 100 iterations, and the 0.01 random coefficient bounds are
    taken from the authors' notebook and ``particle_swarm.py``.
    """

    population_size: int = 100
    iterations: int = 100
    seed: int = 42
    random_coefficient_max: float = 0.01

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1.")
        if self.random_coefficient_max <= 0.0:
            raise ValueError("random_coefficient_max must be positive.")
        if 2 * self.random_coefficient_max >= 1.0:
            raise ValueError("random coefficients must leave a positive inertia term.")


@dataclass(frozen=True)
class PSOStageResult:
    """Summarize the best feasible candidate found during one PSO stage."""

    stage: StageName
    best_score: float
    information_loss: float
    entropy_sum: float
    entropy_by_sensitive_attribute: dict[str, float]
    iteration_found: int
    feasible_evaluations: int
    infeasible_evaluations: int


@dataclass(frozen=True)
class ScalableMOOBAMResult:
    """Contain the final anonymized data and all optimization evidence."""

    membership: np.ndarray
    centroids: pd.DataFrame
    anonymized_data: pd.DataFrame
    metrics: MOOBAMMetrics
    cluster_sizes: tuple[int, ...]
    stages: tuple[PSOStageResult, ...]


@dataclass
class _Candidate:
    assignments: np.ndarray
    information_loss: float
    entropy_by_sa: dict[str, float]
    entropy_sum: float


@dataclass
class _StageSearchResult:
    numeric_centroids: np.ndarray
    categorical_centroids: np.ndarray
    candidate: _Candidate
    summary: PSOStageResult


class MOOBAMParticleSwarmOptimizer:
    """Search centroid variables with the authors' PSO update mechanics.

    For every centroid particle, record membership is assigned to the nearest
    centroid using Equation (12), as specified in arXiv v2 Algorithm 1.
    Infeasible particles are rejected because Equation (20) is a hard model
    constraint. Algorithm 1 prints a negative k-violation term that would
    reward violations during minimization, while the authors' code uses a
    positive term; this implementation does not guess an undocumented
    multiplier or normalization.
    """

    def __init__(
        self,
        formulation: MOOBAMFormulation,
        pso_config: PSOConfig | None = None,
    ) -> None:
        self.formulation = formulation
        self.pso_config = pso_config or PSOConfig()
        self._prepare_arrays()

    def solve(
        self,
        progress: ProgressCallback | None = None,
    ) -> ScalableMOOBAMResult:
        """Minimize the direct arXiv v2 objective in one PSO run."""

        stage = self._run_stage(
            "paper_objective",
            np.random.SeedSequence(self.pso_config.seed),
            progress=progress,
        )

        final_centroids = self._centroid_frame(
            stage.numeric_centroids,
            stage.categorical_centroids,
        )
        membership = self.formulation.membership_from_assignments(
            stage.candidate.assignments
        )
        metrics = MOOBAMMetrics(
            information_loss=stage.candidate.information_loss,
            entropy_by_sensitive_attribute=stage.candidate.entropy_by_sa,
            entropy_sum=stage.candidate.entropy_sum,
            objective=self.formulation.objective(
                stage.candidate.information_loss,
                stage.candidate.entropy_sum,
            ),
        )
        anonymized = self.formulation.anonymize(membership, final_centroids)
        cluster_sizes = tuple(int(value) for value in membership.sum(axis=0))
        return ScalableMOOBAMResult(
            membership=membership,
            centroids=final_centroids,
            anonymized_data=anonymized,
            metrics=metrics,
            cluster_sizes=cluster_sizes,
            stages=(stage.summary,),
        )

    def _run_stage(
        self,
        stage: StageName,
        seed_sequence: np.random.SeedSequence,
        progress: ProgressCallback | None,
    ) -> _StageSearchResult:
        """Execute one deterministic PSO stage and retain its best feasible particle."""

        rng = np.random.default_rng(seed_sequence)
        numeric, categorical = self._initialize_particles(rng)
        numeric_velocity = np.zeros_like(numeric, dtype=float)
        categorical_velocity = np.zeros_like(categorical, dtype=float)
        personal_numeric = numeric.copy()
        personal_categorical = categorical.copy()
        personal_scores = np.full(self.pso_config.population_size, np.inf)

        global_numeric: np.ndarray | None = None
        global_categorical: np.ndarray | None = None
        global_candidate: _Candidate | None = None
        global_score = float("inf")
        iteration_found = -1
        feasible_evaluations = 0
        infeasible_evaluations = 0

        for iteration in range(self.pso_config.iterations):
            for particle_index in range(self.pso_config.population_size):
                candidate = self._evaluate_particle(
                    numeric[particle_index],
                    categorical[particle_index],
                )
                if candidate is None:
                    infeasible_evaluations += 1
                    continue
                feasible_evaluations += 1
                score = self._stage_score(candidate)
                if score < personal_scores[particle_index]:
                    personal_scores[particle_index] = score
                    personal_numeric[particle_index] = numeric[particle_index]
                    personal_categorical[particle_index] = categorical[particle_index]
                if score < global_score:
                    global_score = score
                    global_numeric = numeric[particle_index].copy()
                    global_categorical = categorical[particle_index].copy()
                    global_candidate = candidate
                    iteration_found = iteration

            if global_numeric is None or global_categorical is None:
                raise NoFeasibleParticleError(
                    f"The {stage} stage found no particle satisfying hard k-anonymity."
                )
            if progress is not None:
                progress(
                    stage,
                    iteration + 1,
                    self.pso_config.iterations,
                    global_score,
                    feasible_evaluations,
                    infeasible_evaluations,
                )
            numeric, categorical, numeric_velocity, categorical_velocity = (
                self._update_particles(
                    numeric,
                    categorical,
                    numeric_velocity,
                    categorical_velocity,
                    personal_numeric,
                    personal_categorical,
                    global_numeric,
                    global_categorical,
                    rng,
                )
            )

        assert global_numeric is not None
        assert global_categorical is not None
        global_candidate = self._evaluate_particle(
            global_numeric, global_categorical
        )
        assert global_candidate is not None
        final_score = self._stage_score(global_candidate)
        summary = PSOStageResult(
            stage=stage,
            best_score=final_score,
            information_loss=global_candidate.information_loss,
            entropy_sum=global_candidate.entropy_sum,
            entropy_by_sensitive_attribute=global_candidate.entropy_by_sa,
            iteration_found=iteration_found,
            feasible_evaluations=feasible_evaluations,
            infeasible_evaluations=infeasible_evaluations,
        )
        return _StageSearchResult(
            numeric_centroids=global_numeric,
            categorical_centroids=global_categorical,
            candidate=global_candidate,
            summary=summary,
        )

    def _stage_score(
        self,
        candidate: _Candidate,
    ) -> float:
        """Return IL - lambda * sum(minimum SA entropy), the v2 objective."""

        return self.formulation.objective(
            candidate.information_loss,
            candidate.entropy_sum,
        )

    def _evaluate_particle(
        self,
        numeric_centroids: np.ndarray,
        categorical_centroids: np.ndarray,
    ) -> _Candidate | None:
        """Evaluate Equations (12)-(20) without constructing intermediate frames.

        Records with the same QID tuple have the same distance to every
        centroid and therefore always receive the same nearest-centroid
        assignment.  Evaluating each distinct tuple once and weighting it by
        its record count is mathematically identical to Algorithm 1's
        record-by-record loop, while avoiding duplicate distance calculations.
        """

        distances = self._distances_to_centroids(
            numeric_centroids,
            categorical_centroids,
        )

        profile_assignments = np.argmin(distances, axis=1)
        cluster_sizes = np.bincount(
            profile_assignments,
            weights=self._profile_sizes,
            minlength=self.formulation.config.n_clusters,
        ).astype(int)
        if np.any(cluster_sizes < self.formulation.config.k):
            return None

        selected_distances = distances[np.arange(self._n_profiles), profile_assignments]
        weighted_deviation = np.dot(selected_distances, self._profile_sizes)
        information_loss = float(weighted_deviation / self._denominator)
        entropy_by_sa = self._calculate_entropy(profile_assignments, cluster_sizes)
        entropy_sum = float(sum(entropy_by_sa.values()))
        return _Candidate(
            assignments=profile_assignments[self._record_profile_ids],
            information_loss=information_loss,
            entropy_by_sa=entropy_by_sa,
            entropy_sum=entropy_sum,
        )

    def _distances_to_centroids(
        self,
        numeric_centroids: np.ndarray,
        categorical_centroids: np.ndarray,
    ) -> np.ndarray:
        """Evaluate Equation (12) with compiled pairwise-distance kernels.

        Squared Euclidean distance is the numeric term. Hamming distance is a
        mismatch fraction, so multiplying by the categorical QID count and
        rounding recovers the exact integer mismatch count used by Equation
        (11). This avoids allocating a records-by-clusters-by-QIDs tensor for
        every particle.
        """

        cluster_count = self.formulation.config.n_clusters
        if self._numeric_profiles.shape[1]:
            distances = cdist(
                self._numeric_profiles,
                numeric_centroids,
                metric="sqeuclidean",
            )
        else:
            distances = np.zeros((self._n_profiles, cluster_count), dtype=float)
        categorical_count = self._categorical_profiles.shape[1]
        if categorical_count:
            mismatch_fraction = cdist(
                self._categorical_profiles,
                categorical_centroids,
                metric="hamming",
            )
            distances += np.rint(mismatch_fraction * categorical_count)
        return distances

    def _calculate_entropy(
        self,
        profile_assignments: np.ndarray,
        cluster_sizes: np.ndarray,
    ) -> dict[str, float]:
        """Calculate Equations (15)-(19) for every sensitive attribute.

        The precomputed sparse arrays contain the nonzero sensitive-value
        counts for each distinct QID profile. One indexed accumulation builds
        all cluster/value counts, and ``reduceat`` then sums the entropy terms
        within each sensitive attribute. This is algebraically identical to
        evaluating every attribute separately but avoids a Python loop for
        every attribute and particle.
        """

        assigned_clusters = profile_assignments[self._sensitive_profile_indices]
        flattened_states = (
            assigned_clusters * self._sensitive_state_count
            + self._sensitive_state_indices
        )
        counts = np.rint(
            np.bincount(
                flattened_states,
                weights=self._sensitive_state_counts,
                minlength=(
                    self.formulation.config.n_clusters
                    * self._sensitive_state_count
                ),
            )
        ).astype(np.int64).reshape(
            self.formulation.config.n_clusters,
            self._sensitive_state_count,
        )

        probabilities = counts / cluster_sizes[:, None]
        entropy_terms = np.zeros_like(probabilities, dtype=float)
        positive = probabilities > 0
        entropy_terms[positive] = -(
            probabilities[positive] * np.log2(probabilities[positive])
        )
        entropy_by_cluster = np.add.reduceat(
            entropy_terms,
            self._sensitive_state_offsets[:-1],
            axis=1,
        )
        minimum_entropies = entropy_by_cluster.min(axis=0)
        return dict(
            zip(
                self.formulation.schema.sensitive_attributes,
                minimum_entropies.astype(float),
                strict=True,
            )
        )

    def _initialize_particles(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize every q_cj from observed QI values as Algorithm 1 states."""

        shape = (self.pso_config.population_size, self.formulation.config.n_clusters)
        numeric = np.empty(shape + (self._numeric_data.shape[1],), dtype=float)
        for column_index, levels in enumerate(self._numeric_levels):
            numeric[:, :, column_index] = rng.choice(levels, size=shape)

        categorical = np.empty(shape + (self._categorical_data.shape[1],), dtype=int)
        for column_index, levels in enumerate(self._categorical_levels):
            categorical[:, :, column_index] = rng.integers(
                0, len(levels), size=shape
            )
        return numeric, categorical

    def _update_particles(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray,
        numeric_velocity: np.ndarray,
        categorical_velocity: np.ndarray,
        personal_numeric: np.ndarray,
        personal_categorical: np.ndarray,
        global_numeric: np.ndarray,
        global_categorical: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply the continuous and categorical updates in ``particle_swarm.py``."""

        coefficient_shape = (self.pso_config.population_size, 1, 1)
        cognitive = rng.uniform(
            0.0, self.pso_config.random_coefficient_max, size=coefficient_shape
        )
        social = rng.uniform(
            0.0, self.pso_config.random_coefficient_max, size=coefficient_shape
        )
        inertia = 1.0 - cognitive - social

        numeric_velocity = (
            inertia * numeric_velocity
            + cognitive * (personal_numeric - numeric)
            + social * (global_numeric[None, :, :] - numeric)
        )
        numeric = numeric + numeric_velocity
        for column_index in range(self._numeric_data.shape[1]):
            outside = (numeric[:, :, column_index] < self._numeric_lower[column_index]) | (
                numeric[:, :, column_index] > self._numeric_upper[column_index]
            )
            numeric[:, :, column_index][outside] = self._numeric_means[column_index]

        categorical_velocity = (
            inertia * categorical_velocity
            + cognitive * (personal_categorical != categorical)
            + social * (global_categorical[None, :, :] != categorical)
        )
        transfer = np.abs(
            categorical_velocity / np.sqrt(categorical_velocity**2 + 1.0)
        )
        change = rng.uniform(0.0, 1.0, size=categorical.shape) < transfer
        for column_index, levels in enumerate(self._categorical_levels):
            random_categories = rng.integers(0, len(levels), size=categorical.shape[:2])
            categorical[:, :, column_index] = np.where(
                change[:, :, column_index],
                random_categories,
                categorical[:, :, column_index],
            )
        return numeric, categorical, numeric_velocity, categorical_velocity

    def _prepare_arrays(self) -> None:
        """Encode values and losslessly group duplicate QID tuples for speed."""

        data = self.formulation.data
        self._n_records = len(data)
        if self.formulation.schema.numeric_qids:
            self._numeric_data = data.loc[
                :, self.formulation.schema.numeric_qids
            ].to_numpy(dtype=float)
        else:
            self._numeric_data = np.empty((self._n_records, 0), dtype=float)
        self._numeric_lower = self._numeric_data.min(axis=0, initial=np.inf)
        self._numeric_upper = self._numeric_data.max(axis=0, initial=-np.inf)
        self._numeric_means = self._numeric_data.mean(axis=0)
        self._numeric_levels = [
            pd.unique(self._numeric_data[:, index])
            for index in range(self._numeric_data.shape[1])
        ]

        categorical_columns: list[np.ndarray] = []
        self._categorical_levels: list[np.ndarray] = []
        for column in self.formulation.schema.categorical_qids:
            codes, levels = pd.factorize(data[column], sort=False)
            categorical_columns.append(codes)
            self._categorical_levels.append(np.asarray(levels))
        self._categorical_data = (
            np.column_stack(categorical_columns)
            if categorical_columns
            else np.empty((self._n_records, 0), dtype=int)
        )

        profile_ids = data.groupby(
            list(self.formulation.schema.qids),
            dropna=False,
            observed=True,
            sort=False,
        ).ngroup().to_numpy(dtype=int)
        self._record_profile_ids = profile_ids
        self._n_profiles = int(profile_ids.max()) + 1
        self._profile_sizes = np.bincount(
            profile_ids, minlength=self._n_profiles
        ).astype(int)
        _, first_profile_rows = np.unique(profile_ids, return_index=True)
        self._numeric_profiles = self._numeric_data[first_profile_rows]
        self._categorical_profiles = self._categorical_data[first_profile_rows]

        sensitive_profile_indices: list[np.ndarray] = []
        sensitive_state_indices: list[np.ndarray] = []
        sensitive_state_counts: list[np.ndarray] = []
        state_offsets = [0]
        for column in self.formulation.schema.sensitive_attributes:
            codes, levels = pd.factorize(data[column], sort=False)
            n_levels = len(levels)
            profile_counts = np.bincount(
                profile_ids * n_levels + codes,
                minlength=self._n_profiles * n_levels,
            ).reshape(self._n_profiles, n_levels)
            profile_index, state_index = np.nonzero(profile_counts)
            sensitive_profile_indices.append(profile_index)
            sensitive_state_indices.append(state_index + state_offsets[-1])
            sensitive_state_counts.append(profile_counts[profile_index, state_index])
            state_offsets.append(state_offsets[-1] + n_levels)

        self._sensitive_profile_indices = np.concatenate(
            sensitive_profile_indices
        ).astype(np.intp, copy=False)
        self._sensitive_state_indices = np.concatenate(
            sensitive_state_indices
        ).astype(np.intp, copy=False)
        self._sensitive_state_counts = np.concatenate(
            sensitive_state_counts
        ).astype(np.int64, copy=False)
        self._sensitive_state_offsets = np.asarray(state_offsets, dtype=np.intp)
        self._sensitive_state_count = int(state_offsets[-1])

        self._denominator = self.formulation.original_deviation
        if self._denominator <= 0.0:
            raise ValueError("Equation (14) must be positive for PSO optimization.")

    def _centroid_frame(
        self,
        numeric_centroids: np.ndarray,
        categorical_centroids: np.ndarray,
    ) -> pd.DataFrame:
        """Decode a particle into paper-ordered q_cj centroid values."""

        columns: dict[str, np.ndarray] = {}
        for index, column in enumerate(self.formulation.schema.numeric_qids):
            columns[column] = numeric_centroids[:, index]
        for index, column in enumerate(self.formulation.schema.categorical_qids):
            columns[column] = self._categorical_levels[index][
                categorical_centroids[:, index]
            ]
        return pd.DataFrame(columns, columns=self.formulation.schema.qids)

    @staticmethod
    def _assert_fast_metrics_match(
        candidate: _Candidate,
        metrics: MOOBAMMetrics,
    ) -> None:
        """Confirm optimized array calculations equal the reference formulation."""

        if not np.isclose(candidate.information_loss, metrics.information_loss):
            raise AssertionError("Fast and reference information-loss calculations differ.")
        if not np.isclose(candidate.entropy_sum, metrics.entropy_sum):
            raise AssertionError("Fast and reference entropy calculations differ.")
        for column, value in candidate.entropy_by_sa.items():
            if not np.isclose(value, metrics.entropy_by_sensitive_attribute[column]):
                raise AssertionError(f"Fast and reference entropy differ for {column!r}.")
