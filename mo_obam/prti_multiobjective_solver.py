"""Additive multi-objective PSO extension for PRTI-GH.

The scalar, paper-faithful optimizer in :mod:`mo_obam.pso_solver` remains the
baseline.  This module reuses its particle representation and update mechanics,
but retains a bounded Pareto archive instead of choosing one global best with a
fixed lambda value.  Lambda is therefore analysed after optimization rather
than requiring a separate PSO run for every weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .formulation import MOOBAMFormulation
from .pso_solver import (
    MOOBAMParticleSwarmOptimizer,
    NoFeasibleParticleError,
    PSOConfig,
    _Candidate,
)


MultiObjectiveProgress = Callable[[int, int, int, int, int], None]


@dataclass(frozen=True)
class PRTIParetoCandidate:
    """One non-dominated hard-k-feasible solution returned by the extension."""

    candidate_index: int
    membership: np.ndarray
    centroids: pd.DataFrame
    information_loss: float
    entropy_sum: float
    entropy_by_sensitive_attribute: dict[str, float]
    cluster_sizes: tuple[int, ...]

    def objective(self, lambda_weight: float) -> float:
        """Evaluate the original scalar objective for any lambda in [0, 1]."""

        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError("lambda_weight must be in [0, 1].")
        return self.information_loss - lambda_weight * self.entropy_sum


@dataclass(frozen=True)
class PRTIMultiObjectiveResult:
    """Pareto archive and search diagnostics for one k/C/seed run."""

    candidates: tuple[PRTIParetoCandidate, ...]
    feasible_evaluations: int
    infeasible_evaluations: int
    archive_size_limit: int


@dataclass
class _ArchiveEntry:
    numeric_centroids: np.ndarray
    categorical_centroids: np.ndarray
    candidate: _Candidate


class PRTIMultiObjectiveParticleSwarm(MOOBAMParticleSwarmOptimizer):
    """Search IL/entropy jointly while preserving the existing PSO baseline."""

    def __init__(
        self,
        formulation: MOOBAMFormulation,
        pso_config: PSOConfig | None = None,
        archive_size: int = 100,
        initialization_attempts: int = 20,
    ) -> None:
        if archive_size < 2:
            raise ValueError("archive_size must be at least 2.")
        if initialization_attempts < 1:
            raise ValueError("initialization_attempts must be at least 1.")
        super().__init__(formulation, pso_config)
        self.archive_size = archive_size
        self.initialization_attempts = initialization_attempts

    @staticmethod
    def dominates(left: _Candidate, right: _Candidate) -> bool:
        """Return whether left is no worse in both objectives and better in one."""

        no_worse = (
            left.information_loss <= right.information_loss
            and left.entropy_sum >= right.entropy_sum
        )
        strictly_better = (
            left.information_loss < right.information_loss
            or left.entropy_sum > right.entropy_sum
        )
        return no_worse and strictly_better

    def solve_multiobjective(
        self,
        progress: MultiObjectiveProgress | None = None,
    ) -> PRTIMultiObjectiveResult:
        """Run one Pareto PSO search and return its bounded archive."""

        rng = np.random.default_rng(np.random.SeedSequence(self.pso_config.seed))
        numeric, categorical = self._initialize_particles(rng)
        numeric_velocity = np.zeros_like(numeric, dtype=float)
        categorical_velocity = np.zeros_like(categorical, dtype=float)
        personal_numeric = numeric.copy()
        personal_categorical = categorical.copy()
        personal_candidates: list[_Candidate | None] = [
            None for _ in range(self.pso_config.population_size)
        ]
        archive: list[_ArchiveEntry] = []
        feasible_evaluations = 0
        infeasible_evaluations = 0

        iteration = 0
        initialization_attempt = 1
        while iteration < self.pso_config.iterations:
            for particle_index in range(self.pso_config.population_size):
                candidate = self._evaluate_particle(
                    numeric[particle_index], categorical[particle_index]
                )
                if candidate is None:
                    infeasible_evaluations += 1
                    continue
                feasible_evaluations += 1
                incumbent = personal_candidates[particle_index]
                if (
                    incumbent is None
                    or self.dominates(candidate, incumbent)
                    or (
                        not self.dominates(incumbent, candidate)
                        and bool(rng.integers(0, 2))
                    )
                ):
                    personal_candidates[particle_index] = candidate
                    personal_numeric[particle_index] = numeric[particle_index]
                    personal_categorical[particle_index] = categorical[particle_index]
                self._add_to_archive(
                    archive,
                    numeric[particle_index],
                    categorical[particle_index],
                    candidate,
                )

            if not archive:
                if initialization_attempt >= self.initialization_attempts:
                    raise NoFeasibleParticleError(
                        "The multi-objective stage found no particle satisfying hard "
                        f"k-anonymity after {initialization_attempt} independent "
                        "initial populations."
                    )
                initialization_attempt += 1
                numeric, categorical = self._initialize_particles(rng)
                numeric_velocity = np.zeros_like(numeric, dtype=float)
                categorical_velocity = np.zeros_like(categorical, dtype=float)
                personal_numeric = numeric.copy()
                personal_categorical = categorical.copy()
                personal_candidates = [
                    None for _ in range(self.pso_config.population_size)
                ]
                continue
            if progress is not None:
                progress(
                    iteration + 1,
                    self.pso_config.iterations,
                    len(archive),
                    feasible_evaluations,
                    infeasible_evaluations,
                )
            leader = self._select_leader(archive, rng)
            numeric, categorical, numeric_velocity, categorical_velocity = (
                self._update_particles(
                    numeric,
                    categorical,
                    numeric_velocity,
                    categorical_velocity,
                    personal_numeric,
                    personal_categorical,
                    leader.numeric_centroids,
                    leader.categorical_centroids,
                    rng,
                )
            )
            iteration += 1

        ordered = sorted(
            archive,
            key=lambda entry: (
                entry.candidate.information_loss,
                -entry.candidate.entropy_sum,
            ),
        )
        candidates: list[PRTIParetoCandidate] = []
        for candidate_index, entry in enumerate(ordered):
            membership = self.formulation.membership_from_assignments(
                entry.candidate.assignments
            )
            candidates.append(
                PRTIParetoCandidate(
                    candidate_index=candidate_index,
                    membership=membership,
                    centroids=self._centroid_frame(
                        entry.numeric_centroids,
                        entry.categorical_centroids,
                    ),
                    information_loss=entry.candidate.information_loss,
                    entropy_sum=entry.candidate.entropy_sum,
                    entropy_by_sensitive_attribute=dict(entry.candidate.entropy_by_sa),
                    cluster_sizes=tuple(
                        int(value) for value in membership.sum(axis=0)
                    ),
                )
            )
        return PRTIMultiObjectiveResult(
            candidates=tuple(candidates),
            feasible_evaluations=feasible_evaluations,
            infeasible_evaluations=infeasible_evaluations,
            archive_size_limit=self.archive_size,
        )

    def _add_to_archive(
        self,
        archive: list[_ArchiveEntry],
        numeric_centroids: np.ndarray,
        categorical_centroids: np.ndarray,
        candidate: _Candidate,
    ) -> None:
        if any(self.dominates(entry.candidate, candidate) for entry in archive):
            return
        archive[:] = [
            entry
            for entry in archive
            if not self.dominates(candidate, entry.candidate)
        ]
        for entry in archive:
            if (
                np.array_equal(
                    entry.candidate.assignments, candidate.assignments
                )
                and np.isclose(
                    entry.candidate.information_loss, candidate.information_loss
                )
                and np.isclose(entry.candidate.entropy_sum, candidate.entropy_sum)
            ):
                return
        archive.append(
            _ArchiveEntry(
                numeric_centroids=numeric_centroids.copy(),
                categorical_centroids=categorical_centroids.copy(),
                candidate=candidate,
            )
        )
        while len(archive) > self.archive_size:
            distances = self._crowding_distances(archive)
            finite = [
                (distance, index)
                for index, distance in enumerate(distances)
                if np.isfinite(distance)
            ]
            remove_index = min(finite)[1] if finite else len(archive) // 2
            archive.pop(remove_index)

    @staticmethod
    def _crowding_distances(archive: list[_ArchiveEntry]) -> np.ndarray:
        count = len(archive)
        if count <= 2:
            return np.full(count, np.inf)
        values = np.asarray(
            [
                [entry.candidate.information_loss, entry.candidate.entropy_sum]
                for entry in archive
            ],
            dtype=float,
        )
        distances = np.zeros(count, dtype=float)
        for objective in range(values.shape[1]):
            order = np.argsort(values[:, objective], kind="stable")
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            span = values[order[-1], objective] - values[order[0], objective]
            if span == 0.0:
                continue
            for position in range(1, count - 1):
                index = order[position]
                distances[index] += (
                    values[order[position + 1], objective]
                    - values[order[position - 1], objective]
                ) / span
        return distances

    def _select_leader(
        self,
        archive: list[_ArchiveEntry],
        rng: np.random.Generator,
    ) -> _ArchiveEntry:
        distances = self._crowding_distances(archive)
        finite_distances = distances[np.isfinite(distances)]
        replacement = (
            float(finite_distances.max() + 1.0)
            if finite_distances.size
            else 1.0
        )
        weights = np.where(np.isfinite(distances), distances + 1e-12, replacement)
        weights = weights / weights.sum()
        return archive[int(rng.choice(len(archive), p=weights))]
