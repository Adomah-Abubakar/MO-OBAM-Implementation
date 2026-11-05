"""Exact small-data solver used to verify the published MO-OBAM formulation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .formulation import MOOBAMFormulation, MOOBAMMetrics
from .schema import MOOBAMConfig, MOOBAMSchema


class ExactSearchLimitError(RuntimeError):
    """Raised before an exhaustive search that exceeds its explicit safety limit."""


@dataclass(frozen=True)
class MOOBAMResult:
    """Contain an exact solution and provide reproducible CSV/JSON exports."""

    membership: np.ndarray
    centroids: pd.DataFrame
    anonymized_data: pd.DataFrame
    metrics: MOOBAMMetrics
    feasible_partitions_evaluated: int
    schema: MOOBAMSchema
    config: MOOBAMConfig

    def save(self, output_directory: str | Path, prefix: str = "mo_obam") -> None:
        """Write the anonymized data, assignments, centroids, and summary."""

        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.anonymized_data.to_csv(destination / f"{prefix}_anonymized.csv", index=False)
        membership = pd.DataFrame(
            self.membership,
            columns=[f"cluster_{index}" for index in range(self.membership.shape[1])],
        )
        membership.insert(0, "record_position", np.arange(len(membership)))
        membership.to_csv(destination / f"{prefix}_membership.csv", index=False)

        centroids = self.centroids.copy()
        centroids.insert(0, "cluster", np.arange(len(centroids)))
        centroids.to_csv(destination / f"{prefix}_centroids.csv", index=False)

        summary = {
            "metrics": asdict(self.metrics),
            "feasible_partitions_evaluated": self.feasible_partitions_evaluated,
            "schema": asdict(self.schema),
            "config": asdict(self.config),
        }
        (destination / f"{prefix}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


class ExactMOOBAMSolver:
    """Solve MO-OBAM exactly by enumerating every feasible unlabeled partition.

    This solver exists to verify the active arXiv v2 objective and constraints.
    Exhaustive enumeration is not a scalable substitute for Algorithm 1's PSO.
    """

    def __init__(
        self,
        formulation: MOOBAMFormulation,
        max_records: int = 12,
    ) -> None:
        self.formulation = formulation
        self.max_records = max_records

    def solve(self) -> MOOBAMResult:
        """Return the globally best v2 objective within the full feasible set."""

        if self.formulation.n_records > self.max_records:
            raise ExactSearchLimitError(
                "Exact partition enumeration is intentionally limited to "
                f"{self.max_records} records; received {self.formulation.n_records}. "
                "Use the PSO implementation for larger data."
            )

        feasible_count = 0
        best: tuple[float, np.ndarray, pd.DataFrame, MOOBAMMetrics] | None = None

        for assignments in self._feasible_assignments():
            membership = self.formulation.membership_from_assignments(assignments)
            centroids = self.formulation.optimal_centroids(membership)
            metrics = self.formulation.evaluate(membership, centroids)
            feasible_count += 1
            if best is None or metrics.objective < best[0]:
                best = (metrics.objective, membership, centroids, metrics)

        if best is None:
            raise RuntimeError("No assignment satisfies Equations (9) and (20).")
        _, membership, centroids, metrics = best
        return MOOBAMResult(
            membership=membership,
            centroids=centroids,
            anonymized_data=self.formulation.anonymize(membership, centroids),
            metrics=metrics,
            feasible_partitions_evaluated=feasible_count,
            schema=self.formulation.schema,
            config=self.formulation.config,
        )

    def _feasible_assignments(self) -> Iterator[np.ndarray]:
        """Generate each feasible partition once using restricted-growth labels."""

        n = self.formulation.n_records
        n_clusters = self.formulation.config.n_clusters
        k = self.formulation.config.k
        assignments = np.zeros(n, dtype=int)
        counts = np.zeros(n_clusters, dtype=int)
        counts[0] = 1

        def search(position: int, maximum_label: int) -> Iterator[np.ndarray]:
            remaining = n - position
            deficits = int(np.maximum(k - counts, 0).sum())
            if deficits > remaining:
                return
            if position == n:
                if maximum_label == n_clusters - 1 and np.all(counts >= k):
                    yield assignments.copy()
                return

            largest_allowed = min(maximum_label + 1, n_clusters - 1)
            for label in range(largest_allowed + 1):
                assignments[position] = label
                counts[label] += 1
                yield from search(position + 1, max(maximum_label, label))
                counts[label] -= 1

        yield from search(1, 0)
