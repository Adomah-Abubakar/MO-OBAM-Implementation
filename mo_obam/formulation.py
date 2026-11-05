"""Direct implementation of the MO-OBAM arXiv v2 formulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import MOOBAMConfig, MOOBAMSchema


class UndefinedPaperMetricError(ValueError):
    """Raised when a denominator in the published equations is zero."""


@dataclass(frozen=True)
class MOOBAMMetrics:
    """Values produced by the active objective in arXiv v2."""

    information_loss: float
    entropy_by_sensitive_attribute: dict[str, float]
    entropy_sum: float
    objective: float


class MOOBAMFormulation:
    """Evaluate MO-OBAM centroids and memberships against arXiv v2.

    The input frame is copied and never modified. Missing values are rejected
    because the paper does not define their treatment in any equation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        schema: MOOBAMSchema,
        config: MOOBAMConfig,
    ) -> None:
        self.data = data.copy(deep=True)
        self.schema = schema
        self.config = config
        self._validate_input()
        self._denominator = self._calculate_original_deviation()

    @property
    def n_records(self) -> int:
        """Return the number of records, denoted by n in the paper."""

        return len(self.data)

    @property
    def original_deviation(self) -> float:
        """Return the fixed Equation (14) denominator for this dataset."""

        return self._denominator

    def membership_from_assignments(self, assignments: np.ndarray) -> np.ndarray:
        """Convert cluster labels to the binary w_ic variables from Equation (9)."""

        labels = np.asarray(assignments)
        if labels.shape != (self.n_records,):
            raise ValueError(f"assignments must have shape ({self.n_records},).")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("assignments must contain integer cluster labels.")
        if np.any(labels < 0) or np.any(labels >= self.config.n_clusters):
            raise ValueError("assignments contain a cluster outside the configured range.")

        membership = np.zeros((self.n_records, self.config.n_clusters), dtype=np.int8)
        membership[np.arange(self.n_records), labels] = 1
        return membership

    def validate_membership(self, membership: np.ndarray) -> np.ndarray:
        """Enforce the assignment (9) and hard k-anonymity (20) constraints."""

        values = np.asarray(membership)
        expected = (self.n_records, self.config.n_clusters)
        if values.shape != expected:
            raise ValueError(f"membership must have shape {expected}.")
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("Equation (9) requires binary membership values.")
        if not np.all(values.sum(axis=1) == 1):
            raise ValueError("Equation (9) requires each record to belong to one cluster.")

        cluster_sizes = values.sum(axis=0)
        if np.any(cluster_sizes < self.config.k):
            raise ValueError(
                "Equation (20) requires every configured cluster, including empty "
                f"clusters, to contain at least k={self.config.k} records."
            )
        return values.astype(np.int8, copy=False)

    def optimal_centroids(self, membership: np.ndarray) -> pd.DataFrame:
        """Derive the information-loss-minimizing q_cj values for fixed w_ic.

        Equations (10)-(13) are minimized by the arithmetic mean for a numerical
        QI and by a mode for a categorical QI. Entropy does not depend on q_cj, so
        eliminating q this way does not change the optimum for fixed membership.
        If categorical modes tie, the first observed mode is used; tied modes
        have identical categorical-deviation values.
        """

        w = self.validate_membership(membership)
        rows: list[dict[str, object]] = []
        for cluster in range(self.config.n_clusters):
            cluster_data = self.data.loc[w[:, cluster] == 1]
            centroid: dict[str, object] = {}
            for column in self.schema.numeric_qids:
                centroid[column] = float(cluster_data[column].mean())
            for column in self.schema.categorical_qids:
                centroid[column] = self._first_mode(cluster_data[column])
            rows.append(centroid)
        return pd.DataFrame(rows, columns=self.schema.qids)

    def deviation_after_anonymization(
        self,
        membership: np.ndarray,
        centroids: pd.DataFrame,
    ) -> float:
        """Calculate the total post-anonymization deviation in Equation (13)."""

        w = self.validate_membership(membership)
        q = self._validate_centroids(centroids)
        total = 0.0

        if self.schema.numeric_qids:
            numeric = self.data.loc[:, self.schema.numeric_qids].to_numpy(dtype=float)
            numeric_q = q.loc[:, self.schema.numeric_qids].to_numpy(dtype=float)
            squared = (numeric[:, None, :] - numeric_q[None, :, :]) ** 2
            total += float(np.sum(squared * w[:, :, None]))

        if self.schema.categorical_qids:
            categorical = self.data.loc[:, self.schema.categorical_qids].to_numpy()
            categorical_q = q.loc[:, self.schema.categorical_qids].to_numpy()
            mismatch = categorical[:, None, :] != categorical_q[None, :, :]
            total += float(np.sum(mismatch * w[:, :, None]))
        return total

    def information_loss(
        self,
        membership: np.ndarray,
        centroids: pd.DataFrame,
    ) -> float:
        """Calculate IL as Equation (13) divided by Equation (14)."""

        if self._denominator == 0.0:
            raise UndefinedPaperMetricError(
                "Equation (14) is zero, so the paper's information-loss ratio is undefined."
            )
        numerator = self.deviation_after_anonymization(membership, centroids)
        return numerator / self._denominator

    def cluster_entropies(self, membership: np.ndarray) -> pd.DataFrame:
        """Calculate each cluster/SA entropy using Equations (15)-(17)."""

        w = self.validate_membership(membership)
        result = np.zeros(
            (self.config.n_clusters, len(self.schema.sensitive_attributes)),
            dtype=float,
        )
        for cluster in range(self.config.n_clusters):
            cluster_mask = w[:, cluster] == 1
            cluster_size = int(cluster_mask.sum())
            for column_index, column in enumerate(self.schema.sensitive_attributes):
                counts = self.data.loc[cluster_mask, column].value_counts(dropna=False)
                probabilities = counts.to_numpy(dtype=float) / cluster_size
                result[cluster, column_index] = float(
                    -np.sum(probabilities * np.log2(probabilities))
                )
        return pd.DataFrame(result, columns=self.schema.sensitive_attributes)

    def entropy_by_sensitive_attribute(
        self,
        membership: np.ndarray,
    ) -> dict[str, float]:
        """Take the minimum cluster entropy per SA as Equation (18)."""

        entropies = self.cluster_entropies(membership)
        return {
            column: float(entropies[column].min())
            for column in self.schema.sensitive_attributes
        }

    def entropy_sum(self, membership: np.ndarray) -> float:
        """Calculate Equation (19), the sum of per-SA minimum entropies."""

        return float(sum(self.entropy_by_sensitive_attribute(membership).values()))

    def objective(self, information_loss: float, entropy_sum: float) -> float:
        """Return the active arXiv v2 objective: IL - lambda * Equation (19)."""

        if information_loss < 0.0:
            raise ValueError("Information loss cannot be negative.")
        if entropy_sum < 0.0:
            raise ValueError("Entropy cannot be negative.")
        return information_loss - self.config.lambda_weight * entropy_sum

    def evaluate(
        self,
        membership: np.ndarray,
        centroids: pd.DataFrame,
    ) -> MOOBAMMetrics:
        """Evaluate a feasible pair of paper decision variables q and w."""

        information_loss = self.information_loss(membership, centroids)
        entropy_by_sa = self.entropy_by_sensitive_attribute(membership)
        entropy_sum = float(sum(entropy_by_sa.values()))
        objective = self.objective(information_loss, entropy_sum)
        return MOOBAMMetrics(
            information_loss=information_loss,
            entropy_by_sensitive_attribute=entropy_by_sa,
            entropy_sum=entropy_sum,
            objective=objective,
        )

    def anonymize(
        self,
        membership: np.ndarray,
        centroids: pd.DataFrame,
        cluster_column: str = "cluster",
    ) -> pd.DataFrame:
        """Replace only QID values by q_cj and preserve every other column."""

        if cluster_column in self.data.columns:
            raise ValueError(
                f"Cannot add cluster labels because {cluster_column!r} already exists."
            )
        w = self.validate_membership(membership)
        q = self._validate_centroids(centroids)
        assignments = np.argmax(w, axis=1)
        anonymized = self.data.copy(deep=True)
        for column in self.schema.qids:
            replacement = q[column].to_numpy()[assignments]
            anonymized[column] = replacement
        anonymized[cluster_column] = assignments
        return anonymized

    def _validate_input(self) -> None:
        if not self.data.columns.is_unique:
            raise ValueError("The input data must have unique column names.")
        missing = sorted(set(self.schema.protected_columns) - set(self.data.columns))
        if missing:
            raise ValueError(f"Columns missing from the input data: {missing}")
        if self.n_records == 0:
            raise ValueError("The input data must contain at least one record.")
        if self.config.k * self.config.n_clusters > self.n_records:
            raise ValueError("Equation (20) is infeasible because k * n_clusters > n.")
        if self.data.loc[:, self.schema.protected_columns].isna().any().any():
            raise ValueError(
                "The paper does not define missing-value handling for QIDs or SAs; "
                "resolve it before applying MO-OBAM."
            )
        for column in self.schema.numeric_qids:
            if not pd.api.types.is_numeric_dtype(self.data[column]):
                raise TypeError(f"Numeric QID {column!r} is not numeric.")
            values = self.data[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Numeric QID {column!r} contains a non-finite value.")

    def _validate_centroids(self, centroids: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(centroids, pd.DataFrame):
            raise TypeError("centroids must be a pandas DataFrame.")
        if len(centroids) != self.config.n_clusters:
            raise ValueError("centroids must contain one row per configured cluster.")
        if list(centroids.columns) != list(self.schema.qids):
            raise ValueError(
                "centroid columns must exactly match schema.qids in paper order."
            )
        if centroids.isna().any().any():
            raise ValueError("Centroids cannot contain missing values.")
        for column in self.schema.numeric_qids:
            values = pd.to_numeric(centroids[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Centroid {column!r} contains a non-finite value.")
        return centroids.reset_index(drop=True)

    def _calculate_original_deviation(self) -> float:
        """Calculate the data's original deviation in Equation (14)."""

        total = 0.0
        for column in self.schema.numeric_qids:
            values = self.data[column].to_numpy(dtype=float)
            total += float(np.sum((values - values.mean()) ** 2))
        for column in self.schema.categorical_qids:
            mode = self._first_mode(self.data[column])
            total += float(np.sum(self.data[column].to_numpy() != mode))
        return total

    @staticmethod
    def _first_mode(values: pd.Series) -> object:
        """Return the first observed value among equally frequent modes."""

        counts = values.value_counts(sort=False, dropna=False)
        maximum = counts.max()
        modes = set(counts.index[counts == maximum].tolist())
        for value in values.tolist():
            if value in modes:
                return value
        raise RuntimeError("Cannot calculate a mode for an empty series.")
