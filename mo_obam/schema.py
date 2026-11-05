"""Validated column roles and hyperparameters for MO-OBAM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MOOBAMSchema:
    """Identify the numeric QIs, categorical QIs, and sensitive attributes."""

    numeric_qids: tuple[str, ...]
    categorical_qids: tuple[str, ...]
    sensitive_attributes: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (
            self.numeric_qids,
            self.categorical_qids,
            self.sensitive_attributes,
        )
        if not self.numeric_qids and not self.categorical_qids:
            raise ValueError("At least one quasi-identifier is required.")
        if not self.sensitive_attributes:
            raise ValueError("At least one sensitive attribute is required.")

        all_columns = [column for group in groups for column in group]
        if len(all_columns) != len(set(all_columns)):
            raise ValueError("QID and sensitive-attribute column roles must be disjoint.")

    @property
    def qids(self) -> tuple[str, ...]:
        """Return QIDs in the numerical-then-categorical order used by the paper."""

        return self.numeric_qids + self.categorical_qids

    @property
    def protected_columns(self) -> tuple[str, ...]:
        """Return every column needed to evaluate the paper formulation."""

        return self.qids + self.sensitive_attributes


@dataclass(frozen=True)
class MOOBAMConfig:
    """Store the three MO-OBAM hyperparameters described in Section V-B."""

    k: int
    n_clusters: int
    lambda_weight: float

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("The paper states that k must be at least 2.")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if not 0.0 <= self.lambda_weight <= 1.0:
            raise ValueError("The paper restricts lambda_weight to [0, 1].")
