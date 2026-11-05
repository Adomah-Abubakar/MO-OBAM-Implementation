"""Reproduce the 45,222-record Adult dataset reported in the MO-OBAM paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ADULT_COLUMNS = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
)

ADULT_INTEGER_COLUMNS = (
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
)


@dataclass(frozen=True)
class AdultPreparationReport:
    """Record the raw and retained row counts needed to reproduce Table III."""

    raw_training_rows: int
    raw_test_rows: int
    removed_training_rows: int
    removed_test_rows: int
    prepared_rows: int
    prepared_columns: int


class AdultDataPreparer:
    """Prepare the UCI Adult train/test files used by the MO-OBAM study.

    The UCI documentation and the paper report 30,162 retained training rows
    and 15,060 retained test rows after records containing ``?`` are removed.
    No imputation, category recoding, or duplicate removal is performed.
    """

    EXPECTED_TRAIN_ROWS = 30_162
    EXPECTED_TEST_ROWS = 15_060
    EXPECTED_TOTAL_ROWS = 45_222

    def prepare(self, data_directory: str | Path) -> tuple[pd.DataFrame, AdultPreparationReport]:
        """Load, clean as documented, combine, and validate the Adult files."""

        directory = Path(data_directory)
        training = self._read_file(directory / "adult.data", is_test=False)
        test = self._read_file(directory / "adult.test", is_test=True)

        raw_training_rows = len(training)
        raw_test_rows = len(test)
        clean_training = training.dropna(axis=0, how="any").copy()
        clean_test = test.dropna(axis=0, how="any").copy()

        if len(clean_training) != self.EXPECTED_TRAIN_ROWS:
            raise ValueError(
                f"Expected {self.EXPECTED_TRAIN_ROWS} clean training rows; "
                f"found {len(clean_training)}."
            )
        if len(clean_test) != self.EXPECTED_TEST_ROWS:
            raise ValueError(
                f"Expected {self.EXPECTED_TEST_ROWS} clean test rows; found {len(clean_test)}."
            )

        prepared = pd.concat([clean_training, clean_test], ignore_index=True)
        for column in ADULT_INTEGER_COLUMNS:
            prepared[column] = prepared[column].astype("int64")

        if prepared.shape != (self.EXPECTED_TOTAL_ROWS, len(ADULT_COLUMNS)):
            raise ValueError(
                "Prepared Adult data does not match the paper's 45,222 by 15 shape: "
                f"found {prepared.shape}."
            )

        report = AdultPreparationReport(
            raw_training_rows=raw_training_rows,
            raw_test_rows=raw_test_rows,
            removed_training_rows=raw_training_rows - len(clean_training),
            removed_test_rows=raw_test_rows - len(clean_test),
            prepared_rows=len(prepared),
            prepared_columns=len(prepared.columns),
        )
        return prepared, report

    @staticmethod
    def _read_file(path: Path, is_test: bool) -> pd.DataFrame:
        """Read one headerless UCI file while preserving its documented values."""

        if not path.is_file():
            raise FileNotFoundError(f"Adult source file not found: {path}")
        data = pd.read_csv(
            path,
            names=ADULT_COLUMNS,
            header=None,
            comment="|" if is_test else None,
            skipinitialspace=True,
            skip_blank_lines=True,
            na_values=["?"],
            keep_default_na=False,
        )
        text_columns = [
            column
            for column in data.columns
            if pd.api.types.is_string_dtype(data[column].dtype)
        ]
        for column in text_columns:
            data[column] = data[column].str.strip()
        if is_test:
            data["income"] = data["income"].str.removesuffix(".")
        data.replace("?", pd.NA, inplace=True)
        return data
