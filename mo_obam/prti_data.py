"""Lossless, auditable preparation of the PRTI-GH workbook for MO-OBAM.

This module contains dataset-specific column roles and representation rules.
It does not implement anonymization mathematics. Categorical codes are only
identifiers: MO-OBAM still treats those columns with categorical mismatch,
never as ordered numerical distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .schema import MOOBAMSchema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRTI_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "PRTI-GH-study"
    / "PRTI-GH-study.xls"
)
EXPECTED_PRTI_SHAPE = (1_485, 106)
EXPECTED_PRTI_SHA256 = (
    "0deb1fc54a3dccba3a4c0339696ed1fb4f124a0349afde9f06e4b79dbf89a37e"
)
MISSING_CATEGORY_CODE = 0
MISSING_VALUE_TOKEN = "<MISSING>"
# Known source characteristics are preserved and documented, never auto-cleaned.
KNOWN_EXACT_DUPLICATE_COLUMNS = (("xray", "xray1"),)


# These are the 29 QIDs agreed for the first PRTI-GH MO-OBAM baseline.
# Religion is a QID in this threat model. Temperature and oxygen saturation
# remain SAs unless a later, explicitly documented scenario changes the roles.
PRTI_QIDS = (
    "zonesGhana",
    "region",
    "district",
    "remotenessCommunity",
    "a3Hospital",
    "childsAge",
    "sexOfTheChild",
    "educationLevel",
    "religiousAffiliation",
    "months",
    "year",
    "weatherSeason",
    "dayOfWeek",
    "timeOfDay",
    "modePrehospitalTransport",
    "typeOfRoadUser",
    "carSeatBeltUse",
    "typeOfRoad",
    "natureOfRoad",
    "presenceOfSupervisionsRoadGuideF",
    "crashType",
    "circumstancesOfInjury",
    "mechanismOfInjury",
    "whatIsTheActualCauseTheAccident",
    "overtaking",
    "perceivedSpeeding",
    "vehicleCrossLane",
    "driverWasTiredSleeping",
    "vehicleTyreBust",
)
PRTI_NUMERIC_QIDS = ("childsAge", "year")
PRTI_CATEGORICAL_QIDS = tuple(
    column for column in PRTI_QIDS if column not in PRTI_NUMERIC_QIDS
)


# Regular camel-case conversion cannot reliably split these source headers.
# The mapping is copied from the previously reviewed PRTI preprocessing step.
MANUAL_COLUMN_RENAMES = {
    "SexoftheChild": "sexOfTheChild",
    "Carseatbeltuse": "carSeatBeltUse",
    "Presenceofsupervisionsroadguidef": "presenceOfSupervisionsRoadGuideF",
    "Circumstancesofinjury": "circumstancesOfInjury",
    "Whatistheactualcausetheaccident": "whatIsTheActualCauseTheAccident",
    "Pesceived_speeding": "perceivedSpeeding",
    "Vehiclecrosslane": "vehicleCrossLane",
    "Driverwastiredsleeping": "driverWasTiredSleeping",
    "Vehicletyrebust": "vehicleTyreBust",
    "Radiologicinvestigations": "radiologicInvestigations",
    "CTscan": "ctScan",
    "BodypartsexaminedXray": "bodyPartsExaminedXray",
    "BodypartsexaminedCTscan": "bodyPartsExaminedCtScan",
    "Sufferedmultipleinjuries": "sufferedMultipleInjuries",
    "Commoninjuriessuffered": "commonInjuriesSuffered",
    "Headinjuriessuffered": "headInjuriesSuffered",
    "Facialinjuriessuffered": "facialInjuriesSuffered",
    "Neckinjuriessuffered": "neckInjuriesSuffered",
    "Abdomeninjuriessuffered": "abdomenInjuriesSuffered",
    "Upperlimbsinjuriessuffered": "upperLimbsInjuriesSuffered",
    "Lowerlimbsinjuriessuffered": "lowerLimbsInjuriesSuffered",
    "Superficial_externalinjuriessuff": "superficialExternalInjuriesSuff",
    "Musculoskeletalpain": "musculoskeletalPain",
    "Typeofheadinjuries": "typeOfHeadInjuries",
    "ScalplacerationHematoma": "scalpLacerationHematoma",
    "Intracrainalhemorrhage": "intracranialHemorrhage",
    "Epiduralhemorrhage": "epiduralHemorrhage",
    "Subduralhemorrhage": "subduralHemorrhage",
    "Subarachnoidhemorrhage": "subarachnoidHemorrhage",
    "Cerebralcontusion": "cerebralContusion",
    "Uspecifiedheadinjury": "unspecifiedHeadInjury",
    "Othersspecifyheadinjury": "othersSpecifyHeadInjury",
    "AbbreviatedinjuryscaleAIS": "abbreviatedInjuryScaleAis",
    "AISHeadNeckorspine": "aisHeadNeckOrSpine",
    "AISAbdomenamppelvic": "aisAbdomenAndPelvic",
    "AISExternalorsuperficial": "aisExternalOrSuperficial",
    "TRIAGEampMONITORING": "triageAndMonitoring",
    "Heartratebpm": "heartRateBpm",
    "Temperaturedegreecelcius": "temperatureDegreeCelsius",
    "Oxygensaturation": "oxygenSaturation",
    "IntensivecareunitICUadmission": "intensiveCareUnitIcuAdmission",
    "Majortreatmentmanagement": "majorTreatmentManagement",
    "Majorsurgical": "majorSurgical",
    "Minorsurgical": "minorSurgical",
    "Nonsurgical": "nonSurgical",
    "Nonsurgicaltreatment": "nonSurgicalTreatment",
    "Tetanusinjection": "tetanusInjection",
    "Bloodtransfusion": "bloodTransfusion",
    "IVfluids": "ivFluids",
    "PoP": "pop",
    "Admittingservice": "admittingService",
    "Emergencyserviceonly": "emergencyServiceOnly",
    "Treatmentoutcome": "treatmentOutcome",
    "LOHdays": "lohDays",
}


# Manual ordering is used only to make the exported codebook meaningful and
# stable. MO-OBAM treats all of these QIDs as categorical mismatch variables.
SEMANTIC_ORDERS = {
    "educationLevel": (
        "None/Preschool",
        "Lower Primary",
        "Upper primary",
        "Junior High",
        "Senior high",
        "Tertiary",
    ),
    "months": (
        "January",
        "Febraury",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ),
    "dayOfWeek": (
        "Monday",
        "Teusday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ),
    "timeOfDay": ("Morning", "Afternoon", "Evening", "Night"),
    "catIss": (
        "Mild (<=8)",
        "moderate(9-15)",
        "severe(16-24)",
        "Critical(>=25)",
    ),
    "binaryIss": ("Non severe", "Severe"),
    "catGlasgowOutcomeScale": (
        "Death",
        "Severe Disability",
        "Moderate Disability",
        "Good Recovery",
    ),
}


@dataclass
class PRTIPreparedData:
    """Hold source, encoded model input, mappings, audit, and metadata."""

    original: pd.DataFrame
    model_input: pd.DataFrame
    schema: MOOBAMSchema
    audit: pd.DataFrame
    category_mappings: pd.DataFrame
    decoding_maps: dict[str, dict[int, object]]
    summary: dict[str, object]


class PRTIDataAdapter:
    """Read and losslessly encode PRTI-GH without changing its source file."""

    def __init__(
        self,
        source_path: str | Path = DEFAULT_PRTI_SOURCE,
        expected_shape: tuple[int, int] | None = EXPECTED_PRTI_SHAPE,
    ) -> None:
        self.source_path = Path(source_path)
        self.expected_shape = expected_shape

    def load_source(self) -> pd.DataFrame:
        """Load the original XLS and apply the agreed canonical column names."""

        if not self.source_path.is_file():
            raise FileNotFoundError(f"PRTI-GH source workbook not found: {self.source_path}")
        data = pd.read_excel(self.source_path, engine="xlrd")
        if self.expected_shape is not None:
            unused_renames = set(MANUAL_COLUMN_RENAMES) - set(data.columns)
            if unused_renames:
                raise ValueError(
                    "Reviewed manual headers are missing from the source: "
                    f"{sorted(unused_renames)}"
                )
        data.columns = [
            MANUAL_COLUMN_RENAMES.get(column, self._to_camel_case(column))
            for column in data.columns
        ]
        if not data.columns.is_unique:
            raise ValueError("Canonical PRTI-GH column names are not unique.")
        if self.expected_shape is not None and data.shape != self.expected_shape:
            raise ValueError(
                f"Expected PRTI-GH shape {self.expected_shape}; found {data.shape}."
            )
        return data

    def prepare(self) -> PRTIPreparedData:
        """Prepare a model copy and verify that the XLS source was not modified."""

        hash_before = self.file_sha256(self.source_path)
        if self.expected_shape is not None and hash_before != EXPECTED_PRTI_SHA256:
            raise ValueError(
                "The PRTI-GH source fingerprint differs from the reviewed workbook. "
                f"Expected {EXPECTED_PRTI_SHA256}; found {hash_before}."
            )
        prepared = self.prepare_frame(self.load_source(), source_hash=hash_before)
        self._validate_known_source_characteristics(prepared.original)
        hash_after = self.file_sha256(self.source_path)
        if hash_after != hash_before:
            raise RuntimeError("The PRTI-GH source workbook changed during preparation.")
        prepared.summary["source_sha256_after"] = hash_after
        prepared.summary["source_unchanged"] = True
        return prepared

    def prepare_frame(
        self,
        source: pd.DataFrame,
        source_hash: str = "in-memory-test-frame",
    ) -> PRTIPreparedData:
        """Encode a canonical in-memory frame and validate a lossless round trip."""

        if not source.columns.is_unique:
            raise ValueError("PRTI-GH columns must be unique.")
        missing_qids = sorted(set(PRTI_QIDS) - set(source.columns))
        if missing_qids:
            raise ValueError(f"Agreed PRTI-GH QIDs are missing: {missing_qids}")

        sensitive_attributes = tuple(
            column for column in source.columns if column not in PRTI_QIDS
        )
        schema = MOOBAMSchema(
            numeric_qids=PRTI_NUMERIC_QIDS,
            categorical_qids=PRTI_CATEGORICAL_QIDS,
            sensitive_attributes=sensitive_attributes,
        )
        model_input = source.copy(deep=True)
        audit_rows: list[dict[str, object]] = []
        mapping_rows: list[dict[str, object]] = []
        decoding_maps: dict[str, dict[int, object]] = {}

        for position, column in enumerate(source.columns, start=1):
            original = source[column]
            role = self._role(column, schema)
            missing_count = int(original.isna().sum())
            distinct_count = int(original.nunique(dropna=True))

            if is_numeric_dtype(original.dtype):
                if column in schema.numeric_qids and missing_count:
                    raise ValueError(
                        f"Numeric QID {column!r} contains missing values. The paper "
                        "does not define imputation or a numerical missing distance."
                    )
                treatment = "unchanged numeric"
                if missing_count:
                    model_input[column] = original.astype(object).where(
                        original.notna(), MISSING_VALUE_TOKEN
                    )
                    treatment = "unchanged numeric; missing is an explicit SA state"
            else:
                categories = self._ordered_categories(column, original)
                encoding = {
                    value: code for code, value in enumerate(categories, start=1)
                }
                encoded = original.map(encoding).astype("Int64")
                if missing_count:
                    encoded = encoded.fillna(MISSING_CATEGORY_CODE)
                model_input[column] = encoded
                decoding_maps[column] = {
                    code: value for value, code in encoding.items()
                }
                treatment = "one-to-one categorical identifier"
                for value, code in encoding.items():
                    count = int(original.eq(value).sum())
                    mapping_rows.append(
                        {
                            "attribute": column,
                            "role": role,
                            "code": code,
                            "originalValueExact": value,
                            "displayValue": self._display_value(value),
                            "valueState": "observed value",
                            "count": count,
                            "proportion": count / len(source),
                        }
                    )
                if missing_count:
                    mapping_rows.append(
                        {
                            "attribute": column,
                            "role": role,
                            "code": MISSING_CATEGORY_CODE,
                            "originalValueExact": pd.NA,
                            "displayValue": MISSING_VALUE_TOKEN,
                            "valueState": "missing; not imputed",
                            "count": missing_count,
                            "proportion": missing_count / len(source),
                        }
                    )

            audit_rows.append(
                {
                    "position": position,
                    "attribute": column,
                    "role": role,
                    "sourceDtype": str(original.dtype),
                    "modelDtype": str(model_input[column].dtype),
                    "rows": len(source),
                    "distinctNonMissingValues": distinct_count,
                    "missingValues": missing_count,
                    "missingPercentage": missing_count / len(source),
                    "cardinalityPercentage": distinct_count / len(source),
                    "treatment": treatment,
                    "reviewFlag": self._review_flag(
                        column, role, original, distinct_count, missing_count
                    ),
                }
            )

        decoded = self.decode_model_data(model_input, source, decoding_maps)
        self._validate_round_trip(source, decoded)
        if model_input.loc[:, schema.protected_columns].isna().any().any():
            raise AssertionError("Prepared MO-OBAM input still contains missing values.")

        audit = pd.DataFrame(audit_rows)
        audit["modelDistinctStates"] = [
            int(model_input[column].nunique(dropna=False)) for column in source.columns
        ]
        audit["modelMissingValues"] = [
            int(model_input[column].isna().sum()) for column in source.columns
        ]
        audit["decodedDistinctNonMissingValues"] = [
            int(decoded[column].nunique(dropna=True)) for column in source.columns
        ]
        audit["decodedMissingValues"] = [
            int(decoded[column].isna().sum()) for column in source.columns
        ]
        audit["largestFrequencyDifferenceAfterRoundTrip"] = [
            self._largest_frequency_difference(source[column], decoded[column])
            for column in source.columns
        ]
        audit["distributionStatus"] = "PASS"
        category_mappings = pd.DataFrame(mapping_rows)
        summary = self._build_summary(
            source,
            model_input,
            schema,
            audit,
            source_hash,
        )
        summary["source"] = str(self.source_path)
        return PRTIPreparedData(
            original=source.copy(deep=True),
            model_input=model_input,
            schema=schema,
            audit=audit,
            category_mappings=category_mappings,
            decoding_maps=decoding_maps,
            summary=summary,
        )

    def decode_model_data(
        self,
        model_data: pd.DataFrame,
        original: pd.DataFrame,
        decoding_maps: dict[str, dict[int, object]],
    ) -> pd.DataFrame:
        """Restore exact labels and missing cells from an encoded model frame."""

        decoded = model_data.copy(deep=True)
        for column in original.columns:
            if column in decoding_maps:
                reverse = decoding_maps[column]
                decoded[column] = decoded[column].map(
                    lambda value: pd.NA
                    if int(value) == MISSING_CATEGORY_CODE
                    else reverse[int(value)]
                )
            elif is_numeric_dtype(original[column].dtype):
                decoded[column] = decoded[column].replace(MISSING_VALUE_TOKEN, np.nan)
                if column not in PRTI_NUMERIC_QIDS:
                    decoded[column] = pd.to_numeric(decoded[column], errors="raise")
        return decoded

    def write_audit_outputs(
        self,
        output_directory: str | Path,
        prepared: PRTIPreparedData | None = None,
    ) -> dict[str, Path]:
        """Write the prepared input, mappings, audit table, and JSON summary."""

        import json

        result = prepared or self.prepare()
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "prepared_input": output / "prti_prepared_input.csv",
            "input_audit": output / "prti_input_audit.csv",
            "category_mappings": output / "prti_category_mappings.csv",
            "preparation_summary": output / "prti_preparation_summary.json",
        }
        result.model_input.to_csv(paths["prepared_input"], index=False)
        result.audit.to_csv(paths["input_audit"], index=False)
        result.category_mappings.to_csv(paths["category_mappings"], index=False)
        paths["preparation_summary"].write_text(
            json.dumps(result.summary, indent=2), encoding="utf-8"
        )
        return paths

    @staticmethod
    def file_sha256(path: str | Path) -> str:
        """Return a stable fingerprint used to prove the source stayed intact."""

        digest = sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _to_camel_case(column_name: object) -> str:
        name = str(column_name).strip()
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
        name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
        words = re.findall(r"[A-Za-z0-9]+", name)
        if not words:
            raise ValueError(f"Cannot create a canonical name for {column_name!r}.")
        return words[0].lower() + "".join(
            word[:1].upper() + word[1:].lower() for word in words[1:]
        )

    @staticmethod
    def _ordered_categories(column: str, values: pd.Series) -> list[object]:
        observed = values.dropna().drop_duplicates().tolist()
        if column in SEMANTIC_ORDERS:
            order = SEMANTIC_ORDERS[column]
            unexpected = set(observed) - set(order)
            if unexpected:
                raise ValueError(
                    f"Reviewed order for {column!r} does not contain: "
                    f"{sorted(unexpected)}"
                )
            return [value for value in order if value in set(observed)]
        return sorted(
            observed,
            key=lambda value: (
                isinstance(value, str) and not value.strip(),
                str(value).casefold(),
                str(value),
            ),
        )

    @staticmethod
    def _display_value(value: object) -> str:
        if isinstance(value, str) and not value.strip():
            return f"<BLANK: {len(value)} character(s)>"
        return str(value)

    @staticmethod
    def _role(column: str, schema: MOOBAMSchema) -> str:
        if column in schema.numeric_qids:
            return "QID - numeric"
        if column in schema.categorical_qids:
            return "QID - categorical"
        return "sensitive attribute"

    @staticmethod
    def _review_flag(
        column: str,
        role: str,
        values: pd.Series,
        distinct_count: int,
        missing_count: int,
    ) -> str:
        flags: list[str] = []
        if role.startswith("QID") and (distinct_count > 50 or distinct_count / len(values) > 0.05):
            flags.append("high-cardinality QID")
        if role == "sensitive attribute" and distinct_count > 50:
            flags.append("high-cardinality SA may dominate entropy review")
        if role == "sensitive attribute" and is_numeric_dtype(values.dtype) and distinct_count > 20:
            flags.append("numeric SA is treated as observed discrete states")
        if missing_count:
            flags.append("missing retained as an explicit model state")
        return "; ".join(flags)

    @staticmethod
    def _validate_round_trip(source: pd.DataFrame, decoded: pd.DataFrame) -> None:
        if source.shape != decoded.shape or list(source.columns) != list(decoded.columns):
            raise AssertionError("Preparation changed the PRTI-GH shape or column order.")
        for column in source.columns:
            before = source[column]
            after = decoded[column]
            equal = (before.eq(after)) | (before.isna() & after.isna())
            if not bool(equal.all()):
                raise AssertionError(f"Lossless round-trip validation failed for {column!r}.")

    @staticmethod
    def _largest_frequency_difference(before: pd.Series, after: pd.Series) -> int:
        """Compare exact value-state frequencies, including missing and blanks."""

        def state(value: object) -> tuple[str, str]:
            if pd.isna(value):
                return ("missing", "")
            return (type(value).__name__, repr(value))

        before_counts = before.map(state).value_counts().to_dict()
        after_counts = after.map(state).value_counts().to_dict()
        states = set(before_counts) | set(after_counts)
        return max(
            (abs(before_counts.get(key, 0) - after_counts.get(key, 0)) for key in states),
            default=0,
        )

    @staticmethod
    def _build_summary(
        source: pd.DataFrame,
        model_input: pd.DataFrame,
        schema: MOOBAMSchema,
        audit: pd.DataFrame,
        source_hash: str,
    ) -> dict[str, object]:
        class_sizes = model_input.groupby(
            list(schema.qids), dropna=False, observed=True, sort=False
        ).size()
        flagged = audit.loc[audit["reviewFlag"].ne(""), ["attribute", "reviewFlag"]]
        return {
            "source": str(DEFAULT_PRTI_SOURCE),
            "source_sha256_before": source_hash,
            "rows": len(source),
            "columns": len(source.columns),
            "numeric_qids": list(schema.numeric_qids),
            "categorical_qids": list(schema.categorical_qids),
            "sensitive_attributes": list(schema.sensitive_attributes),
            "role_counts": {
                "numeric_qids": len(schema.numeric_qids),
                "categorical_qids": len(schema.categorical_qids),
                "sensitive_attributes": len(schema.sensitive_attributes),
            },
            "preparation_policy": {
                "categorical_values": "one-to-one deterministic identifiers",
                "numeric_values": "unchanged",
                "missing_values": "explicit model state; no imputation",
                "spelling_and_case": "preserved exactly",
                "normalization": "none",
                "category_merging": "none",
                "duplicate_columns": "preserved as separate source attributes",
            },
            "known_source_characteristics": {
                "exact_duplicate_columns": [
                    list(columns) for columns in KNOWN_EXACT_DUPLICATE_COLUMNS
                ],
                "policy": (
                    "Retain every source column. Duplicate attributes are documented "
                    "and are not merged, removed, or silently reweighted."
                ),
            },
            "validation": {
                "shape_preserved": source.shape == model_input.shape,
                "column_order_preserved": list(source.columns) == list(model_input.columns),
                "lossless_round_trip": True,
                "all_attribute_distributions_preserved": bool(
                    audit["distributionStatus"].eq("PASS").all()
                ),
                "largest_frequency_difference_after_round_trip": int(
                    audit["largestFrequencyDifferenceAfterRoundTrip"].max()
                ),
                "model_input_contains_no_na": not model_input.loc[
                    :, schema.protected_columns
                ].isna().any().any(),
            },
            "baseline_qid_equivalence_classes": {
                "count": int(len(class_sizes)),
                "minimum_size": int(class_sizes.min()),
                "maximum_size": int(class_sizes.max()),
                "singleton_classes": int((class_sizes == 1).sum()),
                "records_in_singleton_classes": int(class_sizes[class_sizes == 1].sum()),
            },
            "review_flags": flagged.to_dict(orient="records"),
        }

    @staticmethod
    def _validate_known_source_characteristics(source: pd.DataFrame) -> None:
        """Confirm documented duplicates remain exact without modifying them."""

        for left, right in KNOWN_EXACT_DUPLICATE_COLUMNS:
            if left not in source or right not in source:
                raise ValueError(
                    f"Documented source columns are missing: {left!r}, {right!r}."
                )
            equal = source[left].eq(source[right]) | (
                source[left].isna() & source[right].isna()
            )
            if not bool(equal.all()):
                raise ValueError(
                    f"Documented duplicate columns {left!r} and {right!r} no longer "
                    "match. Review the source version instead of changing it silently."
                )
