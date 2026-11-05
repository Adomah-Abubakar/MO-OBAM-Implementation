from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mo_obam.prti_data import (
    DEFAULT_PRTI_SOURCE,
    EXPECTED_PRTI_SHA256,
    KNOWN_EXACT_DUPLICATE_COLUMNS,
    MISSING_CATEGORY_CODE,
    PRTI_CATEGORICAL_QIDS,
    PRTI_NUMERIC_QIDS,
    PRTI_QIDS,
    PRTIDataAdapter,
)
from mo_obam.provenance import ArtifactManifestWriter
from mo_obam.privacy_metrics import PrivacyRiskMetrics
from mo_obam.prti_runner import PRTIAnonymizationRunner, PRTIRunConfig
from mo_obam.schema import MOOBAMConfig


class PRTIDataTests(unittest.TestCase):
    def make_frame(self) -> pd.DataFrame:
        values: dict[str, object] = {}
        for column in PRTI_QIDS:
            values[column] = ["A", "B", "A", "B"]
        values["childsAge"] = [4, 5, 6, 7]
        values["year"] = [2018, 2019, 2020, 2021]
        values["educationLevel"] = [
            "None/Preschool",
            "Lower Primary",
            "Upper primary",
            "Junior High",
        ]
        values["months"] = ["January", "Febraury", "March", "April"]
        values["dayOfWeek"] = ["Monday", "Teusday", "Wednesday", "Thursday"]
        values["timeOfDay"] = ["Morning", "Afternoon", "Evening", "Night"]
        values["presenceOfSupervisionsRoadGuideF"] = ["Yes", np.nan, "No", np.nan]
        values["mortality"] = ["Alive", "Alive", "Died", "Alive"]
        values["iss"] = [1, 2, 3, 4]
        values["aisHeadNeckOrSpine"] = [1.0, np.nan, 2.0, np.nan]
        return pd.DataFrame(values)

    def test_recorded_prti_default_configuration_is_explicit(self) -> None:
        config = PRTIRunConfig()

        self.assertEqual(config.model.k, 10)
        self.assertEqual(config.model.n_clusters, 30)
        self.assertEqual(config.model.lambda_weight, 1.0)
        self.assertEqual(config.pso.population_size, 100)
        self.assertEqual(config.pso.iterations, 100)
        self.assertEqual(config.pso.seed, 9)
        self.assertEqual(
            config.output_directory,
            Path("outputs/prti/baseline-v1-seed9"),
        )
        self.assertEqual(config.run_label, "baseline-v1-seed9")

    def test_agreed_roles_contain_29_qids(self) -> None:
        self.assertEqual(len(PRTI_QIDS), 29)
        self.assertEqual(PRTI_NUMERIC_QIDS, ("childsAge", "year"))
        self.assertEqual(len(PRTI_CATEGORICAL_QIDS), 27)
        self.assertIn("religiousAffiliation", PRTI_CATEGORICAL_QIDS)
        self.assertNotIn("temperatureDegreeCelsius", PRTI_QIDS)
        self.assertNotIn("oxygenSaturation", PRTI_QIDS)

    def test_preparation_is_lossless_and_represents_missing_without_imputation(self) -> None:
        source = self.make_frame()
        adapter = PRTIDataAdapter("unused.xls", expected_shape=None)
        prepared = adapter.prepare_frame(source)

        self.assertFalse(prepared.model_input.isna().any().any())
        self.assertEqual(
            prepared.model_input["presenceOfSupervisionsRoadGuideF"].tolist(),
            [2, MISSING_CATEGORY_CODE, 1, MISSING_CATEGORY_CODE],
        )
        decoded = adapter.decode_model_data(
            prepared.model_input, source, prepared.decoding_maps
        )
        for column in source.columns:
            equal = source[column].eq(decoded[column]) | (
                source[column].isna() & decoded[column].isna()
            )
            self.assertTrue(bool(equal.all()), column)

        self.assertEqual(len(prepared.schema.sensitive_attributes), 3)
        self.assertEqual(
            prepared.summary["preparation_policy"]["missing_values"],
            "explicit model state; no imputation",
        )

    def test_audit_outputs_are_complete_and_machine_readable(self) -> None:
        adapter = PRTIDataAdapter("unused.xls", expected_shape=None)
        prepared = adapter.prepare_frame(self.make_frame())
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = adapter.write_audit_outputs(temporary_directory, prepared)
            self.assertEqual(
                set(paths),
                {
                    "prepared_input",
                    "input_audit",
                    "category_mappings",
                    "preparation_summary",
                },
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))
            summary = json.loads(paths["preparation_summary"].read_text())
            self.assertTrue(summary["validation"]["lossless_round_trip"])
            self.assertEqual(pd.read_csv(paths["prepared_input"]).shape, (4, 32))

    def test_before_after_places_each_qid_side_by_side(self) -> None:
        original = self.make_frame()
        anonymized = original.copy()
        anonymized["childsAge"] = [5.5] * len(original)
        anonymized["cluster"] = [0] * len(original)
        paired = PRTIAnonymizationRunner._build_before_after(original, anonymized)

        self.assertIn("childsAgeBefore", paired)
        self.assertIn("childsAgeAfter", paired)
        self.assertIn("mortality", paired)
        self.assertNotIn("mortalityBefore", paired)

    def test_saved_csvs_are_reloaded_and_cross_checked(self) -> None:
        source = self.make_frame()
        adapter = PRTIDataAdapter("unused.xls", expected_shape=None)
        prepared = adapter.prepare_frame(source)
        anonymized = source.copy()
        anonymized["childsAge"] = [5.5] * len(source)
        anonymized["cluster"] = [0] * len(source)
        paired = PRTIAnonymizationRunner._build_before_after(source, anonymized)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            runner = PRTIAnonymizationRunner(
                PRTIRunConfig(
                    source_path=Path("unused.xls"),
                    output_directory=output,
                    model=MOOBAMConfig(k=2, n_clusters=1, lambda_weight=1.0),
                )
            )
            paths = adapter.write_audit_outputs(output, prepared)
            result_paths = {
                "before_after_anonymization": output / "before_after.csv",
                "anonymized_final": output / "final.csv",
                "cluster_details": output / "clusters.csv",
                "sensitive_entropy": output / "entropy.csv",
                "model_evaluation": output / "evaluation.csv",
            }
            paired.to_csv(result_paths["before_after_anonymization"], index=False)
            anonymized.drop(columns="cluster").to_csv(
                result_paths["anonymized_final"], index=False
            )
            pd.DataFrame({"cluster": [0], "records": [len(source)]}).to_csv(
                result_paths["cluster_details"], index=False
            )
            pd.DataFrame(
                {
                    "cluster": [0] * len(prepared.schema.sensitive_attributes),
                    "sensitiveAttribute": prepared.schema.sensitive_attributes,
                    "entropy": [0.0] * len(prepared.schema.sensitive_attributes),
                }
            ).to_csv(result_paths["sensitive_entropy"], index=False)
            risk = PrivacyRiskMetrics(
                equivalence_classes=1,
                minimum_equivalence_class_size=4,
                maximum_equivalence_class_size=4,
                linkage_attack_records={"0.05": 0, "0.075": 0, "0.1": 0},
                homogeneity_attack_records=4,
                homogeneity_attack_records_by_sensitive_attribute={},
            )
            mock_result = SimpleNamespace(
                metrics=SimpleNamespace(information_loss=0.5, entropy_sum=1.0)
            )
            runner._model_evaluation(risk, risk, mock_result).to_csv(
                result_paths["model_evaluation"], index=False
            )
            validation = runner._validate_saved_exports(
                {**paths, **result_paths}, prepared
            )
            self.assertTrue(validation["all_csvs_reloaded"])
            self.assertEqual(validation["final_release_shape"], [4, 32])

            damaged = anonymized.drop(columns="cluster")
            damaged.loc[0, "childsAge"] = 99
            damaged.to_csv(result_paths["anonymized_final"], index=False)
            with self.assertRaisesRegex(AssertionError, "childsAge"):
                runner._validate_saved_exports({**paths, **result_paths}, prepared)

    def test_model_evaluation_separates_clusters_and_equivalence_classes(self) -> None:
        runner = PRTIAnonymizationRunner(
            PRTIRunConfig(
                source_path=Path("unused.xls"),
                model=MOOBAMConfig(k=2, n_clusters=3, lambda_weight=1.0),
                run_label="baseline-v1-seed9",
            )
        )
        baseline = PrivacyRiskMetrics(
            equivalence_classes=11,
            minimum_equivalence_class_size=1,
            maximum_equivalence_class_size=2,
            linkage_attack_records={"0.05": 4, "0.075": 4, "0.1": 4},
            homogeneity_attack_records=4,
            homogeneity_attack_records_by_sensitive_attribute={},
        )
        anonymized = PrivacyRiskMetrics(
            equivalence_classes=3,
            minimum_equivalence_class_size=4,
            maximum_equivalence_class_size=6,
            linkage_attack_records={"0.05": 0, "0.075": 0, "0.1": 0},
            homogeneity_attack_records=0,
            homogeneity_attack_records_by_sensitive_attribute={},
        )
        result = SimpleNamespace(
            metrics=SimpleNamespace(information_loss=0.25, entropy_sum=2.0)
        )

        report = runner._model_evaluation(baseline, anonymized, result)

        self.assertNotIn("numberOfClusters", report.columns)
        self.assertEqual(report["equivalenceClasses"].tolist(), [11, 3])
        self.assertTrue(pd.isna(report.loc[0, "configuredClusters"]))
        self.assertEqual(report.loc[1, "configuredClusters"], 3)

    def test_requirements_match_the_recorded_runtime(self) -> None:
        requirements_path = Path(__file__).resolve().parents[1] / "requirements.txt"
        requirements = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        pins = dict(requirement.split("==", maxsplit=1) for requirement in requirements)
        environment = PRTIAnonymizationRunner._runtime_environment()

        self.assertEqual(set(pins), {"numpy", "pandas", "scipy", "xlrd"})
        self.assertTrue(environment["python"].startswith("3.12."))
        self.assertEqual(environment["packages"], pins)

        provenance = PRTIAnonymizationRunner._paper_provenance()
        self.assertEqual(provenance["paper_identifier"], "arXiv:2501.01002v2")
        self.assertEqual(provenance["paper_revision"], "15 May 2025")

    def test_artifact_manifest_is_refreshable_without_changing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "outputs"
            output.mkdir()
            source = root / "source.xls"
            artifact = output / "result.csv"
            source.write_bytes(b"unchanged source")
            artifact.write_text("value\n1\n", encoding="utf-8")
            summary = {
                "run_label": "test",
                "model_configuration": {
                    "k": 2,
                    "n_clusters": 1,
                    "lambda_weight": 1.0,
                },
                "pso_configuration": {"seed": 7},
                "outputs": {"result": artifact.name},
            }
            writer = ArtifactManifestWriter(Path(__file__).resolve().parents[1])
            source_hash = writer.file_sha256(source)
            artifact_hash = writer.file_sha256(artifact)

            manifest_path = writer.write(output, source, summary)
            first = json.loads(manifest_path.read_text(encoding="utf-8"))
            writer.write(output, source, summary)
            refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(first["manifest_version"], 2)
            self.assertTrue(first["run"]["run_id"].startswith("legacy-prti-test"))
            self.assertEqual(
                first["code"]["snapshot_status"],
                "captured_at_manifest_refresh_not_run_start",
            )
            self.assertEqual(first["source"]["sha256"], source_hash)
            self.assertEqual(first["artifacts"]["result"]["sha256"], artifact_hash)
            self.assertEqual(writer.file_sha256(source), source_hash)
            self.assertEqual(writer.file_sha256(artifact), artifact_hash)
            self.assertEqual(
                refreshed["artifacts"]["result"]["sha256"], artifact_hash
            )

    def test_artifact_refresh_preserves_run_start_generation_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "outputs"
            output.mkdir()
            source = root / "source.xls"
            artifact = output / "result.csv"
            source.write_bytes(b"source")
            artifact.write_text("value\n1\n", encoding="utf-8")
            snapshot = {
                "snapshot_status": "captured_at_run_start",
                "git": {"commit": "fixed-commit", "working_tree_clean": True},
                "generation_files": {"run_prti.py": "fixed-hash"},
                "validation_files": {},
                "documentation_files": {},
            }
            summary = {
                "run_id": "recorded-run",
                "run_label": "baseline",
                "generation_snapshot": snapshot,
                "outputs": {"result": artifact.name},
            }
            writer = ArtifactManifestWriter(Path(__file__).resolve().parents[1])

            path = writer.write(output, source, summary)
            writer.write(output, source, summary)
            refreshed = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(refreshed["code"], snapshot)

    @unittest.skipUnless(
        (
            Path(__file__).resolve().parents[1]
            / "outputs/prti/baseline-v1-seed9/prti_run_summary.json"
        ).is_file(),
        "Recorded PRTI-GH output is not installed",
    )
    def test_saved_prti_result_matches_the_documented_baseline(self) -> None:
        output = (
            Path(__file__).resolve().parents[1]
            / "outputs/prti/baseline-v1-seed9"
        )
        summary = json.loads((output / "prti_run_summary.json").read_text())

        self.assertEqual(summary["model_configuration"], {
            "k": 10,
            "n_clusters": 30,
            "lambda_weight": 1.0,
        })
        self.assertEqual(summary["pso_configuration"]["population_size"], 100)
        self.assertEqual(summary["pso_configuration"]["iterations"], 100)
        self.assertEqual(summary["pso_configuration"]["seed"], 9)
        self.assertEqual(len(summary["cluster_sizes"]), 30)
        self.assertEqual(min(summary["cluster_sizes"]), 16)
        self.assertAlmostEqual(
            summary["final_metrics"]["information_loss"], 0.4366537386166644
        )
        self.assertAlmostEqual(
            summary["final_metrics"]["entropy_sum"], 52.31126541274985
        )
        self.assertAlmostEqual(
            summary["final_metrics"]["objective"], -51.874611674133185
        )
        self.assertEqual(summary["validation"]["saved_csvs"]["entropy_rows"], 2310)
        self.assertEqual(
            summary["final_metrics"]["entropy_by_sensitive_attribute"]["xray"],
            summary["final_metrics"]["entropy_by_sensitive_attribute"]["xray1"],
        )

    @unittest.skipUnless(DEFAULT_PRTI_SOURCE.is_file(), "PRTI-GH source is not installed")
    def test_installed_source_shape_roles_and_hash_are_preserved(self) -> None:
        adapter = PRTIDataAdapter(DEFAULT_PRTI_SOURCE)
        hash_before = adapter.file_sha256(DEFAULT_PRTI_SOURCE)
        prepared = adapter.prepare()
        hash_after = adapter.file_sha256(DEFAULT_PRTI_SOURCE)

        self.assertEqual(prepared.original.shape, (1_485, 106))
        self.assertEqual(len(prepared.schema.numeric_qids), 2)
        self.assertEqual(len(prepared.schema.categorical_qids), 27)
        self.assertEqual(len(prepared.schema.sensitive_attributes), 77)
        self.assertEqual(hash_before, hash_after)
        self.assertEqual(hash_before, EXPECTED_PRTI_SHA256)
        self.assertTrue(prepared.summary["source_unchanged"])

    @unittest.skipUnless(DEFAULT_PRTI_SOURCE.is_file(), "PRTI-GH source is not installed")
    def test_known_duplicate_columns_are_preserved_as_separate_sas(self) -> None:
        prepared = PRTIDataAdapter(DEFAULT_PRTI_SOURCE).prepare()

        self.assertEqual(KNOWN_EXACT_DUPLICATE_COLUMNS, (("xray", "xray1"),))
        self.assertIn("xray", prepared.schema.sensitive_attributes)
        self.assertIn("xray1", prepared.schema.sensitive_attributes)
        self.assertNotEqual(
            prepared.original.columns.get_loc("xray"),
            prepared.original.columns.get_loc("xray1"),
        )
        pd.testing.assert_series_equal(
            prepared.original["xray"],
            prepared.original["xray1"],
            check_names=False,
        )
        self.assertEqual(
            prepared.summary["preparation_policy"]["duplicate_columns"],
            "preserved as separate source attributes",
        )
        final_path = Path(__file__).resolve().parents[1] / (
            "outputs/prti/baseline-v1-seed9/prti_anonymized_final.csv"
        )
        if final_path.is_file():
            final = pd.read_csv(final_path)
            for column in ("xray", "xray1"):
                pd.testing.assert_series_equal(
                    prepared.original[column].reset_index(drop=True),
                    final[column],
                    check_dtype=False,
                    check_names=False,
                )


if __name__ == "__main__":
    unittest.main()
