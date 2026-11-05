from __future__ import annotations

from dataclasses import asdict
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mo_obam.prti_dense_clusters import (
    PRTIDenseClusterConfig,
    PRTIDenseClusterRunner,
    PRTIParameterGridConfig,
    PRTIParameterGridRunner,
)
from mo_obam.prti_runner import PAPER_REVISION
from mo_obam.pso_solver import PSOConfig


class PRTIDenseClusterTests(unittest.TestCase):
    project_root = Path(__file__).resolve().parents[1]
    wrapper_environment = {**os.environ, "PRTI_PYTHON": sys.executable}

    def test_plan_covers_every_cluster_count_and_excludes_patient_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = PRTIDenseClusterRunner(
                PRTIDenseClusterConfig(
                    source_path=Path("unused.xls"),
                    output_directory=Path(temporary_directory),
                )
            )

            plan = json.loads(runner.write_plan().read_text(encoding="utf-8"))

            self.assertEqual(plan["cluster_values"], list(range(4, 31)))
            self.assertEqual(plan["number_of_configurations"], 27)
            self.assertEqual(plan["number_of_planned_runs"], 270)
            self.assertFalse(plan["patient_level_outputs"])
            self.assertIn("lowest mean information loss", plan["selection_rule"])
            self.assertIn("Pareto frontier", plan["analysis_rule"])

    def test_plan_cannot_be_silently_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            PRTIDenseClusterRunner(
                PRTIDenseClusterConfig(output_directory=output)
            ).write_plan()

            changed = PRTIDenseClusterRunner(
                PRTIDenseClusterConfig(
                    output_directory=output,
                    maximum_information_loss=0.55,
                )
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                changed.write_plan()

    def test_parameter_grid_declares_only_mathematically_feasible_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = PRTIParameterGridRunner(
                PRTIParameterGridConfig(
                    source_path=Path("unused.xls"),
                    output_directory=Path(temporary_directory),
                    minimum_clusters=73,
                    maximum_clusters=76,
                    k_values=(10, 20),
                    lambda_weights=(0.5, 1.0),
                    seeds=(0, 1),
                    held_out_seed=11,
                )
            )

            plan = json.loads(runner.write_plan().read_text(encoding="utf-8"))

            self.assertEqual(plan["feasible_maximum_clusters_by_k"], {"10": 76, "20": 74})
            self.assertEqual(plan["number_of_configurations"], 12)
            self.assertEqual(plan["number_of_planned_runs"], 24)
            self.assertEqual(
                {
                    item["model_configuration"]["n_clusters"]
                    for item in plan["configurations"]
                    if item["model_configuration"]["k"] == 20
                },
                {73, 74},
            )
            self.assertFalse(plan["exact_checkpoint_reuse"]["permitted"])

    def test_parameter_grid_cli_writes_an_explicit_dry_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "run_prti_experiments.py",
                    "parameter-grid",
                    "--output-directory",
                    temporary_directory,
                    "--minimum-clusters",
                    "73",
                    "--maximum-clusters",
                    "76",
                    "--k-values",
                    "10",
                    "20",
                    "--lambda-weights",
                    "0.5",
                    "1",
                    "--population",
                    "100",
                    "--iterations",
                    "100",
                    "--seeds",
                    "0",
                    "1",
                    "--held-out-seed",
                    "11",
                    "--workers",
                    "2",
                    "--no-reuse",
                    "--dry-run",
                ],
                cwd=self.project_root,
                env=self.wrapper_environment,
                check=True,
                capture_output=True,
                text=True,
            )
            plan = json.loads(
                (Path(temporary_directory) / "prti_parameter_grid_plan.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertIn("Planned 24 runs", completed.stdout)
            self.assertEqual(plan["number_of_configurations"], 12)
            self.assertEqual(plan["held_out_seed"], 11)

    def test_parameter_grid_requires_unique_values_and_an_unseen_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "k values must be unique"):
            PRTIParameterGridConfig(k_values=(10, 10), held_out_seed=11)
        with self.assertRaisesRegex(ValueError, "Lambda weights must be unique"):
            PRTIParameterGridConfig(
                lambda_weights=(1.0, 1.0),
                held_out_seed=11,
            )
        with self.assertRaisesRegex(ValueError, "must not appear"):
            PRTIParameterGridConfig(seeds=(0, 1), held_out_seed=1)
        runner = PRTIParameterGridRunner(
            PRTIParameterGridConfig(
                minimum_clusters=1,
                maximum_clusters=1,
                k_values=(10,),
                lambda_weights=(0.1234564, 0.1234565),
                seeds=(0,),
                held_out_seed=11,
            )
        )
        identifiers = [item.configuration_id for item in runner.experiments]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_parameter_grid_ranks_across_k_cluster_and_lambda(self) -> None:
        runner = PRTIParameterGridRunner(
            PRTIParameterGridConfig(
                minimum_clusters=4,
                maximum_clusters=4,
                k_values=(10, 20),
                lambda_weights=(0.5, 1.0),
                seeds=(0, 1),
                held_out_seed=11,
            )
        )
        base = {
            "k": 10,
            "lambda": 1.0,
            "seeds_planned": 2,
            "runs_completed": 2,
            "runs_no_feasible_particle": 0,
            "minimum_cluster_size_min": 20,
            "linkage_records_tau_0_05_max": 0,
            "linkage_records_tau_0_075_max": 0,
            "linkage_records_tau_0_1_max": 0,
            "homogeneity_records_any_sa_max": 0,
            "zero_minimum_entropy_sensitive_attributes_max": 0,
            "information_loss_max": 0.59,
        }
        summary = pd.DataFrame(
            [
                {
                    **base,
                    "configuration_id": "k10_c004_lambda1",
                    "k": 10,
                    "configured_clusters": 4,
                    "lambda": 1.0,
                    "information_loss_mean": 0.50,
                },
                {
                    **base,
                    "configuration_id": "k20_c004_lambda0p5",
                    "k": 20,
                    "configured_clusters": 4,
                    "lambda": 0.5,
                    "information_loss_mean": 0.45,
                },
            ]
        )

        assessed = runner.assess_summary(summary).set_index("configuration_id")

        self.assertTrue(bool(assessed.loc["k20_c004_lambda0p5", "selected"]))
        self.assertEqual(
            assessed.loc["k20_c004_lambda0p5", "selection_rank"],
            1,
        )

    def test_parameter_grid_reports_privacy_eligible_il_entropy_pareto_frontier(
        self,
    ) -> None:
        runner = PRTIParameterGridRunner(
            PRTIParameterGridConfig(
                minimum_clusters=4,
                maximum_clusters=4,
                k_values=(10,),
                lambda_weights=(1.0,),
                seeds=(0, 1),
                held_out_seed=11,
            )
        )
        base = {
            "k": 10,
            "configured_clusters": 4,
            "lambda": 1.0,
            "seeds_planned": 2,
            "runs_completed": 2,
            "runs_no_feasible_particle": 0,
            "minimum_cluster_size_min": 10,
            "linkage_records_tau_0_05_max": 0,
            "linkage_records_tau_0_075_max": 0,
            "linkage_records_tau_0_1_max": 0,
            "homogeneity_records_any_sa_max": 0,
            "zero_minimum_entropy_sensitive_attributes_max": 0,
            "information_loss_max": 0.59,
        }
        summary = pd.DataFrame(
            [
                {
                    **base,
                    "configuration_id": "low_il",
                    "information_loss_mean": 0.40,
                    "entropy_sum_mean": 10.0,
                },
                {
                    **base,
                    "configuration_id": "high_entropy",
                    "information_loss_mean": 0.50,
                    "entropy_sum_mean": 12.0,
                },
                {
                    **base,
                    "configuration_id": "entropy_dominated",
                    "information_loss_mean": 0.60,
                    "entropy_sum_mean": 11.0,
                },
                {
                    **base,
                    "configuration_id": "utility_dominated",
                    "information_loss_mean": 0.45,
                    "entropy_sum_mean": 9.0,
                },
            ]
        )

        assessed = runner.assess_summary(summary).set_index("configuration_id")

        self.assertTrue(
            bool(assessed.loc["low_il", "pareto_efficient_il_entropy"])
        )
        self.assertTrue(
            bool(assessed.loc["high_entropy", "pareto_efficient_il_entropy"])
        )
        self.assertFalse(
            bool(assessed.loc["entropy_dominated", "pareto_efficient_il_entropy"])
        )
        self.assertFalse(
            bool(assessed.loc["utility_dominated", "pareto_efficient_il_entropy"])
        )

    def test_selection_prefers_lowest_il_only_among_privacy_passes(self) -> None:
        runner = PRTIDenseClusterRunner(
            PRTIDenseClusterConfig(
                minimum_clusters=4,
                maximum_clusters=6,
                seeds=(0, 1),
                reuse_directory=None,
            )
        )
        base = {
            "k": 10,
            "lambda": 1.0,
            "seeds_planned": 2,
            "runs_completed": 2,
            "runs_no_feasible_particle": 0,
            "minimum_cluster_size_min": 10,
            "linkage_records_tau_0_05_max": 0,
            "linkage_records_tau_0_075_max": 0,
            "linkage_records_tau_0_1_max": 0,
            "zero_minimum_entropy_sensitive_attributes_max": 1,
            "information_loss_max": 0.59,
        }
        summary = pd.DataFrame(
            [
                {
                    **base,
                    "configuration_id": "clusters_04",
                    "configured_clusters": 4,
                    "information_loss_mean": 0.55,
                    "homogeneity_records_any_sa_max": 0,
                },
                {
                    **base,
                    "configuration_id": "clusters_05",
                    "configured_clusters": 5,
                    "information_loss_mean": 0.50,
                    "homogeneity_records_any_sa_max": 100,
                },
                {
                    **base,
                    "configuration_id": "clusters_06",
                    "configured_clusters": 6,
                    "information_loss_mean": 0.40,
                    "homogeneity_records_any_sa_max": 400,
                },
            ]
        )

        assessed = runner.assess_summary(summary).set_index("configured_clusters")

        self.assertTrue(bool(assessed.loc[5, "selected"]))
        self.assertFalse(bool(assessed.loc[6, "passes_all_predeclared_rules"]))
        self.assertEqual(assessed.loc[5, "selection_rank"], 1)

    def test_incomplete_study_withholds_provisional_rank_and_selection(self) -> None:
        provisional = pd.DataFrame(
            [
                {
                    "configured_clusters": 4,
                    "passes_all_predeclared_rules": True,
                    "selection_rank": 1,
                    "selected": True,
                },
                {
                    "configured_clusters": 5,
                    "passes_all_predeclared_rules": False,
                    "selection_rank": None,
                    "selected": False,
                },
            ]
        )

        withheld = PRTIDenseClusterRunner.withhold_partial_selection(provisional)

        self.assertTrue(withheld["selection_rank"].isna().all())
        self.assertFalse(withheld["selected"].any())
        self.assertTrue(bool(withheld.loc[0, "passes_all_predeclared_rules"]))

    def test_exact_aggregate_checkpoint_can_be_reused_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reuse = root / "old"
            output = root / "new"
            config = PRTIDenseClusterConfig(
                output_directory=output,
                minimum_clusters=4,
                maximum_clusters=4,
                seeds=(0,),
                reuse_directory=reuse,
            )
            runner = PRTIDenseClusterRunner(config)
            experiment = runner.experiments[0]
            source_hash = "reviewed-source"
            pso = PSOConfig(population_size=100, iterations=100, seed=0)
            old_path = reuse / "runs/privacy_candidate/seed_0000.json"
            old_path.parent.mkdir(parents=True)
            old_path.write_text(
                json.dumps(
                    {
                        "paper_revision": PAPER_REVISION,
                        "source_sha256": source_hash,
                        "configuration_id": "privacy_candidate",
                        "model_configuration": asdict(experiment.model),
                        "pso_configuration": asdict(pso),
                        "patient_level_outputs": False,
                        "result_row": {
                            "configuration_id": "privacy_candidate",
                            "model_run_id": experiment.model_run_id,
                            "purpose": "old",
                            "k": 10,
                            "configured_clusters": 4,
                            "lambda": 1.0,
                            "population": 100,
                            "iterations": 100,
                            "status": "completed",
                            "issue": None,
                            "seed": 0,
                            "information_loss": 0.5,
                            "entropy_sum": 1.0,
                            "objective": -0.5,
                            "equivalence_classes": 4,
                            "minimum_cluster_size": 10,
                            "maximum_cluster_size": 1000,
                            "linkage_records_tau_0_05": 0,
                            "linkage_records_tau_0_075": 0,
                            "linkage_records_tau_0_1": 0,
                            "homogeneity_records_any_sa": 0,
                            "zero_minimum_entropy_sensitive_attributes": 0,
                            "best_iteration_zero_based": 0,
                            "feasible_evaluations": 1,
                            "infeasible_evaluations": 9999,
                            "feasible_evaluation_percent": 0.01,
                            "runtime_seconds": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(runner.import_exact_checkpoints(source_hash), 1)
            imported = json.loads(
                runner._checkpoint_path(experiment, 0).read_text(encoding="utf-8")
            )
            self.assertEqual(imported["configuration_id"], "clusters_04")
            self.assertEqual(
                imported["result_row"]["configuration_id"], "clusters_04"
            )
            self.assertIn("sha256", imported["reused_aggregate_checkpoint"])

    def test_terminal_wrapper_passes_custom_configuration_to_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            subprocess.run(
                [
                    "./run_prti_cluster_study.sh",
                    "12",
                    "--k",
                    "15",
                    "--lambda-weight",
                    "0.1",
                    "--population",
                    "150",
                    "--iterations",
                    "200",
                    "--seeds",
                    "0,1,2",
                    "--held-out-seed",
                    "11",
                    "--workers",
                    "2",
                    "--output-directory",
                    temporary_directory,
                    "--dry-run",
                ],
                cwd=self.project_root,
                env=self.wrapper_environment,
                check=True,
                capture_output=True,
                text=True,
            )
            plan = json.loads(
                (Path(temporary_directory) / "prti_dense_cluster_plan.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(plan["cluster_values"], [12])
            self.assertEqual(
                plan["fixed_model_configuration"],
                {"k": 15, "lambda_weight": 0.1},
            )
            self.assertEqual(
                plan["pso_configuration_except_seed"]["population_size"], 150
            )
            self.assertEqual(
                plan["pso_configuration_except_seed"]["iterations"], 200
            )
            self.assertEqual(plan["seeds"], [0, 1, 2])
            self.assertEqual(plan["held_out_seed"], 11)

    def test_terminal_wrapper_rejects_cluster_count_impossible_for_k(self) -> None:
        completed = subprocess.run(
            [
                "./run_prti_cluster_study.sh",
                "100",
                "--k",
                "15",
                "--lambda-weight",
                "1",
                "--population",
                "100",
                "--iterations",
                "100",
                "--seeds",
                "0,1,2",
                "--held-out-seed",
                "11",
                "--dry-run",
            ],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("maximum is 99", completed.stderr)

    def test_terminal_wrapper_requires_explicit_model_and_optimizer_values(self) -> None:
        completed = subprocess.run(
            ["./run_prti_cluster_study.sh", "8", "--dry-run"],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "--k --lambda-weight --population --iterations --seeds --held-out-seed",
            completed.stderr,
        )

    def test_dense_python_cli_also_requires_explicit_study_values(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "run_prti_experiments.py",
                "dense-clusters",
                "--dry-run",
            ],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--minimum-clusters", completed.stderr)
        self.assertIn("--maximum-clusters", completed.stderr)
        self.assertIn("--k", completed.stderr)
        self.assertIn("--lambda-weight", completed.stderr)
        self.assertIn("--population", completed.stderr)
        self.assertIn("--iterations", completed.stderr)
        self.assertIn("--seeds", completed.stderr)
        self.assertIn("--held-out-seed", completed.stderr)

    def test_terminal_wrapper_help_works_without_a_cluster_argument(self) -> None:
        completed = subprocess.run(
            ["./run_prti_cluster_study.sh", "--help"],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Usage:", completed.stdout)
        self.assertIn("--k N", completed.stdout)
        self.assertIn("--lambda-weight VALUE", completed.stdout)
        self.assertIn("--population N", completed.stdout)
        self.assertIn("--iterations N", completed.stdout)
        self.assertIn("--seeds CSV", completed.stdout)
        self.assertIn(
            "8 --k 10 --lambda-weight 1 --population 100 --iterations 100",
            completed.stdout,
        )

    def test_core_rejects_mathematically_infeasible_cluster_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum is 148"):
            PRTIDenseClusterConfig(
                minimum_clusters=1,
                maximum_clusters=149,
                k=10,
            )

    def test_core_requires_held_out_seed_to_be_unseen(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not appear"):
            PRTIDenseClusterConfig(seeds=(0, 1), held_out_seed=1)

    def test_range_wrapper_writes_full_explicit_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            subprocess.run(
                [
                    "./run_prti_cluster_range_study.sh",
                    "1",
                    "5",
                    "--k",
                    "10",
                    "--lambda-weight",
                    "1",
                    "--population",
                    "100",
                    "--iterations",
                    "100",
                    "--seeds",
                    "0,1,2",
                    "--held-out-seed",
                    "11",
                    "--output-directory",
                    temporary_directory,
                    "--dry-run",
                ],
                cwd=self.project_root,
                env=self.wrapper_environment,
                check=True,
                capture_output=True,
                text=True,
            )
            plan = json.loads(
                (Path(temporary_directory) / "prti_dense_cluster_plan.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(plan["cluster_values"], [1, 2, 3, 4, 5])
            self.assertEqual(plan["number_of_planned_runs"], 15)
            self.assertEqual(plan["seed_protocol"]["comparison_seeds"], [0, 1, 2])
            self.assertEqual(plan["seed_protocol"]["held_out_seed"], 11)
            self.assertFalse(plan["patient_level_outputs"])

    def test_range_wrapper_can_explicitly_disable_checkpoint_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            subprocess.run(
                [
                    "./run_prti_cluster_range_study.sh",
                    "1",
                    "2",
                    "--k",
                    "10",
                    "--lambda-weight",
                    "1",
                    "--population",
                    "100",
                    "--iterations",
                    "100",
                    "--seeds",
                    "0,1",
                    "--held-out-seed",
                    "11",
                    "--output-directory",
                    temporary_directory,
                    "--no-reuse",
                    "--dry-run",
                ],
                cwd=self.project_root,
                env=self.wrapper_environment,
                check=True,
                capture_output=True,
                text=True,
            )
            plan = json.loads(
                (Path(temporary_directory) / "prti_dense_cluster_plan.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertFalse(plan["exact_checkpoint_reuse"]["permitted"])
            self.assertIsNone(plan["exact_checkpoint_reuse"]["source_directory"])


if __name__ == "__main__":
    unittest.main()
