"""Run aggregate PRTI-GH seed, sensitivity, and comparison workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mo_obam.prti_data import DEFAULT_PRTI_SOURCE
from mo_obam.prti_dense_clusters import (
    PRTIDenseClusterConfig,
    PRTIDenseClusterRunner,
    PRTIParameterGridConfig,
    PRTIParameterGridRunner,
)
from mo_obam.prti_experiments import (
    PRTIBaselineComparator,
    PRTISeedScreenConfig,
    PRTISeedScreenRunner,
    PRTISensitivityConfig,
    PRTISensitivityExperiment,
    PRTISensitivityRunner,
)
from mo_obam.prti_robustness import (
    PRTIRobustnessConfig,
    PRTIRobustnessExperiment,
    PRTIRobustnessRunner,
)
from mo_obam.pso_solver import PSOConfig
from mo_obam.schema import MOOBAMConfig


def build_parser() -> argparse.ArgumentParser:
    """Build explicit subcommands for safe aggregate experiment evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    screen = commands.add_parser("seed-screen")
    screen.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    screen.add_argument(
        "--output-directory",
        type=Path,
        default=Path("evidence/prti-baseline-v1"),
    )
    screen.add_argument("--k", type=int, default=10)
    screen.add_argument("--clusters", type=int, default=30)
    screen.add_argument("--lambda-weight", type=float, default=1.0)
    screen.add_argument("--population", type=int, default=100)
    screen.add_argument("--iterations", type=int, default=20)
    screen.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))

    compare = commands.add_parser("compare")
    compare.add_argument(
        "--legacy-directory",
        type=Path,
        default=Path("outputs/prti/archive/legacy-seed9"),
    )
    compare.add_argument("--fresh-directory", type=Path, required=True)
    compare.add_argument(
        "--evidence-directory",
        type=Path,
        default=Path("evidence/prti-baseline-v1"),
    )

    sensitivity = commands.add_parser("sensitivity")
    sensitivity.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    sensitivity.add_argument(
        "--output-directory",
        type=Path,
        default=Path("evidence/prti-sensitivity-v1"),
    )
    sensitivity.add_argument("--population", type=int, default=100)
    sensitivity.add_argument("--iterations", type=int, default=100)
    sensitivity.add_argument("--seed", type=int, default=9)
    sensitivity.add_argument(
        "--max-new-runs",
        type=int,
        help="Run only this many new checkpoints; rerun the command to resume.",
    )
    sensitivity.add_argument(
        "--run-id",
        action="append",
        help="Run only the named Appendix A grid point; may be repeated.",
    )
    sensitivity.add_argument(
        "--retry-blocked",
        action="store_true",
        help="Retry matching checkpoints that found no feasible particle.",
    )
    sensitivity.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the complete plan without running PSO.",
    )

    robustness = commands.add_parser("robustness")
    robustness.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    robustness.add_argument(
        "--output-directory",
        type=Path,
        default=Path("evidence/prti-robustness-v1"),
    )
    robustness.add_argument("--population", type=int, default=100)
    robustness.add_argument("--iterations", type=int, default=100)
    robustness.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    robustness.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent seed runs to execute concurrently.",
    )
    robustness.add_argument(
        "--worker-backend",
        choices=("thread", "process"),
        default="thread",
        help="Concurrency engine; use process for CPU-dense Linux server runs.",
    )
    robustness.add_argument(
        "--max-new-runs",
        type=int,
        help="Run only this many new seed checkpoints; rerun to resume.",
    )
    robustness.add_argument(
        "--configuration-id",
        action="append",
        help="Run only the named robustness configuration; may be repeated.",
    )
    robustness.add_argument(
        "--retry-blocked",
        action="store_true",
        help="Retry matching seeds that found no feasible initial particle.",
    )
    robustness.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the multi-seed plan without running PSO.",
    )

    dense = commands.add_parser("dense-clusters")
    dense.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    dense.add_argument(
        "--output-directory",
        type=Path,
        default=Path("evidence/prti-dense-clusters-v1"),
    )
    dense.add_argument("--minimum-clusters", type=int, required=True)
    dense.add_argument("--maximum-clusters", type=int, required=True)
    dense.add_argument("--k", type=int, required=True)
    dense.add_argument("--lambda-weight", type=float, required=True)
    dense.add_argument("--population", type=int, required=True)
    dense.add_argument("--iterations", type=int, required=True)
    dense.add_argument("--seeds", type=int, nargs="+", required=True)
    dense.add_argument(
        "--held-out-seed",
        type=int,
        required=True,
        help=(
            "Reserve this unseen seed for the selected final candidate; it must "
            "not appear in --seeds."
        ),
    )
    dense.add_argument("--workers", type=int, default=1)
    dense.add_argument(
        "--worker-backend",
        choices=("thread", "process"),
        default="thread",
        help="Concurrency engine; use process for CPU-dense Linux server runs.",
    )
    reuse = dense.add_mutually_exclusive_group()
    reuse.add_argument(
        "--reuse-directory",
        type=Path,
        default=Path("evidence/prti-robustness-v1"),
        help="Reuse only exactly matching aggregate robustness checkpoints.",
    )
    reuse.add_argument(
        "--no-reuse",
        dest="reuse_directory",
        action="store_const",
        const=None,
        help="Start without importing any historical aggregate checkpoints.",
    )
    dense.add_argument("--max-new-runs", type=int)
    dense.add_argument("--configuration-id", action="append")
    dense.add_argument("--retry-blocked", action="store_true")
    dense.add_argument("--dry-run", action="store_true")

    grid = commands.add_parser(
        "parameter-grid",
        help="Compare a declared feasible grid of k, C, and lambda values.",
        description=(
            "Run or resume an aggregate-only joint PRTI-GH parameter study. "
            "For each k, cluster counts above floor(1485/k) are excluded before "
            "optimization. Every retained configuration uses every comparison seed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (write the immutable plan only):\n"
            "  .venv/bin/python run_prti_experiments.py parameter-grid \\\n"
            "    --minimum-clusters 1 --maximum-clusters 148 \\\n"
            "    --k-values 10 15 20 \\\n"
            "    --lambda-weights 0.0001 0.001 0.01 0.1 1 \\\n"
            "    --population 100 --iterations 100 \\\n"
            "    --seeds 0 1 2 3 4 5 6 7 8 9 --held-out-seed 11 \\\n"
            "    --workers 48 --worker-backend process --no-reuse \\\n"
            "    --output-directory evidence/prti-parameter-grid-v1 --dry-run"
        ),
    )
    grid.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_PRTI_SOURCE,
        help="Reviewed PRTI-GH workbook (default: repository-adjacent source).",
    )
    grid.add_argument(
        "--output-directory",
        type=Path,
        default=Path("evidence/prti-parameter-grid-v1"),
        help="Dedicated aggregate evidence directory for this immutable plan.",
    )
    grid.add_argument(
        "--minimum-clusters",
        type=int,
        required=True,
        help="First cluster count C to consider (at least 1).",
    )
    grid.add_argument(
        "--maximum-clusters",
        type=int,
        required=True,
        help="Requested last C; infeasible endpoints are bounded separately for each k.",
    )
    grid.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        required=True,
        help="Distinct hard minimum equivalence-class sizes, e.g. 10 15 20.",
    )
    grid.add_argument(
        "--lambda-weights",
        type=float,
        nargs="+",
        required=True,
        help=(
            "Distinct objective privacy weights in [0,1], e.g. the paper-derived "
            "0.0001 0.001 0.01 0.1 1 grid."
        ),
    )
    grid.add_argument(
        "--population",
        type=int,
        required=True,
        help="PSO particles per run (at least 2).",
    )
    grid.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="PSO iterations per run (at least 1).",
    )
    grid.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Distinct non-negative comparison seeds.",
    )
    grid.add_argument(
        "--held-out-seed",
        type=int,
        required=True,
        help="Prospectively reserved non-negative seed absent from --seeds.",
    )
    grid.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent independent runs on this server (default: 1).",
    )
    grid.add_argument(
        "--worker-backend",
        choices=("thread", "process"),
        default="thread",
        help=(
            "Concurrency engine (default: thread). Use process on a POSIX/Linux "
            "server to avoid Python thread contention between independent runs."
        ),
    )
    grid_reuse = grid.add_mutually_exclusive_group(required=True)
    grid_reuse.add_argument(
        "--reuse-directory",
        type=Path,
        help="Reuse only exactly matching aggregate robustness checkpoints.",
    )
    grid_reuse.add_argument(
        "--no-reuse",
        dest="reuse_directory",
        action="store_const",
        const=None,
        help="Import no historical checkpoints.",
    )
    grid.add_argument(
        "--max-new-runs",
        type=int,
        help="Calculate at most N missing checkpoints, then stop cleanly.",
    )
    grid.add_argument(
        "--configuration-id",
        action="append",
        help="Run only this declared ID; repeat for multiple IDs.",
    )
    grid.add_argument(
        "--retry-blocked",
        action="store_true",
        help="Retry recorded no-feasible-particle outcomes.",
    )
    grid.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and inspect the immutable plan without reading patient rows.",
    )
    return parser


def main() -> None:
    """Execute the selected aggregate experiment workflow."""

    args = build_parser().parse_args()
    if args.command == "compare":
        result = PRTIBaselineComparator().compare(
            args.legacy_directory,
            args.fresh_directory,
            args.evidence_directory,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "sensitivity":
        config = PRTISensitivityConfig(
            source_path=args.source,
            output_directory=args.output_directory,
            pso=PSOConfig(
                population_size=args.population,
                iterations=args.iterations,
                seed=args.seed,
            ),
        )
        runner = PRTISensitivityRunner(config)
        selected_run_ids = set(args.run_id) if args.run_id else None
        if selected_run_ids is not None:
            known = {experiment.run_id for experiment in runner.experiments}
            unknown = sorted(selected_run_ids - known)
            if unknown:
                raise SystemExit(f"Unknown sensitivity run IDs: {unknown}")
        if args.dry_run:
            path = runner.write_plan()
            print(f"Planned {len(runner.experiments)} configurations: {path.resolve()}")
            print("k values: 5, 10, 15, 20")
            print("cluster values: 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100")
            print("lambda values: 0.0001, 0.001, 0.01, 0.1, 1")
            return

        def sensitivity_progress(
            experiment: PRTISensitivityExperiment,
            iteration: int,
            total: int,
            best_score: float,
            feasible: int,
            infeasible: int,
        ) -> None:
            if iteration == 1 or iteration == total or iteration % 10 == 0:
                print(
                    f"{experiment.run_id}: iteration {iteration}/{total}, "
                    f"best={best_score:.8f}, feasible={feasible}, "
                    f"infeasible={infeasible}",
                    flush=True,
                )

        results = runner.run(
            progress=sensitivity_progress,
            max_new_runs=args.max_new_runs,
            run_ids=selected_run_ids,
            retry_blocked=args.retry_blocked,
        )
        display = [
            "status",
            "run_id",
            "information_loss",
            "entropy_sum",
            "equivalence_classes",
            "linkage_records_tau_0_05",
            "homogeneity_records_any_sa",
        ]
        print(results.reindex(columns=display).to_string(index=False, na_rep="-"))
        print(
            "Aggregate results: "
            f"{(config.output_directory / runner.RESULTS_FILE).resolve()}"
        )
        return

    if args.command == "robustness":
        config = PRTIRobustnessConfig(
            source_path=args.source,
            output_directory=args.output_directory,
            population_size=args.population,
            iterations=args.iterations,
            seeds=tuple(args.seeds),
            workers=args.workers,
            worker_backend=args.worker_backend,
        )
        runner = PRTIRobustnessRunner(config)
        selected_ids = set(args.configuration_id) if args.configuration_id else None
        if selected_ids is not None:
            known = {
                experiment.configuration_id for experiment in runner.experiments
            }
            unknown = sorted(selected_ids - known)
            if unknown:
                raise SystemExit(f"Unknown robustness configuration IDs: {unknown}")
        if args.dry_run:
            path = runner.write_plan()
            print(
                f"Planned {len(runner.experiments) * len(config.seeds)} runs: "
                f"{path.resolve()}"
            )
            print("Configurations: " + ", ".join(
                experiment.configuration_id for experiment in runner.experiments
            ))
            print("Seeds: " + ", ".join(str(seed) for seed in config.seeds))
            return

        def robustness_progress(
            experiment: PRTIRobustnessExperiment,
            seed: int,
            iteration: int,
            total: int,
            best_score: float,
            feasible: int,
            infeasible: int,
        ) -> None:
            if iteration == 1 or iteration == total or iteration % 10 == 0:
                print(
                    f"{experiment.configuration_id}/seed={seed}: "
                    f"iteration {iteration}/{total}, best={best_score:.8f}, "
                    f"feasible={feasible}, infeasible={infeasible}",
                    flush=True,
                )

        result = runner.run(
            progress=robustness_progress,
            max_new_runs=args.max_new_runs,
            configuration_ids=selected_ids,
            retry_blocked=args.retry_blocked,
        )
        display = [
            "configuration_id",
            "runs_completed",
            "runs_no_feasible_particle",
            "information_loss_mean",
            "information_loss_std",
            "entropy_sum_mean",
            "linkage_records_tau_0_05_mean",
            "homogeneity_records_any_sa_mean",
        ]
        print(result["summary"].reindex(columns=display).to_string(index=False))
        print(f"Run-level evidence: {result['paths']['runs'].resolve()}")
        print(f"Summary evidence: {result['paths']['summary'].resolve()}")
        return

    if args.command in {"dense-clusters", "parameter-grid"}:
        if args.command == "parameter-grid":
            config = PRTIParameterGridConfig(
                source_path=args.source,
                output_directory=args.output_directory,
                minimum_clusters=args.minimum_clusters,
                maximum_clusters=args.maximum_clusters,
                k_values=tuple(args.k_values),
                lambda_weights=tuple(args.lambda_weights),
                population_size=args.population,
                iterations=args.iterations,
                seeds=tuple(args.seeds),
                workers=args.workers,
                worker_backend=args.worker_backend,
                reuse_directory=args.reuse_directory,
                held_out_seed=args.held_out_seed,
            )
            runner = PRTIParameterGridRunner(config)
        else:
            config = PRTIDenseClusterConfig(
                source_path=args.source,
                output_directory=args.output_directory,
                minimum_clusters=args.minimum_clusters,
                maximum_clusters=args.maximum_clusters,
                k=args.k,
                lambda_weight=args.lambda_weight,
                population_size=args.population,
                iterations=args.iterations,
                seeds=tuple(args.seeds),
                workers=args.workers,
                worker_backend=args.worker_backend,
                reuse_directory=args.reuse_directory,
                held_out_seed=args.held_out_seed,
            )
            runner = PRTIDenseClusterRunner(config)
        selected_ids = set(args.configuration_id) if args.configuration_id else None
        if selected_ids is not None:
            known = {experiment.configuration_id for experiment in runner.experiments}
            unknown = sorted(selected_ids - known)
            if unknown:
                raise SystemExit(f"Unknown study configuration IDs: {unknown}")
        if args.dry_run:
            path = runner.write_plan()
            print(
                f"Planned {len(runner.experiments) * len(config.seeds)} runs: "
                f"{path.resolve()}"
            )
            print(
                f"Configurations: {len(runner.experiments)}; comparison seeds: "
                f"{len(config.seeds)}"
            )
            if args.command == "parameter-grid":
                endpoints = {
                    k: min(args.maximum_clusters, 1485 // k)
                    for k in args.k_values
                }
                print(f"Feasible maximum C by k: {endpoints}")
            else:
                print(
                    f"Cluster values: {config.minimum_clusters} through "
                    f"{config.maximum_clusters} inclusive"
                )
            print("Patient-level outputs: disabled")
            return

        def study_progress(
            experiment: PRTIRobustnessExperiment,
            seed: int,
            iteration: int,
            total: int,
            best_score: float,
            feasible: int,
            infeasible: int,
        ) -> None:
            if iteration == 1 or iteration == total or iteration % 10 == 0:
                print(
                    f"{experiment.configuration_id}/seed={seed}: "
                    f"iteration {iteration}/{total}, best={best_score:.8f}, "
                    f"feasible={feasible}, infeasible={infeasible}",
                    flush=True,
                )

        result = runner.run(
            progress=study_progress,
            max_new_runs=args.max_new_runs,
            configuration_ids=selected_ids,
            retry_blocked=args.retry_blocked,
        )
        display = [
            "k",
            "configured_clusters",
            "lambda",
            "runs_completed",
            "information_loss_mean",
            "homogeneity_fraction_any_sa_max",
            "passes_all_predeclared_rules",
            "selection_rank",
            "selected",
        ]
        print(result["selection"].reindex(columns=display).to_string(index=False))
        print(f"Selection evidence: {result['paths']['selection_json'].resolve()}")
        return

    config = PRTISeedScreenConfig(
        source_path=args.source,
        output_directory=args.output_directory,
        model=MOOBAMConfig(
            k=args.k,
            n_clusters=args.clusters,
            lambda_weight=args.lambda_weight,
        ),
        population_size=args.population,
        iterations=args.iterations,
        seeds=tuple(args.seeds),
    )

    def progress(
        seed: int,
        iteration: int,
        total: int,
        best_score: float,
        feasible: int,
        infeasible: int,
    ) -> None:
        if iteration == 1 or iteration == total or iteration % 10 == 0:
            print(
                f"seed={seed}: iteration {iteration}/{total}, "
                f"best={best_score:.8f}, feasible={feasible}, "
                f"infeasible={infeasible}",
                flush=True,
            )

    summary = PRTISeedScreenRunner(config).run(progress=progress)
    print(
        json.dumps(
            {
                "selected_seed": summary["selected_seed"],
                "completed_runs": sum(
                    run["status"] == "completed" for run in summary["runs"]
                ),
                "outputs": summary["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
