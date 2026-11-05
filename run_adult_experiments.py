"""Run the paper's non-ML Adult MO-OBAM experiment configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from mo_obam.adult_experiments import (
    AdultExperiment,
    AdultExperimentRunConfig,
    AdultExperimentRunner,
)
from mo_obam.pso_solver import PSOConfig


def build_parser() -> argparse.ArgumentParser:
    """Expose the paper presets and explicit reproducibility controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("data/adult"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Output directory. Defaults to outputs/adult_experiments/<preset> "
            "so Table 6 and full-grid checkpoints cannot be mixed."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=("table-6", "full-grid"),
        default="table-6",
        help="Eight reported Table 6 boundary runs or all 220 Appendix A points.",
    )
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-new-runs",
        type=int,
        help="Run only this many new checkpoints; rerun the command to resume.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        help="Run only the named planned configuration; may be supplied repeatedly.",
    )
    parser.add_argument(
        "--retry-blocked",
        action="store_true",
        help="Retry configurations previously recorded as blocked with the same settings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and display the experiment plan without running PSO.",
    )
    parser.add_argument(
        "--export-run",
        metavar="RUN_ID",
        help="Build full before/after CSVs from one completed checkpoint.",
    )
    return parser


def main() -> None:
    """Plan, execute, resume, or export the Adult anonymization study."""

    args = build_parser().parse_args()
    if args.dry_run and args.export_run:
        raise SystemExit("Choose --dry-run or --export-run, not both.")

    output_directory = args.output_directory or (
        Path("outputs/adult_experiments") / args.preset.replace("-", "_")
    )
    config = AdultExperimentRunConfig(
        data_directory=args.data_directory,
        output_directory=output_directory,
        preset=args.preset,
        pso=PSOConfig(
            population_size=args.population,
            iterations=args.iterations,
            seed=args.seed,
        ),
    )
    runner = AdultExperimentRunner(config)
    selected_run_ids = set(args.run_id) if args.run_id else None
    if selected_run_ids is not None:
        known_run_ids = {experiment.run_id for experiment in runner.experiments}
        unknown = sorted(selected_run_ids - known_run_ids)
        if unknown:
            raise SystemExit(f"Unknown run IDs for preset {args.preset!r}: {unknown}")

    if args.dry_run:
        path = runner.write_plan()
        print(f"Planned {len(runner.experiments)} configurations: {path.resolve()}")
        if len(runner.experiments) <= 20:
            for experiment in runner.experiments:
                print(
                    f"{experiment.run_id}: k={experiment.k}, "
                    f"clusters={experiment.n_clusters}, "
                    f"lambda={experiment.lambda_weight:g}"
                )
        else:
            print(
                "k values: "
                + ", ".join(
                    str(value) for value in sorted({x.k for x in runner.experiments})
                )
            )
            print(
                "cluster values: "
                + ", ".join(
                    str(value)
                    for value in sorted({x.n_clusters for x in runner.experiments})
                )
            )
            print(
                "lambda values: "
                + ", ".join(
                    f"{value:g}"
                    for value in sorted({x.lambda_weight for x in runner.experiments})
                )
            )
        return

    if args.export_run:
        path = runner.export_run(args.export_run)
        print(f"Full before/after release exported to: {path.resolve()}")
        return

    def report_progress(
        experiment: AdultExperiment,
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
        progress=report_progress,
        max_new_runs=args.max_new_runs,
        run_ids=selected_run_ids,
        retry_blocked=args.retry_blocked,
    )
    if results.empty:
        print("No experiment checkpoint has been completed yet.")
    else:
        display_columns = [
            "status",
            "run_id",
            "information_loss",
            "paper_information_loss",
            "equivalence_classes",
            "linkage_tau_0.05",
            "linkage_tau_0.075",
            "linkage_tau_0.1",
            "homogeneity_attack_records",
        ]
        print(
            results.reindex(columns=display_columns).to_string(
                index=False,
                na_rep="-",
            )
        )
    print(
        "Completed-results table: "
        f"{(config.output_directory / runner.RESULTS_FILE).resolve()}"
    )


if __name__ == "__main__":
    main()
