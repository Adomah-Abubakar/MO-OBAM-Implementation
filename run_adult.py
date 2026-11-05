"""Run the paper's high-protection Adult configuration and save requested outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mo_obam import MOOBAMConfig, PSOConfig
from mo_obam.adult_reporting import AdultOutputReporter
from mo_obam.adult_runner import AdultAnonymizationRunner, AdultRunConfig


def build_parser() -> argparse.ArgumentParser:
    """Expose paper parameters and repository PSO controls for reproducibility."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("data/adult"))
    parser.add_argument("--output-directory", type=Path, default=Path("outputs"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-existing",
        action="store_true",
        help="Create/display final CSVs and tables from an existing completed run.",
    )
    parser.add_argument(
        "--show-tables",
        action="store_true",
        help="Display previously saved result tables without running PSO.",
    )
    return parser


def main() -> None:
    """Build the configuration, run anonymization, and print its final summary."""

    args = build_parser().parse_args()
    if args.report_existing and args.show_tables:
        raise SystemExit("Choose either --report-existing or --show-tables, not both.")
    if args.show_tables:
        AdultOutputReporter.print_saved_tables(args.output_directory)
        return

    config = AdultRunConfig(
        data_directory=args.data_directory,
        output_directory=args.output_directory,
        model=MOOBAMConfig(
            k=args.k,
            n_clusters=args.clusters,
            lambda_weight=args.lambda_weight,
        ),
        pso=PSOConfig(
            population_size=args.population,
            iterations=args.iterations,
            seed=args.seed,
        ),
    )

    def report_progress(
        stage: str,
        iteration: int,
        total: int,
        best_score: float,
        feasible: int,
        infeasible: int,
    ) -> None:
        if iteration == 1 or iteration == total or iteration % 10 == 0:
            print(
                f"{stage}: iteration {iteration}/{total}, best={best_score:.8f}, "
                f"feasible={feasible}, infeasible={infeasible}",
                flush=True,
            )

    runner = AdultAnonymizationRunner(config)
    if args.report_existing:
        summary = runner.report_existing()
        print("Rebuilt release data and tables from the existing completed run.")
    else:
        summary = runner.run(progress=report_progress)
    print(json.dumps(summary["final_metrics"], indent=2))
    AdultOutputReporter.print_saved_tables(config.output_directory)
    print(f"Outputs saved in: {config.output_directory.resolve()}")


if __name__ == "__main__":
    main()
