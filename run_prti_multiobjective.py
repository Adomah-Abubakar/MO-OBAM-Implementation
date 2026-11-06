#!/usr/bin/env python3
"""Run or resume the additive PRTI-GH multi-objective PSO study."""

from __future__ import annotations

import argparse
from pathlib import Path

from mo_obam.prti_data import DEFAULT_PRTI_SOURCE, PRTIDataAdapter
from mo_obam.prti_multiobjective import (
    PRTIMultiObjectiveConfig,
    PRTIMultiObjectiveStudy,
)


def parse_configuration_pair(value: str) -> tuple[int, int]:
    """Parse one explicit K:C configuration pair."""

    try:
        k_text, clusters_text = value.split(":", maxsplit=1)
        return int(k_text), int(clusters_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            f"invalid configuration pair {value!r}; expected K:C, for example 20:5"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search information loss and sensitive-attribute entropy jointly. "
            "The scalar PSO baseline is not modified and lambda is analysed after "
            "optimization from the discovered Pareto archive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Full C=1..30 plan example:\n"
            "  .venv/bin/python run_prti_multiobjective.py \\\n"
            "    --source data/PRTI-GH-study/PRTI-GH-study.xls \\\n"
            "    --output-directory evidence/prti-multiobjective-v1 \\\n"
            "    --minimum-clusters 1 --maximum-clusters 30 \\\n"
            "    --k-values 10 15 20 25 30 \\\n"
            "    --population 100 --iterations 100 --archive-size 100 \\\n"
            "    --initialization-attempts 20 \\\n"
            "    --random-coefficient-max 0.01 \\\n"
            "    --seeds 0 1 2 3 4 5 6 7 8 9 --held-out-seed 12 \\\n"
            "    --maximum-information-loss 0.60 \\\n"
            "    --maximum-homogeneity-fraction 0.20 \\\n"
            "    --maximum-zero-entropy-sensitive-attributes 1 \\\n"
            "    --workers 32 --dry-run\n\n"
            "Remove --dry-run to execute or resume all missing comparison runs. "
            "This command never executes the held-out seed."
        ),
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    parser.add_argument(
        "--output-directory", type=Path, required=True,
        help="New versioned evidence directory; an incompatible plan is never overwritten.",
    )
    parser.add_argument("--minimum-clusters", type=int)
    parser.add_argument("--maximum-clusters", type=int)
    parser.add_argument("--k-values", type=int, nargs="+")
    parser.add_argument(
        "--configuration-pairs",
        type=parse_configuration_pair,
        nargs="+",
        metavar="K:C",
        help=(
            "Run only explicit k/cluster pairs; mutually exclusive with "
            "--minimum-clusters, --maximum-clusters, and --k-values."
        ),
    )
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--archive-size", type=int, required=True)
    parser.add_argument(
        "--initialization-attempts",
        type=int,
        required=True,
        help="Independent random initial populations tried before recording no feasible particle.",
    )
    parser.add_argument(
        "--random-coefficient-max",
        type=float,
        required=True,
        help=(
            "Upper bound for each PSO cognitive/social random coefficient; "
            "the authors' implementation uses 0.01."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--held-out-seed", type=int, required=True)
    parser.add_argument("--maximum-information-loss", type=float, required=True)
    parser.add_argument("--maximum-homogeneity-fraction", type=float, required=True)
    parser.add_argument(
        "--maximum-zero-entropy-sensitive-attributes", type=int, required=True
    )
    parser.add_argument(
        "--selection-information-loss-margin",
        type=float,
        default=0.0,
        help=(
            "If positive, retain all-seed-passing configurations within this "
            "absolute mean-IL margin, then minimize maximum homogeneity."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-new-runs",
        type=int,
        help="Execute at most this many missing runs, then write an incomplete report.",
    )
    parser.add_argument(
        "--retry-blocked",
        action="store_true",
        help="Retry checkpoints previously recorded as no_feasible_particle.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate the Excel and selection JSON from SQLite without PSO.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and inspect the immutable full plan without reading patient rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run and args.report_only:
        raise SystemExit("--dry-run and --report-only cannot be used together.")
    range_arguments = (
        args.minimum_clusters,
        args.maximum_clusters,
        args.k_values,
    )
    if args.configuration_pairs and any(value is not None for value in range_arguments):
        raise SystemExit(
            "--configuration-pairs cannot be combined with cluster ranges or --k-values."
        )
    if not args.configuration_pairs and any(value is None for value in range_arguments):
        raise SystemExit(
            "Provide --configuration-pairs or all of --minimum-clusters, "
            "--maximum-clusters, and --k-values."
        )
    configuration_pairs = tuple(args.configuration_pairs or ())
    if configuration_pairs:
        minimum_clusters = min(clusters for _, clusters in configuration_pairs)
        maximum_clusters = max(clusters for _, clusters in configuration_pairs)
        k_values = tuple(dict.fromkeys(k for k, _ in configuration_pairs))
    else:
        minimum_clusters = args.minimum_clusters
        maximum_clusters = args.maximum_clusters
        k_values = tuple(args.k_values)
    config = PRTIMultiObjectiveConfig(
        source_path=args.source,
        output_directory=args.output_directory,
        minimum_clusters=minimum_clusters,
        maximum_clusters=maximum_clusters,
        k_values=k_values,
        seeds=tuple(args.seeds),
        held_out_seed=args.held_out_seed,
        population_size=args.population,
        iterations=args.iterations,
        archive_size=args.archive_size,
        initialization_attempts=args.initialization_attempts,
        random_coefficient_max=args.random_coefficient_max,
        workers=args.workers,
        maximum_information_loss=args.maximum_information_loss,
        maximum_homogeneity_fraction=args.maximum_homogeneity_fraction,
        maximum_zero_minimum_entropy_sensitive_attributes=(
            args.maximum_zero_entropy_sensitive_attributes
        ),
        configuration_pairs=configuration_pairs,
        selection_information_loss_margin=args.selection_information_loss_margin,
    )
    study = PRTIMultiObjectiveStudy(config)
    if args.dry_run:
        source_hash = PRTIDataAdapter.file_sha256(args.source)
        path = study.write_plan(source_hash)
        print("PRTI-GH multi-objective study plan")
        print(f"  configurations: {len(config.configurations)}")
        print(f"  comparison runs: {config.planned_runs}")
        if config.configuration_pairs:
            pairs = ", ".join(
                f"k={k}/C={clusters}" for k, clusters in config.configuration_pairs
            )
            print(f"  configuration pairs: {pairs}")
        else:
            print(
                f"  clusters: {config.minimum_clusters} through "
                f"{config.maximum_clusters} inclusive"
            )
            print(f"  k values: {', '.join(map(str, config.k_values))}")
        print(
            "  selection IL margin: "
            f"{config.selection_information_loss_margin:g}"
        )
        print("  lambda: continuous post-optimization analysis over [0, 1]")
        print(f"  plan: {path.resolve()}")
        print("  patient-level outputs: disabled")
        return
    if args.report_only:
        result = study.refresh_reports()
    else:
        def progress(
            completed: int,
            total: int,
            k: int,
            clusters: int,
            seed: int,
            status: str,
        ) -> None:
            print(
                f"[{completed}/{total}] k={k}, C={clusters}, seed={seed}: {status}",
                flush=True,
            )

        result = study.run(
            max_new_runs=args.max_new_runs,
            retry_blocked=args.retry_blocked,
            progress=progress,
        )
    print(f"Status: {result['status']}")
    print(f"Recorded runs: {result['recorded_runs']}/{result['planned_runs']}")
    for name, path in result["paths"].items():
        print(f"{name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
