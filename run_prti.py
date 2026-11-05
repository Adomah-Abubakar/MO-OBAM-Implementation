"""Audit or anonymize PRTI-GH with the paper-aligned MO-OBAM implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mo_obam.prti_data import DEFAULT_PRTI_SOURCE
from mo_obam.prti_runner import PRTIAnonymizationRunner, PRTIRunConfig
from mo_obam.pso_solver import NoFeasibleParticleError, PSOConfig
from mo_obam.schema import MOOBAMConfig


def build_parser() -> argparse.ArgumentParser:
    """Expose data paths and every optimization parameter explicitly."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Output directory. Defaults to the labelled baseline directory, "
            "or a separate audit/smoke-test directory for those modes."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Prepare, validate, and export the audit without running PSO.",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Label the result as a smoke test; parameters remain explicit.",
    )
    mode.add_argument(
        "--show-tables",
        action="store_true",
        help="Print previously saved result tables without running PSO.",
    )
    mode.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Refresh hashes and code provenance for the saved run without running PSO.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=30)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument(
        "--run-label",
        default=None,
        help="Stable label written to the run ID and saved result metadata.",
    )
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=9,
        help="PSO seed; 9 is the recorded PRTI run selected by the documented screen.",
    )
    return parser


def main() -> None:
    """Run the requested phase and print only its most useful summary."""

    args = build_parser().parse_args()
    if args.output_directory is not None:
        output_directory = args.output_directory
    elif args.audit_only:
        output_directory = Path("outputs/prti/audit-only")
    elif args.smoke_test:
        output_directory = Path("outputs/prti/smoke-test")
    else:
        output_directory = Path("outputs/prti/baseline-v1-seed9")

    if args.show_tables:
        PRTIAnonymizationRunner.print_saved_tables(output_directory)
        return

    population = args.population if args.population is not None else 100
    iterations = args.iterations if args.iterations is not None else (
        5 if args.smoke_test else 100
    )
    config = PRTIRunConfig(
        source_path=args.source,
        output_directory=output_directory,
        model=MOOBAMConfig(
            k=args.k,
            n_clusters=args.clusters,
            lambda_weight=args.lambda_weight,
        ),
        pso=PSOConfig(
            population_size=population,
            iterations=iterations,
            seed=args.seed,
        ),
        run_label=(
            "smoke-test"
            if args.smoke_test
            else (args.run_label or "baseline-v1-seed9")
        ),
    )
    runner = PRTIAnonymizationRunner(config)
    if args.refresh_manifest:
        path = runner.refresh_saved_manifest()
        print(f"Artifact manifest refreshed: {path.resolve()}")
        return
    if args.audit_only:
        summary = runner.audit_only()
        print(
            json.dumps(
                {
                    "mode": summary["mode"],
                    "rows": summary["rows"],
                    "columns": summary["columns"],
                    "role_counts": summary["role_counts"],
                    "baseline_qid_equivalence_classes": summary[
                        "baseline_qid_equivalence_classes"
                    ],
                    "review_flag_count": len(summary["review_flags"]),
                    "source_unchanged": summary["source_unchanged"],
                    "outputs": summary["outputs"],
                },
                indent=2,
            )
        )
        print(f"Audit outputs saved in: {config.output_directory.resolve()}")
        return

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

    try:
        summary = runner.run(progress=report_progress)
    except NoFeasibleParticleError as error:
        raise SystemExit(
            f"{error}\nNo result was reported. This means the PSO initialization "
            "did not find a hard-k-feasible particle; it does not prove that the "
            "mathematical model is infeasible. Try a lower --clusters value or a "
            "deliberately recorded seed/population change."
        ) from error
    metrics = summary["final_metrics"]
    print(
        json.dumps(
            {
                "run_label": summary["run_label"],
                "information_loss": metrics["information_loss"],
                "entropy_sum": metrics["entropy_sum"],
                "objective": metrics["objective"],
                "sensitive_attributes": len(
                    metrics["entropy_by_sensitive_attribute"]
                ),
            },
            indent=2,
        )
    )
    runner.print_saved_tables(config.output_directory)
    print(f"Outputs saved in: {config.output_directory.resolve()}")


if __name__ == "__main__":
    main()
