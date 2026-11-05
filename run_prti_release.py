"""Run the prospectively declared held-out PRTI-GH release candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mo_obam.prti_data import DEFAULT_PRTI_SOURCE
from mo_obam.prti_release import PRTIFinalCandidateRunner, PRTIReleasePolicy
from mo_obam.pso_solver import NoFeasibleParticleError, PSOConfig
from mo_obam.schema import MOOBAMConfig


def build_parser() -> argparse.ArgumentParser:
    """Expose the selection evidence, candidate settings, and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    parser.add_argument(
        "--robustness-summary",
        type=Path,
        default=Path(
            "evidence/prti-robustness-v1/prti_robustness_summary.json"
        ),
    )
    parser.add_argument(
        "--evidence-directory",
        type=Path,
        default=Path("evidence/prti-final-selection-v1"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/prti/privacy-candidate-heldout-seed10"),
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--run-label", default="privacy-candidate-heldout-seed10"
    )
    parser.add_argument("--maximum-information-loss", type=float, default=0.60)
    parser.add_argument(
        "--maximum-homogeneity-fraction", type=float, default=0.20
    )
    parser.add_argument(
        "--maximum-zero-entropy-sensitive-attributes", type=int, default=1
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only", action="store_true", help="Write or verify the plan only."
    )
    mode.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Assess an already saved matching candidate without rerunning PSO.",
    )
    return parser


def main() -> None:
    """Run the requested phase and print its decision and observed metrics."""

    args = build_parser().parse_args()
    policy = PRTIReleasePolicy(
        source_path=args.source,
        robustness_summary_path=args.robustness_summary,
        evidence_directory=args.evidence_directory,
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
        run_label=args.run_label,
        maximum_information_loss=args.maximum_information_loss,
        maximum_homogeneity_fraction=args.maximum_homogeneity_fraction,
        maximum_zero_minimum_entropy_sensitive_attributes=(
            args.maximum_zero_entropy_sensitive_attributes
        ),
    )
    runner = PRTIFinalCandidateRunner(policy)
    if args.plan_only:
        print(f"Final-selection plan: {runner.predeclare().resolve()}")
        return
    if args.evaluate_only:
        assessment = runner.evaluate_existing()
    else:
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
                    f"{stage}: iteration {iteration}/{total}, "
                    f"best={best_score:.8f}, feasible={feasible}, "
                    f"infeasible={infeasible}",
                    flush=True,
                )

        try:
            assessment = runner.run(progress=report_progress)
        except NoFeasibleParticleError as error:
            raise SystemExit(
                f"{error}\nThe prospective plan remains recorded, but no result "
                "or replacement seed was invented."
            ) from error
    print(
        json.dumps(
            {
                "status": assessment["status"],
                "all_checks_passed": assessment["all_checks_passed"],
                "run_id": assessment["run_id"],
                "observed": assessment["observed"],
                "release_boundary": assessment["release_boundary"],
                "outputs": assessment["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
