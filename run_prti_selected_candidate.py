"""Generate the held-out candidate selected by a completed PRTI-GH range study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mo_obam.prti_data import DEFAULT_PRTI_SOURCE
from mo_obam.prti_dense_release import (
    PRTIDenseSelectedCandidateRunner,
    PRTIDenseSelectedPolicy,
)
from mo_obam.pso_solver import NoFeasibleParticleError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    parser.add_argument("--run-label")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = PRTIDenseSelectedCandidateRunner(
        PRTIDenseSelectedPolicy(
            study_directory=args.study_directory,
            output_directory=args.output_directory,
            evidence_directory=args.evidence_directory,
            source_path=args.source,
            run_label=args.run_label,
        )
    )
    if args.plan_only:
        print(f"Held-out plan: {runner.predeclare().resolve()}")
        return
    if args.evaluate_only:
        assessment = runner.evaluate_existing()
    else:
        def progress(
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
            assessment = runner.run(progress=progress)
        except NoFeasibleParticleError as error:
            raise SystemExit(
                f"{error}\nThe predeclared held-out candidate failed. Do not "
                "substitute another seed after inspecting this result."
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
