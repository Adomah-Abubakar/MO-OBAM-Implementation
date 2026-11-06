#!/usr/bin/env python3
"""Generate anonymized data from a completed multi-objective selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mo_obam.prti_data import DEFAULT_PRTI_SOURCE
from mo_obam.prti_multiobjective_release import (
    PRTIMultiObjectiveSelectedCandidateRunner,
    PRTIMultiObjectiveSelectedPolicy,
)
from mo_obam.pso_solver import NoFeasibleParticleError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    parser.add_argument("--run-label", default="multiobjective-selected-heldout")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = PRTIMultiObjectiveSelectedCandidateRunner(
        PRTIMultiObjectiveSelectedPolicy(
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

    def progress(
        iteration: int,
        total: int,
        archive_size: int,
        feasible: int,
        infeasible: int,
    ) -> None:
        if iteration == 1 or iteration == total or iteration % 10 == 0:
            print(
                f"iteration {iteration}/{total}, archive={archive_size}, "
                f"feasible={feasible}, infeasible={infeasible}",
                flush=True,
            )

    try:
        result = runner.run(progress=progress)
    except (NoFeasibleParticleError, RuntimeError) as error:
        raise SystemExit(
            f"{error}\nThe held-out run failed. Do not substitute another seed."
        ) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
