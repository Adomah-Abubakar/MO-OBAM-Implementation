"""Command-line entry point for the exact MO-OBAM reference implementation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import pandas as pd

from mo_obam import ExactMOOBAMSolver, MOOBAMConfig, MOOBAMFormulation, MOOBAMSchema


def build_parser() -> argparse.ArgumentParser:
    """Define explicit, reusable command-line inputs without dataset hard-coding."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--numeric-qid", action="append", default=[])
    parser.add_argument("--categorical-qid", action="append", default=[])
    parser.add_argument("--sensitive-attribute", action="append", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--lambda-weight", type=float, required=True)
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument("--prefix", default="mo_obam")
    return parser


def main() -> None:
    """Load CSV data, solve the exact formulation, and save reproducible outputs."""

    args = build_parser().parse_args()
    data = pd.read_csv(args.input)
    schema = MOOBAMSchema(
        numeric_qids=tuple(args.numeric_qid),
        categorical_qids=tuple(args.categorical_qid),
        sensitive_attributes=tuple(args.sensitive_attribute),
    )
    config = MOOBAMConfig(
        k=args.k,
        n_clusters=args.clusters,
        lambda_weight=args.lambda_weight,
    )
    formulation = MOOBAMFormulation(data, schema, config)
    result = ExactMOOBAMSolver(formulation, max_records=args.max_records).solve()
    result.save(args.output_directory, prefix=args.prefix)

    print(
        json.dumps(
            {
                "metrics": asdict(result.metrics),
                "feasible_partitions_evaluated": result.feasible_partitions_evaluated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
