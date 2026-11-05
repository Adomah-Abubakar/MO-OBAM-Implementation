"""Generate aggregate-only PRTI-GH release-readiness review artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from mo_obam.prti_data import DEFAULT_PRTI_SOURCE
from mo_obam.prti_release_readiness import (
    DEFAULT_CANDIDATE_DIRECTORY,
    DEFAULT_EVIDENCE_DIRECTORY,
    generate_release_readiness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_PRTI_SOURCE)
    parser.add_argument(
        "--candidate-directory", type=Path, default=DEFAULT_CANDIDATE_DIRECTORY
    )
    parser.add_argument(
        "--evidence-directory", type=Path, default=DEFAULT_EVIDENCE_DIRECTORY
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = generate_release_readiness(
        source_path=args.source,
        candidate_directory=args.candidate_directory,
        evidence_directory=args.evidence_directory,
    )
    print(f"Attribute review: {paths.attribute_review.resolve()}")
    print(f"Readiness decision record: {paths.readiness.resolve()}")


if __name__ == "__main__":
    main()
