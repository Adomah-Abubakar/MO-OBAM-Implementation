"""Generate aggregate PRTI-GH MO-OBAM tables for reporting and write-up."""

from __future__ import annotations

import argparse
from pathlib import Path

from mo_obam.prti_results import PRTIResultsConfig, PRTIResultsReporter


def build_parser() -> argparse.ArgumentParser:
    """Expose explicit input and output locations without changing source data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-directory",
        type=Path,
        default=Path("outputs/prti/baseline-v1-seed9"),
    )
    parser.add_argument(
        "--sensitivity-results",
        type=Path,
        default=Path("evidence/prti-sensitivity-v1/prti_sensitivity_results.csv"),
    )
    parser.add_argument(
        "--robustness-summary",
        type=Path,
        default=Path(
            "evidence/prti-robustness-v1/prti_robustness_summary.json"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("evidence/prti-results-v1"),
    )
    return parser


def main() -> None:
    """Build and display the aggregate reporting artifacts."""

    args = build_parser().parse_args()
    result = PRTIResultsReporter(
        PRTIResultsConfig(
            baseline_directory=args.baseline_directory,
            sensitivity_results=args.sensitivity_results,
            robustness_summary=args.robustness_summary,
            output_directory=args.output_directory,
        )
    ).write()
    print("Generated aggregate PRTI-GH result files:")
    for name, path in result["paths"].items():
        print(f"- {name}: {path.resolve()}")
    print("\nBaseline privacy/utility table:")
    print(result["tables"]["baseline_privacy_utility"].to_string(index=False))


if __name__ == "__main__":
    main()
