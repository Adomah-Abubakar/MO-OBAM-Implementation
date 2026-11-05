"""Generate the aggregate PRTI-GH result tables, figures, and manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from mo_obam.prti_results_package import (
    PRTIResultsPackageBuilder,
    PRTIResultsPackageConfig,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose only evidence locations; the command never reruns anonymization."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("evidence/prti-results-v2"),
    )
    return parser


def main() -> None:
    """Build the package and print its reporting artifacts."""

    args = build_parser().parse_args()
    result = PRTIResultsPackageBuilder(
        PRTIResultsPackageConfig(output_directory=args.output_directory)
    ).write()
    print("Generated aggregate PRTI-GH results package:")
    for path in result["table_paths"].values():
        print(f"- table: {path.resolve()}")
    for path in result["figure_paths"]:
        print(f"- figure: {path.resolve()}")
    print(f"- index: {result['readme_path'].resolve()}")
    print(f"- manifest: {result['manifest_path'].resolve()}")


if __name__ == "__main__":
    main()
