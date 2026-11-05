"""CLI entry point: ``python -m evaluation.run``.

Exit codes:
    0  every case passed (and any --fail-under threshold was met)
    1  at least one case failed or errored
    2  the dataset could not be loaded or validated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.cases import DatasetError, discover_datasets, load_dataset
from evaluation.report import render_text, write_reports
from evaluation.runner import run_dataset

DEFAULT_DATASET_DIR = Path(__file__).parent / "datasets"
DEFAULT_REPORT_DIR = Path(__file__).parent.parent / "reports"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run",
        description="Run evaluation datasets against the application.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="PATH",
        help="Dataset JSON file. Repeatable. Defaults to every dataset in evaluation/datasets/.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help=f"Where to write reports (default: {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Print results without writing report files.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=1.0,
        metavar="RATE",
        help="Exit non-zero if the pass rate is below this (0.0-1.0, default 1.0).",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only the summary line.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    paths = [Path(p) for p in args.dataset] if args.dataset else discover_datasets(DEFAULT_DATASET_DIR)
    if not paths:
        print(f"No datasets found in {DEFAULT_DATASET_DIR}", file=sys.stderr)
        return 2

    exit_code = 0
    for path in paths:
        try:
            dataset = load_dataset(path)
        except DatasetError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        run = run_dataset(dataset)

        if args.quiet:
            m = run.metrics
            print(
                f"{run.dataset_name}: {m.passed}/{m.total_cases} passed "
                f"({m.pass_rate:.1%}) fingerprint={run.fingerprint}"
            )
        else:
            print(render_text(run))

        if not args.no_report:
            written = write_reports(run, args.report_dir)
            if not args.quiet:
                print(f"Reports written:\n  {written['json']}\n  {written['text']}\n")

        if run.metrics.pass_rate < args.fail_under:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
