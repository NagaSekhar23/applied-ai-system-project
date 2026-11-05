"""Evaluation reports: human-readable text and machine-readable JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.runner import EvaluationRun

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-") or "dataset"


def render_text(run: EvaluationRun, max_failures: int = 20) -> str:
    """Render a run as readable text for a terminal."""
    m = run.metrics
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append(f"Evaluation: {run.dataset_name} (v{run.dataset_version})")
    lines.append(f"Started:    {run.started_at}Z")
    lines.append(f"Measuring:  {m.kind.value}")
    lines.append(f"Fingerprint:{run.fingerprint}")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"  Total cases : {m.total_cases}")
    lines.append(f"  Passed      : {m.passed}")
    lines.append(f"  Failed      : {m.failed}")
    lines.append(f"  Errored     : {m.errored}")
    lines.append(f"  Pass rate   : {m.pass_rate:.1%}")
    lines.append(f"  Failure rate: {m.failure_rate:.1%}")
    lines.append(f"  Duration    : {m.duration_seconds:.3f}s")
    lines.append("")

    if m.by_category:
        lines.append("By category")
        lines.append("-" * 68)
        width = max(len(c.category) for c in m.by_category)
        for cat in m.by_category:
            bar_cells = round(cat.pass_rate * 20)
            bar = "#" * bar_cells + "." * (20 - bar_cells)
            lines.append(
                f"  {cat.category.ljust(width)}  {bar}  "
                f"{cat.passed}/{cat.total}  ({cat.pass_rate:.0%})"
            )
        lines.append("")

    failures = run.failed_results
    if failures:
        lines.append(f"Failures ({len(failures)})")
        lines.append("-" * 68)
        for result in failures[:max_failures]:
            lines.append(f"  [{result.status.upper()}] {result.case_id}  ({result.target})")
            if result.expected_behavior:
                lines.append(f"      expected: {result.expected_behavior}")
            for diff in result.diffs:
                lines.append(f"      - {diff.describe()}")
            if result.error:
                first_line = result.error.splitlines()[0]
                lines.append(f"      ! {first_line}")
            lines.append("")
        if len(failures) > max_failures:
            lines.append(f"  ... and {len(failures) - max_failures} more (see JSON report)")
            lines.append("")
    else:
        lines.append("All cases passed.")
        lines.append("")

    return "\n".join(lines)


def write_reports(run: EvaluationRun, report_dir: str | Path) -> dict[str, Path]:
    """Write timestamped JSON and text reports, plus a stable ``latest`` copy.

    The timestamped files build a history of runs; the ``latest`` files give
    tooling (and the Streamlit dashboard) a fixed path to read.
    """
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    stamp = run.started_at.replace(":", "").replace("-", "")
    base = f"{_slug(run.dataset_name)}-{stamp}"

    json_path = directory / f"{base}.json"
    text_path = directory / f"{base}.txt"
    latest_json = directory / f"latest-{_slug(run.dataset_name)}.json"

    payload = json.dumps(run.to_dict(), indent=2, sort_keys=False)
    json_path.write_text(payload, encoding="utf-8")
    text_path.write_text(render_text(run), encoding="utf-8")
    latest_json.write_text(payload, encoding="utf-8")

    return {"json": json_path, "text": text_path, "latest_json": latest_json}


def load_latest_reports(report_dir: str | Path) -> list[dict]:
    """Read every ``latest-*.json`` report. Used by the observability dashboard.

    Unreadable or malformed files are skipped rather than crashing a dashboard.
    """
    directory = Path(report_dir)
    if not directory.is_dir():
        return []
    reports = []
    for path in sorted(directory.glob("latest-*.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return reports
