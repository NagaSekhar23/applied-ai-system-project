"""Evaluation runner: load, execute, compare, aggregate.

The runner is deliberately generic. It knows how to run cases and compare
results; it knows nothing about the game. Everything application-specific lives
in an executor, which is how later phases reuse this runner unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from evaluation.cases import Dataset, EvalCase, load_dataset
from evaluation.executors import Executor, default_registry, resolve
from evaluation.metrics import EvaluationMetrics, MetricKind, compute_metrics

Status = Literal["pass", "fail", "error"]


@dataclass(frozen=True)
class FieldDiff:
    """One mismatched field in a failed case."""

    field_name: str
    expected: Any
    actual: Any

    def describe(self) -> str:
        return f"{self.field_name}: expected {self.expected!r}, got {self.actual!r}"


@dataclass(frozen=True)
class CaseResult:
    """Outcome of one evaluation case."""

    case_id: str
    category: str
    target: str
    status: Status
    expected: dict[str, Any]
    actual: dict[str, Any]
    diffs: list[FieldDiff] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0
    expected_behavior: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "target": self.target,
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "diffs": [
                {"field": d.field_name, "expected": d.expected, "actual": d.actual}
                for d in self.diffs
            ],
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 6),
            "expected_behavior": self.expected_behavior,
        }


@dataclass(frozen=True)
class EvaluationRun:
    """A complete evaluation run: dataset identity, results and metrics."""

    dataset_name: str
    dataset_version: int
    results: list[CaseResult]
    metrics: EvaluationMetrics
    started_at: str
    fingerprint: str

    @property
    def failed_results(self) -> list[CaseResult]:
        """Failures and errors, in dataset order."""
        return [r for r in self.results if r.status != "pass"]

    @property
    def ok(self) -> bool:
        """True when every case passed."""
        return not self.failed_results

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": {"name": self.dataset_name, "version": self.dataset_version},
            "started_at": self.started_at,
            "fingerprint": self.fingerprint,
            "metrics": self.metrics.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }


def _normalise(value: Any) -> Any:
    """Make values comparable across the JSON boundary.

    JSON has no tuple type, so a tuple from the application and a list from the
    dataset must compare equal. Nested structures are normalised too.
    """
    if isinstance(value, tuple):
        return [_normalise(v) for v in value]
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    return value


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[FieldDiff]:
    """Subset comparison: every expected key must match.

    Extra keys in ``actual`` are ignored, so a case asserts only what it cares
    about. A missing key is reported as a diff rather than raising.
    """
    diffs: list[FieldDiff] = []
    for key, expected_value in expected.items():
        if key not in actual:
            diffs.append(FieldDiff(key, _normalise(expected_value), "<missing>"))
            continue
        if _normalise(expected_value) != _normalise(actual[key]):
            diffs.append(FieldDiff(key, _normalise(expected_value), _normalise(actual[key])))
    return diffs


def run_case(case: EvalCase, registry: dict[str, Executor]) -> CaseResult:
    """Execute one case and compare its result.

    An exception from the executor becomes an ``error`` result rather than
    aborting the run, so one broken case cannot hide the other results.
    """
    started = time.perf_counter()
    try:
        executor = resolve(registry, case.target)
        actual = executor(case)
    except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
        return CaseResult(
            case_id=case.id,
            category=case.category,
            target=case.target,
            status="error",
            expected=case.expected,
            actual={},
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
            duration_seconds=time.perf_counter() - started,
            expected_behavior=case.expected_behavior,
        )

    duration = time.perf_counter() - started
    diffs = compare(case.expected, actual)
    return CaseResult(
        case_id=case.id,
        category=case.category,
        target=case.target,
        status="pass" if not diffs else "fail",
        expected=case.expected,
        actual=_normalise(actual),
        diffs=diffs,
        duration_seconds=duration,
        expected_behavior=case.expected_behavior,
    )


def fingerprint_results(results: list[CaseResult]) -> str:
    """Stable hash of (case id, status) pairs.

    Two runs of the same dataset against the same code produce the same
    fingerprint. Durations and timestamps are excluded precisely because they
    vary between runs — this is what makes reproducibility testable.
    """
    payload = json.dumps(
        [[r.case_id, r.status] for r in results], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_dataset(
    dataset: Dataset,
    registry: dict[str, Executor] | None = None,
    kind: MetricKind = MetricKind.DETERMINISTIC_CORRECTNESS,
) -> EvaluationRun:
    """Run every case in a dataset, in dataset order."""
    active_registry = registry if registry is not None else default_registry()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    started = time.perf_counter()

    results = [run_case(case, active_registry) for case in dataset.cases]

    duration = time.perf_counter() - started
    return EvaluationRun(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        results=results,
        metrics=compute_metrics(results, duration, kind=kind),
        started_at=started_at,
        fingerprint=fingerprint_results(results),
    )


def run_dataset_file(
    path: str | Path,
    registry: dict[str, Executor] | None = None,
    kind: MetricKind = MetricKind.DETERMINISTIC_CORRECTNESS,
) -> EvaluationRun:
    """Load a dataset file and run it."""
    return run_dataset(load_dataset(path), registry=registry, kind=kind)
