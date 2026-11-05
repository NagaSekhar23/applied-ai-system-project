"""Aggregate metrics for an evaluation run.

## What these metrics mean

Phase 2 metrics measure **deterministic correctness**: did the application
produce exactly the output the case specified? Because the system under test is
deterministic, a pass rate of 1.0 means the implementation matches the dataset,
nothing more. It is not a measure of model quality, user satisfaction, or
coverage of behavior the dataset does not describe.

Later phases add model-based metrics. Those are reported under a different
``kind`` so the two are never averaged together or confused in a report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum


class MetricKind(str, Enum):
    """What a metric block is actually measuring."""

    DETERMINISTIC_CORRECTNESS = "deterministic_correctness"
    MODEL_QUALITY = "model_quality"
    RETRIEVAL_QUALITY = "retrieval_quality"


@dataclass(frozen=True)
class CategoryMetrics:
    """Per-category breakdown."""

    category: str
    total: int
    passed: int
    failed: int
    errored: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate result of one evaluation run.

    Attributes:
        kind: What these numbers measure. See module docstring.
        total_cases: Cases attempted.
        passed: Cases whose actual output matched ``expected``.
        failed: Cases that ran but did not match.
        errored: Cases where the executor raised an unexpected exception.
            Counted as not passing, but tracked separately because an error is
            an infrastructure problem, not a wrong answer.
        duration_seconds: Wall-clock duration of the run.
    """

    kind: MetricKind
    total_cases: int
    passed: int
    failed: int
    errored: int
    duration_seconds: float
    by_category: list[CategoryMetrics] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Passed / total. 0.0 for an empty run."""
        return self.passed / self.total_cases if self.total_cases else 0.0

    @property
    def failure_rate(self) -> float:
        """(failed + errored) / total. The complement of ``pass_rate``."""
        return (
            (self.failed + self.errored) / self.total_cases if self.total_cases else 0.0
        )

    @property
    def accuracy(self) -> float:
        """Alias of ``pass_rate``.

        For deterministic cases with one correct answer each, "accuracy" and
        "pass rate" are the same number. Kept as a separate name because
        reviewers look for it, but it carries no extra meaning here.
        """
        return self.pass_rate

    def to_dict(self) -> dict:
        """JSON-serialisable form, with derived rates made explicit."""
        return {
            "kind": self.kind.value,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "pass_rate": round(self.pass_rate, 6),
            "failure_rate": round(self.failure_rate, 6),
            "accuracy": round(self.accuracy, 6),
            "duration_seconds": round(self.duration_seconds, 6),
            "by_category": [
                {**asdict(cat), "pass_rate": round(cat.pass_rate, 6)}
                for cat in self.by_category
            ],
        }


def compute_metrics(
    results,
    duration_seconds: float,
    kind: MetricKind = MetricKind.DETERMINISTIC_CORRECTNESS,
) -> EvaluationMetrics:
    """Aggregate a list of ``CaseResult`` into metrics.

    Args:
        results: Iterable of case results with ``.status`` and ``.category``.
        duration_seconds: Measured wall-clock duration of the run.
        kind: What the resulting numbers measure.
    """
    results = list(results)
    status_counts = Counter(r.status for r in results)

    per_category: dict[str, Counter] = {}
    for result in results:
        per_category.setdefault(result.category, Counter())[result.status] += 1

    by_category = [
        CategoryMetrics(
            category=category,
            total=sum(counts.values()),
            passed=counts.get("pass", 0),
            failed=counts.get("fail", 0),
            errored=counts.get("error", 0),
        )
        for category, counts in sorted(per_category.items())
    ]

    return EvaluationMetrics(
        kind=kind,
        total_cases=len(results),
        passed=status_counts.get("pass", 0),
        failed=status_counts.get("fail", 0),
        errored=status_counts.get("error", 0),
        duration_seconds=duration_seconds,
        by_category=by_category,
    )
