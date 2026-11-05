"""Metrics computed from recorded events.

## How each metric is calculated, and what it cannot tell you

All of these are computed over whatever events are currently in the store —
a local, in-process window, not a production time series. With a handful of
events the percentages are noisy; with none they are zero rather than
undefined. The dashboard shows the sample size next to every figure for
exactly that reason.

| Metric | Formula | Caveat |
|---|---|---|
| success_rate | ok events / total events | "ok" includes `degraded` (a fallback still answered the user) |
| failure_rate | 1 − success_rate | — |
| avg_latency_ms | mean of `duration_ms` | A mean hides bimodal cache-hit/miss behaviour |
| p95_latency_ms | nearest-rank 95th percentile | Meaningless below ~20 samples |
| llm_failure_rate | failed LLM events / LLM events | Counts a failed *call*, after the client's own retries |
| tool_failure_rate | failed tool events / tool events | A recoverable failure the agent corrected still counts |
| retry_rate | total retries / LLM calls | Retries per call, so it can exceed 1.0 |
| fallback_rate | degraded events / events that could degrade | — |
| evaluation_pass_rate | read from the latest report files | Not computed here; it comes from `python -m evaluation.run` |
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from observability.events import Component, Event, Status


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty list.

    Nearest-rank rather than interpolating, so the result is always a value
    that actually occurred.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class ComponentMetrics:
    """Per-component rollup."""

    component: str
    total: int = 0
    ok: int = 0
    errors: int = 0
    degraded: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return _rate(self.ok, self.total)

    @property
    def failure_rate(self) -> float:
        return _rate(self.errors, self.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "total": self.total,
            "ok": self.ok,
            "errors": self.errors,
            "degraded": self.degraded,
            "success_rate": round(self.success_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


@dataclass
class SystemMetrics:
    """Everything the dashboard shows, computed from recorded events."""

    total_events: int = 0
    ok_events: int = 0
    error_events: int = 0
    degraded_events: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    llm_calls: int = 0
    llm_failures: int = 0
    llm_retries: int = 0
    llm_timeouts: int = 0
    llm_validation_failures: int = 0

    tool_calls: int = 0
    tool_failures: int = 0

    agent_runs: int = 0
    agent_iterations: int = 0

    known_tokens_in: int = 0
    known_tokens_out: int = 0
    priced_calls: int = 0
    unpriced_calls: int = 0
    estimated_cost_usd: float = 0.0

    by_component: list[ComponentMetrics] = field(default_factory=list)
    tool_usage: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return _rate(self.ok_events, self.total_events)

    @property
    def failure_rate(self) -> float:
        return _rate(self.error_events, self.total_events)

    @property
    def llm_failure_rate(self) -> float:
        return _rate(self.llm_failures, self.llm_calls)

    @property
    def tool_failure_rate(self) -> float:
        return _rate(self.tool_failures, self.tool_calls)

    @property
    def retry_rate(self) -> float:
        """Retries per LLM call. Can exceed 1.0 when calls retry repeatedly."""
        return _rate(self.llm_retries, self.llm_calls)

    @property
    def fallback_rate(self) -> float:
        return _rate(self.degraded_events, self.total_events)

    @property
    def avg_agent_iterations(self) -> float:
        return _rate(self.agent_iterations, self.agent_runs)

    @property
    def cost_is_complete(self) -> bool:
        """True only if every LLM call could be priced."""
        return self.unpriced_calls == 0 and self.priced_calls > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "ok_events": self.ok_events,
            "error_events": self.error_events,
            "degraded_events": self.degraded_events,
            "success_rate": round(self.success_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "llm": {
                "calls": self.llm_calls,
                "failures": self.llm_failures,
                "failure_rate": round(self.llm_failure_rate, 4),
                "retries": self.llm_retries,
                "retry_rate": round(self.retry_rate, 4),
                "timeouts": self.llm_timeouts,
                "validation_failures": self.llm_validation_failures,
            },
            "tools": {
                "calls": self.tool_calls,
                "failures": self.tool_failures,
                "failure_rate": round(self.tool_failure_rate, 4),
                "usage": dict(sorted(self.tool_usage.items())),
            },
            "agent": {
                "runs": self.agent_runs,
                "total_iterations": self.agent_iterations,
                "avg_iterations": round(self.avg_agent_iterations, 2),
            },
            "cost": {
                "known_input_tokens": self.known_tokens_in,
                "known_output_tokens": self.known_tokens_out,
                "priced_calls": self.priced_calls,
                "unpriced_calls": self.unpriced_calls,
                "estimated_usd": round(self.estimated_cost_usd, 6),
                "complete": self.cost_is_complete,
            },
            "fallback_rate": round(self.fallback_rate, 4),
        }


def compute_metrics(events: Iterable[Event]) -> SystemMetrics:
    """Roll a list of events up into system metrics."""
    events = list(events)
    metrics = SystemMetrics(total_events=len(events))

    latencies: list[float] = []
    per_component: dict[str, dict[str, Any]] = {}

    for event in events:
        latencies.append(event.duration_ms)

        if event.status is Status.ERROR:
            metrics.error_events += 1
        else:
            metrics.ok_events += 1
            if event.status is Status.DEGRADED:
                metrics.degraded_events += 1

        bucket = per_component.setdefault(
            event.component.value,
            {"total": 0, "ok": 0, "errors": 0, "degraded": 0, "latencies": []},
        )
        bucket["total"] += 1
        bucket["latencies"].append(event.duration_ms)
        if event.status is Status.ERROR:
            bucket["errors"] += 1
        else:
            bucket["ok"] += 1
            if event.status is Status.DEGRADED:
                bucket["degraded"] += 1

        attrs = event.attributes

        if event.component is Component.LLM:
            metrics.llm_calls += 1
            if event.status is Status.ERROR:
                metrics.llm_failures += 1
            metrics.llm_retries += int(attrs.get("retries", 0) or 0)
            metrics.llm_timeouts += int(attrs.get("timeouts", 0) or 0)
            metrics.llm_validation_failures += int(attrs.get("validation_failures", 0) or 0)

            if event.cost:
                if event.cost.input_tokens is not None:
                    metrics.known_tokens_in += event.cost.input_tokens
                if event.cost.output_tokens is not None:
                    metrics.known_tokens_out += event.cost.output_tokens
                if event.cost.priced and event.cost.usd is not None:
                    metrics.priced_calls += 1
                    metrics.estimated_cost_usd += event.cost.usd
                else:
                    metrics.unpriced_calls += 1

        elif event.component is Component.TOOL:
            metrics.tool_calls += 1
            if event.status is Status.ERROR:
                metrics.tool_failures += 1
            name = attrs.get("tool")
            if name:
                metrics.tool_usage[name] = metrics.tool_usage.get(name, 0) + 1

        elif event.component is Component.AGENT:
            metrics.agent_runs += 1
            metrics.agent_iterations += int(attrs.get("iterations", 0) or 0)

    if latencies:
        metrics.avg_latency_ms = sum(latencies) / len(latencies)
        metrics.p95_latency_ms = percentile(latencies, 0.95)

    metrics.by_component = [
        ComponentMetrics(
            component=name,
            total=data["total"],
            ok=data["ok"],
            errors=data["errors"],
            degraded=data["degraded"],
            avg_latency_ms=sum(data["latencies"]) / len(data["latencies"]),
            p95_latency_ms=percentile(data["latencies"], 0.95),
        )
        for name, data in sorted(per_component.items())
    ]

    return metrics
