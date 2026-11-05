"""Agent execution trace.

Every decision and tool call is recorded as a span with a parent, so a run can
be reconstructed afterwards. Phase 6 consumes these; the structure exists from
the start because retro-fitting tracing means changing every call site.

Nothing sensitive is recorded: tool arguments are captured, but the game's
secret never appears in a tool's public output while a game is live.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

SpanKind = Literal["agent", "decision", "tool", "response"]


@dataclass
class Span:
    """One timed step within a trace."""

    name: str
    kind: SpanKind
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    iteration: int = 0
    status: str = "ok"
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "iteration": self.iteration,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "error": self.error,
        }


@dataclass
class AgentTrace:
    """The full record of one agent run."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    request: str = ""
    spans: list[Span] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    iterations: int = 0
    stop_reason: str = ""

    def add(self, span: Span) -> Span:
        self.spans.append(span)
        return span

    @property
    def tool_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "tool"]

    @property
    def tools_called(self) -> list[str]:
        return [s.attributes.get("tool", "") for s in self.tool_spans]

    @property
    def failed_spans(self) -> list[Span]:
        return [s for s in self.spans if s.status != "ok"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "request": self.request,
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "duration_ms": round(self.duration_ms, 2),
            "tools_called": self.tools_called,
            "spans": [s.to_dict() for s in self.spans],
        }


class SpanTimer:
    """Context manager that times a span and records failures.

    Exceptions are recorded on the span and re-raised — never swallowed.
    """

    def __init__(self, trace: AgentTrace, span: Span):
        self.trace = trace
        self.span = span
        self._started = 0.0

    def __enter__(self) -> Span:
        self._started = time.perf_counter()
        self.trace.add(self.span)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        self.span.duration_ms = (time.perf_counter() - self._started) * 1000
        if exc is not None:
            self.span.status = "error"
            self.span.error = f"{exc_type.__name__}: {exc}"
        return False
