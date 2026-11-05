"""The tracker: request context and the recording API.

Request and trace ids live in ``contextvars``, so every event recorded during
one user action is correlated without threading an id through every function
signature. That is what keeps observability out of the business logic: no
function in ``game.py``, ``logic_utils.py`` or ``assistant.py`` imports this
module.

Services produce data (``CallStats``, ``AgentTrace``); the caller hands that
data here. The dependency points one way only.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from observability.events import Component, Event, Status, TokenCost
from observability.metrics import SystemMetrics, compute_metrics
from observability.pricing import estimate_cost
from observability.redaction import summarise_text
from observability.store import EventStore

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def current_request_id() -> str | None:
    return _request_id.get()


def current_trace_id() -> str | None:
    return _trace_id.get()


class Tracker:
    """Records events into a store and computes metrics from them."""

    def __init__(self, store: EventStore | None = None, clock=time.perf_counter):
        self.store = store or EventStore()
        self._clock = clock

    # --- Request context -----------------------------------------------------

    @contextmanager
    def request(self, operation: str, request_id: str | None = None) -> Iterator[str]:
        """Bind a request id for everything recorded inside the block."""
        rid = request_id or new_id()
        token = _request_id.set(rid)
        try:
            yield rid
        finally:
            _request_id.reset(token)

    @contextmanager
    def operation(
        self,
        operation: str,
        component: Component,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Event]:
        """Time an operation and record it, including on failure.

        Exceptions are recorded and then re-raised — never swallowed.
        """
        event = Event(
            operation=operation,
            component=component,
            request_id=current_request_id() or new_id(),
            trace_id=current_trace_id(),
            attributes=dict(attributes or {}),
        )
        started = self._clock()
        try:
            yield event
        except Exception as exc:
            event.duration_ms = (self._clock() - started) * 1000
            event.status = Status.ERROR
            event.error_type = type(exc).__name__
            event.error_message = str(exc)
            self.store.record(event)
            raise
        else:
            event.duration_ms = (self._clock() - started) * 1000
            self.store.record(event)

    # --- Recording helpers ---------------------------------------------------

    def record(self, event: Event) -> Event:
        return self.store.record(event)

    def record_llm_call(
        self,
        operation: str,
        stats,
        status: Status = Status.OK,
        extra: dict[str, Any] | None = None,
    ) -> Event:
        """Record one LLM call from a ``providers.CallStats``."""
        cost = estimate_cost(stats.model or "", stats.usage, provider=stats.provider)
        attributes: dict[str, Any] = {
            "provider": stats.provider,
            "model": stats.model,
            "attempts": stats.attempts,
            "retries": stats.retries,
            "timeouts": stats.timeouts,
            "validation_failures": stats.validation_failures,
        }
        if extra:
            attributes.update(extra)

        return self.record(
            Event(
                operation=operation,
                component=Component.LLM,
                status=status,
                duration_ms=stats.latency_ms,
                request_id=current_request_id() or new_id(),
                trace_id=current_trace_id(),
                error_type=_first_error_type(stats.errors),
                error_message=stats.errors[-1] if stats.errors else None,
                attributes=attributes,
                cost=cost,
            )
        )

    def record_assistant_reply(self, reply, question: str | None = None) -> Event:
        """Record an assistant reply, including its LLM call when there was one."""
        if reply.stats is not None:
            self.record_llm_call(
                "assistant.hint",
                reply.stats,
                status=Status.OK if reply.used_llm else Status.ERROR,
            )

        return self.record(
            Event(
                operation="assistant.reply",
                component=Component.ASSISTANT,
                status=Status.OK if reply.used_llm else Status.DEGRADED,
                duration_ms=reply.stats.latency_ms if reply.stats else 0.0,
                request_id=current_request_id() or new_id(),
                trace_id=current_trace_id(),
                error_message=reply.fallback_reason,
                attributes={
                    "source": reply.source,
                    "category": reply.category.value,
                    "prompt_id": reply.prompt_id,
                    "warnings": len(reply.warnings),
                    "question": summarise_text(question),
                },
            )
        )

    def record_parse(self, outcome, message: str | None = None) -> Event:
        """Record one natural language parse."""
        if outcome.stats is not None:
            self.record_llm_call("parser.parse", outcome.stats)

        return self.record(
            Event(
                operation="parser.parse",
                component=Component.PARSER,
                status=Status.OK if outcome.source == "llm" else Status.DEGRADED,
                duration_ms=outcome.stats.latency_ms if outcome.stats else 0.0,
                request_id=current_request_id() or new_id(),
                trace_id=current_trace_id(),
                error_message=outcome.fallback_reason,
                attributes={
                    "command": outcome.command.command.value,
                    "source": outcome.source,
                    "confidence": outcome.command.confidence,
                    "message": summarise_text(message),
                },
            )
        )

    def record_agent_run(self, run) -> list[Event]:
        """Record an agent run: the run itself, plus one event per tool call.

        Trace ids connect them, so the dashboard can show a whole run.
        """
        trace_id = run.trace.trace_id
        token = _trace_id.set(trace_id)
        request_id = current_request_id() or new_id()
        recorded: list[Event] = []

        try:
            for span in run.trace.tool_spans:
                recorded.append(
                    self.record(
                        Event(
                            operation=f"tool.{span.attributes.get('tool', 'unknown')}",
                            component=Component.TOOL,
                            status=Status.ERROR if span.status == "error" else Status.OK,
                            duration_ms=span.duration_ms,
                            request_id=request_id,
                            trace_id=trace_id,
                            parent_id=span.parent_id,
                            error_message=span.error,
                            attributes={
                                "tool": span.attributes.get("tool"),
                                "arguments": span.attributes.get("arguments", {}),
                                "iteration": span.iteration,
                            },
                        )
                    )
                )

            for plan in run.plans:
                if plan.stats is not None:
                    recorded.append(self.record_llm_call("agent.plan", plan.stats))

            recorded.append(
                self.record(
                    Event(
                        operation="agent.run",
                        component=Component.AGENT,
                        status=Status.OK if run.succeeded else Status.ERROR,
                        duration_ms=run.trace.duration_ms,
                        request_id=request_id,
                        trace_id=trace_id,
                        attributes={
                            "iterations": run.iterations,
                            "stop_reason": run.stop_reason,
                            "tools_called": run.tools_called,
                            "tool_count": len(run.tool_results),
                            "request": summarise_text(run.request),
                        },
                    )
                )
            )
        finally:
            _trace_id.reset(token)

        return recorded

    def record_game_action(self, operation: str, status: Status, **attributes) -> Event:
        """Record a deterministic game action."""
        return self.record(
            Event(
                operation=operation,
                component=Component.GAME,
                status=status,
                request_id=current_request_id() or new_id(),
                attributes=attributes,
            )
        )

    # --- Reading -------------------------------------------------------------

    def metrics(self) -> SystemMetrics:
        return compute_metrics(self.store.all())

    def clear(self) -> None:
        self.store.clear()


def _first_error_type(errors: list[str]) -> str | None:
    """Pull the exception class name out of a recorded error string."""
    if not errors:
        return None
    return errors[-1].split(":", 1)[0].strip() or None


_default_tracker: Tracker | None = None


def get_tracker() -> Tracker:
    """The process-wide tracker used by the UI."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = Tracker()
    return _default_tracker


def set_tracker(tracker: Tracker | None) -> None:
    """Replace the process-wide tracker. Used by tests for isolation."""
    global _default_tracker
    _default_tracker = tracker
