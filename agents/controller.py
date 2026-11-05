"""Agent controller: the bounded execution loop.

Every loop here terminates. Four independent guards ensure it:

1. ``max_iterations`` — a hard ceiling on passes.
2. ``timeout_seconds`` — wall-clock budget, checked before each pass.
3. Repeated-call protection — the same tool with the same arguments is
   refused the second time, which is how a confused planner would otherwise
   spin.
4. ``max_tool_errors`` — a planner that keeps producing invalid calls stops.

A tool error is recoverable: it is fed back as an observation so the planner
can correct itself. A guard breach is terminal.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agents.planner import PlanOutcome, Planner
from agents.tools import (
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
)
from agents.trace import AgentTrace, Span, SpanTimer
from assistant import Assistant
from game import GameState


@dataclass
class AgentConfig:
    """Guardrail settings."""

    max_iterations: int = 5
    timeout_seconds: float = 30.0
    max_tool_errors: int = 3
    allow_repeat_calls: bool = False


@dataclass
class AgentRun:
    """Everything one agent run produced."""

    request: str
    response: str
    state: GameState | None
    last_outcome: str | None
    trace: AgentTrace
    tool_results: list[ToolResult] = field(default_factory=list)
    plans: list[PlanOutcome] = field(default_factory=list)
    succeeded: bool = True

    @property
    def tools_called(self) -> list[str]:
        return [r.tool for r in self.tool_results]

    @property
    def iterations(self) -> int:
        return self.trace.iterations

    @property
    def stop_reason(self) -> str:
        return self.trace.stop_reason


def _call_signature(tool: str, arguments: dict[str, Any]) -> str:
    """Stable identity for a tool call, used for repeat detection."""
    return f"{tool}:{json.dumps(arguments or {}, sort_keys=True)}"


class AgentController:
    """Runs the plan → validate → execute → observe loop, under guardrails."""

    def __init__(
        self,
        registry: ToolRegistry,
        planner: Planner,
        assistant: Assistant | None = None,
        config: AgentConfig | None = None,
        clock=time.monotonic,
    ):
        self.registry = registry
        self.planner = planner
        self.assistant = assistant
        self.config = config or AgentConfig()
        self._clock = clock

    def run(
        self,
        message: str,
        state: GameState | None,
        last_outcome: str | None = None,
    ) -> AgentRun:
        """Handle one user request."""
        trace = AgentTrace(request=message)
        root = trace.add(Span(name="agent_run", kind="agent", attributes={"request_chars": len(message)}))

        started = self._clock()
        completed: list[str] = []
        observations: list[str] = []  # for the planner: prefixed with the tool name
        messages: list[str] = []  # for the user: plain text, no tool names
        seen_calls: set[str] = set()
        tool_results: list[ToolResult] = []
        plans: list[PlanOutcome] = []
        error_count = 0
        response: str | None = None
        stop_reason = "completed"

        for iteration in range(1, self.config.max_iterations + 1):
            trace.iterations = iteration

            if self._clock() - started > self.config.timeout_seconds:
                stop_reason = "timeout"
                break

            plan = self._plan(message, state, completed, observations, trace, root, iteration)
            plans.append(plan)
            decision = plan.decision

            if decision.action in ("respond", "clarify"):
                response = decision.response
                stop_reason = "responded" if decision.action == "respond" else "clarification"
                break

            tool_name = decision.tool or ""
            signature = _call_signature(tool_name, decision.arguments)
            if not self.config.allow_repeat_calls and signature in seen_calls:
                stop_reason = "repeated_call"
                observations.append(f"{tool_name} was already called with those arguments.")
                break
            seen_calls.add(signature)

            result = self._execute(tool_name, decision.arguments, state, last_outcome, trace, root, iteration)
            tool_results.append(result)

            if not result.ok:
                error_count += 1
                observations.append(f"{tool_name} failed: {result.error}")
                if error_count >= self.config.max_tool_errors:
                    stop_reason = "too_many_tool_errors"
                    break
                continue

            completed.append(result.tool)
            observations.append(f"{result.tool}: {result.message}")
            messages.append(result.message)
            if result.new_state is not None:
                state = result.new_state
            if result.new_last_outcome is not None:
                last_outcome = result.new_last_outcome
            elif result.tool in ("start_game", "reset_game", "change_difficulty"):
                last_outcome = None
        else:
            stop_reason = "max_iterations"

        final = self._finalise(response, messages, stop_reason, trace, root)
        root.duration_ms = (self._clock() - started) * 1000
        trace.duration_ms = root.duration_ms
        trace.stop_reason = stop_reason

        return AgentRun(
            request=message,
            response=final,
            state=state,
            last_outcome=last_outcome,
            trace=trace,
            tool_results=tool_results,
            plans=plans,
            succeeded=stop_reason in ("completed", "responded", "clarification"),
        )

    # --- Internals -----------------------------------------------------------

    def _plan(self, message, state, completed, observations, trace, root, iteration) -> PlanOutcome:
        span = Span(name="plan", kind="decision", parent_id=root.span_id, iteration=iteration)
        with SpanTimer(trace, span):
            plan = self.planner.plan(message, state, completed, observations)
            span.attributes.update(
                {
                    "action": plan.decision.action,
                    "tool": plan.decision.tool,
                    "source": plan.source,
                    "reasoning": plan.decision.reasoning,
                }
            )
            if plan.fallback_reason:
                span.attributes["fallback_reason"] = plan.fallback_reason
                span.status = "degraded"
        return plan

    def _execute(
        self, tool_name, arguments, state, last_outcome, trace, root, iteration
    ) -> ToolResult:
        """Validate arguments, run the tool, validate its output."""
        span = Span(
            name=f"tool:{tool_name}",
            kind="tool",
            parent_id=root.span_id,
            iteration=iteration,
            attributes={"tool": tool_name, "arguments": arguments},
        )
        with SpanTimer(trace, span):
            try:
                tool = self.registry.get(tool_name)
                validated_args = tool.validate_arguments(arguments)
                context = ToolContext(state=state, last_outcome=last_outcome, assistant=self.assistant)
                result = tool.run(validated_args, context)
                tool.validate_output(result)
            except ToolError as exc:
                span.status = "error"
                span.error = str(exc)
                return ToolResult(
                    tool=tool_name, ok=False, output={}, message=str(exc), error=str(exc)
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                span.status = "error"
                span.error = f"{type(exc).__name__}: {exc}"
                return ToolResult(
                    tool=tool_name,
                    ok=False,
                    output={},
                    message=f"{tool_name} raised an unexpected error.",
                    error=f"{type(exc).__name__}: {exc}",
                )

            span.attributes["ok"] = result.ok
            if not result.ok:
                span.status = "error"
                span.error = result.error
        return result

    def _finalise(self, response, messages, stop_reason, trace, root) -> str:
        """Compose the user-facing answer from plain tool messages.

        Tool names are deliberately absent here: they belong in the trace, not
        in what the player reads.
        """
        span = Span(name="respond", kind="response", parent_id=root.span_id)
        done = " ".join(messages)
        with SpanTimer(trace, span):
            if response:
                text = response
            elif stop_reason == "timeout":
                text = f"That took too long, so I stopped. {done}"
            elif stop_reason == "max_iterations":
                text = (
                    f"I reached my step limit before finishing. {done}"
                    if messages
                    else "I reached my step limit without completing anything."
                )
            elif stop_reason == "repeated_call":
                text = done or "I was about to repeat myself, so I stopped."
            elif stop_reason == "too_many_tool_errors":
                text = f"I could not complete that. {done}"
            elif messages:
                text = done
            else:
                text = "I was not able to do anything with that request."

            span.attributes["stop_reason"] = stop_reason
            span.attributes["response_chars"] = len(text)
        return text.strip()


def build_agent(
    assistant: Assistant | None = None,
    client=None,
    config: AgentConfig | None = None,
) -> AgentController:
    """Build an agent with the default tool registry."""
    from agents.tools import default_registry

    registry = default_registry()
    return AgentController(
        registry=registry,
        planner=Planner(registry, client),
        assistant=assistant,
        config=config,
    )
