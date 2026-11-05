"""Executors for the agent layer.

Measured here: tool-selection correctness, argument validity, execution
success, unnecessary-call rate, iteration counts, and the guardrails.

With the mock provider, planning is the deterministic rule-based planner, so
these are ``deterministic_correctness`` checks on orchestration, not a measure
of how well a real model plans.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from agents import AgentConfig, AgentController, Planner, default_registry
from assistant import Assistant
from evaluation.cases import EvalCase
from evaluation.executors import Executor, _capture
from game import new_game
from providers import MockProvider, ReliableLLMClient, ScriptedProvider

READ_ONLY_TOOLS = {"get_game_status", "request_hint", "explain_rules"}


def _no_sleep(_seconds: float) -> None:
    pass


def _state_from(case: EvalCase):
    spec = case.initial_state or {}
    if not spec.get("game_in_progress", True):
        return None
    state = new_game(spec.get("difficulty", "Normal"), rng=random.Random(spec.get("seed", 0)))
    changes = {
        key: spec[key]
        for key in ("secret", "status", "attempts_used", "score")
        if key in spec
    }
    return replace(state, **changes) if changes else state


def _build_agent(case: EvalCase) -> AgentController:
    registry = default_registry()
    mock = ReliableLLMClient(MockProvider(), sleep=_no_sleep)

    if case.input.get("provider") == "scripted":
        planner_client = ReliableLLMClient(
            ScriptedProvider(list(case.input.get("script", []))),
            max_retries=case.input.get("max_retries", 1),
            sleep=_no_sleep,
        )
    else:
        planner_client = mock

    config = AgentConfig(
        max_iterations=case.input.get("max_iterations", 5),
        timeout_seconds=case.input.get("timeout_seconds", 30.0),
        max_tool_errors=case.input.get("max_tool_errors", 3),
        allow_repeat_calls=case.input.get("allow_repeat_calls", False),
    )
    return AgentController(
        registry=registry,
        planner=Planner(registry, planner_client),
        assistant=Assistant(mock),
        config=config,
    )


def execute_agent_run(case: EvalCase) -> dict[str, Any]:
    """Run the agent on one request and report observable properties."""

    def run() -> dict[str, Any]:
        state = _state_from(case)
        agent = _build_agent(case)
        result = agent.run(
            case.input["message"], state, last_outcome=case.input.get("last_outcome")
        )

        after = result.state
        called = result.tools_called
        expected_tools = set(case.input.get("expected_tools", called))
        unnecessary = [t for t in called if t not in expected_tools]
        registry_names = set(agent.registry.names)

        return {
            "tools_called": called,
            "first_tool": called[0] if called else None,
            "tool_count": len(called),
            "all_tools_registered": all(t in registry_names for t in called),
            "succeeded": result.succeeded,
            "stop_reason": result.stop_reason,
            "iterations": result.iterations,
            "unnecessary_tool_calls": len(unnecessary),
            "failed_tool_calls": sum(1 for r in result.tool_results if not r.ok),
            "mutated_state": after is not state and after != state,
            "attempts_used": after.attempts_used if after else None,
            "status": after.status if after else None,
            "difficulty": after.difficulty if after else None,
            "score": after.score if after else None,
            "response_is_non_empty": bool(result.response.strip()),
            "trace_span_count": len(result.trace.spans),
        }

    return _capture(run)


def execute_tool_contract(case: EvalCase) -> dict[str, Any]:
    """Validate arguments against a tool's declared input schema."""

    def run() -> dict[str, Any]:
        registry = default_registry()
        name = case.input["tool"]
        if name not in registry:
            return {"registered": False, "arguments_valid": False}
        tool = registry.get(name)
        try:
            tool.validate_arguments(case.input.get("arguments", {}))
        except Exception as exc:
            return {
                "registered": True,
                "arguments_valid": False,
                "error_type": type(exc).__name__,
            }
        return {
            "registered": True,
            "arguments_valid": True,
            "mutates_state": tool.mutates_state,
        }

    return _capture(run)


AGENT_EXECUTORS: dict[str, Executor] = {
    "agent_run": execute_agent_run,
    "tool_contract": execute_tool_contract,
}
