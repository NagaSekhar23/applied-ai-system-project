"""Executors for the natural language layer.

Measured here: intent classification, argument extraction, schema validity,
correct execution, and safe rejection. With the mock provider these are
deterministic, so they are reported as ``deterministic_correctness``.

Not measured here: how well a *real* model handles unusual phrasing. The
rule-based parser is what runs in mock mode, so a passing result shows the
pipeline is correct, not that an LLM is accurate.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from assistant import Assistant
from evaluation.cases import EvalCase
from evaluation.executors import Executor, _capture
from game import new_game
from natural_language import (
    CommandExecutor,
    CommandParser,
    handle_message,
    rule_based_parse,
    validate,
)
from providers import MockProvider, ReliableLLMClient, ScriptedProvider


def _no_sleep(_seconds: float) -> None:
    pass


def _state_from(case: EvalCase):
    spec = case.initial_state or {}
    if not spec.get("game_in_progress", True):
        return None
    state = new_game(spec.get("difficulty", "Normal"), rng=random.Random(spec.get("seed", 0)))
    changes: dict[str, Any] = {
        key: spec[key]
        for key in ("secret", "status", "attempts_used", "score")
        if key in spec
    }
    if "history" in spec:
        changes["history"] = tuple(spec["history"])
    return replace(state, **changes) if changes else state


def execute_intent_parse(case: EvalCase) -> dict[str, Any]:
    """Classify intent and extract arguments, without executing anything."""

    def run() -> dict[str, Any]:
        command = rule_based_parse(case.input["message"])
        return {
            "command": command.command.value,
            "guess": command.guess,
            "difficulty": command.difficulty,
            "has_clarification": command.clarification is not None,
            "schema_valid": True,
        }

    return _capture(run)


def execute_command_validation(case: EvalCase) -> dict[str, Any]:
    """Validate a parsed command against game state."""

    def run() -> dict[str, Any]:
        state = _state_from(case)
        command = rule_based_parse(case.input["message"])
        verdict = validate(command, state)
        return {
            "command": command.command.value,
            "valid": verdict.ok,
            "needs_clarification": verdict.needs_clarification,
            "issue_count": len(verdict.issues),
            "summary": verdict.summary,
        }

    return _capture(run)


def execute_nl_pipeline(case: EvalCase) -> dict[str, Any]:
    """Run the full parse → validate → execute pipeline."""

    def run() -> dict[str, Any]:
        state = _state_from(case)
        mode = case.input.get("provider", "mock")

        if mode == "scripted":
            provider = ScriptedProvider(list(case.input.get("script", [])))
        else:
            provider = MockProvider()

        client = ReliableLLMClient(provider, max_retries=case.input.get("max_retries", 1), sleep=_no_sleep)
        parser = CommandParser(client)
        executor = CommandExecutor(Assistant(ReliableLLMClient(MockProvider(), sleep=_no_sleep)))

        turn = handle_message(
            case.input["message"],
            state,
            parser,
            executor,
            last_outcome=case.input.get("last_outcome"),
        )

        result = turn.result
        after = result.state
        return {
            "command": turn.command_type,
            "executed": result.executed,
            "parse_source": turn.parse.source,
            "needs_clarification": turn.validated.needs_clarification,
            "was_rejected": not result.executed,
            "message_is_non_empty": bool(result.message.strip()),
            "attempts_used": after.attempts_used if after else None,
            "status": after.status if after else None,
            "difficulty": after.difficulty if after else None,
            "score": after.score if after else None,
            "state_unchanged": (after is state) or (after == state),
        }

    return _capture(run)


NL_EXECUTORS: dict[str, Executor] = {
    "intent_parse": execute_intent_parse,
    "command_validation": execute_command_validation,
    "nl_pipeline": execute_nl_pipeline,
}
