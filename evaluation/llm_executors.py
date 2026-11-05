"""Executors for the LLM assistant layer.

**What these measure.** With the mock provider these are still *deterministic*
checks: does the assistant validate output, enforce the deterministic outcome,
refuse to leak the secret, and fall back correctly when the provider fails?
Those properties are objectively right or wrong, so they belong under
``deterministic_correctness``.

**What they do not measure.** Whether a hint is well phrased, natural or
genuinely helpful. That is model quality, it needs a real provider, and it is
not claimed anywhere in this dataset.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from assistant import Assistant, build_facts, leaks_secret
from evaluation.cases import EvalCase
from evaluation.executors import Executor, _capture
from game import new_game
from prompts import get_prompt
from providers import (
    LLMRateLimitError,
    LLMTimeoutError,
    MockProvider,
    ReliableLLMClient,
    ScriptedProvider,
)
from schemas import AssistantHint

ERROR_TYPES = {
    "timeout": LLMTimeoutError("evaluation-induced timeout"),
    "rate_limit": LLMRateLimitError("evaluation-induced rate limit", 0.0),
}


def _no_sleep(_seconds: float) -> None:
    """Evaluation must not spend wall-clock time on retry backoff."""


def _state_from(case: EvalCase):
    spec = case.initial_state or {}
    state = new_game(spec.get("difficulty", "Normal"), rng=random.Random(spec.get("seed", 0)))
    changes: dict[str, Any] = {}
    for field_name in ("secret", "status", "attempts_used", "score"):
        if field_name in spec:
            changes[field_name] = spec[field_name]
    if "history" in spec:
        changes["history"] = tuple(spec["history"])
    return replace(state, **changes) if changes else state


def _build_assistant(case: EvalCase) -> Assistant:
    """Build an assistant whose provider matches what the case is testing."""
    mode = case.input.get("provider", "mock")

    if mode == "mock":
        provider = MockProvider()
    elif mode == "scripted":
        script = [
            ERROR_TYPES[item] if item in ERROR_TYPES else item
            for item in case.input.get("script", [])
        ]
        provider = ScriptedProvider(script)
    else:
        raise ValueError(f"Unknown provider mode {mode!r}")

    return Assistant(
        ReliableLLMClient(provider, max_retries=case.input.get("max_retries", 1), sleep=_no_sleep)
    )


def execute_assistant_hint(case: EvalCase) -> dict[str, Any]:
    """Run the assistant and report observable, checkable properties."""

    def run() -> dict[str, Any]:
        state = _state_from(case)
        assistant = _build_assistant(case)
        reply = assistant.hint(
            state,
            last_outcome=case.input.get("last_outcome"),
            question=case.input.get("question", ""),
            purpose=case.input.get("purpose", "hint"),
        )
        return {
            "source": reply.source,
            "category": reply.category.value,
            "used_llm": reply.used_llm,
            "message_is_non_empty": bool(reply.message.strip()),
            "leaks_secret": leaks_secret(reply.message, state),
            "has_fallback_reason": reply.fallback_reason is not None,
            "warning_count": len(reply.warnings),
            "attempts": reply.stats.attempts if reply.stats else 0,
            "prompt_id": reply.prompt_id,
        }

    return _capture(run)


def execute_prompt_safety(case: EvalCase) -> dict[str, Any]:
    """Check that a rendered prompt cannot disclose the secret mid-game."""

    def run() -> dict[str, Any]:
        state = _state_from(case)
        prompt = get_prompt(case.input.get("purpose", "hint"))
        facts = build_facts(state, case.input.get("last_outcome"))
        rendered = prompt.render_user(facts, case.input.get("question", ""))
        return {
            "facts_contain_secret": "secret" in facts,
            "prompt_leaks_secret": leaks_secret(rendered, state),
            "prompt_id": prompt.id,
            "mentions_restrictions": "never" in prompt.system.lower(),
        }

    return _capture(run)


def execute_schema_validation(case: EvalCase) -> dict[str, Any]:
    """Check that the response schema accepts good output and rejects bad."""

    def run() -> dict[str, Any]:
        try:
            hint = AssistantHint.model_validate(case.input.get("payload", {}))
        except Exception as exc:
            return {"valid": False, "error_type": type(exc).__name__}
        return {
            "valid": True,
            "category": hint.hint_category.value,
            "message": hint.message,
        }

    return _capture(run)


LLM_EXECUTORS: dict[str, Executor] = {
    "assistant_hint": execute_assistant_hint,
    "prompt_safety": execute_prompt_safety,
    "schema_validation": execute_schema_validation,
}


def registry_with_llm() -> dict[str, Executor]:
    """Deterministic executors plus the LLM ones."""
    from evaluation.executors import default_registry

    registry = default_registry()
    registry.update(LLM_EXECUTORS)
    return registry
