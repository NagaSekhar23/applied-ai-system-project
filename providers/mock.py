"""Mock and scripted providers.

The mock provider is what makes this project runnable, testable and evaluable
with no API key, no network and no cost. It is not a stub that returns
"lorem ipsum": it reads the same structured facts block the real provider
receives and produces schema-valid output derived from it, so the surrounding
validation, fallback and evaluation code is genuinely exercised.

What it cannot do is produce natural, varied language. Mock-mode output is
therefore useful for testing plumbing, not for judging phrasing quality.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from typing import Any, Callable

from providers.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

FACTS_BLOCK = re.compile(r"<facts>\s*(\{.*?\})\s*</facts>", re.DOTALL)


def extract_facts(user_message: str) -> dict[str, Any]:
    """Pull the JSON facts block out of a prompt. Returns {} when absent."""
    match = FACTS_BLOCK.search(user_message)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for the mock only.

    The real provider reports exact counts. This exists so mock runs have
    *some* number to exercise the accounting path; it is labelled as an
    estimate everywhere it surfaces.
    """
    return max(1, len(text) // 4)


# --- Default handlers, one per prompt purpose --------------------------------


def _hint_handler(facts: dict[str, Any]) -> dict[str, Any]:
    outcome = facts.get("last_outcome")
    attempts_left = facts.get("attempts_left", 0)
    low, high = facts.get("low", 1), facts.get("high", 100)
    difficulty = facts.get("difficulty", "Normal")

    if outcome == "Too High":
        message = (
            f"Your last guess was above the number. Try something lower, "
            f"somewhere between {low} and that guess. You have {attempts_left} attempts left."
        )
        category = "too_high"
    elif outcome == "Too Low":
        message = (
            f"Your last guess was below the number. Aim higher, between that guess "
            f"and {high}. You have {attempts_left} attempts left."
        )
        category = "too_low"
    elif outcome == "Win":
        message = "You found it. Nicely done."
        category = "win"
    else:
        message = (
            f"No guesses yet on {difficulty}. The range is {low} to {high}; "
            f"starting in the middle splits it fastest."
        )
        category = "opening"

    return {
        "message": message,
        "hint_category": category,
        "difficulty_level": difficulty,
        "reveals_secret": False,
    }


def _explain_handler(facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": (
            f"You are playing {facts.get('difficulty', 'Normal')}: guess a number "
            f"between {facts.get('low', 1)} and {facts.get('high', 100)} within "
            f"{facts.get('max_attempts', 8)} attempts. A correct guess scores "
            "max(10, 100 - 10 x guess number); a wrong guess costs 5 points and the "
            "score never drops below zero."
        ),
        "hint_category": "rules",
        "difficulty_level": facts.get("difficulty", "Normal"),
        "reveals_secret": False,
    }


DEFAULT_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "hint": _hint_handler,
    "explain_rules": _explain_handler,
}


class MockProvider:
    """Deterministic provider for development, tests and offline evaluation."""

    def __init__(
        self,
        model: str = "mock-model",
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        latency_ms: float = 0.0,
    ):
        self.name = "mock"
        self.model = model
        self.handlers = dict(DEFAULT_HANDLERS)
        if handlers:
            self.handlers.update(handlers)
        self._latency_ms = latency_ms
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        self.calls.append(request)

        facts = extract_facts(request.user)
        handler = self.handlers.get(request.purpose)
        payload = handler(facts) if handler else {"message": "No mock handler for this purpose."}
        text = json.dumps(payload) if request.json_schema else str(payload.get("message", ""))

        if self._latency_ms:
            time.sleep(self._latency_ms / 1000)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage=TokenUsage(
                input_tokens=_estimate_tokens(request.system + request.user),
                output_tokens=_estimate_tokens(text),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            stop_reason="end_turn",
        )


class ScriptedProvider:
    """Returns queued responses or raises queued errors, in order.

    Used by tests to drive failure paths (timeouts, rate limits, malformed
    output, retry exhaustion) without patching a real SDK.
    """

    def __init__(self, script: list[str | Exception], model: str = "scripted-model"):
        self.name = "scripted"
        self.model = model
        self._script: deque[str | Exception] = deque(script)
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if not self._script:
            raise LLMError("ScriptedProvider ran out of scripted responses")

        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            text=item,
            provider=self.name,
            model=self.model,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_ms=1.0,
            stop_reason="end_turn",
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)
