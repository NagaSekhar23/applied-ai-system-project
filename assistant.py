"""AI assistant service.

This module is the boundary between deterministic game logic and the LLM.

Three guarantees, each enforced here rather than trusted to the prompt:

1. **The secret never leaves the process while a game is in progress.** The
   facts block is built from a redacted view of ``GameState``.
2. **The model never decides an outcome.** The outcome comes from
   ``check_guess``; the model's ``hint_category`` is cross-checked against it
   and overridden when it disagrees.
3. **The game always works.** Every failure path — provider down, timeout,
   malformed output, leaked secret — falls back to a deterministic hint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from game import GameState
from prompts import get_prompt
from providers import CallStats, LLMRequest, ReliableLLMClient
from schemas import AssistantHint, HintCategory

OUTCOME_TO_CATEGORY = {
    "Too High": HintCategory.TOO_HIGH,
    "Too Low": HintCategory.TOO_LOW,
    "Win": HintCategory.WIN,
}

DETERMINISTIC_HINTS = {
    HintCategory.TOO_HIGH: "That guess was too high. Try a lower number.",
    HintCategory.TOO_LOW: "That guess was too low. Try a higher number.",
    HintCategory.WIN: "Correct! You found the number.",
    HintCategory.OPENING: "No guesses yet. Starting near the middle of the range narrows it fastest.",
}


@dataclass(frozen=True)
class AssistantReply:
    """What the assistant produced, and how.

    ``source`` is "llm" or "fallback". It is surfaced in the UI so a user can
    always tell whether they are reading model output or a canned hint.
    """

    message: str
    category: HintCategory
    source: str
    stats: CallStats | None = None
    fallback_reason: str | None = None
    prompt_id: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def used_llm(self) -> bool:
        return self.source == "llm"


def build_facts(state: GameState, last_outcome: str | None = None) -> dict[str, Any]:
    """Build the redacted fact block describing the game.

    The secret is included **only** once the game is over, when revealing it
    is already the correct behavior. While ``status == "playing"`` there is no
    key that carries it, so no prompt can leak what it never received.
    """
    facts: dict[str, Any] = {
        "difficulty": state.difficulty,
        "low": state.low,
        "high": state.high,
        "max_attempts": state.max_attempts,
        "attempts_used": state.attempts_used,
        "attempts_left": state.attempts_left,
        "score": state.score,
        "status": state.status,
        "guess_history": list(state.history),
        "last_guess": state.history[-1] if state.history else None,
        "last_outcome": last_outcome,
    }
    if state.status != "playing":
        facts["secret"] = state.secret
    return facts


def deterministic_hint(state: GameState, last_outcome: str | None) -> tuple[str, HintCategory]:
    """The always-available hint, computed with no model involved."""
    category = OUTCOME_TO_CATEGORY.get(last_outcome or "", HintCategory.OPENING)
    base = DETERMINISTIC_HINTS.get(category, DETERMINISTIC_HINTS[HintCategory.OPENING])
    if state.status == "playing" and category is not HintCategory.WIN:
        base = f"{base} You have {state.attempts_left} attempts left."
    return base, category


def leaks_secret(message: str, state: GameState) -> bool:
    """True if the message appears to disclose the secret during play.

    Matches the number as a standalone token, so "50" is caught but the "50"
    inside "150" is not. Only applies while the game is in progress; once it
    is over the secret is public.
    """
    if state.status != "playing":
        return False
    return re.search(rf"(?<!\d){state.secret}(?!\d)", message) is not None


class Assistant:
    """Generates hints and explanations, with a deterministic safety net."""

    def __init__(self, client: ReliableLLMClient, max_tokens: int = 1024, effort: str = "low"):
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort

    def hint(
        self,
        state: GameState,
        last_outcome: str | None = None,
        question: str = "",
        purpose: str = "hint",
    ) -> AssistantReply:
        """Produce a hint for the current state.

        Falls back to a deterministic hint whenever the model is unavailable,
        returns unusable output, or says something it must not.
        """
        fallback_message, true_category = deterministic_hint(state, last_outcome)

        def fallback(reason: str, stats: CallStats | None = None) -> AssistantReply:
            return AssistantReply(
                message=fallback_message,
                category=true_category,
                source="fallback",
                stats=stats,
                fallback_reason=reason,
            )

        try:
            prompt = get_prompt(purpose)
        except KeyError as exc:
            return fallback(f"unknown prompt: {exc}")

        facts = build_facts(state, last_outcome)
        request = LLMRequest(
            system=prompt.system,
            user=prompt.render_user(facts, question),
            json_schema=AssistantHint.json_schema(),
            max_tokens=self.max_tokens,
            effort=self.effort,
            purpose=purpose,
        )

        result = self.client.complete_structured(request, AssistantHint)
        if not result.ok or result.value is None:
            reason = result.stats.errors[-1] if result.stats.errors else "no response"
            return fallback(f"provider or validation failure: {reason}", result.stats)

        hint: AssistantHint = result.value
        warnings: list[str] = []

        # Guarantee 1: the model must not disclose the secret mid-game.
        if leaks_secret(hint.message, state):
            return fallback("model output disclosed the secret", result.stats)

        # Guarantee 2: deterministic logic wins any disagreement about outcome.
        category = hint.hint_category
        if purpose == "hint" and category is not true_category:
            warnings.append(
                f"model said {category.value!r}; deterministic outcome is "
                f"{true_category.value!r} and was used instead"
            )
            category = true_category

        return AssistantReply(
            message=hint.message,
            category=category,
            source="llm",
            stats=result.stats,
            prompt_id=prompt.id,
            warnings=warnings,
        )

    def explain_rules(self, state: GameState, question: str = "") -> AssistantReply:
        """Explain the rules and current state for a beginner."""
        return self.hint(
            state,
            last_outcome=None,
            question=question or "Explain the rules of this game.",
            purpose="explain_rules",
        )
