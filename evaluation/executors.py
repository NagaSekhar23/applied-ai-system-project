"""Executors: the bridge from an evaluation case to real application code.

An executor takes a case and returns a flat dict of observed values. The runner
compares that dict against the case's ``expected`` block. Executors never
re-implement application logic — they call it, so the evaluation measures the
real system.

Later phases register their own executors (LLM, natural language, agent,
retrieval) in the same registry.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from evaluation.cases import EvalCase
from game import new_game, submit_guess
from logic_utils import (
    check_guess,
    get_attempt_limit,
    get_difficulty_config,
    parse_guess,
    update_score,
)

Executor = Callable[[EvalCase], dict[str, Any]]


class UnknownExecutorError(Exception):
    """Raised when a case names a target with no registered executor."""


def _capture(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run ``fn``, converting an expected exception into comparable fields.

    Cases that assert on failure can then check ``raised`` instead of the
    runner treating the exception as an infrastructure error.
    """
    try:
        return fn()
    except (ValueError, TypeError, KeyError) as exc:
        return {"raised": type(exc).__name__, "error_message": str(exc)}


# --- Phase 1 deterministic executors ----------------------------------------


def execute_check_guess(case: EvalCase) -> dict[str, Any]:
    """Compare a guess to a secret. Input: ``guess``, ``secret``."""

    def run() -> dict[str, Any]:
        outcome, hint = check_guess(case.input["guess"], case.input["secret"])
        return {"outcome": outcome, "hint": hint}

    return _capture(run)


def execute_parse_guess(case: EvalCase) -> dict[str, Any]:
    """Validate raw input. Input: ``raw``, optional ``low`` / ``high``."""

    def run() -> dict[str, Any]:
        result = parse_guess(
            case.input.get("raw"),
            case.input.get("low", 1),
            case.input.get("high", 100),
        )
        return {"ok": result.ok, "value": result.value, "error": result.error}

    return _capture(run)


def execute_update_score(case: EvalCase) -> dict[str, Any]:
    """Apply scoring. Input: ``current_score``, ``outcome``, ``guess_number``."""

    def run() -> dict[str, Any]:
        return {
            "score": update_score(
                case.input["current_score"],
                case.input["outcome"],
                case.input["guess_number"],
            )
        }

    return _capture(run)


def execute_difficulty_config(case: EvalCase) -> dict[str, Any]:
    """Look up difficulty settings. Input: ``difficulty``."""

    def run() -> dict[str, Any]:
        config = get_difficulty_config(case.input["difficulty"])
        return {
            "low": config.low,
            "high": config.high,
            "max_attempts": config.max_attempts,
            "attempt_limit": get_attempt_limit(case.input["difficulty"]),
        }

    return _capture(run)


def execute_game_flow(case: EvalCase) -> dict[str, Any]:
    """Play a scripted sequence of guesses against the real game service.

    ``initial_state`` sets ``difficulty`` and a fixed ``secret`` so the run is
    deterministic. ``input.guesses`` is a list of raw strings, exactly what a
    player would type. An optional ``input.change_difficulty_to`` starts a fresh
    game at that difficulty partway through, after the listed guesses.
    """

    def run() -> dict[str, Any]:
        state = case.initial_state or {}
        difficulty = state.get("difficulty", "Normal")
        game = new_game(difficulty, rng=random.Random(state.get("seed", 0)))
        if "secret" in state:
            game = replace(game, secret=state["secret"])

        errors: list[str] = []
        outcomes: list[str] = []
        for raw in case.input.get("guesses", []):
            result = submit_guess(game, raw)
            game = result.state
            if result.error:
                errors.append(result.error)
            if result.outcome:
                outcomes.append(result.outcome)

        if "change_difficulty_to" in case.input:
            game = new_game(
                case.input["change_difficulty_to"],
                rng=random.Random(state.get("seed", 0)),
            )

        return {
            "status": game.status,
            "attempts_used": game.attempts_used,
            "attempts_left": game.attempts_left,
            "score": game.score,
            "history": list(game.history),
            "difficulty": game.difficulty,
            "low": game.low,
            "high": game.high,
            "max_attempts": game.max_attempts,
            "errors": errors,
            "outcomes": outcomes,
            "error_count": len(errors),
        }

    return _capture(run)


# --- Registry ----------------------------------------------------------------

DETERMINISTIC_EXECUTORS: dict[str, Executor] = {
    "check_guess": execute_check_guess,
    "parse_guess": execute_parse_guess,
    "update_score": execute_update_score,
    "difficulty_config": execute_difficulty_config,
    "game_flow": execute_game_flow,
}


def default_registry() -> dict[str, Executor]:
    """Return a fresh copy of the built-in executors.

    A copy, so a caller registering extra executors cannot mutate the shared
    defaults for everyone else.
    """
    return dict(DETERMINISTIC_EXECUTORS)


def resolve(registry: dict[str, Executor], target: str) -> Executor:
    """Look up an executor, with an error that lists what is available."""
    try:
        return registry[target]
    except KeyError:
        raise UnknownExecutorError(
            f"No executor registered for target {target!r}. "
            f"Available: {sorted(registry)}"
        ) from None
