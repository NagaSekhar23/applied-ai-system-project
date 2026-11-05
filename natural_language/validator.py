"""Command validation.

Schema validation (Pydantic) proves a command is *well-formed*. This module
proves it is *permitted right now*: arguments present, values in range, and the
action legal in the current game state.

Both run before the executor sees anything.
"""

from __future__ import annotations

from game import GameState
from logic_utils import DIFFICULTY_NAMES, parse_guess
from natural_language.commands import (
    Command,
    CommandType,
    ValidatedCommand,
    ValidationIssue,
)

REQUIRES_ACTIVE_GAME = (CommandType.SUBMIT_GUESS,)


def validate(command: Command, state: GameState | None) -> ValidatedCommand:
    """Check a parsed command against the current game state.

    Returns a verdict rather than raising, because a rejected command is a
    normal conversational outcome that the UI explains to the user.
    """
    issues: list[ValidationIssue] = []

    if command.command is CommandType.UNKNOWN:
        return ValidatedCommand(command=command, needs_clarification=True)

    if state is None and command.command in REQUIRES_ACTIVE_GAME:
        issues.append(
            ValidationIssue(
                field_name="state",
                message="There is no game in progress. Start one first.",
            )
        )
        return ValidatedCommand(command=command, issues=issues)

    if command.command is CommandType.SUBMIT_GUESS:
        issues.extend(_validate_guess(command, state))

    if command.command is CommandType.CHANGE_DIFFICULTY:
        issues.extend(_validate_difficulty(command))

    if command.command is CommandType.START_GAME and command.difficulty is not None:
        issues.extend(_validate_difficulty(command))

    return ValidatedCommand(command=command, issues=issues)


def _validate_guess(command: Command, state: GameState | None) -> list[ValidationIssue]:
    """A guess needs a value, a live game, and a number inside the range."""
    if command.guess is None:
        return [
            ValidationIssue(
                field_name="guess",
                message="I did not catch a number. Try 'guess 42'.",
            )
        ]

    if state is None:
        return [ValidationIssue(field_name="state", message="There is no game in progress.")]

    if state.status != "playing":
        return [
            ValidationIssue(
                field_name="state",
                message="This game is over. Start a new game to play again.",
            )
        ]

    # Reuse the game's own validator so the natural language path cannot
    # accept a guess the typed path would reject.
    parsed = parse_guess(str(command.guess), state.low, state.high)
    if not parsed.ok:
        return [ValidationIssue(field_name="guess", message=parsed.error or "Invalid guess.")]

    return []


def _validate_difficulty(command: Command) -> list[ValidationIssue]:
    """Difficulty must be named and known."""
    if command.difficulty is None:
        return [
            ValidationIssue(
                field_name="difficulty",
                message=f"Which difficulty? Choose from {', '.join(DIFFICULTY_NAMES)}.",
            )
        ]
    if command.difficulty not in DIFFICULTY_NAMES:
        return [
            ValidationIssue(
                field_name="difficulty",
                message=f"{command.difficulty!r} is not a difficulty I know.",
            )
        ]
    return []
