"""Pure game rules: difficulty settings, input parsing, guess checking, and scoring.

Nothing in this module imports Streamlit, so every rule can be tested directly.
"""

import re
from dataclasses import dataclass
from typing import NamedTuple

OUTCOME_WIN = "Win"
OUTCOME_TOO_HIGH = "Too High"
OUTCOME_TOO_LOW = "Too Low"

WRONG_GUESS_PENALTY = 5
MIN_WIN_POINTS = 10
WIN_BASE_POINTS = 100
WIN_POINTS_PER_GUESS = 10

_INTEGER_PATTERN = re.compile(r"[+-]?[0-9]+")

# Python refuses to convert very long digit strings to int (default limit is
# 4300 digits). Any value with more than 18 significant digits is far outside
# every difficulty range, so it is rejected as out of range before conversion.
_MAX_SIGNIFICANT_DIGITS = 18


@dataclass(frozen=True)
class DifficultyConfig:
    """Settings for one difficulty level."""

    low: int
    high: int
    max_attempts: int


DIFFICULTY_CONFIGS: dict[str, DifficultyConfig] = {
    "Easy": DifficultyConfig(low=1, high=20, max_attempts=6),
    "Normal": DifficultyConfig(low=1, high=100, max_attempts=8),
    "Hard": DifficultyConfig(low=1, high=50, max_attempts=5),
}

DIFFICULTY_NAMES: tuple[str, ...] = tuple(DIFFICULTY_CONFIGS)


class ParseResult(NamedTuple):
    """Result of parsing raw player input. ``value`` is set only when ``ok`` is True."""

    ok: bool
    value: int | None
    error: str | None


def get_difficulty_config(difficulty: str) -> DifficultyConfig:
    """Return the settings for a difficulty name.

    Raises:
        ValueError: if the difficulty is unknown. We fail loudly rather than
            silently falling back to a default.
    """
    try:
        return DIFFICULTY_CONFIGS[difficulty]
    except KeyError:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}; expected one of {DIFFICULTY_NAMES}"
        ) from None


def get_range_for_difficulty(difficulty: str) -> tuple[int, int]:
    """Return the inclusive (low, high) secret range for a difficulty."""
    config = get_difficulty_config(difficulty)
    return config.low, config.high


def get_attempt_limit(difficulty: str) -> int:
    """Return how many valid guesses a player gets at this difficulty."""
    return get_difficulty_config(difficulty).max_attempts


def parse_guess(raw: str | None, low: int = 1, high: int = 100) -> ParseResult:
    """Validate raw text input as a whole number within [low, high].

    Surrounding whitespace is ignored. Decimals such as "7.9" are rejected
    rather than silently truncated.
    """
    if raw is None or raw.strip() == "":
        return ParseResult(False, None, "Enter a guess.")

    text = raw.strip()
    if not _INTEGER_PATTERN.fullmatch(text):
        return ParseResult(False, None, "That is not a whole number.")

    out_of_range = ParseResult(False, None, f"Your guess must be between {low} and {high}.")
    sign = -1 if text.startswith("-") else 1
    digits = text.lstrip("+-").lstrip("0") or "0"  # leading zeros are not significant
    if len(digits) > _MAX_SIGNIFICANT_DIGITS:
        return out_of_range

    value = sign * int(digits)
    if value < low or value > high:
        return out_of_range

    return ParseResult(True, value, None)


def check_guess(guess: int, secret: int) -> tuple[str, str]:
    """Compare a guess to the secret and return ``(outcome, hint)``.

    This is the only place that decides whether a guess is correct.
    """
    if guess == secret:
        return OUTCOME_WIN, "🎉 Correct!"
    if guess > secret:
        return OUTCOME_TOO_HIGH, "📉 Go LOWER!"
    return OUTCOME_TOO_LOW, "📈 Go HIGHER!"


def update_score(current_score: int, outcome: str, guess_number: int) -> int:
    """Return the new score after a guess.

    A win awards ``max(10, 100 - 10 * guess_number)`` points. A wrong guess
    costs 5 points, whichever direction it was in.

    Args:
        guess_number: 1-based index of this valid guess in the game.

    Raises:
        ValueError: if the outcome is not recognised.
    """
    if outcome == OUTCOME_WIN:
        points = max(MIN_WIN_POINTS, WIN_BASE_POINTS - WIN_POINTS_PER_GUESS * guess_number)
        return current_score + points
    if outcome in (OUTCOME_TOO_HIGH, OUTCOME_TOO_LOW):
        return current_score - WRONG_GUESS_PENALTY
    raise ValueError(f"Unknown outcome {outcome!r}")
