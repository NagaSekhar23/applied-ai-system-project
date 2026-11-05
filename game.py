"""Game state and state transitions, independent of Streamlit.

The UI keeps one ``GameState`` in session state and replaces it after each
action. ``GameState`` is frozen, so a transition returns a new value instead of
mutating shared state. That makes every rule testable with plain asserts.
"""

import random
from dataclasses import dataclass, replace
from typing import Literal

from logic_utils import (
    OUTCOME_WIN,
    check_guess,
    get_difficulty_config,
    parse_guess,
    update_score,
)

Status = Literal["playing", "won", "lost"]


@dataclass(frozen=True)
class GameState:
    """Everything the game remembers between player actions."""

    difficulty: str
    secret: int
    low: int
    high: int
    max_attempts: int
    attempts_used: int = 0
    score: int = 0
    status: Status = "playing"
    history: tuple[int, ...] = ()

    @property
    def attempts_left(self) -> int:
        """Valid guesses the player may still make."""
        return self.max_attempts - self.attempts_used


@dataclass(frozen=True)
class GuessResult:
    """Outcome of submitting one guess.

    If ``error`` is set, the guess was rejected and ``state`` is unchanged.
    """

    state: GameState
    error: str | None = None
    outcome: str | None = None
    hint: str | None = None


def new_game(difficulty: str, rng: random.Random | None = None) -> GameState:
    """Start a fresh game with a secret drawn from the difficulty's range.

    Args:
        difficulty: One of the names in ``DIFFICULTY_CONFIGS``.
        rng: Random source. Pass a seeded ``random.Random`` in tests.

    Raises:
        ValueError: if the difficulty is unknown.
    """
    config = get_difficulty_config(difficulty)
    source = rng if rng is not None else random.Random()
    return GameState(
        difficulty=difficulty,
        secret=source.randint(config.low, config.high),
        low=config.low,
        high=config.high,
        max_attempts=config.max_attempts,
    )


def submit_guess(state: GameState, raw: str | None) -> GuessResult:
    """Apply one player guess and return the new state.

    Invalid input and guesses after the game is over do not consume an attempt.
    """
    if state.status != "playing":
        return GuessResult(state, error="This game is over. Start a new game to play again.")

    parsed = parse_guess(raw, state.low, state.high)
    if not parsed.ok or parsed.value is None:
        return GuessResult(state, error=parsed.error)

    guess = parsed.value
    guess_number = state.attempts_used + 1
    outcome, hint = check_guess(guess, state.secret)
    score = update_score(state.score, outcome, guess_number)

    if outcome == OUTCOME_WIN:
        status: Status = "won"
    elif guess_number >= state.max_attempts:
        status = "lost"
    else:
        status = "playing"

    new_state = replace(
        state,
        attempts_used=guess_number,
        score=score,
        status=status,
        history=state.history + (guess,),
    )
    return GuessResult(new_state, outcome=outcome, hint=hint)
