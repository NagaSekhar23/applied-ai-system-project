import random
from dataclasses import FrozenInstanceError, replace

import pytest

from game import GameState, new_game, submit_guess


def make_game(secret: int = 50, difficulty: str = "Normal") -> GameState:
    return replace(new_game(difficulty, rng=random.Random(0)), secret=secret)


# --- new_game --------------------------------------------------------------

@pytest.mark.parametrize(
    "difficulty, low, high, attempts",
    [("Easy", 1, 20, 6), ("Normal", 1, 100, 8), ("Hard", 1, 50, 5)],
)
def test_new_game_uses_difficulty_range_and_limit(difficulty, low, high, attempts):
    game = new_game(difficulty, rng=random.Random(1))
    assert (game.low, game.high, game.max_attempts) == (low, high, attempts)
    assert low <= game.secret <= high


def test_new_game_starts_clean():
    game = new_game("Normal", rng=random.Random(2))
    assert game.attempts_used == 0
    assert game.attempts_left == 8
    assert game.score == 0
    assert game.status == "playing"
    assert game.history == ()


def test_new_game_with_seeded_rng_is_reproducible():
    assert new_game("Normal", random.Random(7)).secret == new_game("Normal", random.Random(7)).secret


def test_new_game_rejects_unknown_difficulty():
    with pytest.raises(ValueError):
        new_game("Impossible")


# --- submit_guess: outcomes and scoring ------------------------------------

def test_winning_guess_ends_game_and_scores():
    result = submit_guess(make_game(secret=50), "50")
    assert result.error is None
    assert result.outcome == "Win"
    assert result.state.status == "won"
    assert result.state.score == 90  # first guess: 100 - 10 * 1


def test_wrong_guess_consumes_one_attempt_and_costs_five():
    result = submit_guess(make_game(secret=50), "60")
    assert result.outcome == "Too High"
    assert result.hint == "📉 Go LOWER!"
    assert result.state.attempts_used == 1
    assert result.state.attempts_left == 7
    assert result.state.score == -5
    assert result.state.status == "playing"


def test_history_records_valid_guesses_only():
    game = make_game(secret=50)
    game = submit_guess(game, "abc").state
    game = submit_guess(game, "10").state
    game = submit_guess(game, "500").state
    game = submit_guess(game, "20").state
    assert game.history == (10, 20)


# --- submit_guess: attempt rules -------------------------------------------

@pytest.mark.parametrize("raw", ["", "abc", "7.5", "500", "0"])
def test_invalid_input_is_rejected_without_consuming_an_attempt(raw):
    game = make_game(secret=50)
    result = submit_guess(game, raw)
    assert result.error is not None
    assert result.outcome is None
    assert result.state == game  # unchanged
    assert result.state.attempts_used == 0


def test_last_allowed_guess_loses_the_game():
    game = make_game(secret=99, difficulty="Normal")
    for _ in range(7):
        game = submit_guess(game, "1").state
    assert game.status == "playing"
    result = submit_guess(game, "1")
    assert result.state.attempts_used == 8
    assert result.state.status == "lost"


def test_finished_game_rejects_further_guesses():
    game = submit_guess(make_game(secret=50), "50").state
    result = submit_guess(game, "50")
    assert result.error == "This game is over. Start a new game to play again."
    assert result.state == game


def test_game_state_is_immutable():
    game = make_game()
    with pytest.raises(FrozenInstanceError):
        game.score = 999  # type: ignore[misc]
