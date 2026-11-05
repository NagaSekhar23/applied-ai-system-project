import random

import pytest

from logic_utils import update_score


@pytest.mark.parametrize(
    "guess_number, expected_points",
    [(1, 90), (2, 80), (5, 50), (8, 20), (9, 10), (10, 10), (50, 10)],
)
def test_win_points_decrease_per_guess_with_a_floor_of_ten(guess_number, expected_points):
    assert update_score(0, "Win", guess_number) == expected_points


def test_win_adds_to_existing_score():
    assert update_score(40, "Win", 2) == 40 + 80


@pytest.mark.parametrize("outcome", ["Too High", "Too Low"])
@pytest.mark.parametrize("guess_number", [1, 2, 3, 4])
def test_wrong_guess_costs_five_points_in_both_directions(outcome, guess_number):
    # Regression: the old code gave +5 for "Too High" on even attempts.
    assert update_score(20, outcome, guess_number) == 15


def test_incorrect_guess_when_score_is_zero_stays_at_zero():
    assert update_score(0, "Too Low", 1) == 0
    assert update_score(0, "Too High", 1) == 0


def test_incorrect_guess_when_score_is_positive_subtracts_five():
    assert update_score(10, "Too High", 2) == 5


def test_multiple_incorrect_guesses_stop_at_the_floor():
    score = 12
    for guess_number, outcome in enumerate(["Too High", "Too Low", "Too High", "Too Low"], 1):
        score = update_score(score, outcome, guess_number)
    assert score == 0


def test_winning_after_incorrect_guesses_adds_win_points_to_floored_score():
    score = update_score(0, "Too High", 1)  # floored to 0
    score = update_score(score, "Too Low", 2)  # still 0
    assert update_score(score, "Win", 3) == 70  # 100 - 10 * 3


def test_score_never_becomes_negative_for_any_outcome_sequence():
    rng = random.Random(0)  # deterministic sequences
    outcomes = ["Win", "Too High", "Too Low"]
    for _ in range(500):
        score = 0
        for guess_number in range(1, 10):
            score = update_score(score, rng.choice(outcomes), guess_number)
            assert score >= 0


def test_unknown_outcome_raises():
    with pytest.raises(ValueError, match="Unknown outcome"):
        update_score(0, "Maybe", 1)
