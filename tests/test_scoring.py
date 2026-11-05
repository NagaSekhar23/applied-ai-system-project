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


def test_score_can_go_negative_on_wrong_guesses():
    assert update_score(0, "Too Low", 1) == -5


def test_unknown_outcome_raises():
    with pytest.raises(ValueError, match="Unknown outcome"):
        update_score(0, "Maybe", 1)
