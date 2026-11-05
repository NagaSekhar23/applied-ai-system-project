"""Regression tests from the Phase 1 final review.

Written before the fixes. Each test describes behavior that the original
Phase 1 commit does not provide yet.
"""

from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

from logic_utils import parse_guess

APP = "app.py"


@pytest.fixture(autouse=True)
def no_debug_env(monkeypatch):
    monkeypatch.delenv("GAME_DEBUG", raising=False)


def _game(at: AppTest):
    return at.session_state["game"]


def _set_secret(at: AppTest, secret: int) -> AppTest:
    at.session_state["game"] = replace(_game(at), secret=secret)
    return at.run()


def _button(at: AppTest, label_part: str):
    return next(b for b in at.button if label_part in b.label)


def _submit(at: AppTest, text: str) -> AppTest:
    at.text_input[0].input(text)
    _button(at, "Submit").click()
    return at.run()


# --- Game-over experience ----------------------------------------------------

def test_guess_input_is_disabled_on_the_run_that_wins():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 50)
    _submit(at, "50")
    assert at.text_input[0].disabled is True
    assert _button(at, "Submit").disabled is True


def test_guess_input_is_disabled_on_the_run_that_loses():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 99)
    for n in range(1, 9):
        _submit(at, str(n))
    assert at.text_input[0].disabled is True
    assert _button(at, "Submit").disabled is True


def test_new_game_button_stays_enabled_after_game_over():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 50)
    _submit(at, "50")
    assert _button(at, "New Game").disabled is False


def test_new_game_after_loss_is_playable_immediately():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 99)
    for n in range(1, 9):
        _submit(at, str(n))
    assert _game(at).status == "lost"

    _button(at, "New Game").click()
    at.run()
    assert _game(at).status == "playing"
    assert _game(at).attempts_used == 0
    assert at.text_input[0].disabled is False
    _submit(at, "10")
    assert _game(at).attempts_used == 1


def test_final_result_message_is_shown_after_win():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 50)
    _submit(at, "50")
    assert any("You won" in s.value for s in at.success)


# --- Input validation ----------------------------------------------------------

def test_very_long_numeric_input_does_not_crash_the_app():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _submit(at, "9" * 5000)
    assert not at.exception
    assert _game(at).attempts_used == 0


def test_parse_guess_rejects_very_long_numeric_input():
    result = parse_guess("9" * 5000)
    assert result.ok is False


@pytest.mark.parametrize("raw", ["٤٢", "４２"])
def test_non_ascii_digits_are_rejected(raw):
    # Only ASCII digits are accepted, so input is predictable across keyboards.
    result = parse_guess(raw)
    assert result.ok is False


def test_leading_zeros_are_still_accepted():
    assert parse_guess("000042").value == 42
    assert parse_guess("0000000000000000000042").value == 42


@pytest.mark.parametrize("raw", ["+0", "-0", "-1", "0"])
def test_zero_and_negative_values_are_out_of_range(raw):
    assert parse_guess(raw, low=1, high=100).ok is False


# --- Game-state transitions in the UI -----------------------------------------

def test_changing_difficulty_mid_game_resets_score_and_attempts():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 99)
    _submit(at, "1")
    assert _game(at).attempts_used == 1
    assert _game(at).score == -5

    at.sidebar.selectbox[0].set_value("Hard").run()
    assert _game(at).difficulty == "Hard"
    assert _game(at).attempts_used == 0
    assert _game(at).score == 0


def test_hard_difficulty_accepts_bounds_and_rejects_outside():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("Hard").run()
    _submit(at, "51")
    assert _game(at).attempts_used == 0
    _submit(at, "50")
    assert _game(at).attempts_used == 1


def test_feedback_is_shown_once_not_on_every_rerun():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 99)
    _submit(at, "1")
    assert any("Go HIGHER" in w.value for w in at.warning)

    at.checkbox[0].set_value(False).run()  # unrelated rerun
    assert not any("Go HIGHER" in w.value for w in at.warning)


def test_hint_is_hidden_when_show_hint_is_off():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 99)
    at.checkbox[0].set_value(False).run()
    _submit(at, "1")
    assert not any("Go HIGHER" in w.value for w in at.warning)
    assert _game(at).attempts_used == 1  # the guess still counts


def test_new_game_clears_the_guess_box():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.text_input[0].input("42")
    _button(at, "New Game").click()
    at.run()
    assert at.text_input[0].value == ""


# --- Scoring behavior currently implemented (pending owner approval) ---------

def test_current_scoring_penalizes_wrong_guesses_in_both_directions():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 50)
    _submit(at, "60")  # too high, guess 1
    _submit(at, "40")  # too low, guess 2
    assert _game(at).score == -10


def test_current_scoring_can_go_negative_when_the_game_is_lost():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("Hard").run()  # 5 attempts
    _set_secret(at, 25)
    for n in (1, 2, 3, 4, 5):
        _submit(at, str(n))
    assert _game(at).status == "lost"
    assert _game(at).score == -25
