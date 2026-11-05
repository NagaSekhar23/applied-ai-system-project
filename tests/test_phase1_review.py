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
