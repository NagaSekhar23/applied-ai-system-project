"""Regression tests for the Phase 1 bugs.

Written before the fixes, and expected to FAIL on the original code. After the
state refactor, the helpers read and write ``session_state["game"]`` instead of
the old loose keys. The asserted behavior is unchanged.
"""

from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

from logic_utils import check_guess

APP = "app.py"


@pytest.fixture(autouse=True)
def no_debug_env(monkeypatch):
    """Debug panel is hidden unless GAME_DEBUG=true; keep tests independent of the shell."""
    monkeypatch.delenv("GAME_DEBUG", raising=False)


def _game(at: AppTest):
    return at.session_state["game"]


def _set_secret(at: AppTest, secret: int) -> AppTest:
    """Force the secret for the current game, then rerun."""
    at.session_state["game"] = replace(_game(at), secret=secret)
    return at.run()


def _button(at: AppTest, label_part: str):
    return next(b for b in at.button if label_part in b.label)


def _submit(at: AppTest, text: str) -> AppTest:
    at.text_input[0].input(text)
    _button(at, "Submit").click()
    return at.run()


def _all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error]
    parts += [w.value for w in at.warning] + [s.value for s in at.success]
    parts += [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


# --- Bug: string-secret path gives wrong hints on even attempts --------------

def test_check_guess_is_correct_for_numbers_that_sort_differently_as_text():
    # "9" > "10" lexicographically, but 9 < 10 numerically.
    outcome, _ = check_guess(9, 10)
    assert outcome == "Too Low"


# --- Bug: New Game leaves the player stuck after a win ----------------------

def test_new_game_after_win_allows_play_again():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 50)
    _submit(at, "50")
    assert _game(at).status == "won"

    _button(at, "New Game").click()
    at.run()
    _submit(at, "10")

    assert "already won" not in _all_text(at)
    assert _game(at).status == "playing"


# --- Bug: New Game ignores difficulty range ---------------------------------

def test_new_game_uses_selected_difficulty_range():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("Easy").run()
    for _ in range(50):  # new secret each time; check the range holds
        _button(at, "New Game").click()
        at.run()
        assert 1 <= _game(at).secret <= 20


# --- Bug: difficulty change keeps the old (possibly out-of-range) secret ----

def test_changing_difficulty_regenerates_secret_within_new_range():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 77)  # valid for Normal (1-100), invalid for Easy (1-20)
    at.sidebar.selectbox[0].set_value("Easy").run()
    assert 1 <= _game(at).secret <= 20


# --- Bug: invalid input consumes an attempt ---------------------------------

def test_invalid_input_does_not_consume_an_attempt():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _submit(at, "abc")
    # Normal allows 8 attempts; an invalid entry must leave all 8 available.
    assert "Attempts left: 8" in _all_text(at)


# --- Bug: first game allows one fewer guess than advertised -----------------

def test_normal_difficulty_allows_exactly_eight_valid_guesses():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 99)
    for n in range(1, 9):
        _submit(at, str(n))
    assert _game(at).status == "lost"  # the eighth wrong guess ends the game


def test_seventh_wrong_guess_does_not_end_normal_game():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _set_secret(at, 99)
    for n in range(1, 8):
        _submit(at, str(n))
    assert _game(at).status == "playing"


# --- Bug: out-of-range guesses are accepted ---------------------------------

def test_out_of_range_guess_is_rejected_without_consuming_attempt():
    at = AppTest.from_file(APP, default_timeout=10).run()
    _submit(at, "500")
    assert any("between" in e.value for e in at.error)
    assert "Attempts left: 8" in _all_text(at)


# --- Feature: debug panel hidden by default ---------------------------------

def test_debug_panel_is_hidden_by_default():
    at = AppTest.from_file(APP, default_timeout=10).run()
    text = _all_text(at)
    assert "Secret:" not in text
    assert "Developer Debug Info" not in text
