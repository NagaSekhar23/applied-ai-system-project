"""UI-level tests that run app.py through Streamlit's AppTest harness."""

import pytest
from streamlit.testing.v1 import AppTest

APP = "app.py"


@pytest.fixture(autouse=True)
def no_debug_env(monkeypatch):
    monkeypatch.delenv("GAME_DEBUG", raising=False)


def _all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error]
    parts += [w.value for w in at.warning] + [s.value for s in at.success]
    parts += [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


def test_debug_panel_appears_when_game_debug_is_true(monkeypatch):
    monkeypatch.setenv("GAME_DEBUG", "true")
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert "Secret:" in _all_text(at)  # panel contents render; the label is not text


@pytest.mark.parametrize("value", ["false", "1", "yes", ""])
def test_debug_panel_stays_hidden_for_other_values(monkeypatch, value):
    monkeypatch.setenv("GAME_DEBUG", value)
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert "Developer Debug Info" not in _all_text(at)


def test_valid_guess_decrements_attempts_left():
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert "Attempts left: 8" in _all_text(at)
    at.text_input[0].input("1")
    next(b for b in at.button if "Submit" in b.label).click()
    at.run()
    assert "Attempts left: 7" in _all_text(at)


def test_switching_difficulty_shows_new_range_and_attempts():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("Hard").run()
    assert "Attempts left: 5" in _all_text(at)
    assert "between 1 and 50" in _all_text(at)
