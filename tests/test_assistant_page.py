"""AppTest coverage for navigation and the AI Assistant page."""

from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

APP = "app.py"

LLM_ENV = ["LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "LLM_EFFORT", "GAME_DEBUG"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)


def _all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error]
    parts += [w.value for w in at.warning] + [s.value for s in at.success]
    parts += [i.value for i in at.info] + [c.value for c in at.caption]
    parts += [h.value for h in at.subheader] + [h.value for h in at.title]
    return "\n".join(str(p) for p in parts)


def _button(at: AppTest, label_part: str):
    return next(b for b in at.button if label_part in b.label)


def goto_assistant(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("AI Assistant")
    return at.run()


# --- Navigation --------------------------------------------------------------


def test_game_is_the_default_page():
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert at.sidebar.radio[0].value == "Game"
    assert any("Submit" in b.label for b in at.button)


def test_can_switch_to_the_assistant_page():
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    assert "AI Assistant" in _all_text(at)
    assert any("hint" in b.label.lower() for b in at.button)


def test_switching_pages_preserves_the_game():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    at.text_input[0].input("60")
    _button(at, "Submit").click()
    at.run()
    assert at.session_state["game"].attempts_used == 1

    goto_assistant(at)
    assert at.session_state["game"].attempts_used == 1


# --- Mode banner and secret safety -------------------------------------------


def test_mock_mode_is_announced_by_default():
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    assert "Mock mode" in _all_text(at)


def test_live_mode_is_announced_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    assert "Live mode" in _all_text(at)


def test_missing_credentials_are_flagged(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    assert "ANTHROPIC_API_KEY" in _all_text(at)


def test_api_key_is_never_rendered(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-value")
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    assert "sk-ant-super-secret-value" not in _all_text(at)


def test_invalid_configuration_shows_an_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonsense")
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert any("Configuration error" in e.value for e in at.error)


# --- Getting a hint ----------------------------------------------------------


def test_hint_button_produces_a_reply():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    at.text_input[0].input("60")
    _button(at, "Submit").click()
    at.run()

    goto_assistant(at)
    _button(at, "hint").click()
    at.run()

    reply = at.session_state["assistant_reply"]
    assert reply.message
    assert reply.category.value == "too_high"
    assert "Assistant" in _all_text(at)


def test_hint_never_shows_the_secret_while_playing():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=77)
    at.run()
    at.text_input[0].input("10")
    _button(at, "Submit").click()
    at.run()

    goto_assistant(at)
    _button(at, "hint").click()
    at.run()

    assert "77" not in at.session_state["assistant_reply"].message


def test_rules_button_uses_the_rules_prompt():
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    _button(at, "rules").click()
    at.run()
    assert at.session_state["assistant_reply"].prompt_id == "assistant_rules@1.0.0"


def test_reply_is_labelled_with_its_source():
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    _button(at, "hint").click()
    at.run()
    assert "Source: model output" in _all_text(at)


def test_new_game_clears_a_previous_reply():
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    _button(at, "hint").click()
    at.run()
    assert "assistant_reply" in at.session_state

    at.sidebar.radio[0].set_value("Game")
    at.run()
    _button(at, "New Game").click()
    at.run()
    assert "assistant_reply" not in at.session_state


def test_assistant_page_renders_before_any_guess():
    at = goto_assistant(AppTest.from_file(APP, default_timeout=10).run())
    _button(at, "hint").click()
    at.run()
    assert at.session_state["assistant_reply"].category.value == "opening"
