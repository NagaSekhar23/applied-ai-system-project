"""AppTest coverage for the Observability page and end-to-end instrumentation."""

from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

from observability import Tracker
from observability.tracker import set_tracker

APP = "app.py"


@pytest.fixture(autouse=True)
def isolated_tracker(monkeypatch):
    """Each test gets a clean tracker, and the global one is restored after."""
    for name in ["LLM_PROVIDER", "ANTHROPIC_API_KEY", "GAME_DEBUG"]:
        monkeypatch.delenv(name, raising=False)
    set_tracker(Tracker())
    yield
    set_tracker(None)


def goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio[0].set_value(page)
    return at.run()


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [i.value for i in at.info]
    parts += [c.value for c in at.caption] + [s.value for s in at.success]
    parts += [w.value for w in at.warning] + [e.value for e in at.error]
    parts += [h.value for h in at.subheader]
    return "\n".join(str(p) for p in parts)


def test_dashboard_is_reachable():
    at = goto(AppTest.from_file(APP, default_timeout=10).run(), "Observability")
    assert not at.exception
    assert "Observability" in all_text(at)


def test_dashboard_shows_an_empty_state_rather_than_fake_data():
    at = goto(AppTest.from_file(APP, default_timeout=10).run(), "Observability")
    text = all_text(at)
    assert "No activity recorded yet" in text
    assert "never displays sample data" in text


def test_assistant_use_is_recorded_and_shown():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()

    goto(at, "AI Assistant")
    next(b for b in at.button if "hint" in b.label.lower()).click()
    at.run()

    goto(at, "Observability")
    assert "No activity recorded yet" not in all_text(at)
    assert at.metric


def test_agent_use_is_recorded_with_tool_metrics():
    at = goto(AppTest.from_file(APP, default_timeout=10).run(), "Agent")
    at.chat_input[0].set_value("what is my status")
    at.run()

    goto(at, "Observability")
    labels = {m.label for m in at.metric}
    assert "Events" in labels
    assert "Runs" in labels  # the agent section rendered


def test_natural_language_use_is_recorded():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()

    goto(at, "Natural Language")
    at.chat_input[0].set_value("guess 60")
    at.run()

    goto(at, "Observability")
    assert "No activity recorded yet" not in all_text(at)


def test_mock_calls_are_reported_as_unpriced():
    at = goto(AppTest.from_file(APP, default_timeout=10).run(), "Agent")
    at.chat_input[0].set_value("explain the rules")
    at.run()

    goto(at, "Observability")
    text = all_text(at)
    # Mock usage must never be presented as a real cost.
    assert "could not be priced" in text or "not available" in str(
        [m.value for m in at.metric]
    )


def test_clearing_events_returns_to_the_empty_state():
    at = goto(AppTest.from_file(APP, default_timeout=10).run(), "Agent")
    at.chat_input[0].set_value("what is my status")
    at.run()

    goto(at, "Observability")
    next(b for b in at.button if "Clear" in b.label).click()
    at.run()
    assert "No activity recorded yet" in all_text(at)


def test_dashboard_warns_about_small_sample_sizes():
    at = goto(AppTest.from_file(APP, default_timeout=10).run(), "Agent")
    at.chat_input[0].set_value("what is my status")
    at.run()

    goto(at, "Observability")
    assert "percentiles are noisy" in all_text(at)
