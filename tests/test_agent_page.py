"""AppTest coverage for the Agent page."""

from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

APP = "app.py"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ["LLM_PROVIDER", "ANTHROPIC_API_KEY", "GAME_DEBUG"]:
        monkeypatch.delenv(name, raising=False)


def goto_agent(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("Agent")
    return at.run()


def ask(at: AppTest, message: str) -> AppTest:
    at.chat_input[0].set_value(message)
    return at.run()


def last_run(at: AppTest):
    return at.session_state["agent_history"][-1]


def test_agent_page_is_reachable():
    at = goto_agent(AppTest.from_file(APP, default_timeout=10).run())
    assert at.chat_input
    assert not at.exception


def test_agent_answers_a_status_question():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_agent(at)

    ask(at, "what is my status")
    run = last_run(at)
    assert run.tools_called == ["get_game_status"]
    assert run.succeeded


def test_agent_completes_a_multi_step_request():
    at = goto_agent(AppTest.from_file(APP, default_timeout=10).run())
    ask(at, "start a hard game and explain the rules")
    run = last_run(at)
    assert run.tools_called == ["start_game", "explain_rules"]
    assert at.session_state["game"].difficulty == "Hard"


def test_agent_guess_updates_shared_state():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_agent(at)

    ask(at, "guess 60")
    assert at.session_state["game"].attempts_used == 1
    assert at.session_state["last_outcome"] == "Too High"


def test_agent_does_not_act_on_an_unsupported_request():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_agent(at)

    ask(at, "delete everything")
    assert last_run(at).tools_called == []
    assert at.session_state["game"].attempts_used == 0


def test_agent_page_lists_registered_tools():
    at = goto_agent(AppTest.from_file(APP, default_timeout=10).run())
    text = "\n".join(str(m.value) for m in at.markdown)
    assert "submit_guess" in text
    assert "get_game_status" in text


def test_agent_history_accumulates():
    at = goto_agent(AppTest.from_file(APP, default_timeout=10).run())
    ask(at, "explain the rules")
    ask(at, "what is my status")
    assert len(at.session_state["agent_history"]) == 2


def test_agent_never_reveals_the_secret_while_playing():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=77)
    at.run()
    goto_agent(at)

    ask(at, "what is my status")
    assert "77" not in last_run(at).response


def test_agent_and_game_pages_share_one_game():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_agent(at)
    ask(at, "guess 60")

    at.sidebar.radio[0].set_value("Game")
    at.run()
    assert "Attempts left: 7" in "\n".join(str(i.value) for i in at.info)
