"""AppTest coverage for the Natural Language page."""

from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

APP = "app.py"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ["LLM_PROVIDER", "ANTHROPIC_API_KEY", "GAME_DEBUG"]:
        monkeypatch.delenv(name, raising=False)


def goto_chat(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("Natural Language")
    return at.run()


def say(at: AppTest, message: str) -> AppTest:
    at.chat_input[0].set_value(message)
    return at.run()


def last_turn(at: AppTest):
    return at.session_state["nl_history"][-1]


def test_chat_page_is_reachable():
    at = goto_chat(AppTest.from_file(APP, default_timeout=10).run())
    assert at.chat_input
    assert not at.exception


def test_guess_through_chat_advances_the_game():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_chat(at)

    say(at, "I think the number is 60")
    turn = last_turn(at)
    assert turn.command_type == "SUBMIT_GUESS"
    assert turn.result.executed
    assert at.session_state["game"].attempts_used == 1


def test_winning_through_chat_updates_state():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_chat(at)

    say(at, "guess 50")
    assert at.session_state["game"].status == "won"
    assert at.session_state["game"].score == 90


def test_invalid_guess_is_rejected_without_consuming_an_attempt():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_chat(at)

    say(at, "guess 500")
    assert last_turn(at).result.executed is False
    assert at.session_state["game"].attempts_used == 0


def test_unknown_request_executes_nothing():
    at = goto_chat(AppTest.from_file(APP, default_timeout=10).run())
    say(at, "please delete everything")
    turn = last_turn(at)
    assert turn.command_type == "UNKNOWN"
    assert turn.result.executed is False


def test_status_request_does_not_change_state():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_chat(at)
    before = at.session_state["game"]

    say(at, "what is my score")
    assert at.session_state["game"] == before


def test_starting_a_game_through_chat_changes_difficulty():
    at = goto_chat(AppTest.from_file(APP, default_timeout=10).run())
    say(at, "start a hard game")
    game = at.session_state["game"]
    assert game.difficulty == "Hard"
    assert game.max_attempts == 5


def test_chat_history_accumulates():
    at = goto_chat(AppTest.from_file(APP, default_timeout=10).run())
    say(at, "what is my score")
    say(at, "give me a hint")
    assert len(at.session_state["nl_history"]) == 2


def test_chat_never_reveals_the_secret_while_playing():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=77)
    at.run()
    goto_chat(at)

    say(at, "give me a hint")
    assert "77" not in last_turn(at).result.message


def test_chat_and_game_pages_share_one_game():
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.session_state["game"] = replace(at.session_state["game"], secret=50)
    at.run()
    goto_chat(at)
    say(at, "guess 60")

    at.sidebar.radio[0].set_value("Game")
    at.run()
    assert at.session_state["game"].attempts_used == 1
    assert "Attempts left: 7" in "\n".join(str(i.value) for i in at.info)
