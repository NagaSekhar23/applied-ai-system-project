"""Phase 4 tests: command schema, parser, validator and executor.

Parser, validator and executor are tested independently, then together. A
recurring theme: malformed or hostile model output must never cause an
unintended action.
"""

import json
import random
from dataclasses import replace

import pytest
from pydantic import ValidationError

from assistant import Assistant
from game import new_game
from natural_language import (
    Command,
    CommandExecutor,
    CommandParser,
    CommandType,
    handle_message,
    rule_based_parse,
    validate,
)
from natural_language.parser import find_number, find_difficulty
from providers import LLMTimeoutError, MockProvider, ReliableLLMClient, ScriptedProvider


def no_sleep(_seconds):
    pass


def game(secret: int = 50, difficulty: str = "Normal", **changes):
    state = replace(new_game(difficulty, rng=random.Random(0)), secret=secret)
    return replace(state, **changes) if changes else state


def mock_pipeline():
    client = ReliableLLMClient(MockProvider(), sleep=no_sleep)
    return CommandParser(client), CommandExecutor(Assistant(client))


def command_json(**overrides) -> str:
    payload = {
        "command": "SUBMIT_GUESS",
        "guess": 42,
        "difficulty": None,
        "clarification": None,
        "confidence": 0.95,
    }
    payload.update(overrides)
    return json.dumps(payload)


# --- Command schema ----------------------------------------------------------


def test_difficulty_is_normalised_to_canonical_casing():
    assert Command(command=CommandType.START_GAME, difficulty="hard").difficulty == "Hard"
    assert Command(command=CommandType.START_GAME, difficulty="EASY").difficulty == "Easy"


def test_unknown_difficulty_is_rejected_at_the_schema():
    with pytest.raises(ValidationError, match="difficulty"):
        Command(command=CommandType.START_GAME, difficulty="nightmare")


def test_unknown_command_type_is_rejected():
    with pytest.raises(ValidationError):
        Command(command="DROP_TABLE")


def test_confidence_is_bounded():
    with pytest.raises(ValidationError):
        Command(command=CommandType.REQUEST_HINT, confidence=1.5)


def test_command_schema_is_valid_json_schema():
    schema = Command.json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["command"]["enum"]) == {c.value for c in CommandType}


# --- Rule-based parser -------------------------------------------------------


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Start a hard game.", CommandType.START_GAME),
        ("start a new game", CommandType.START_GAME),
        ("let's play", CommandType.START_GAME),
        ("I think the number is 45.", CommandType.SUBMIT_GUESS),
        ("guess 42", CommandType.SUBMIT_GUESS),
        ("my guess is 7", CommandType.SUBMIT_GUESS),
        ("Give me a hint.", CommandType.REQUEST_HINT),
        ("I'm stuck", CommandType.REQUEST_HINT),
        ("what should i do", CommandType.REQUEST_HINT),
        ("explain the rules", CommandType.REQUEST_HINT),
        ("How many tries do I have left?", CommandType.GET_GAME_STATUS),
        ("what is my score", CommandType.GET_GAME_STATUS),
        ("how am i doing", CommandType.GET_GAME_STATUS),
        ("reset the game", CommandType.RESET_GAME),
        ("start over", CommandType.RESET_GAME),
        ("change difficulty to easy", CommandType.CHANGE_DIFFICULTY),
        ("switch to hard", CommandType.CHANGE_DIFFICULTY),
        ("Tell me something.", CommandType.UNKNOWN),
        ("fly me to the moon", CommandType.UNKNOWN),
        ("", CommandType.UNKNOWN),
        ("   ", CommandType.UNKNOWN),
    ],
)
def test_rule_based_parser_classifies_intent(message, expected):
    assert rule_based_parse(message).command is expected


@pytest.mark.parametrize(
    "message, expected",
    [
        ("guess 42", 42),
        ("I think it is 7", 7),
        ("guess forty-five", 45),
        ("my guess is fifty", 50),
        ("try twenty one", 21),
        ("is it one hundred", 100),
    ],
)
def test_rule_based_parser_extracts_numbers(message, expected):
    assert rule_based_parse(message).guess == expected


@pytest.mark.parametrize(
    "message, expected", [("start a hard game", "Hard"), ("new easy game", "Easy")]
)
def test_rule_based_parser_extracts_difficulty(message, expected):
    assert rule_based_parse(message).difficulty == expected


def test_bare_difficulty_word_asks_for_clarification():
    command = rule_based_parse("hard")
    assert command.command is CommandType.UNKNOWN
    assert "new Hard game" in command.clarification
    assert command.confidence < 0.5


def test_unknown_input_suggests_what_to_say():
    assert "guess 42" in rule_based_parse("asdfgh").clarification


def test_hint_request_keeps_the_original_question():
    assert rule_based_parse("give me a hint please").question == "give me a hint please"


def test_find_number_returns_none_without_a_number():
    assert find_number("no digits here") is None


def test_find_difficulty_returns_none_when_absent():
    assert find_difficulty("just a sentence") is None


def test_status_beats_guess_when_both_could_match():
    # "how many attempts" contains no digits, but this guards the ordering:
    # a status question that happens to contain a number stays a status query.
    assert rule_based_parse("how many of my 3 guesses are left").command is CommandType.GET_GAME_STATUS


# --- LLM parser, with fallback ----------------------------------------------


def test_llm_parser_uses_model_output():
    provider = ScriptedProvider([command_json(guess=45)])
    parser = CommandParser(ReliableLLMClient(provider, sleep=no_sleep))
    outcome = parser.parse("I reckon forty five")
    assert outcome.source == "llm"
    assert outcome.command.guess == 45
    assert outcome.prompt_id == "parse_command@1.0.0"


def test_llm_parser_falls_back_to_rules_on_failure():
    provider = ScriptedProvider([LLMTimeoutError("down")] * 4)
    parser = CommandParser(ReliableLLMClient(provider, max_retries=1, sleep=no_sleep))
    outcome = parser.parse("guess 42")
    assert outcome.source == "rules"
    assert outcome.command.command is CommandType.SUBMIT_GUESS
    assert outcome.command.guess == 42
    assert "parser fell back" in outcome.fallback_reason


def test_llm_parser_falls_back_on_malformed_output():
    provider = ScriptedProvider(["not json", "still not json"])
    parser = CommandParser(ReliableLLMClient(provider, max_retries=1, sleep=no_sleep))
    outcome = parser.parse("give me a hint")
    assert outcome.source == "rules"
    assert outcome.command.command is CommandType.REQUEST_HINT


def test_llm_parser_falls_back_on_invalid_difficulty():
    provider = ScriptedProvider([command_json(command="START_GAME", difficulty="nightmare")] * 2)
    parser = CommandParser(ReliableLLMClient(provider, max_retries=1, sleep=no_sleep))
    outcome = parser.parse("start a nightmare game")
    assert outcome.source == "rules"


def test_low_confidence_becomes_a_clarification_request():
    provider = ScriptedProvider([command_json(confidence=0.1)])
    parser = CommandParser(ReliableLLMClient(provider, sleep=no_sleep), min_confidence=0.4)
    outcome = parser.parse("maybe something")
    assert outcome.command.command is CommandType.UNKNOWN
    assert outcome.warnings


def test_parser_without_a_client_uses_rules():
    outcome = CommandParser(client=None).parse("guess 42")
    assert outcome.source == "rules"
    assert outcome.command.guess == 42


def test_parser_never_mutates_game_state():
    state = game(secret=50)
    parser, _ = mock_pipeline()
    parser.parse("guess 42", context={"difficulty": "Normal"})
    assert state == game(secret=50)


# --- Validator ---------------------------------------------------------------


def test_valid_guess_passes_validation():
    result = validate(Command(command=CommandType.SUBMIT_GUESS, guess=42), game())
    assert result.ok


def test_guess_without_a_number_is_rejected():
    result = validate(Command(command=CommandType.SUBMIT_GUESS), game())
    assert not result.ok
    assert result.issues[0].field_name == "guess"


def test_out_of_range_guess_is_rejected_by_the_games_own_validator():
    result = validate(Command(command=CommandType.SUBMIT_GUESS, guess=500), game())
    assert not result.ok
    assert "between 1 and 100" in result.summary


def test_guess_outside_hard_range_is_rejected():
    result = validate(
        Command(command=CommandType.SUBMIT_GUESS, guess=51), game(difficulty="Hard", secret=25)
    )
    assert not result.ok


def test_guess_on_a_finished_game_is_rejected():
    result = validate(
        Command(command=CommandType.SUBMIT_GUESS, guess=42), game(status="won")
    )
    assert not result.ok
    assert "over" in result.summary


def test_guess_without_a_game_is_rejected():
    result = validate(Command(command=CommandType.SUBMIT_GUESS, guess=42), None)
    assert not result.ok
    assert "no game in progress" in result.summary.lower()


def test_unknown_command_needs_clarification():
    result = validate(Command(command=CommandType.UNKNOWN, clarification="What?"), game())
    assert result.needs_clarification
    assert not result.ok
    assert result.summary == "What?"


def test_change_difficulty_without_a_value_is_rejected():
    result = validate(Command(command=CommandType.CHANGE_DIFFICULTY), game())
    assert not result.ok
    assert "Which difficulty" in result.summary


def test_status_and_hint_are_allowed_on_a_finished_game():
    finished = game(status="lost")
    assert validate(Command(command=CommandType.GET_GAME_STATUS), finished).ok
    assert validate(Command(command=CommandType.REQUEST_HINT), finished).ok


def test_start_game_is_allowed_without_an_existing_game():
    assert validate(Command(command=CommandType.START_GAME), None).ok


# --- Executor ----------------------------------------------------------------


def executor() -> CommandExecutor:
    return CommandExecutor(Assistant(ReliableLLMClient(MockProvider(), sleep=no_sleep)))


def test_executor_submits_a_valid_guess():
    result = executor().run(Command(command=CommandType.SUBMIT_GUESS, guess=60), game(secret=50))
    assert result.ok
    assert result.state.attempts_used == 1
    assert result.last_outcome == "Too High"
    assert "LOWER" in result.message


def test_executor_reports_a_win():
    result = executor().run(Command(command=CommandType.SUBMIT_GUESS, guess=50), game(secret=50))
    assert result.state.status == "won"
    assert "You won" in result.message
    assert "50" in result.message  # the secret is public once the game ends


def test_executor_does_not_run_an_invalid_command():
    state = game(secret=50)
    result = executor().run(Command(command=CommandType.SUBMIT_GUESS, guess=500), state)
    assert not result.ok
    assert result.state.attempts_used == 0
    assert result.rejection_reason


def test_executor_refuses_a_guess_after_the_game_ends():
    finished = game(secret=50, status="won")
    result = executor().run(Command(command=CommandType.SUBMIT_GUESS, guess=10), finished)
    assert not result.ok
    assert finished.attempts_used == 0


def test_executor_starts_a_game_at_the_requested_difficulty():
    result = executor().run(
        Command(command=CommandType.START_GAME, difficulty="Hard"), None
    )
    assert result.ok
    assert result.state.difficulty == "Hard"
    assert result.state.max_attempts == 5
    assert 1 <= result.state.secret <= 50


def test_executor_change_difficulty_starts_a_fresh_game():
    played = game(secret=50, attempts_used=3, score=20)
    result = executor().run(
        Command(command=CommandType.CHANGE_DIFFICULTY, difficulty="Easy"), played
    )
    assert result.state.difficulty == "Easy"
    assert result.state.attempts_used == 0
    assert result.state.score == 0


def test_executor_reset_clears_progress():
    played = game(secret=50, attempts_used=4, score=15)
    result = executor().run(Command(command=CommandType.RESET_GAME), played)
    assert result.state.attempts_used == 0
    assert result.state.difficulty == "Normal"


def test_executor_reports_status_without_revealing_the_secret():
    state = game(secret=77, attempts_used=2, score=10)
    result = executor().run(Command(command=CommandType.GET_GAME_STATUS), state)
    assert result.ok
    assert "Attempts left: 6" in result.message
    assert "77" not in result.message


def test_executor_status_reveals_the_secret_once_the_game_is_over():
    result = executor().run(
        Command(command=CommandType.GET_GAME_STATUS), game(secret=77, status="lost")
    )
    assert "77" in result.message


def test_executor_requests_a_hint_through_the_assistant():
    result = executor().run(Command(command=CommandType.REQUEST_HINT), game(secret=50))
    assert result.ok
    assert result.assistant_reply is not None
    assert result.message


def test_executor_hint_works_without_an_assistant():
    result = CommandExecutor(assistant=None).run(
        Command(command=CommandType.REQUEST_HINT), game(secret=50)
    )
    assert result.ok
    assert result.message


def test_executor_handles_a_missing_game_for_status_and_hint():
    for command_type in (CommandType.GET_GAME_STATUS, CommandType.REQUEST_HINT):
        result = executor().run(Command(command=command_type), None)
        assert result.ok
        assert "start a game" in result.message.lower()


def test_executor_never_runs_an_unclarified_command():
    state = game(secret=50)
    result = executor().run(Command(command=CommandType.UNKNOWN, clarification="?"), state)
    assert not result.ok
    assert result.state is state


# --- Full pipeline -----------------------------------------------------------


def test_pipeline_submits_a_guess_end_to_end():
    parser, ex = mock_pipeline()
    turn = handle_message("I think the number is 60", game(secret=50), parser, ex)
    assert turn.command_type == "SUBMIT_GUESS"
    assert turn.result.ok
    assert turn.result.state.attempts_used == 1


def test_pipeline_rejects_an_out_of_range_guess_end_to_end():
    parser, ex = mock_pipeline()
    turn = handle_message("guess 500", game(secret=50), parser, ex)
    assert not turn.result.ok
    assert turn.result.state.attempts_used == 0


def test_pipeline_asks_for_clarification_on_nonsense():
    parser, ex = mock_pipeline()
    turn = handle_message("purple monkey dishwasher", game(secret=50), parser, ex)
    assert turn.command_type == "UNKNOWN"
    assert turn.validated.needs_clarification
    assert not turn.result.executed


def test_pipeline_context_never_includes_the_secret():
    from natural_language.executor import _context_for

    context = _context_for(game(secret=42))
    assert 42 not in context.values()
    assert "secret" not in context


def test_pipeline_survives_a_hostile_command_string():
    # A model emitting something outside the enum cannot reach the executor:
    # Pydantic rejects it, and the parser falls back to rules.
    provider = ScriptedProvider([command_json(command="DELETE_EVERYTHING")] * 2)
    parser = CommandParser(ReliableLLMClient(provider, max_retries=1, sleep=no_sleep))
    ex = executor()
    state = game(secret=50)
    turn = handle_message("do something bad", state, parser, ex)
    assert turn.result.command.command in set(CommandType)
    assert state.attempts_used == 0


def test_pipeline_multi_turn_sequence():
    parser, ex = mock_pipeline()
    state = game(secret=50)
    last_outcome = None

    for message, expected in [
        ("guess 60", "SUBMIT_GUESS"),
        ("what is my score", "GET_GAME_STATUS"),
        ("give me a hint", "REQUEST_HINT"),
        ("guess 50", "SUBMIT_GUESS"),
    ]:
        turn = handle_message(message, state, parser, ex, last_outcome)
        assert turn.command_type == expected
        if turn.result.state is not None:
            state = turn.result.state
        if turn.result.last_outcome:
            last_outcome = turn.result.last_outcome

    assert state.status == "won"
    assert state.attempts_used == 2
