"""Phase 5 tests: tool registry, planner, execution loop and guardrails.

The central claims under test: the agent can only call registered tools, it
cannot bypass game validation, and every loop terminates.
"""

import json
import random
from dataclasses import replace

import pytest
from pydantic import BaseModel, ValidationError

from agents import (
    AgentConfig,
    AgentController,
    AgentDecision,
    Planner,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
    UnknownToolError,
    build_agent,
    default_registry,
    rule_based_plan,
)
from agents.tools import GuessInput, NoInput, snapshot_of
from assistant import Assistant
from game import new_game
from providers import LLMTimeoutError, MockProvider, ReliableLLMClient, ScriptedProvider


def no_sleep(_seconds):
    pass


def game(secret: int = 50, difficulty: str = "Normal", **changes):
    state = replace(new_game(difficulty, rng=random.Random(0)), secret=secret)
    return replace(state, **changes) if changes else state


def mock_client() -> ReliableLLMClient:
    return ReliableLLMClient(MockProvider(), sleep=no_sleep)


def mock_agent(config: AgentConfig | None = None) -> AgentController:
    client = mock_client()
    return build_agent(assistant=Assistant(client), client=client, config=config)


def decision_json(**overrides) -> str:
    payload = {
        "action": "call_tool",
        "tool": "get_game_status",
        "arguments": {},
        "response": None,
        "reasoning": "checking status",
    }
    payload.update(overrides)
    return json.dumps(payload)


# --- Tool registry -----------------------------------------------------------


def test_default_registry_exposes_the_expected_tools():
    registry = default_registry()
    assert set(registry.names) == {
        "start_game",
        "submit_guess",
        "request_hint",
        "get_game_status",
        "change_difficulty",
        "reset_game",
        "explain_rules",
    }


def test_unknown_tool_lookup_lists_available_tools():
    with pytest.raises(UnknownToolError, match="submit_guess"):
        default_registry().get("rm_rf")


def test_registry_rejects_duplicate_registration():
    registry = default_registry()
    tool = registry.get("start_game")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_tool_schemas_describe_inputs_for_the_planner():
    schemas = {s["name"]: s for s in default_registry().schemas()}
    assert "guess" in schemas["submit_guess"]["input_schema"]["properties"]
    assert schemas["submit_guess"]["mutates_state"] is True
    assert schemas["get_game_status"]["mutates_state"] is False


def test_tool_validates_arguments_against_its_schema():
    tool = default_registry().get("submit_guess")
    assert tool.validate_arguments({"guess": 42}).guess == 42
    with pytest.raises(ToolValidationError):
        tool.validate_arguments({"guess": "not a number"})
    with pytest.raises(ToolValidationError):
        tool.validate_arguments({})


def test_tool_rejects_unexpected_arguments():
    tool = default_registry().get("submit_guess")
    with pytest.raises(ToolValidationError):
        tool.validate_arguments({"guess": 42, "sneaky": "extra"})


def test_tool_output_is_validated_against_its_schema():
    tool = default_registry().get("get_game_status")
    bad = ToolResult(tool="get_game_status", ok=True, output={"nonsense": 1}, message="x")
    with pytest.raises(Exception, match="violates its schema"):
        tool.validate_output(bad)


def test_snapshot_hides_the_secret_while_playing():
    assert snapshot_of(game(secret=77)).secret is None


def test_snapshot_reveals_the_secret_once_the_game_ends():
    assert snapshot_of(game(secret=77, status="won")).secret == 77


# --- Individual tools --------------------------------------------------------


def run_tool(name: str, arguments: dict, state=None, assistant=None) -> ToolResult:
    registry = default_registry()
    tool = registry.get(name)
    args = tool.validate_arguments(arguments)
    return tool.run(args, ToolContext(state=state, assistant=assistant))


def test_start_game_tool_uses_the_requested_difficulty():
    result = run_tool("start_game", {"difficulty": "Hard"})
    assert result.ok
    assert result.new_state.difficulty == "Hard"
    assert result.new_state.max_attempts == 5


def test_start_game_tool_rejects_an_unknown_difficulty():
    with pytest.raises(ValueError, match="difficulty must be one of"):
        run_tool("start_game", {"difficulty": "Nightmare"})


def test_submit_guess_tool_advances_the_game():
    result = run_tool("submit_guess", {"guess": 60}, game(secret=50))
    assert result.ok
    assert result.output["outcome"] == "Too High"
    assert result.new_state.attempts_used == 1


def test_submit_guess_tool_cannot_bypass_range_validation():
    state = game(secret=25, difficulty="Hard")
    result = run_tool("submit_guess", {"guess": 999}, state)
    assert not result.ok
    assert result.new_state.attempts_used == 0


def test_submit_guess_tool_refuses_a_finished_game():
    result = run_tool("submit_guess", {"guess": 10}, game(secret=50, status="won"))
    assert not result.ok


def test_tools_requiring_a_game_fail_cleanly_without_one():
    for name in ("submit_guess", "get_game_status", "request_hint"):
        arguments = {"guess": 5} if name == "submit_guess" else {}
        result = run_tool(name, arguments, None)
        assert not result.ok
        assert result.error


def test_status_tool_does_not_leak_the_secret_while_playing():
    result = run_tool("get_game_status", {}, game(secret=77))
    assert "77" not in result.message
    assert result.output["secret"] is None


def test_hint_tool_falls_back_without_an_assistant():
    result = run_tool("request_hint", {}, game(secret=50))
    assert result.ok
    assert result.output["source"] == "fallback"


def test_explain_rules_tool_describes_the_active_difficulty():
    result = run_tool("explain_rules", {}, game(difficulty="Hard", secret=25))
    assert "Hard" in result.message
    assert "1 and 50" in result.message


# --- Rule-based planner ------------------------------------------------------


@pytest.mark.parametrize(
    "message, expected_tool",
    [
        ("start a hard game", "start_game"),
        ("guess 42", "submit_guess"),
        ("give me a hint", "request_hint"),
        ("what is my score", "get_game_status"),
        ("change difficulty to easy", "change_difficulty"),
        ("reset the game", "reset_game"),
        ("explain the rules", "explain_rules"),
    ],
)
def test_rule_planner_selects_the_right_tool(message, expected_tool):
    decision = rule_based_plan(message, game(), [], default_registry())
    assert decision.action == "call_tool"
    assert decision.tool == expected_tool


def test_rule_planner_clarifies_an_unsupported_request():
    decision = rule_based_plan("do a backflip", game(), [], default_registry())
    assert decision.action == "clarify"
    assert decision.response


def test_rule_planner_responds_once_steps_are_done():
    decision = rule_based_plan("start a game", game(), ["start_game"], default_registry())
    assert decision.action == "respond"


def test_rule_planner_never_invents_a_guess():
    decision = rule_based_plan("guess for me", game(), [], default_registry())
    assert decision.tool != "submit_guess"


# --- LLM planner -------------------------------------------------------------


def planner_with(script) -> Planner:
    registry = default_registry()
    return Planner(registry, ReliableLLMClient(ScriptedProvider(script), max_retries=1, sleep=no_sleep))


def test_llm_planner_uses_model_decisions():
    outcome = planner_with([decision_json()]).plan("status?", game(), [], [])
    assert outcome.source == "llm"
    assert outcome.decision.tool == "get_game_status"


def test_llm_planner_rejects_an_unregistered_tool():
    outcome = planner_with([decision_json(tool="delete_everything")]).plan(
        "give me a hint", game(), [], []
    )
    assert outcome.source == "rules"
    assert outcome.decision.tool == "request_hint"
    assert outcome.warnings


def test_llm_planner_falls_back_when_the_provider_fails():
    outcome = planner_with([LLMTimeoutError("down")] * 4).plan("guess 42", game(), [], [])
    assert outcome.source == "rules"
    assert outcome.decision.tool == "submit_guess"


def test_llm_planner_falls_back_on_malformed_output():
    outcome = planner_with(["not json", "still not json"]).plan("hint", game(), [], [])
    assert outcome.source == "rules"


def test_agent_decision_rejects_an_unknown_action():
    with pytest.raises(ValidationError):
        AgentDecision(action="delete_everything")


# --- The loop and its guardrails ---------------------------------------------


def test_agent_completes_a_single_step_request():
    run = mock_agent().run("what is my status", game(secret=50))
    assert run.succeeded
    assert run.tools_called == ["get_game_status"]
    assert run.response


def test_agent_completes_a_multi_step_request():
    run = mock_agent().run("start a hard game and explain the rules", game(secret=50))
    assert run.tools_called == ["start_game", "explain_rules"]
    assert run.state.difficulty == "Hard"
    assert run.iterations == 3  # two tool calls, then respond


def test_agent_response_does_not_leak_tool_names():
    run = mock_agent().run("start a hard game and explain the rules", game(secret=50))
    assert "start_game:" not in run.response
    assert "explain_rules:" not in run.response


def test_agent_submits_a_guess_and_updates_state():
    run = mock_agent().run("guess 60", game(secret=50))
    assert run.state.attempts_used == 1
    assert run.last_outcome == "Too High"


def test_agent_clarifies_an_unsupported_request_without_calling_tools():
    run = mock_agent().run("do a backflip", game(secret=50))
    assert run.tools_called == []
    assert run.stop_reason == "clarification"


def test_agent_stops_at_the_iteration_limit():
    """A planner that always calls a tool must still terminate."""
    registry = default_registry()
    always_call = Planner(registry, ReliableLLMClient(
        ScriptedProvider([decision_json(arguments={}) for _ in range(20)]), sleep=no_sleep
    ))
    agent = AgentController(
        registry, always_call, config=AgentConfig(max_iterations=3, allow_repeat_calls=True)
    )
    run = agent.run("status", game(secret=50))
    assert run.iterations == 3
    assert run.stop_reason == "max_iterations"
    assert not run.succeeded


def test_agent_refuses_a_repeated_identical_call():
    registry = default_registry()
    repeat = Planner(registry, ReliableLLMClient(
        ScriptedProvider([decision_json() for _ in range(5)]), sleep=no_sleep
    ))
    agent = AgentController(registry, repeat, config=AgentConfig(max_iterations=5))
    run = agent.run("status", game(secret=50))
    assert run.stop_reason == "repeated_call"
    assert run.tools_called == ["get_game_status"]


def test_agent_stops_on_timeout():
    ticks = iter([0.0, 0.0, 99.0, 99.0, 99.0, 99.0])
    registry = default_registry()
    planner = Planner(registry, ReliableLLMClient(
        ScriptedProvider([decision_json() for _ in range(5)]), sleep=no_sleep
    ))
    agent = AgentController(
        registry, planner, config=AgentConfig(timeout_seconds=1.0), clock=lambda: next(ticks)
    )
    run = agent.run("status", game(secret=50))
    assert run.stop_reason == "timeout"
    assert not run.succeeded


def test_agent_recovers_from_a_single_tool_error():
    """An invalid call is reported back, and the agent carries on."""
    registry = default_registry()
    script = [
        decision_json(tool="submit_guess", arguments={"guess": 9999}),  # out of range
        decision_json(tool="get_game_status", arguments={}),
        decision_json(action="respond", tool=None, response="Done."),
    ]
    planner = Planner(registry, ReliableLLMClient(ScriptedProvider(script), sleep=no_sleep))
    agent = AgentController(registry, planner, config=AgentConfig(max_iterations=5))
    run = agent.run("guess then check", game(secret=50))
    assert run.response == "Done."
    assert run.state.attempts_used == 0  # the invalid guess never landed


def test_agent_stops_after_too_many_tool_errors():
    registry = default_registry()
    script = [decision_json(tool="submit_guess", arguments={"guess": 9999}) for _ in range(6)]
    planner = Planner(registry, ReliableLLMClient(ScriptedProvider(script), sleep=no_sleep))
    agent = AgentController(
        registry, planner,
        config=AgentConfig(max_iterations=10, max_tool_errors=2, allow_repeat_calls=True),
    )
    run = agent.run("break it", game(secret=50))
    assert run.stop_reason == "too_many_tool_errors"
    assert run.state.attempts_used == 0


def test_agent_handles_a_tool_that_raises():
    def boom(args, ctx):
        raise RuntimeError("tool exploded")

    registry = default_registry()
    registry.register(
        Tool(
            name="explodes",
            description="always fails",
            input_model=NoInput,
            output_model=NoInput,
            run=boom,
        )
    )
    planner = Planner(registry, ReliableLLMClient(
        ScriptedProvider([
            decision_json(tool="explodes", arguments={}),
            decision_json(action="respond", tool=None, response="Recovered."),
        ]), sleep=no_sleep,
    ))
    run = AgentController(registry, planner).run("explode", game(secret=50))
    assert run.response == "Recovered."
    assert any(s.status == "error" for s in run.trace.spans)


def test_agent_cannot_guess_on_a_finished_game():
    run = mock_agent().run("guess 42", game(secret=50, status="won"))
    assert run.state.attempts_used == 0
    assert not any(r.ok for r in run.tool_results if r.tool == "submit_guess")


def test_agent_does_not_start_a_game_unprompted():
    state = game(secret=50, attempts_used=3)
    run = mock_agent().run("what is my score", state)
    assert "start_game" not in run.tools_called
    assert run.state.attempts_used == 3


def test_agent_never_executes_an_unregistered_tool():
    registry = default_registry()
    planner = Planner(registry, ReliableLLMClient(
        ScriptedProvider([decision_json(tool="os.system", arguments={"cmd": "rm -rf /"})] * 3),
        max_retries=1, sleep=no_sleep,
    ))
    run = AgentController(registry, planner).run("delete my files", game(secret=50))
    for called in run.tools_called:
        assert called in registry


# --- Tracing -----------------------------------------------------------------


def test_trace_records_decisions_and_tool_calls():
    run = mock_agent().run("what is my status", game(secret=50))
    kinds = [s.kind for s in run.trace.spans]
    assert "agent" in kinds and "decision" in kinds and "tool" in kinds and "response" in kinds


def test_trace_spans_have_a_parent():
    run = mock_agent().run("guess 60", game(secret=50))
    root = next(s for s in run.trace.spans if s.kind == "agent")
    children = [s for s in run.trace.spans if s.kind != "agent"]
    assert all(child.parent_id == root.span_id for child in children)


def test_trace_records_iteration_numbers():
    run = mock_agent().run("start a hard game and explain the rules", game(secret=50))
    assert sorted({s.iteration for s in run.trace.tool_spans}) == [1, 2]


def test_trace_serialises_to_a_dict():
    run = mock_agent().run("what is my status", game(secret=50))
    payload = run.trace.to_dict()
    assert payload["trace_id"]
    assert payload["tools_called"] == ["get_game_status"]
    assert payload["stop_reason"] == "responded"
    json.dumps(payload)  # must be JSON-serialisable for Phase 6


def test_trace_records_tool_failures():
    run = mock_agent().run("guess 42", game(secret=50, status="won"))
    assert run.trace.failed_spans
