"""Phase 6 tests: redaction, events, store, metrics, pricing and tracing."""

import json
import random
from dataclasses import replace

import pytest

from agents import build_agent
from assistant import Assistant
from game import new_game
from observability import (
    REDACTED,
    Component,
    Event,
    EventStore,
    Status,
    Tracker,
    compute_metrics,
    estimate_cost,
    get_pricing,
    load_events,
    percentile,
    redact,
    scrub_text,
    summarise_text,
)
from observability.tracker import current_request_id, get_tracker, set_tracker
from providers import CallStats, MockProvider, ReliableLLMClient, ScriptedProvider, TokenUsage
from providers.base import LLMTimeoutError


def no_sleep(_seconds):
    pass


def game(secret: int = 50, **changes):
    state = replace(new_game("Normal", rng=random.Random(0)), secret=secret)
    return replace(state, **changes) if changes else state


@pytest.fixture
def tracker():
    """A fresh tracker per test, and the global one restored afterwards."""
    instance = Tracker()
    set_tracker(instance)
    yield instance
    set_tracker(None)


# --- Redaction ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "my key is sk-ant-api03-abcdef123456",
        "Authorization: Bearer abcdef0123456789",
        "sk-abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_key_shaped_strings_are_scrubbed(text):
    assert REDACTED in scrub_text(text)
    assert "abcdef" not in scrub_text(text) or "sk-" not in scrub_text(text)


def test_sensitive_field_names_are_dropped():
    out = redact({"api_key": "anything", "ANTHROPIC_API_KEY": "x", "safe": "kept"})
    assert out["api_key"] == REDACTED
    assert out["ANTHROPIC_API_KEY"] == REDACTED
    assert out["safe"] == "kept"


def test_game_secret_is_redacted_by_default():
    assert redact({"secret": 42})["secret"] == REDACTED


def test_game_secret_can_be_kept_for_a_finished_game():
    assert redact({"secret": 42}, drop_game_secret=False)["secret"] == 42


def test_redaction_is_recursive():
    out = redact({"outer": {"inner": [{"api_key": "x"}]}})
    assert out["outer"]["inner"][0]["api_key"] == REDACTED


def test_long_text_is_truncated():
    scrubbed = scrub_text("a" * 500)
    assert len(scrubbed) < 300
    assert "500 chars" in scrubbed


def test_summarise_text_records_shape_not_content():
    assert summarise_text("my secret question") == {"present": True, "chars": 18}
    assert summarise_text(None) == {"present": False, "chars": 0}


def test_event_redacts_attributes_at_construction():
    event = Event(
        operation="x",
        component=Component.LLM,
        attributes={"api_key": "sk-ant-leak", "note": "sk-ant-inline-leak"},
    )
    serialised = json.dumps(event.to_dict())
    assert "sk-ant-leak" not in serialised
    assert "sk-ant-inline-leak" not in serialised


def test_event_redacts_error_messages():
    event = Event(
        operation="x",
        component=Component.LLM,
        error_message="failed with key sk-ant-abcdef123456",
    )
    assert "sk-ant-abcdef123456" not in event.error_message


# --- Events and the store ----------------------------------------------------


def test_event_serialises_to_json():
    event = Event(operation="op", component=Component.GAME, duration_ms=12.345)
    payload = event.to_dict()
    assert payload["operation"] == "op"
    assert payload["duration_ms"] == 12.35
    json.dumps(payload)


def test_degraded_counts_as_ok():
    assert Event(operation="o", component=Component.LLM, status=Status.DEGRADED).ok
    assert not Event(operation="o", component=Component.LLM, status=Status.ERROR).ok


def test_store_is_bounded_and_reports_evictions():
    store = EventStore(max_events=3)
    for i in range(5):
        store.record(Event(operation=f"op{i}", component=Component.GAME))
    assert len(store) == 3
    assert store.dropped_count == 2


def test_store_filters_by_component_and_request():
    store = EventStore()
    store.record(Event(operation="a", component=Component.LLM, request_id="r1"))
    store.record(Event(operation="b", component=Component.TOOL, request_id="r1"))
    store.record(Event(operation="c", component=Component.TOOL, request_id="r2"))

    assert len(store.by_component(Component.TOOL)) == 2
    assert len(store.by_request("r1")) == 2
    assert store.request_ids() == ["r2", "r1"]


def test_store_lists_failures():
    store = EventStore()
    store.record(Event(operation="ok", component=Component.GAME))
    store.record(Event(operation="bad", component=Component.GAME, status=Status.ERROR))
    assert [e.operation for e in store.failures()] == ["bad"]


def test_store_writes_and_reads_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    store = EventStore(log_path=path)
    store.record(Event(operation="a", component=Component.GAME))
    store.record(Event(operation="b", component=Component.LLM))

    loaded = load_events(path)
    assert [e["operation"] for e in loaded] == ["a", "b"]


def test_log_write_failure_does_not_break_recording(tmp_path):
    # A directory where the file should be: writing must fail, recording must not.
    path = tmp_path / "blocked"
    path.mkdir()
    store = EventStore(log_path=path)
    event = store.record(Event(operation="a", component=Component.GAME))
    assert len(store) == 1
    assert "log_write_error" in event.attributes


def test_load_events_tolerates_missing_and_malformed_files(tmp_path):
    assert load_events(tmp_path / "absent.jsonl") == []
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    assert len(load_events(bad)) == 1


# --- Metrics -----------------------------------------------------------------


def test_percentile_uses_nearest_rank():
    assert percentile([10, 20, 30, 40, 50], 0.95) == 50
    assert percentile([10], 0.95) == 10
    assert percentile([], 0.95) == 0.0


def test_metrics_on_an_empty_store_do_not_divide_by_zero():
    metrics = compute_metrics([])
    assert metrics.total_events == 0
    assert metrics.success_rate == 0.0
    assert metrics.llm_failure_rate == 0.0
    assert metrics.retry_rate == 0.0


def test_success_and_failure_rates():
    events = [
        Event(operation="a", component=Component.GAME),
        Event(operation="b", component=Component.GAME),
        Event(operation="c", component=Component.GAME, status=Status.ERROR),
        Event(operation="d", component=Component.GAME, status=Status.DEGRADED),
    ]
    metrics = compute_metrics(events)
    assert metrics.ok_events == 3  # degraded counts as ok
    assert metrics.error_events == 1
    assert metrics.success_rate == 0.75
    assert metrics.failure_rate == 0.25
    assert metrics.degraded_events == 1


def test_latency_metrics():
    events = [
        Event(operation="a", component=Component.GAME, duration_ms=float(ms))
        for ms in (10, 20, 30, 40, 1000)
    ]
    metrics = compute_metrics(events)
    assert metrics.avg_latency_ms == 220.0
    assert metrics.p95_latency_ms == 1000.0


def test_llm_metrics_aggregate_retries_and_timeouts():
    events = [
        Event(
            operation="llm",
            component=Component.LLM,
            attributes={"retries": 2, "timeouts": 1, "validation_failures": 1},
        ),
        Event(
            operation="llm",
            component=Component.LLM,
            status=Status.ERROR,
            attributes={"retries": 1, "timeouts": 1},
        ),
    ]
    metrics = compute_metrics(events)
    assert metrics.llm_calls == 2
    assert metrics.llm_failures == 1
    assert metrics.llm_failure_rate == 0.5
    assert metrics.llm_retries == 3
    assert metrics.retry_rate == 1.5  # can exceed 1.0
    assert metrics.llm_timeouts == 2
    assert metrics.llm_validation_failures == 1


def test_tool_metrics_count_usage():
    events = [
        Event(operation="t", component=Component.TOOL, attributes={"tool": "submit_guess"}),
        Event(operation="t", component=Component.TOOL, attributes={"tool": "submit_guess"}),
        Event(
            operation="t",
            component=Component.TOOL,
            status=Status.ERROR,
            attributes={"tool": "request_hint"},
        ),
    ]
    metrics = compute_metrics(events)
    assert metrics.tool_calls == 3
    assert metrics.tool_failures == 1
    assert metrics.tool_usage == {"submit_guess": 2, "request_hint": 1}


def test_agent_metrics_average_iterations():
    events = [
        Event(operation="a", component=Component.AGENT, attributes={"iterations": 3}),
        Event(operation="a", component=Component.AGENT, attributes={"iterations": 1}),
    ]
    metrics = compute_metrics(events)
    assert metrics.agent_runs == 2
    assert metrics.avg_agent_iterations == 2.0


def test_per_component_breakdown():
    events = [
        Event(operation="a", component=Component.LLM, duration_ms=10),
        Event(operation="b", component=Component.TOOL, duration_ms=20, status=Status.ERROR),
    ]
    by_name = {c.component: c for c in compute_metrics(events).by_component}
    assert by_name["llm"].success_rate == 1.0
    assert by_name["tool"].failure_rate == 1.0


def test_metrics_serialise_to_json():
    metrics = compute_metrics([Event(operation="a", component=Component.LLM)])
    json.dumps(metrics.to_dict())


# --- Pricing -----------------------------------------------------------------


def test_known_model_is_priced():
    cost = estimate_cost(
        "claude-opus-5-5", TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000), "anthropic"
    )
    assert cost.priced
    assert cost.usd == pytest.approx(24.0)  # 4 + 20 per MTok


def test_unknown_model_is_not_priced_and_says_why():
    cost = estimate_cost("some-future-model", TokenUsage(100, 100), "anthropic")
    assert not cost.priced
    assert cost.usd is None
    assert "no configured price" in cost.reason


def test_missing_usage_is_never_estimated():
    cost = estimate_cost("claude-opus-5-5", TokenUsage(None, None), "anthropic")
    assert cost.usd is None
    assert "did not report token usage" in cost.reason


def test_mock_provider_is_never_costed():
    cost = estimate_cost("claude-opus-5-5", TokenUsage(100, 100), "mock")
    assert cost.usd is None
    assert "mock provider" in cost.reason


def test_price_override_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_MTOK", "1.0")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_MTOK", "2.0")
    cost = estimate_cost("unlisted", TokenUsage(1_000_000, 1_000_000), "anthropic")
    assert cost.priced
    assert cost.usd == pytest.approx(3.0)


def test_malformed_price_override_is_ignored(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_MTOK", "free")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_MTOK", "2.0")
    assert get_pricing("claude-opus-5-5") is not None  # falls back to the table


# --- Tracker: context and recording ------------------------------------------


def test_request_id_propagates_within_a_request(tracker):
    with tracker.request("demo") as request_id:
        assert current_request_id() == request_id
        with tracker.operation("step1", Component.GAME):
            pass
        with tracker.operation("step2", Component.LLM):
            pass

    recorded = tracker.store.all()
    assert {e.request_id for e in recorded} == {request_id}


def test_request_id_is_cleared_after_the_block(tracker):
    with tracker.request("demo"):
        pass
    assert current_request_id() is None


def test_operation_records_duration_and_success(tracker):
    with tracker.operation("work", Component.GAME):
        pass
    event = tracker.store.all()[0]
    assert event.status is Status.OK
    assert event.duration_ms >= 0


def test_operation_records_failures_and_reraises(tracker):
    with pytest.raises(ValueError, match="boom"):
        with tracker.operation("work", Component.GAME):
            raise ValueError("boom")

    event = tracker.store.all()[0]
    assert event.status is Status.ERROR
    assert event.error_type == "ValueError"
    assert "boom" in event.error_message


def test_record_llm_call_captures_stats_and_cost(tracker):
    stats = CallStats(
        attempts=2,
        retries=1,
        timeouts=1,
        provider="anthropic",
        model="claude-opus-5-5",
        usage=TokenUsage(1000, 500),
        latency_ms=123.0,
    )
    event = tracker.record_llm_call("test.call", stats)
    assert event.attributes["retries"] == 1
    assert event.cost.priced
    assert event.cost.usd > 0


def test_record_assistant_reply_marks_fallback_as_degraded(tracker):
    assistant = Assistant(
        ReliableLLMClient(ScriptedProvider([LLMTimeoutError("down")] * 3), max_retries=1, sleep=no_sleep)
    )
    reply = assistant.hint(game(), last_outcome="Too High")
    tracker.record_assistant_reply(reply, "what next?")

    assistant_events = tracker.store.by_component(Component.ASSISTANT)
    assert assistant_events[0].status is Status.DEGRADED
    assert assistant_events[0].attributes["source"] == "fallback"


def test_record_assistant_reply_does_not_store_the_question_text(tracker):
    assistant = Assistant(ReliableLLMClient(MockProvider(), sleep=no_sleep))
    reply = assistant.hint(game(), last_outcome="Too High")
    tracker.record_assistant_reply(reply, "my very private question")

    serialised = json.dumps([e.to_dict() for e in tracker.store.all()])
    assert "my very private question" not in serialised
    assert '"chars": 24' in serialised  # only the length is kept


def test_record_agent_run_links_tools_to_a_trace(tracker):
    client = ReliableLLMClient(MockProvider(), sleep=no_sleep)
    agent = build_agent(assistant=Assistant(client), client=client)

    with tracker.request("agent.request"):
        run = agent.run("start a hard game and explain the rules", game())
        tracker.record_agent_run(run)

    trace_id = run.trace.trace_id
    linked = tracker.store.by_trace(trace_id)
    assert len(linked) >= 3

    tool_events = [e for e in linked if e.component is Component.TOOL]
    assert {e.attributes["tool"] for e in tool_events} == {"start_game", "explain_rules"}

    agent_events = [e for e in linked if e.component is Component.AGENT]
    assert agent_events[0].attributes["iterations"] == 3


def test_agent_events_share_one_request_id(tracker):
    client = ReliableLLMClient(MockProvider(), sleep=no_sleep)
    agent = build_agent(assistant=Assistant(client), client=client)
    with tracker.request("agent.request") as request_id:
        tracker.record_agent_run(agent.run("what is my status", game()))

    assert {e.request_id for e in tracker.store.all()} == {request_id}


def test_agent_trace_never_records_the_secret(tracker):
    client = ReliableLLMClient(MockProvider(), sleep=no_sleep)
    agent = build_agent(assistant=Assistant(client), client=client)
    with tracker.request("r"):
        tracker.record_agent_run(agent.run("what is my status", game(secret=77)))

    # Scoped to attributes and error text: timestamps and random ids legitimately
    # contain the digits "77", so a whole-payload substring search false-positives.
    for event in tracker.store.all():
        payload = json.dumps({"attrs": event.attributes, "error": event.error_message})
        assert "77" not in payload, f"{event.operation} recorded the secret: {payload}"


def test_record_parse_marks_rule_fallback_as_degraded(tracker):
    from natural_language import CommandParser

    parser = CommandParser(
        ReliableLLMClient(ScriptedProvider([LLMTimeoutError("x")] * 3), max_retries=1, sleep=no_sleep)
    )
    tracker.record_parse(parser.parse("guess 42"), "guess 42")
    events = tracker.store.by_component(Component.PARSER)
    assert events[0].status is Status.DEGRADED
    assert events[0].attributes["command"] == "SUBMIT_GUESS"


def test_tracker_metrics_reflect_recorded_events(tracker):
    with tracker.request("r"):
        with tracker.operation("a", Component.GAME):
            pass
        with pytest.raises(RuntimeError):
            with tracker.operation("b", Component.TOOL, {"tool": "submit_guess"}):
                raise RuntimeError("fail")

    metrics = tracker.metrics()
    assert metrics.total_events == 2
    assert metrics.error_events == 1
    assert metrics.tool_failures == 1


def test_tracker_clear_empties_the_store(tracker):
    with tracker.operation("a", Component.GAME):
        pass
    tracker.clear()
    assert tracker.metrics().total_events == 0


def test_global_tracker_is_a_singleton():
    set_tracker(None)
    assert get_tracker() is get_tracker()
    set_tracker(None)
