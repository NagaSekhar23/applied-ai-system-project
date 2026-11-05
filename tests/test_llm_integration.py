"""Phase 3 tests: config, providers, reliability, prompts and the assistant.

No test here requires an API key or network access. The real provider is
exercised through an injected fake client that mimics the SDK's surface.
"""

import json
import random
from dataclasses import replace

import pytest
from pydantic import ValidationError

from assistant import (
    Assistant,
    build_facts,
    deterministic_hint,
    leaks_secret,
)
from config import ConfigError, Settings, load_settings
from game import new_game
from prompts import get_prompt, render_facts
from providers import (
    LLMAuthError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMRequest,
    LLMTimeoutError,
    MockProvider,
    ReliableLLMClient,
    ScriptedProvider,
    StructuredOutputError,
    build_client,
    build_provider,
    extract_json,
)
from providers.mock import extract_facts
from schemas import AssistantHint, HintCategory

LLM_ENV = [
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_MAX_TOKENS",
    "LLM_EFFORT",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests must not inherit the developer's real configuration."""
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)


def game(secret: int = 50, difficulty: str = "Normal"):
    return replace(new_game(difficulty, rng=random.Random(0)), secret=secret)


def valid_hint_json(**overrides) -> str:
    payload = {
        "message": "That guess was above the number. Try lower.",
        "hint_category": "too_high",
        "difficulty_level": "Normal",
        "reveals_secret": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def no_sleep(_seconds):
    """Retry backoff must not slow the test suite."""


# --- Configuration -----------------------------------------------------------


def test_defaults_to_mock_provider_without_credentials():
    settings = load_settings()
    assert settings.provider == "mock"
    assert settings.is_mock
    assert settings.live_llm_available is False


def test_reads_provider_and_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    settings = load_settings()
    assert settings.provider == "anthropic"
    assert settings.model == "claude-haiku-5-5"
    assert settings.live_llm_available is True


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "wishful")
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        load_settings()


def test_invalid_effort_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_EFFORT", "turbo")
    with pytest.raises(ConfigError, match="LLM_EFFORT"):
        load_settings()


@pytest.mark.parametrize("name", ["LLM_MAX_RETRIES", "LLM_MAX_TOKENS"])
def test_non_numeric_ints_are_rejected(monkeypatch, name):
    monkeypatch.setenv(name, "lots")
    with pytest.raises(ConfigError, match=name):
        load_settings()


def test_negative_retries_rejected(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "-1")
    with pytest.raises(ConfigError):
        load_settings()


def test_blank_env_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "   ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    settings = load_settings()
    assert settings.model == "claude-opus-5-5"
    assert settings.api_key is None


def test_redacted_settings_never_include_the_api_key():
    settings = Settings(provider="anthropic", api_key="sk-ant-super-secret")
    redacted = settings.redacted()
    assert "sk-ant-super-secret" not in json.dumps(redacted)
    assert redacted["api_key_configured"] is True
    assert redacted["mode"] == "live"


# --- Mock provider -----------------------------------------------------------


def test_mock_provider_returns_schema_valid_hint():
    provider = MockProvider()
    facts = build_facts(game(), last_outcome="Too High")
    prompt = get_prompt("hint")
    response = provider.complete(
        LLMRequest(
            system=prompt.system,
            user=prompt.render_user(facts),
            json_schema=AssistantHint.json_schema(),
            purpose="hint",
        )
    )
    hint = AssistantHint.model_validate(json.loads(response.text))
    assert hint.hint_category is HintCategory.TOO_HIGH
    assert response.usage.input_tokens is not None


def test_mock_provider_reads_the_facts_block():
    facts = {"difficulty": "Hard", "low": 1, "high": 50, "last_outcome": "Too Low"}
    assert extract_facts(render_facts(facts)) == facts


def test_extract_facts_tolerates_a_missing_block():
    assert extract_facts("no facts here") == {}


def test_extract_facts_tolerates_malformed_json():
    assert extract_facts("<facts>\n{broken\n</facts>") == {}


def test_mock_provider_records_calls():
    provider = MockProvider()
    provider.complete(LLMRequest(system="s", user="u", purpose="hint"))
    assert len(provider.calls) == 1


# --- JSON extraction ---------------------------------------------------------


def test_extract_json_parses_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_tolerates_fenced_and_surrounding_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nHope that helps') == {"a": 1}


@pytest.mark.parametrize("text", ["", "   ", "no json at all", "[1, 2, 3]", '"a string"'])
def test_extract_json_rejects_unusable_output(text):
    with pytest.raises(StructuredOutputError):
        extract_json(text)


# --- Reliability: retries, timeouts, failures --------------------------------


def test_successful_call_does_not_retry():
    provider = ScriptedProvider([valid_hint_json()])
    client = ReliableLLMClient(provider, max_retries=2, sleep=no_sleep)
    result = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint)
    assert result.ok
    assert result.stats.attempts == 1
    assert result.stats.retries == 0


def test_retries_after_a_timeout_then_succeeds():
    provider = ScriptedProvider([LLMTimeoutError("timed out"), valid_hint_json()])
    client = ReliableLLMClient(provider, max_retries=2, sleep=no_sleep)
    result = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint)
    assert result.ok
    assert result.stats.attempts == 2
    assert result.stats.retries == 1
    assert result.stats.timeouts == 1


def test_retries_are_bounded_and_exhaust():
    provider = ScriptedProvider([LLMTimeoutError("t")] * 5)
    client = ReliableLLMClient(provider, max_retries=2, sleep=no_sleep)
    result = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint)
    assert not result.ok
    assert result.stats.attempts == 3  # the first try plus two retries
    assert provider.call_count == 3


def test_rate_limit_is_retried_and_respects_retry_after():
    delays = []
    provider = ScriptedProvider([LLMRateLimitError("slow down", 7.0), valid_hint_json()])
    client = ReliableLLMClient(provider, max_retries=2, sleep=delays.append)
    result = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint)
    assert result.ok
    assert delays == [7.0]


def test_backoff_grows_exponentially():
    delays = []
    provider = ScriptedProvider([LLMTimeoutError("t"), LLMTimeoutError("t"), valid_hint_json()])
    client = ReliableLLMClient(provider, max_retries=3, backoff_seconds=0.5, sleep=delays.append)
    assert client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint).ok
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize(
    "error", [LLMAuthError("bad key"), LLMBadRequestError("bad request")]
)
def test_non_retryable_errors_fail_immediately(error):
    provider = ScriptedProvider([error, valid_hint_json()])
    client = ReliableLLMClient(provider, max_retries=3, sleep=no_sleep)
    result = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint)
    assert not result.ok
    assert provider.call_count == 1


def test_malformed_json_is_retried_then_accepted():
    provider = ScriptedProvider(["not json at all", valid_hint_json()])
    client = ReliableLLMClient(provider, max_retries=2, sleep=no_sleep)
    result = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint)
    assert result.ok
    assert result.stats.validation_failures == 1


def test_schema_violation_is_counted_as_a_validation_failure():
    bad = json.dumps({"message": "hi", "hint_category": "not_a_category"})
    provider = ScriptedProvider([bad, bad, bad])
    client = ReliableLLMClient(provider, max_retries=2, sleep=no_sleep)
    result = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint)
    assert not result.ok
    assert result.stats.validation_failures == 3


def test_errors_are_recorded_never_swallowed():
    provider = ScriptedProvider([LLMTimeoutError("boom")] * 3)
    client = ReliableLLMClient(provider, max_retries=2, sleep=no_sleep)
    result = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint)
    assert len(result.stats.errors) == 3
    assert "boom" in result.stats.errors[0]


def test_stats_round_trip_to_dict():
    provider = ScriptedProvider([valid_hint_json()])
    client = ReliableLLMClient(provider, sleep=no_sleep)
    stats = client.complete_structured(LLMRequest(system="s", user="u"), AssistantHint).stats
    payload = stats.to_dict()
    assert payload["attempts"] == 1
    assert payload["usage"]["input_tokens"] == 10


# --- Schemas -----------------------------------------------------------------


def test_hint_schema_rejects_blank_message():
    with pytest.raises(ValidationError):
        AssistantHint(message="   ", hint_category="too_high")


def test_hint_schema_rejects_unknown_category():
    with pytest.raises(ValidationError):
        AssistantHint(message="ok", hint_category="sideways")


def test_hint_schema_ignores_extra_fields():
    hint = AssistantHint.model_validate(
        {"message": "ok", "hint_category": "win", "surprise": 1}
    )
    assert hint.hint_category is HintCategory.WIN


def test_hint_schema_is_valid_json_schema():
    schema = AssistantHint.json_schema()
    assert schema["additionalProperties"] is False
    assert "message" in schema["required"]


# --- Prompts -----------------------------------------------------------------


def test_prompts_are_versioned():
    assert get_prompt("hint").id == "assistant_hint@1.0.0"


def test_unknown_prompt_lists_available_names():
    with pytest.raises(KeyError, match="hint"):
        get_prompt("nonexistent")


def test_rendered_facts_are_byte_stable():
    facts = {"b": 2, "a": 1}
    assert render_facts(facts) == render_facts({"a": 1, "b": 2})


def test_prompt_states_its_restrictions():
    system = get_prompt("hint").system
    assert "never" in system.lower()
    assert "secret" in system.lower()


# --- The deterministic boundary ----------------------------------------------


def test_facts_omit_the_secret_while_playing():
    facts = build_facts(game(secret=73), last_outcome="Too Low")
    assert "secret" not in facts
    assert 73 not in facts.values()


def test_facts_include_the_secret_once_the_game_is_over():
    finished = replace(game(secret=73), status="won")
    assert build_facts(finished)["secret"] == 73


def test_rendered_prompt_never_contains_the_secret_mid_game():
    state = game(secret=42)
    rendered = get_prompt("hint").render_user(build_facts(state, "Too High"))
    assert not leaks_secret(rendered, state)


def test_leak_detector_matches_whole_numbers_only():
    state = game(secret=50)
    assert leaks_secret("the answer is 50", state) is True
    assert leaks_secret("try 150 next", state) is False
    assert leaks_secret("try 500 next", state) is False


def test_leak_detector_is_inactive_once_the_game_is_over():
    finished = replace(game(secret=50), status="won")
    assert leaks_secret("the number was 50", finished) is False


@pytest.mark.parametrize(
    "outcome, expected",
    [
        ("Too High", HintCategory.TOO_HIGH),
        ("Too Low", HintCategory.TOO_LOW),
        ("Win", HintCategory.WIN),
        (None, HintCategory.OPENING),
    ],
)
def test_deterministic_hint_covers_every_outcome(outcome, expected):
    message, category = deterministic_hint(game(), outcome)
    assert category is expected
    assert message


# --- Assistant behavior ------------------------------------------------------


def assistant_with(script) -> tuple[Assistant, ScriptedProvider]:
    provider = ScriptedProvider(script)
    return Assistant(ReliableLLMClient(provider, max_retries=1, sleep=no_sleep)), provider


def test_assistant_returns_model_output_on_success():
    assistant, _ = assistant_with([valid_hint_json(message="Aim lower, around 25.")])
    reply = assistant.hint(game(), last_outcome="Too High")
    assert reply.source == "llm"
    assert reply.message == "Aim lower, around 25."
    assert reply.category is HintCategory.TOO_HIGH


def test_assistant_falls_back_when_the_provider_fails():
    assistant, _ = assistant_with([LLMTimeoutError("down")] * 3)
    reply = assistant.hint(game(), last_outcome="Too Low")
    assert reply.source == "fallback"
    assert "too low" in reply.message.lower()
    assert "provider or validation failure" in reply.fallback_reason


def test_assistant_falls_back_on_unusable_output():
    assistant, _ = assistant_with(["garbage", "still garbage"])
    reply = assistant.hint(game(), last_outcome="Too High")
    assert reply.source == "fallback"


def test_assistant_falls_back_when_the_model_leaks_the_secret():
    leak = valid_hint_json(message="Just guess 50, that is the number.")
    assistant, _ = assistant_with([leak, leak])
    reply = assistant.hint(game(secret=50), last_outcome="Too High")
    assert reply.source == "fallback"
    assert reply.fallback_reason == "model output disclosed the secret"


def test_deterministic_outcome_overrides_a_disagreeing_model():
    # The model claims "too_low"; the game says Too High. The game wins.
    assistant, _ = assistant_with([valid_hint_json(hint_category="too_low")])
    reply = assistant.hint(game(), last_outcome="Too High")
    assert reply.category is HintCategory.TOO_HIGH
    assert reply.warnings
    assert "deterministic outcome" in reply.warnings[0]


def test_assistant_never_claims_a_win_the_game_did_not_award():
    assistant, _ = assistant_with([valid_hint_json(hint_category="win")])
    reply = assistant.hint(game(), last_outcome="Too Low")
    assert reply.category is HintCategory.TOO_LOW


def test_assistant_reports_which_prompt_version_ran():
    assistant, _ = assistant_with([valid_hint_json()])
    assert assistant.hint(game(), "Too High").prompt_id == "assistant_hint@1.0.0"


def test_assistant_explains_rules():
    rules = valid_hint_json(hint_category="rules", message="Guess between 1 and 100.")
    assistant, provider = assistant_with([rules])
    reply = assistant.explain_rules(game())
    assert reply.source == "llm"
    assert provider.calls[0].purpose == "explain_rules"


def test_assistant_works_end_to_end_with_the_mock_provider():
    assistant = Assistant(ReliableLLMClient(MockProvider(), sleep=no_sleep))
    reply = assistant.hint(game(), last_outcome="Too High")
    assert reply.source == "llm"
    assert reply.category is HintCategory.TOO_HIGH
    assert not leaks_secret(reply.message, game())


def test_assistant_stats_are_available_for_observability():
    assistant, _ = assistant_with([valid_hint_json()])
    stats = assistant.hint(game(), "Too High").stats
    assert stats.attempts == 1
    assert stats.provider == "scripted"


# --- Factory -----------------------------------------------------------------


def test_factory_builds_a_mock_provider_by_default():
    assert build_provider(load_settings()).name == "mock"


def test_factory_builds_a_retry_wrapped_client():
    client = build_client(Settings(provider="mock", max_retries=3))
    assert client.max_retries == 3
    assert client.provider.name == "mock"


def test_anthropic_provider_requires_a_key():
    pytest.importorskip("anthropic")
    from providers.anthropic_provider import AnthropicProvider

    with pytest.raises(LLMAuthError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(model="claude-opus-5-5", api_key=None)


# --- Anthropic provider, exercised with an injected fake client --------------
# These cover request construction, response parsing and error translation
# without the SDK installed and without any network call.


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeUsage:
    def __init__(self, input_tokens=12, output_tokens=7):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeMessage:
    def __init__(self, text="{}", stop_reason="end_turn", usage=None, model="claude-opus-5-5"):
        self.content = [FakeBlock(text)]
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = usage if usage is not None else FakeUsage()
        self.model = model
        self._request_id = "req_fake_123"


class FakeMessages:
    def __init__(self, result):
        self._result = result
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)


def anthropic_provider_with(result):
    from providers.anthropic_provider import AnthropicProvider

    client = FakeClient(result)
    provider = AnthropicProvider(model="claude-opus-5-5", client=client)
    return provider, client


def test_anthropic_provider_parses_a_successful_response():
    provider, _ = anthropic_provider_with(FakeMessage(text=valid_hint_json()))
    response = provider.complete(LLMRequest(system="s", user="u", purpose="hint"))
    assert json.loads(response.text)["hint_category"] == "too_high"
    assert response.usage.input_tokens == 12
    assert response.usage.total == 19
    assert response.request_id == "req_fake_123"
    assert response.latency_ms >= 0


def test_anthropic_provider_sends_schema_and_effort():
    provider, client = anthropic_provider_with(FakeMessage())
    provider.complete(
        LLMRequest(
            system="sys",
            user="usr",
            json_schema=AssistantHint.json_schema(),
            max_tokens=256,
            effort="low",
        )
    )
    kwargs = client.messages.last_kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] == 256
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "usr"}]
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_provider_omits_format_when_no_schema():
    provider, client = anthropic_provider_with(FakeMessage())
    provider.complete(LLMRequest(system="s", user="u"))
    assert "format" not in client.messages.last_kwargs["output_config"]


def test_anthropic_provider_treats_a_refusal_as_a_bad_request():
    provider, _ = anthropic_provider_with(FakeMessage(stop_reason="refusal"))
    with pytest.raises(LLMBadRequestError, match="declined"):
        provider.complete(LLMRequest(system="s", user="u"))


def test_anthropic_provider_handles_missing_usage_without_inventing_numbers():
    message = FakeMessage()
    message.usage = None
    provider, _ = anthropic_provider_with(message)
    response = provider.complete(LLMRequest(system="s", user="u"))
    assert response.usage.input_tokens is None
    assert response.usage.total is None


@pytest.mark.parametrize(
    "exc_name, expected",
    [
        ("APITimeoutError", LLMTimeoutError),
        ("RateLimitError", LLMRateLimitError),
        ("AuthenticationError", LLMAuthError),
        ("BadRequestError", LLMBadRequestError),
        ("APIConnectionError", type(None)),  # replaced below
    ],
)
def test_anthropic_provider_translates_vendor_errors(exc_name, expected):
    from providers.base import LLMConnectionError

    if exc_name == "APIConnectionError":
        expected = LLMConnectionError

    vendor_exc = type(exc_name, (Exception,), {})("vendor failure")
    provider, _ = anthropic_provider_with(vendor_exc)
    with pytest.raises(expected):
        provider.complete(LLMRequest(system="s", user="u"))


def test_anthropic_provider_wraps_unknown_errors():
    from providers.base import LLMError

    provider, _ = anthropic_provider_with(RuntimeError("something odd"))
    with pytest.raises(LLMError, match="something odd"):
        provider.complete(LLMRequest(system="s", user="u"))
