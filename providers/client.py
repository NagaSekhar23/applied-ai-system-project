"""Reliable client wrapper: bounded retries and structured-output validation.

Separating this from the providers means retry and validation behavior is
written once and tested once, against a scripted provider, rather than being
reimplemented per vendor.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from providers.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

T = TypeVar("T", bound=BaseModel)

# Errors worth retrying: transient, and a retry can plausibly succeed.
RETRYABLE = (LLMRateLimitError,)
# Errors never worth retrying: the same request will fail the same way.
NON_RETRYABLE = (LLMAuthError, LLMBadRequestError)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class StructuredOutputError(LLMError):
    """The provider replied, but not with output matching the schema."""


@dataclass
class CallStats:
    """What happened during one logical call, including its retries."""

    attempts: int = 0
    retries: int = 0
    timeouts: int = 0
    validation_failures: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "retries": self.retries,
            "timeouts": self.timeouts,
            "validation_failures": self.validation_failures,
            "latency_ms": round(self.latency_ms, 2),
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "errors": list(self.errors),
        }


@dataclass
class StructuredResult:
    """Outcome of a structured call. ``value`` is None when it failed."""

    value: BaseModel | None
    stats: CallStats
    raw_text: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output.

    Tolerates a fenced code block or surrounding prose, because a model that
    was asked for JSON does not always return *only* JSON. Never evaluates the
    text — only ``json.loads``.

    Raises:
        StructuredOutputError: if no JSON object can be parsed.
    """
    stripped = text.strip()
    if not stripped:
        raise StructuredOutputError("Provider returned an empty response")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(stripped)
        if not match:
            raise StructuredOutputError("No JSON object found in the response") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(f"Response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise StructuredOutputError(
            f"Expected a JSON object, got {type(parsed).__name__}"
        )
    return parsed


class ReliableLLMClient:
    """Wraps a provider with bounded retries and schema validation.

    Args:
        provider: Any object satisfying ``LLMProvider``.
        max_retries: Extra attempts after the first. 0 disables retrying.
        backoff_seconds: Base delay; attempt *n* waits ``backoff * 2**n``.
        sleep: Injected for tests, so retry logic is tested without real delay.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        sleep=time.sleep,
    ):
        self.provider = provider
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep

    def complete(self, request: LLMRequest) -> tuple[LLMResponse | None, CallStats]:
        """Call the provider with bounded retries.

        Returns ``(response, stats)``; ``response`` is None if every attempt
        failed. Errors are recorded in ``stats.errors`` — never swallowed
        silently.
        """
        stats = CallStats(provider=self.provider.name, model=self.provider.model)
        started = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            stats.attempts += 1
            try:
                response = self.provider.complete(request)
            except NON_RETRYABLE as exc:
                stats.errors.append(f"{type(exc).__name__}: {exc}")
                break
            except Exception as exc:
                stats.errors.append(f"{type(exc).__name__}: {exc}")
                if type(exc).__name__ == "LLMTimeoutError":
                    stats.timeouts += 1
                if attempt < self.max_retries:
                    stats.retries += 1
                    self._sleep(self._delay_for(attempt, exc))
                    continue
                break
            else:
                stats.usage = response.usage
                stats.model = response.model
                stats.latency_ms = (time.perf_counter() - started) * 1000
                return response, stats

        stats.latency_ms = (time.perf_counter() - started) * 1000
        return None, stats

    def complete_structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResult:
        """Call the provider and validate the reply against a Pydantic model.

        A reply that parses but fails validation is retried like a transient
        error, because models do occasionally emit a near-miss. Retries are
        bounded by the same ``max_retries``.
        """
        stats = CallStats(provider=self.provider.name, model=self.provider.model)
        started = time.perf_counter()
        last_text: str | None = None

        for attempt in range(self.max_retries + 1):
            stats.attempts += 1
            try:
                response = self.provider.complete(request)
            except NON_RETRYABLE as exc:
                stats.errors.append(f"{type(exc).__name__}: {exc}")
                break
            except Exception as exc:
                stats.errors.append(f"{type(exc).__name__}: {exc}")
                if type(exc).__name__ == "LLMTimeoutError":
                    stats.timeouts += 1
                if attempt < self.max_retries:
                    stats.retries += 1
                    self._sleep(self._delay_for(attempt, exc))
                    continue
                break

            stats.usage = response.usage
            stats.model = response.model
            last_text = response.text

            try:
                payload = extract_json(response.text)
                value = schema.model_validate(payload)
            except (StructuredOutputError, ValidationError) as exc:
                stats.validation_failures += 1
                stats.errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < self.max_retries:
                    stats.retries += 1
                    self._sleep(self._delay_for(attempt, exc))
                    continue
                break
            else:
                stats.latency_ms = (time.perf_counter() - started) * 1000
                return StructuredResult(value=value, stats=stats, raw_text=response.text)

        stats.latency_ms = (time.perf_counter() - started) * 1000
        return StructuredResult(value=None, stats=stats, raw_text=last_text)

    def _delay_for(self, attempt: int, exc: Exception) -> float:
        """Exponential backoff, respecting a provider's retry-after hint."""
        if isinstance(exc, LLMRateLimitError) and exc.retry_after_seconds:
            return exc.retry_after_seconds
        return self.backoff_seconds * (2**attempt)
