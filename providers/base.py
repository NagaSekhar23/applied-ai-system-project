"""Provider-neutral LLM interface.

Application code depends on this module, never on a vendor SDK. Swapping
providers means adding a class here, not editing the assistant, the natural
language layer or the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMError(Exception):
    """Base class for provider failures."""


class LLMTimeoutError(LLMError):
    """The provider did not respond within the configured timeout."""


class LLMRateLimitError(LLMError):
    """The provider rejected the request for rate limiting (HTTP 429)."""

    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMAuthError(LLMError):
    """Credentials are missing, invalid, or lack permission. Not retryable."""


class LLMBadRequestError(LLMError):
    """The request itself is invalid. Not retryable."""


class LLMServerError(LLMError):
    """The provider failed server-side (HTTP 5xx). Retryable."""


class LLMConnectionError(LLMError):
    """The provider could not be reached. Retryable."""


@dataclass(frozen=True)
class LLMRequest:
    """One provider-neutral completion request.

    Attributes:
        system: System instructions.
        user: The user-turn content.
        json_schema: When set, the provider must return JSON matching it.
        max_tokens: Output cap.
        effort: Reasoning effort hint; providers that lack one ignore it.
        purpose: Short label for logging and metrics (e.g. "hint"). Never
            contains user content.
    """

    system: str
    user: str
    json_schema: dict[str, Any] | None = None
    max_tokens: int = 1024
    effort: str = "low"
    purpose: str = "general"


@dataclass(frozen=True)
class TokenUsage:
    """Token counts as reported by the provider.

    ``None`` means the provider did not report that number. It is never
    guessed or estimated: a fabricated token count would corrupt cost
    reporting downstream.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
        }


@dataclass(frozen=True)
class LLMResponse:
    """One provider-neutral completion response."""

    text: str
    provider: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    stop_reason: str | None = None
    request_id: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """What every provider must implement.

    Deliberately small: one synchronous call. Streaming, tool use and batching
    are not part of this project's needs, and an interface with unused methods
    is harder to implement for a new provider.
    """

    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion.

        Raises:
            LLMError: or a subclass, for any provider failure.
        """
        ...
