"""LLM provider abstraction and factory.

Application code imports from here, never from a vendor SDK.
"""

from __future__ import annotations

from config import Settings, load_settings
from providers.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMServerError,
    LLMTimeoutError,
    TokenUsage,
)
from providers.client import (
    CallStats,
    ReliableLLMClient,
    StructuredOutputError,
    StructuredResult,
    extract_json,
)
from providers.mock import MockProvider, ScriptedProvider

__all__ = [
    "CallStats",
    "LLMAuthError",
    "LLMBadRequestError",
    "LLMConnectionError",
    "LLMError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMServerError",
    "LLMTimeoutError",
    "MockProvider",
    "ReliableLLMClient",
    "ScriptedProvider",
    "StructuredOutputError",
    "StructuredResult",
    "TokenUsage",
    "build_client",
    "build_provider",
    "extract_json",
]


def build_provider(settings: Settings | None = None) -> LLMProvider:
    """Build the provider named by settings.

    Defaults to the mock provider, so the application starts with no
    credentials configured.
    """
    settings = settings or load_settings()

    if settings.is_mock:
        return MockProvider(model=settings.model)

    if settings.provider == "anthropic":
        from providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=settings.model,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
            max_retries=0,  # retries are owned by ReliableLLMClient, not the SDK
        )

    raise LLMError(f"Unknown provider: {settings.provider!r}")


def build_client(settings: Settings | None = None) -> ReliableLLMClient:
    """Build a retry-wrapped client for the configured provider."""
    settings = settings or load_settings()
    return ReliableLLMClient(
        provider=build_provider(settings),
        max_retries=settings.max_retries,
    )
