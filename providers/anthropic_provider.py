"""Anthropic provider.

The ``anthropic`` SDK is imported lazily inside the constructor so the rest of
the application — and the whole test suite — runs without the package
installed. Nothing else in the codebase imports the SDK.

Vendor exceptions are translated into this project's own error types, so
retry logic and the UI never branch on a vendor class.
"""

from __future__ import annotations

import time
from typing import Any

from providers.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMServerError,
    LLMTimeoutError,
    TokenUsage,
)


class AnthropicProvider:
    """Calls the Claude API through the official Anthropic SDK."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        client: Any | None = None,
    ):
        self.name = "anthropic"
        self.model = model
        self._timeout = timeout_seconds

        if client is not None:
            self._client = client
            self._sdk = None
            return

        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "The 'anthropic' package is required for the anthropic provider. "
                "Install it with: pip install anthropic"
            ) from exc

        if not api_key:
            raise LLMAuthError(
                "ANTHROPIC_API_KEY is not set. Set it in the environment, or use "
                "LLM_PROVIDER=mock to run without credentials."
            )

        self._sdk = anthropic
        # The SDK retries connection errors, 408/409/429 and 5xx itself.
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "output_config": {"effort": request.effort},
        }
        if request.json_schema is not None:
            kwargs["output_config"]["format"] = {
                "type": "json_schema",
                "schema": request.json_schema,
            }

        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise self._translate(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMBadRequestError(
                f"The model declined this request (category: {getattr(details, 'category', None)})."
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=getattr(response, "model", self.model),
            usage=TokenUsage(
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
            ),
            latency_ms=latency_ms,
            stop_reason=getattr(response, "stop_reason", None),
            request_id=getattr(response, "_request_id", None),
        )

    def _translate(self, exc: Exception) -> LLMError:
        """Map a vendor exception onto this project's error types.

        Most specific first. When the SDK is absent (injected test client) we
        fall back to matching on class name so the mapping is still testable.
        """
        anthropic = self._sdk
        if anthropic is not None:
            if isinstance(exc, anthropic.APITimeoutError):
                return LLMTimeoutError(f"Request timed out after {self._timeout}s")
            if isinstance(exc, anthropic.RateLimitError):
                retry_after = None
                response = getattr(exc, "response", None)
                if response is not None:
                    raw = response.headers.get("retry-after")
                    if raw:
                        try:
                            retry_after = float(raw)
                        except ValueError:
                            retry_after = None
                return LLMRateLimitError("Rate limited by the provider", retry_after)
            if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
                return LLMAuthError("Provider rejected the credentials")
            if isinstance(exc, anthropic.BadRequestError):
                return LLMBadRequestError(f"Invalid request: {exc}")
            if isinstance(exc, anthropic.NotFoundError):
                return LLMBadRequestError(f"Unknown model or endpoint: {self.model}")
            if isinstance(exc, anthropic.APIStatusError):
                status = getattr(exc, "status_code", 0)
                if status >= 500:
                    return LLMServerError(f"Provider server error ({status})")
                return LLMBadRequestError(f"Provider error ({status})")
            if isinstance(exc, anthropic.APIConnectionError):
                return LLMConnectionError("Could not reach the provider")

        name = type(exc).__name__
        mapping = {
            "APITimeoutError": LLMTimeoutError,
            "RateLimitError": LLMRateLimitError,
            "AuthenticationError": LLMAuthError,
            "PermissionDeniedError": LLMAuthError,
            "BadRequestError": LLMBadRequestError,
            "NotFoundError": LLMBadRequestError,
            "APIConnectionError": LLMConnectionError,
        }
        if name in mapping:
            return mapping[name](str(exc) or name)
        return LLMError(f"{name}: {exc}")
