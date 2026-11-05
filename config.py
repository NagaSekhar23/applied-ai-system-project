"""Application configuration, read from environment variables.

Secrets are never hardcoded and never written to a committed file. See
``.env.example`` for the variables this reads; copy it to ``.env`` and fill in
real values locally (``.env`` is git-ignored).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# The default model for this project. Overridable with LLM_MODEL.
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_PROVIDER = "mock"

KNOWN_PROVIDERS = ("mock", "anthropic")


class ConfigError(Exception):
    """Raised when configuration is present but invalid."""


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


@dataclass(frozen=True)
class Settings:
    """Resolved application settings.

    ``api_key`` is held here but must never be logged, rendered in the UI, or
    written to a report. ``redacted()`` produces the safe-to-display form.
    """

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    timeout_seconds: float = 20.0
    max_retries: int = 2
    max_tokens: int = 1024
    effort: str = "low"
    game_debug: bool = False
    log_level: str = "INFO"

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key)

    @property
    def live_llm_available(self) -> bool:
        """True when a real provider is both selected and credentialed."""
        return not self.is_mock and self.has_credentials

    def redacted(self) -> dict[str, object]:
        """Settings safe to display or log. The API key is never included."""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_configured": self.has_credentials,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "mode": "mock" if self.is_mock else "live",
        }


def load_settings() -> Settings:
    """Build settings from the environment.

    Defaults to the mock provider so the application, its tests and its
    evaluations run with no API key and no network access.

    Raises:
        ConfigError: if a value is present but unusable.
    """
    provider = _env_str("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    if provider not in KNOWN_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER must be one of {KNOWN_PROVIDERS}, got {provider!r}"
        )

    effort = _env_str("LLM_EFFORT", "low").lower()
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        raise ConfigError(f"LLM_EFFORT must be a valid effort level, got {effort!r}")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    return Settings(
        provider=provider,
        model=_env_str("LLM_MODEL", DEFAULT_MODEL),
        api_key=api_key.strip() if api_key and api_key.strip() else None,
        timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 20.0, minimum=0.1),
        max_retries=_env_int("LLM_MAX_RETRIES", 2, minimum=0),
        max_tokens=_env_int("LLM_MAX_TOKENS", 1024, minimum=1),
        effort=effort,
        game_debug=_env_bool("GAME_DEBUG"),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )
