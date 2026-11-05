"""Redaction for anything that gets logged.

Two rules:

* **Secrets never appear.** API keys are matched by shape, not only by field
  name, because a key pasted into a message is still a key.
* **User content is minimised.** Free text is truncated and, where the field
  is not needed for debugging, replaced by its length.

Redaction runs on the way *into* the store, not on the way out, so a leaked
value is never written to disk in the first place.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
MAX_TEXT_LENGTH = 200

# Key-shaped strings. Matched anywhere, including inside free text.
KEY_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{4,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
)

# Field names whose value is always dropped, whatever it looks like.
SENSITIVE_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "anthropic_api_key",
        "authorization",
        "auth_token",
        "password",
        "secret_key",
        "token",
    }
)

# Game-specific: the answer must never be logged while a game is in progress.
GAME_SENSITIVE_FIELDS = frozenset({"secret"})


def scrub_text(value: str) -> str:
    """Remove key-shaped substrings and truncate long text."""
    cleaned = value
    for pattern in KEY_PATTERNS:
        cleaned = pattern.sub(REDACTED, cleaned)
    if len(cleaned) > MAX_TEXT_LENGTH:
        cleaned = cleaned[:MAX_TEXT_LENGTH] + f"… [{len(value)} chars]"
    return cleaned


def redact(value: Any, drop_game_secret: bool = True) -> Any:
    """Recursively redact a value for logging.

    Args:
        value: Any JSON-like structure.
        drop_game_secret: Also redact a ``secret`` field. True by default;
            callers that are logging a finished game may pass False.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in SENSITIVE_FIELDS:
                out[key] = REDACTED
            elif drop_game_secret and lowered in GAME_SENSITIVE_FIELDS and item is not None:
                out[key] = REDACTED
            else:
                out[key] = redact(item, drop_game_secret)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item, drop_game_secret) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def summarise_text(value: str | None) -> dict[str, Any]:
    """Describe user text without storing it.

    Used for message bodies, where the length and whether it was empty are
    useful for debugging but the content itself is not needed.
    """
    if value is None:
        return {"present": False, "chars": 0}
    return {"present": True, "chars": len(value)}
