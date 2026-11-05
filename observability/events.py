"""The structured event record.

Every observable operation becomes one `Event` with the same fields, so the
metrics layer never has to special-case a component. Attributes are redacted
when the event is built, not when it is read.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from observability.redaction import redact


class Status(str, Enum):
    OK = "ok"
    ERROR = "error"
    DEGRADED = "degraded"  # succeeded, but via a fallback


class Component(str, Enum):
    GAME = "game"
    LLM = "llm"
    ASSISTANT = "assistant"
    PARSER = "parser"
    AGENT = "agent"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class TokenCost:
    """Estimated cost of one call.

    Every field is ``None`` when the provider did not report usage or the
    model has no configured price. A cost is never guessed: an invented
    number here would quietly corrupt every cost figure downstream.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None
    priced: bool = False
    reason: str | None = None  # why there is no cost, when there is none

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6) if self.usd is not None else None,
            "priced": self.priced,
            "reason": self.reason,
        }


@dataclass
class Event:
    """One observable operation."""

    operation: str
    component: Component
    status: Status = Status.OK
    duration_ms: float = 0.0
    request_id: str = ""
    trace_id: str | None = None
    parent_id: str | None = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    error_type: str | None = None
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    cost: TokenCost | None = None

    def __post_init__(self) -> None:
        # Redact once, at construction, so nothing unsafe is ever stored.
        self.attributes = redact(self.attributes)
        if self.error_message:
            from observability.redaction import scrub_text

            self.error_message = scrub_text(self.error_message)

    @property
    def ok(self) -> bool:
        """Degraded counts as success: the user got a usable answer."""
        return self.status in (Status.OK, Status.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.timestamp)),
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "component": self.component.value,
            "operation": self.operation,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "cost": self.cost.to_dict() if self.cost else None,
        }
