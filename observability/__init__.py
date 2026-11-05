"""Observability: structured events, metrics, tracing and redaction.

Dependency direction is one-way. Services (`game`, `assistant`, `agents`,
`natural_language`) produce data; callers hand that data to the tracker.
No business-logic module imports this package.
"""

from observability.events import Component, Event, Status, TokenCost
from observability.metrics import (
    ComponentMetrics,
    SystemMetrics,
    compute_metrics,
    percentile,
)
from observability.pricing import PRICING, PRICING_SNAPSHOT_DATE, estimate_cost, get_pricing
from observability.redaction import REDACTED, redact, scrub_text, summarise_text
from observability.store import EventStore, load_events, summarise
from observability.tracker import (
    Tracker,
    current_request_id,
    current_trace_id,
    get_tracker,
    new_id,
    set_tracker,
)

__all__ = [
    "Component",
    "ComponentMetrics",
    "Event",
    "EventStore",
    "PRICING",
    "PRICING_SNAPSHOT_DATE",
    "REDACTED",
    "Status",
    "SystemMetrics",
    "TokenCost",
    "Tracker",
    "compute_metrics",
    "current_request_id",
    "current_trace_id",
    "estimate_cost",
    "get_pricing",
    "get_tracker",
    "load_events",
    "new_id",
    "percentile",
    "redact",
    "scrub_text",
    "set_tracker",
    "summarise",
    "summarise_text",
]
