"""Event storage.

A bounded in-memory deque plus optional JSONL on disk. Bounded because an
unbounded list in a long-running Streamlit session is a slow memory leak;
JSONL because it is append-only, survives a restart, and is greppable with
ordinary tools.

Retention is explicit: ``max_events`` in memory, and a file the operator
rotates or deletes. Nothing is kept forever by accident.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator

from observability.events import Component, Event, Status

DEFAULT_MAX_EVENTS = 500


class EventStore:
    """Thread-safe, bounded event storage.

    Streamlit can run callbacks on different threads, so the lock is not
    optional.
    """

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS, log_path: str | Path | None = None):
        self.max_events = max_events
        self.log_path = Path(log_path) if log_path else None
        self._events: deque[Event] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._dropped = 0

    def record(self, event: Event) -> Event:
        """Store an event, and append it to the log file when configured.

        A failure to write the log must never break the application, so the
        error is noted on the event rather than raised. It is not silently
        discarded: the attribute is visible in the dashboard.
        """
        with self._lock:
            if len(self._events) == self.max_events:
                self._dropped += 1
            self._events.append(event)

        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event.to_dict(), sort_keys=False) + "\n")
            except OSError as exc:
                event.attributes["log_write_error"] = str(exc)

        return event

    def all(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def recent(self, limit: int = 50) -> list[Event]:
        """The most recent events, newest first."""
        return list(reversed(self.all()))[:limit]

    def by_component(self, component: Component) -> list[Event]:
        return [e for e in self.all() if e.component is component]

    def by_request(self, request_id: str) -> list[Event]:
        return [e for e in self.all() if e.request_id == request_id]

    def by_trace(self, trace_id: str) -> list[Event]:
        return [e for e in self.all() if e.trace_id == trace_id]

    def failures(self) -> list[Event]:
        return [e for e in self.all() if e.status is Status.ERROR]

    def request_ids(self) -> list[str]:
        """Distinct request ids, most recent first."""
        seen: list[str] = []
        for event in reversed(self.all()):
            if event.request_id and event.request_id not in seen:
                seen.append(event.request_id)
        return seen

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._dropped = 0

    @property
    def dropped_count(self) -> int:
        """Events evicted by the size bound. Surfaced so the cap is visible."""
        return self._dropped

    def __len__(self) -> int:
        return len(self.all())

    def __iter__(self) -> Iterator[Event]:
        return iter(self.all())


def load_events(path: str | Path) -> list[dict]:
    """Read events back from a JSONL log. Malformed lines are skipped."""
    file_path = Path(path)
    if not file_path.is_file():
        return []
    events: list[dict] = []
    try:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return events


def summarise(events: Iterable[Event]) -> dict[str, int]:
    """Count events per status. Handy for a quick assertion in tests."""
    counts: dict[str, int] = {}
    for event in events:
        counts[event.status.value] = counts.get(event.status.value, 0) + 1
    return counts
