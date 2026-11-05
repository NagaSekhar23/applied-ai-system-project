"""The Observability page.

Every number here is computed from events this process actually recorded. If
nothing has been done yet, the page says so rather than showing a plausible
sample dashboard — a fake metric is worse than an empty one.
"""

from __future__ import annotations

import streamlit as st

from evaluation.report import load_latest_reports
from observability import (
    PRICING_SNAPSHOT_DATE,
    Component,
    Status,
    get_tracker,
)

REPORT_DIR = "reports"


def render_empty_state() -> None:
    st.info(
        "**No activity recorded yet.** Use the AI Assistant, Natural Language or "
        "Agent pages, then come back. This dashboard only ever shows events from "
        "this session — it never displays sample data."
    )


def render_overview(metrics) -> None:
    st.markdown("#### Requests")
    cols = st.columns(4)
    cols[0].metric("Events", metrics.total_events)
    cols[1].metric("Success rate", f"{metrics.success_rate:.0%}")
    cols[2].metric("Failure rate", f"{metrics.failure_rate:.0%}")
    cols[3].metric("Fallbacks", metrics.degraded_events)

    st.markdown("#### Latency")
    cols = st.columns(2)
    cols[0].metric("Average", f"{metrics.avg_latency_ms:.0f} ms")
    cols[1].metric(
        "P95",
        f"{metrics.p95_latency_ms:.0f} ms",
        help="Nearest-rank 95th percentile. Not meaningful below roughly 20 samples.",
    )
    if metrics.total_events < 20:
        st.caption(
            f"Only {metrics.total_events} events recorded — percentiles are noisy at this sample size."
        )


def render_llm(metrics) -> None:
    st.markdown("#### LLM")
    if metrics.llm_calls == 0:
        st.caption("No LLM calls recorded yet.")
        return

    cols = st.columns(4)
    cols[0].metric("Calls", metrics.llm_calls)
    cols[1].metric("Failure rate", f"{metrics.llm_failure_rate:.0%}")
    cols[2].metric(
        "Retries / call",
        f"{metrics.retry_rate:.2f}",
        help="Total retries divided by calls, so it can exceed 1.0.",
    )
    cols[3].metric("Timeouts", metrics.llm_timeouts)

    cols = st.columns(3)
    cols[0].metric("Input tokens", f"{metrics.known_tokens_in:,}")
    cols[1].metric("Output tokens", f"{metrics.known_tokens_out:,}")

    if metrics.priced_calls:
        cols[2].metric("Estimated cost", f"${metrics.estimated_cost_usd:.4f}")
    else:
        cols[2].metric("Estimated cost", "not available")

    if metrics.unpriced_calls:
        st.caption(
            f"{metrics.unpriced_calls} call(s) could not be priced — either the provider "
            "reported no usage, or the model has no configured price. Those calls are "
            "excluded from the figure above rather than guessed."
        )
    if metrics.priced_calls:
        st.caption(
            f"Cost is an estimate from a price list recorded {PRICING_SNAPSHOT_DATE}. "
            "The provider's billing is authoritative."
        )
    if metrics.llm_validation_failures:
        st.caption(
            f"{metrics.llm_validation_failures} response(s) failed schema validation and were retried."
        )


def render_tools_and_agent(metrics) -> None:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Tools")
        if metrics.tool_calls == 0:
            st.caption("No tool calls recorded yet.")
        else:
            st.metric("Calls", metrics.tool_calls)
            st.metric("Failure rate", f"{metrics.tool_failure_rate:.0%}")
            if metrics.tool_usage:
                st.bar_chart(metrics.tool_usage, horizontal=True)

    with col_right:
        st.markdown("#### Agent")
        if metrics.agent_runs == 0:
            st.caption("No agent runs recorded yet.")
        else:
            st.metric("Runs", metrics.agent_runs)
            st.metric("Average iterations", f"{metrics.avg_agent_iterations:.1f}")


def render_components(metrics) -> None:
    st.markdown("#### By component")
    st.dataframe(
        [c.to_dict() for c in metrics.by_component],
        use_container_width=True,
        hide_index=True,
    )


def render_traces(tracker) -> None:
    st.markdown("#### Recent requests")
    request_ids = tracker.store.request_ids()[:10]
    if not request_ids:
        st.caption("No requests recorded yet.")
        return

    for request_id in request_ids:
        events = sorted(tracker.store.by_request(request_id), key=lambda e: e.timestamp)
        if not events:
            continue
        failed = any(e.status is Status.ERROR for e in events)
        total_ms = sum(e.duration_ms for e in events)
        icon = "❌" if failed else "✅"
        label = f"{icon} `{request_id}` · {len(events)} event(s) · {total_ms:.0f} ms"

        with st.expander(label):
            for event in events:
                marker = {"ok": "✅", "error": "❌", "degraded": "⚠️"}[event.status.value]
                st.markdown(
                    f"{marker} **{event.operation}** "
                    f"(`{event.component.value}`) · {event.duration_ms:.1f} ms"
                )
                if event.trace_id:
                    st.caption(f"trace `{event.trace_id}`")
                if event.error_message:
                    st.error(event.error_message)
                if event.attributes:
                    st.json(event.attributes, expanded=False)
                if event.cost:
                    st.caption(f"cost: {event.cost.to_dict()}")


def render_failures(tracker) -> None:
    failures = tracker.store.failures()
    st.markdown("#### Failures")
    if not failures:
        st.success("No failures recorded.")
        return
    for event in reversed(failures[-10:]):
        st.error(
            f"**{event.operation}** (`{event.component.value}`) — "
            f"{event.error_type or 'error'}: {event.error_message or 'no detail'}"
        )


def render_evaluation() -> None:
    st.markdown("#### Evaluation")
    reports = load_latest_reports(REPORT_DIR)
    if not reports:
        st.caption(
            "No evaluation reports found. Run `python -m evaluation.run` to generate them."
        )
        return

    for report in reports:
        metrics = report.get("metrics", {})
        dataset = report.get("dataset", {})
        cols = st.columns(4)
        cols[0].metric(dataset.get("name", "dataset"), f"{metrics.get('pass_rate', 0):.0%}")
        cols[1].metric("Passed", metrics.get("passed", 0))
        cols[2].metric("Failed", metrics.get("failed", 0) + metrics.get("errored", 0))
        cols[3].metric("Cases", metrics.get("total_cases", 0))
        st.caption(
            f"Measuring `{metrics.get('kind')}` · run {report.get('started_at')}Z · "
            f"fingerprint `{report.get('fingerprint')}`"
        )


def render(settings) -> None:
    """Render the Observability page."""
    st.subheader("📊 Observability")
    st.caption(
        "Metrics computed from events recorded in this session. Secrets are redacted "
        "before anything is stored, and costs are only shown when they can be computed honestly."
    )

    tracker = get_tracker()
    metrics = tracker.metrics()

    if metrics.total_events == 0:
        render_empty_state()
        render_evaluation()
        return

    render_overview(metrics)
    st.divider()
    render_llm(metrics)
    st.divider()
    render_tools_and_agent(metrics)
    st.divider()
    render_components(metrics)
    st.divider()
    render_traces(tracker)
    st.divider()
    render_failures(tracker)
    st.divider()
    render_evaluation()

    st.divider()
    cols = st.columns(2)
    with cols[0]:
        if st.button("Clear recorded events"):
            tracker.clear()
            st.rerun()
    with cols[1]:
        st.caption(
            f"Store holds up to {tracker.store.max_events} events "
            f"({tracker.store.dropped_count} evicted so far)."
        )

    with st.expander("Raw metrics (JSON)"):
        st.json(metrics.to_dict())
