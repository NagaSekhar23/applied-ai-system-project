"""The Agent page.

Shows the execution trace alongside the answer: which tools were chosen, with
what arguments, how long each took, and why the loop stopped. The guardrails
are visible rather than implied.
"""

from __future__ import annotations

import streamlit as st

from agents import AgentConfig, AgentRun, build_agent
from assistant import Assistant
from config import Settings
from providers import build_client

EXAMPLES = [
    "Start a hard game and explain the rules",
    "What is my status?",
    "Give me a hint",
    "Guess 42",
]

STOP_REASON_HELP = {
    "responded": "The agent finished every requested step.",
    "clarification": "The request was ambiguous, so nothing was run.",
    "max_iterations": "Hit the step limit — the loop is bounded.",
    "timeout": "Exceeded the wall-clock budget.",
    "repeated_call": "The agent was about to repeat an identical call.",
    "too_many_tool_errors": "Too many invalid tool calls in a row.",
}


def get_agent(settings: Settings):
    """Build the agent once per session, keyed by configuration."""
    key = (settings.provider, settings.model, settings.max_retries)
    if st.session_state.get("agent_key") != key:
        client = build_client(settings)
        st.session_state.agent = build_agent(
            assistant=Assistant(client, max_tokens=settings.max_tokens, effort=settings.effort),
            client=client,
            config=AgentConfig(max_iterations=5, timeout_seconds=30.0),
        )
        st.session_state.agent_key = key
    return st.session_state.agent


def render_run(run: AgentRun) -> None:
    """Render one agent run and its trace."""
    with st.chat_message("user"):
        st.write(run.request)

    with st.chat_message("assistant"):
        st.write(run.response)
        tools = ", ".join(f"`{t}`" for t in run.tools_called) or "none"
        st.caption(f"Tools used: {tools} · {run.iterations} iteration(s) · stop: `{run.stop_reason}`")

        if not run.succeeded:
            st.warning(STOP_REASON_HELP.get(run.stop_reason, "The agent stopped early."))

        with st.expander("Execution trace"):
            cols = st.columns(3)
            cols[0].metric("Iterations", run.iterations)
            cols[1].metric("Tool calls", len(run.tool_results))
            cols[2].metric("Duration", f"{run.trace.duration_ms:.0f} ms")

            st.caption(f"Trace `{run.trace.trace_id}`")
            for span in run.trace.spans:
                if span.kind == "agent":
                    continue
                icon = {"ok": "✅", "error": "❌", "degraded": "⚠️"}.get(span.status, "•")
                st.markdown(
                    f"{icon} **{span.name}** · iteration {span.iteration} · "
                    f"{span.duration_ms:.1f} ms"
                )
                if span.attributes:
                    st.json(span.attributes, expanded=False)
                if span.error:
                    st.error(span.error)


def render(settings: Settings) -> None:
    """Render the Agent page."""
    st.subheader("🛠️ AI Agent")
    st.caption(
        "The agent plans, calls registered tools, inspects results, and answers. "
        "It can only call tools in its registry, and every loop is bounded."
    )

    if settings.is_mock:
        st.info(
            "**Mock mode.** Planning uses a deterministic rule-based planner. "
            "Set `LLM_PROVIDER=anthropic` with a key for model-driven planning."
        )

    registry = get_agent(settings).registry
    with st.expander(f"Registered tools ({len(registry)})"):
        for schema in registry.schemas():
            marker = "✏️" if schema["mutates_state"] else "👁️"
            st.markdown(f"{marker} **`{schema['name']}`** — {schema['description']}")
        st.caption("✏️ changes game state · 👁️ read-only")

    st.caption("Try: " + " · ".join(f"*{e}*" for e in EXAMPLES))

    for run in st.session_state.get("agent_history", []):
        render_run(run)

    message = st.chat_input("Ask the agent to do something…")
    if message:
        agent = get_agent(settings)
        with st.spinner("Working…"):
            run = agent.run(
                message,
                st.session_state.get("game"),
                st.session_state.get("last_outcome"),
            )
        if run.state is not None:
            st.session_state.game = run.state
            if run.last_outcome is not None:
                st.session_state.last_outcome = run.last_outcome
            else:
                st.session_state.pop("last_outcome", None)
        st.session_state.setdefault("agent_history", []).append(run)
        st.rerun()

    if st.session_state.get("agent_history") and st.button("Clear agent history"):
        st.session_state.agent_history = []
        st.rerun()
