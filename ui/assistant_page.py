"""The AI Assistant page.

The assistant explains the game state the deterministic logic produced. It
never decides anything. The page always shows whether a reply came from the
model or from the deterministic fallback, so the distinction is visible rather
than hidden.
"""

from __future__ import annotations

import streamlit as st

from assistant import Assistant, AssistantReply
from config import Settings
from observability import Component, get_tracker
from providers import build_client


def get_assistant(settings: Settings) -> Assistant:
    """Build the assistant once per session, keyed by the active configuration.

    Rebuilt if the configuration changes, so switching provider in the
    environment does not leave a stale client behind.
    """
    key = (settings.provider, settings.model, settings.max_retries)
    if st.session_state.get("assistant_key") != key:
        st.session_state.assistant = Assistant(
            build_client(settings),
            max_tokens=settings.max_tokens,
            effort=settings.effort,
        )
        st.session_state.assistant_key = key
    return st.session_state.assistant


def render_mode_banner(settings: Settings) -> None:
    """State plainly which provider mode is active. Never shows the API key."""
    if settings.is_mock:
        st.info(
            "**Mock mode.** Replies come from a deterministic local stand-in, not a "
            "real model. Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` for live replies."
        )
    elif not settings.has_credentials:
        st.warning(
            f"**{settings.provider}** is selected but `ANTHROPIC_API_KEY` is not set. "
            "Requests will fail and the deterministic fallback will be used."
        )
    else:
        st.success(f"**Live mode.** Provider `{settings.provider}`, model `{settings.model}`.")


def render_reply(reply: AssistantReply) -> None:
    """Render a reply, labelled with its source."""
    if reply.used_llm:
        st.markdown(f"**Assistant:** {reply.message}")
        st.caption(f"Source: model output, validated. Prompt `{reply.prompt_id}`.")
    else:
        st.markdown(f"**Assistant (fallback):** {reply.message}")
        st.caption(f"Source: deterministic fallback. Reason: {reply.fallback_reason}")

    for warning in reply.warnings:
        st.warning(f"Corrected: {warning}")

    if reply.stats:
        stats = reply.stats
        usage = stats.usage
        with st.expander("Call details"):
            cols = st.columns(4)
            cols[0].metric("Attempts", stats.attempts)
            cols[1].metric("Retries", stats.retries)
            cols[2].metric("Latency", f"{stats.latency_ms:.0f} ms")
            total = usage.total
            cols[3].metric("Tokens", total if total is not None else "not reported")
            if stats.errors:
                st.write("Errors encountered:")
                for error in stats.errors:
                    st.code(error, language=None)


def render(settings: Settings) -> None:
    """Render the AI Assistant page."""
    st.subheader("🤖 AI Assistant")
    st.caption(
        "The assistant explains what the game already decided. It never decides "
        "whether a guess is correct, and it cannot see the secret while you are playing."
    )

    render_mode_banner(settings)

    game = st.session_state.get("game")
    if game is None:
        st.info("Start a game on the **Game** page first, then come back for a hint.")
        return

    cols = st.columns(3)
    with cols[0]:
        st.metric("Attempts left", game.attempts_left)
    with cols[1]:
        st.metric("Score", game.score)
    with cols[2]:
        st.metric("Range", f"{game.low}–{game.high}")

    question = st.text_input(
        "Ask the assistant:",
        key="assistant_question",
        placeholder="e.g. What should I try next?",
    )

    action_cols = st.columns(2)
    with action_cols[0]:
        want_hint = st.button("Get a hint 💡")
    with action_cols[1]:
        want_rules = st.button("Explain the rules 📖")

    assistant = get_assistant(settings)

    if want_hint or want_rules:
        tracker = get_tracker()
        with st.spinner("Thinking…"), tracker.request("assistant.request"):
            if want_rules:
                reply = assistant.explain_rules(game, question)
            else:
                reply = assistant.hint(
                    game,
                    last_outcome=st.session_state.get("last_outcome"),
                    question=question,
                )
            tracker.record_assistant_reply(reply, question)
        st.session_state.assistant_reply = reply

    reply = st.session_state.get("assistant_reply")
    if reply is not None:
        st.divider()
        render_reply(reply)
