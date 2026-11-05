"""The Natural Language page.

Shows the whole pipeline, not just the answer: what the parser understood,
what the validator decided, and what the executor did. For a portfolio project
that transparency is the point — a reviewer can see that the model proposes and
deterministic code disposes.
"""

from __future__ import annotations

import streamlit as st

from assistant import Assistant
from config import Settings
from observability import Status, get_tracker
from natural_language import (
    CommandExecutor,
    CommandParser,
    ConversationTurn,
    handle_message,
)
from providers import build_client

EXAMPLES = [
    "Start a hard game",
    "I think the number is 45",
    "Give me a hint",
    "How many tries do I have left?",
]


def get_pipeline(settings: Settings) -> tuple[CommandParser, CommandExecutor]:
    """Build parser and executor once per session, keyed by configuration."""
    key = (settings.provider, settings.model, settings.max_retries)
    if st.session_state.get("nl_key") != key:
        client = build_client(settings)
        st.session_state.nl_parser = CommandParser(
            client, max_tokens=settings.max_tokens, effort=settings.effort
        )
        st.session_state.nl_executor = CommandExecutor(
            Assistant(client, max_tokens=settings.max_tokens, effort=settings.effort)
        )
        st.session_state.nl_key = key
    return st.session_state.nl_parser, st.session_state.nl_executor


def run_turn(message: str, settings: Settings) -> None:
    """Run one message through the pipeline and record the turn."""
    parser, executor = get_pipeline(settings)
    tracker = get_tracker()

    with tracker.request("nl.message"):
        turn = handle_message(
            message,
            st.session_state.get("game"),
            parser,
            executor,
            last_outcome=st.session_state.get("last_outcome"),
        )
        tracker.record_parse(turn.parse, message)
        tracker.record_game_action(
            f"nl.{turn.command_type.lower()}",
            Status.OK if turn.result.executed else Status.ERROR,
            command=turn.command_type,
            executed=turn.result.executed,
        )

    # Only a successfully executed command may change game state.
    if turn.result.executed and turn.result.state is not None:
        st.session_state.game = turn.result.state
        if turn.result.last_outcome is not None:
            st.session_state.last_outcome = turn.result.last_outcome
        elif turn.result.command.command.value in ("START_GAME", "RESET_GAME", "CHANGE_DIFFICULTY"):
            st.session_state.pop("last_outcome", None)

    st.session_state.setdefault("nl_history", []).append(turn)


def render_turn(turn: ConversationTurn) -> None:
    """Render one turn: the user's message, the parse, and the result."""
    with st.chat_message("user"):
        st.write(turn.user_message)

    with st.chat_message("assistant"):
        st.write(turn.result.message)

        command = turn.result.command
        if turn.result.executed:
            st.caption(f"Ran `{command.command.value}`")
        elif turn.validated.needs_clarification:
            st.caption("Needed clarification — nothing was run")
        else:
            st.caption(f"Rejected `{command.command.value}` — nothing was run")

        with st.expander("How this was handled"):
            st.markdown("**1. Parsed intent**")
            st.json(
                {
                    "command": command.command.value,
                    "guess": command.guess,
                    "difficulty": command.difficulty,
                    "confidence": command.confidence,
                    "source": turn.parse.source,
                    "prompt": turn.parse.prompt_id,
                }
            )
            if turn.parse.fallback_reason:
                st.caption(f"Parser fallback: {turn.parse.fallback_reason}")
            for warning in turn.parse.warnings:
                st.warning(warning)

            st.markdown("**2. Validation**")
            if turn.validated.ok:
                st.success("Passed — permitted in the current game state")
            else:
                for issue in turn.validated.issues:
                    st.error(f"{issue.field_name}: {issue.message}")
                if turn.validated.needs_clarification:
                    st.info(turn.validated.summary)

            st.markdown("**3. Execution**")
            st.write("Executed" if turn.result.executed else "Not executed")
            if turn.result.details:
                st.json(turn.result.details)


def render(settings: Settings) -> None:
    """Render the Natural Language page."""
    st.subheader("💬 Natural Language")
    st.caption(
        "Type an instruction in plain English. The model only classifies intent; "
        "a validator checks it and deterministic game code performs the action."
    )

    if settings.is_mock:
        st.info(
            "**Mock mode.** Intent is classified by a deterministic rule-based parser. "
            "Set `LLM_PROVIDER=anthropic` with a key for model-based parsing of "
            "unusual phrasings and typos."
        )

    st.caption("Try: " + " · ".join(f"*{example}*" for example in EXAMPLES))

    for turn in st.session_state.get("nl_history", []):
        render_turn(turn)

    message = st.chat_input("Tell me what to do…")
    if message:
        run_turn(message, settings)
        st.rerun()

    if st.session_state.get("nl_history") and st.button("Clear conversation"):
        st.session_state.nl_history = []
        st.rerun()
