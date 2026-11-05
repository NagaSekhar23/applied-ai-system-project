"""The game page: the original deterministic experience, unchanged.

State changes happen in ``on_click`` callbacks. Streamlit runs callbacks before
the script body, so widgets render from the updated state on the same run.
Without that, the guess box stayed enabled for one run after a win or loss.
"""

from __future__ import annotations

import streamlit as st

from game import GameState, GuessResult, new_game, submit_guess
from logic_utils import DIFFICULTY_NAMES, get_difficulty_config


def text_key_for(difficulty: str) -> str:
    """Widget key for the guess box, keyed by difficulty so switching clears it."""
    return f"guess_input_{difficulty}"


def current_game(difficulty: str) -> GameState:
    """Return the stored game, starting a fresh one if the difficulty changed."""
    game = st.session_state.get("game")
    if game is None or game.difficulty != difficulty:
        st.session_state.game = new_game(difficulty)
        st.session_state.pop("feedback", None)
        st.session_state.pop("assistant_reply", None)
    return st.session_state.game


def on_submit(text_key: str) -> None:
    """Callback: apply the guess and remember the result for display."""
    result = submit_guess(st.session_state.game, st.session_state.get(text_key))
    st.session_state.game = result.state
    st.session_state.feedback = result
    if result.outcome:
        st.session_state.last_outcome = result.outcome
    st.session_state.pop("assistant_reply", None)


def on_new_game(difficulty: str, text_key: str) -> None:
    """Callback: start a fresh game and clear the guess box."""
    st.session_state.game = new_game(difficulty)
    st.session_state.pop("feedback", None)
    st.session_state.pop("assistant_reply", None)
    st.session_state.pop("last_outcome", None)
    st.session_state[text_key] = ""


def show_feedback(feedback: GuessResult, show_hint: bool) -> None:
    """Render the result of the most recent submission, once."""
    if feedback.error:
        st.error(feedback.error)
        return
    if show_hint and feedback.hint:
        st.warning(feedback.hint)
    if feedback.outcome == "Win":
        st.balloons()


def render_sidebar() -> str:
    """Render difficulty settings and return the selected difficulty."""
    st.sidebar.header("Settings")
    difficulty = st.sidebar.selectbox("Difficulty", list(DIFFICULTY_NAMES), index=1)
    config = get_difficulty_config(difficulty)
    st.sidebar.caption(f"Range: {config.low} to {config.high}")
    st.sidebar.caption(f"Attempts allowed: {config.max_attempts}")
    return difficulty


def render(difficulty: str, debug: bool) -> None:
    """Render the game page."""
    game = current_game(difficulty)
    text_key = text_key_for(difficulty)
    game_over = game.status != "playing"

    st.subheader("Make a guess")
    st.text_input("Enter your guess:", key=text_key, disabled=game_over)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Submit Guess 🚀", on_click=on_submit, args=(text_key,), disabled=game_over)
    with col2:
        st.button("New Game 🔁", on_click=on_new_game, args=(difficulty, text_key))
    with col3:
        show_hint = st.checkbox("Show hint", value=True)

    feedback = st.session_state.pop("feedback", None)
    if feedback is not None:
        show_feedback(feedback, show_hint)

    if game.status == "won":
        st.success(
            f"You won! The secret was {game.secret}. Final score: {game.score}. "
            "Click New Game to play again."
        )
    elif game.status == "lost":
        st.error(
            f"Out of attempts! The secret was {game.secret}. Score: {game.score}. "
            "Click New Game to try again."
        )
    else:
        st.info(
            f"Guess a number between {game.low} and {game.high}. "
            f"Attempts left: {game.attempts_left}"
        )

    if debug:
        with st.expander("Developer Debug Info"):
            st.write("Secret:", game.secret)
            st.write("Attempts used:", game.attempts_used)
            st.write("Score:", game.score)
            st.write("Difficulty:", game.difficulty)
            st.write("History:", list(game.history))

    st.divider()
