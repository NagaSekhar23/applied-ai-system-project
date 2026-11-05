"""Streamlit UI for Game Glitch Investigator.

This file only handles widgets and rendering. All rules live in ``game.py``
and ``logic_utils.py``.
"""

import os

import streamlit as st

from game import GameState, new_game, submit_guess
from logic_utils import DIFFICULTY_NAMES, get_difficulty_config


def debug_enabled() -> bool:
    """The debug panel is shown only when GAME_DEBUG=true."""
    return os.getenv("GAME_DEBUG", "").strip().lower() == "true"


def current_game(difficulty: str) -> GameState:
    """Return the stored game, starting a fresh one if none exists or the difficulty changed."""
    game = st.session_state.get("game")
    if game is None or game.difficulty != difficulty:
        game = new_game(difficulty)
        st.session_state.game = game
    return game


st.set_page_config(page_title="Glitchy Guesser", page_icon="🎮")

st.title("🎮 Game Glitch Investigator")
st.caption("An AI-generated guessing game. Something is off.")

st.sidebar.header("Settings")
difficulty = st.sidebar.selectbox("Difficulty", list(DIFFICULTY_NAMES), index=1)
config = get_difficulty_config(difficulty)
st.sidebar.caption(f"Range: {config.low} to {config.high}")
st.sidebar.caption(f"Attempts allowed: {config.max_attempts}")

game = current_game(difficulty)

st.subheader("Make a guess")

raw_guess = st.text_input("Enter your guess:", key=f"guess_input_{difficulty}")

col1, col2, col3 = st.columns(3)
with col1:
    submit = st.button("Submit Guess 🚀")
with col2:
    new_game_clicked = st.button("New Game 🔁")
with col3:
    show_hint = st.checkbox("Show hint", value=True)

if new_game_clicked:
    st.session_state.game = new_game(difficulty)
    st.rerun()

if submit:
    result = submit_guess(game, raw_guess)
    st.session_state.game = result.state
    game = result.state

    if result.error:
        st.error(result.error)
    else:
        if show_hint and result.hint:
            st.warning(result.hint)
        if result.outcome == "Win":
            st.balloons()

if game.status == "won":
    st.success(
        f"You won! The secret was {game.secret}. Final score: {game.score}. "
        "Start a new game to play again."
    )
elif game.status == "lost":
    st.error(
        f"Out of attempts! The secret was {game.secret}. Score: {game.score}. "
        "Start a new game to try again."
    )
else:
    st.info(
        f"Guess a number between {game.low} and {game.high}. "
        f"Attempts left: {game.attempts_left}"
    )

if debug_enabled():
    with st.expander("Developer Debug Info"):
        st.write("Secret:", game.secret)
        st.write("Attempts used:", game.attempts_used)
        st.write("Score:", game.score)
        st.write("Difficulty:", game.difficulty)
        st.write("History:", list(game.history))

st.divider()
