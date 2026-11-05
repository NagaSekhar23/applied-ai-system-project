"""Streamlit UI for Game Glitch Investigator.

This file only handles widgets and rendering. All rules live in ``game.py``
and ``logic_utils.py``.

State changes happen in ``on_click`` callbacks. Streamlit runs callbacks before
the script body, so widgets are rendered from the updated state on the same
run. Without callbacks, the input and buttons would show stale enabled state
for one extra run after a win or loss.
"""

import os

import streamlit as st

from game import GameState, GuessResult, new_game, submit_guess
from logic_utils import DIFFICULTY_NAMES, get_difficulty_config


def debug_enabled() -> bool:
    """The debug panel is shown only when GAME_DEBUG=true."""
    return os.getenv("GAME_DEBUG", "").strip().lower() == "true"


def text_key_for(difficulty: str) -> str:
    """Widget key for the guess box. Keyed by difficulty so switching clears it."""
    return f"guess_input_{difficulty}"


def current_game(difficulty: str) -> GameState:
    """Return the stored game, starting a fresh one if none exists or the difficulty changed."""
    game = st.session_state.get("game")
    if game is None or game.difficulty != difficulty:
        st.session_state.game = new_game(difficulty)
        st.session_state.pop("feedback", None)
    return st.session_state.game


def on_submit(text_key: str) -> None:
    """Callback: apply the guess in the text box and remember the result for display."""
    result = submit_guess(st.session_state.game, st.session_state.get(text_key))
    st.session_state.game = result.state
    st.session_state.feedback = result


def on_new_game(difficulty: str, text_key: str) -> None:
    """Callback: start a fresh game and clear the guess box."""
    st.session_state.game = new_game(difficulty)
    st.session_state.pop("feedback", None)
    st.session_state[text_key] = ""


def show_feedback(feedback: GuessResult, show_hint: bool) -> None:
    """Render the result of the most recent submission. Shown once, then cleared."""
    if feedback.error:
        st.error(feedback.error)
        return
    if show_hint and feedback.hint:
        st.warning(feedback.hint)
    if feedback.outcome == "Win":
        st.balloons()


st.set_page_config(page_title="Glitchy Guesser", page_icon="🎮")

st.title("🎮 Game Glitch Investigator")
st.caption("An AI-generated guessing game. Something is off.")

st.sidebar.header("Settings")
difficulty = st.sidebar.selectbox("Difficulty", list(DIFFICULTY_NAMES), index=1)
config = get_difficulty_config(difficulty)
st.sidebar.caption(f"Range: {config.low} to {config.high}")
st.sidebar.caption(f"Attempts allowed: {config.max_attempts}")

game = current_game(difficulty)
text_key = text_key_for(difficulty)
game_over = game.status != "playing"

st.subheader("Make a guess")

st.text_input("Enter your guess:", key=text_key, disabled=game_over)

col1, col2, col3 = st.columns(3)
with col1:
    st.button(
        "Submit Guess 🚀",
        on_click=on_submit,
        args=(text_key,),
        disabled=game_over,
    )
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

if debug_enabled():
    with st.expander("Developer Debug Info"):
        st.write("Secret:", game.secret)
        st.write("Attempts used:", game.attempts_used)
        st.write("Score:", game.score)
        st.write("Difficulty:", game.difficulty)
        st.write("History:", list(game.history))

st.divider()
