"""Applied AI Engineering System — Streamlit entry point.

This file is a navigation shell. Each page lives in ``ui/`` and calls
application services; no page contains game rules, prompts or provider code.

Run with:
    python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import ConfigError, load_settings
from ui import assistant_page, game as game_page

PAGES = ["Game", "AI Assistant"]

st.set_page_config(page_title="Applied AI System", page_icon="🎮")

st.title("🎮 Game Glitch Investigator")
st.caption("A deterministic game with an AI layer that explains but never decides.")

try:
    settings = load_settings()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

# Difficulty stays in the sidebar above navigation so the game page's own
# controls keep their familiar position.
difficulty = game_page.render_sidebar()

st.sidebar.divider()
page = st.sidebar.radio("Page", PAGES, index=0)
st.sidebar.divider()
st.sidebar.caption(f"LLM mode: {'mock' if settings.is_mock else 'live'}")

# The game is created even when another page is showing, so the assistant
# always has state to describe.
game_page.current_game(difficulty)

if page == "Game":
    game_page.render(difficulty, debug=settings.game_debug)
elif page == "AI Assistant":
    assistant_page.render(settings)
