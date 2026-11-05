# 🎮 Game Glitch Investigator: The Impossible Guesser

A number-guessing game built with Streamlit. The starter app was AI-generated and contained several bugs. This project documents how those bugs were investigated and fixed, how the core guessing logic was refactored into its own module, and how it was tested.

## 🛠️ Setup

1. Install application dependencies: `pip install -r requirements.txt`
2. For development and testing, also install: `pip install -r requirements-dev.txt`
3. Run the app: `python -m streamlit run app.py`

### Configuration

| Variable | Default | Effect |
|---|---|---|
| `GAME_DEBUG` | unset | Set to `true` to show the Developer Debug Info panel (secret, attempts, score, history). Hidden otherwise. |

### Running the tests

```bash
python -m pytest
```

## 🏗️ Architecture

- `logic_utils.py`: pure rules (difficulty settings, input parsing, `check_guess`, scoring). No Streamlit imports.
- `game.py`: `GameState` (a frozen dataclass) and the transitions `new_game` and `submit_guess`.
- `app.py`: Streamlit UI only. It reads widgets, calls `game.py`, and renders results.
- `tests/`: unit tests per module, plus `test_app_flow.py` and `test_regressions.py`, which run the app through Streamlit's `AppTest`.

## 🐞 Bugs Investigated and Fixed

- **Reversed hint messages:** `check_guess()` told the player to "Go HIGHER!" when the guess was too high and "Go LOWER!" when it was too low. The hints now match the outcome.
- **New Game did not reset the score:** clicking "New Game" reset attempts but kept the old score. The score now resets to its initial value.
- **Incorrect submit button call:** the submit button used `st.menu_button`, which raised a `TypeError`. It was corrected to `st.button`.

## 🔧 Refactoring

`check_guess()` was moved from `app.py` into `logic_utils.py`, and `app.py` now imports it from there. Its return structure, `(outcome, message)`, is unchanged.

## ✅ Testing

The suite covers parsing, difficulty settings, scoring, game-state transitions, and the Streamlit flow. Checks include that a guess too high gets "Go LOWER!", that invalid or out-of-range input never uses up an attempt, and that a new game after a win is playable again.

Run it with `python -m pytest`. Current result: **106 passed**.

## 📋 Rules (current behavior)

- **Difficulty:** Easy 1–20 with 6 attempts, Normal 1–100 with 8 attempts, Hard 1–50 with 5 attempts. Changing difficulty starts a new game.
- **Attempts:** only valid guesses count. Invalid text and out-of-range numbers are rejected with a message and do not use an attempt.
- **Winning:** the game ends when you guess the secret. Points are `max(10, 100 − 10 × guess number)`.
- **Losing:** the game ends after your last attempt. The secret is revealed.
- **Wrong guesses:** each costs 5 points, whether the guess was too high or too low. The score never drops below 0. For example, a score of 0 stays at 0, and a score of 10 becomes 5.
- **Score example:** a score of 0 that guesses wrong, then wins on guess 3, ends with 70 points (100 − 10 × 3).
- **Game over:** the guess box and Submit button are disabled. New Game is always available.

## 📸 Demo Walkthrough

1. Start the app with `python -m streamlit run app.py`.
2. With `GAME_DEBUG=true` set, expand **Developer Debug Info** to see the secret number, attempts, and score.
3. Enter a guess in the text box and click **Submit Guess 🚀**.
4. Verify the hint: a guess above the secret should show "Go LOWER!", and a guess below it should show "Go HIGHER!".
5. Click **New Game 🔁** and check **Developer Debug Info** again: attempts used and score are back to 0.

## 📝 Documentation

- `reflection.md` contains the investigation and reflection.
- `ai_interactions.md` documents the AI-assisted debugging and refactoring.

## 🔀 Git

The project was committed and pushed to GitHub.
