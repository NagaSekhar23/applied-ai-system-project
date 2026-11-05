# Architecture

This document describes what is actually implemented in this repository.
Each phase appends to it; nothing here describes future plans.

## Guiding idea

The application has a **deterministic core** and an **intelligent outer layer**.

```
                 ┌──────────────────────────────┐
                 │  Outer layer (interpretive)  │
                 │  LLM, NL parsing, agent, RAG │
                 └──────────────┬───────────────┘
                                │ validated calls only
                 ┌──────────────▼───────────────┐
                 │  Deterministic core (truth)  │
                 │  logic_utils.py, game.py     │
                 └──────────────────────────────┘
```

The core decides what is true: whether a guess is correct, what the score is,
how many attempts are left. The outer layer may interpret language and explain
results, but it can never substitute its own judgement for the core's.

This split is the single most important architectural decision in the project.
It is what makes the system testable without an API key, and what keeps a
model failure from corrupting game state.

## Phase 1 components

### `logic_utils.py` — pure rules

No imports from Streamlit. Every function is a pure function of its arguments,
so each rule can be tested with a plain assert.

| Function | Responsibility |
|---|---|
| `get_difficulty_config(name)` | Look up one difficulty's settings. Raises `ValueError` on unknown names. |
| `get_range_for_difficulty(name)` | `(low, high)` for a difficulty. |
| `get_attempt_limit(name)` | How many valid guesses the player gets. |
| `parse_guess(raw, low, high)` | Validate raw text into an in-range integer. |
| `check_guess(guess, secret)` | The **only** place that decides Win / Too High / Too Low. |
| `update_score(score, outcome, guess_number)` | Apply scoring rules. |

`DIFFICULTY_CONFIGS` is a single source of truth for ranges and attempt limits.
Keeping them in one table is what prevents the range and the limit from
drifting apart — an earlier version of this project had exactly that bug.

### `game.py` — state and transitions

```python
@dataclass(frozen=True)
class GameState:
    difficulty: str
    secret: int
    low: int
    high: int
    max_attempts: int
    attempts_used: int = 0
    score: int = 0
    status: Status = "playing"   # "playing" | "won" | "lost"
    history: tuple[int, ...] = ()
```

`GameState` is **frozen**. A transition returns a new state rather than mutating
a shared object:

| Function | Signature |
|---|---|
| `new_game(difficulty, rng=None)` | `-> GameState` |
| `submit_guess(state, raw)` | `-> GuessResult` |

`GuessResult` carries the new `state` plus either an `error` (input rejected,
state unchanged) or an `outcome` and `hint`.

`new_game` accepts an injected `random.Random` so tests can pin the secret.
This is dependency injection in its simplest useful form.

### `app.py` — Streamlit UI only

`app.py` reads widgets, calls `game.py`, and renders. It holds no rules.

State changes happen in **`on_click` callbacks**, not in the script body.
Streamlit runs callbacks *before* re-running the script, so widgets render from
the already-updated state. Without callbacks the guess box stayed enabled for
one extra run after a win or loss — a real bug this design fixed.

## Data flow for one guess

```
user types "42", clicks Submit
   │
   ▼
on_submit callback (app.py)
   │
   ▼
submit_guess(state, "42")                    [game.py]
   │
   ├── parse_guess("42", low, high)          [logic_utils.py]
   │     └── invalid? -> GuessResult(state unchanged, error=...)   attempt NOT consumed
   │
   ├── check_guess(42, secret)               [logic_utils.py]
   ├── update_score(score, outcome, n)       [logic_utils.py]
   └── returns new frozen GameState
   │
   ▼
session_state.game = new state; script re-runs and renders
```

## Game rules as implemented

- **Difficulties:** Easy 1–20 / 6 attempts, Normal 1–100 / 8, Hard 1–50 / 5.
- **Attempts:** only valid guesses consume one. Invalid text and out-of-range
  numbers are rejected with a message and cost nothing.
- **Win points:** `max(10, 100 - 10 * guess_number)`.
- **Wrong guess:** `-5`, floored so the total never goes below `0`.
- **Game over:** guess box and Submit are disabled; New Game stays enabled.
- **Difficulty change:** starts a fresh game at the new difficulty.
- **Input:** ASCII digits only, at most 18 significant digits, leading zeros OK.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `GAME_DEBUG` | unset | `true` shows the Developer Debug Info panel. Hidden otherwise. |

## Testing strategy

| Layer | Files | What it proves |
|---|---|---|
| Pure rules | `test_parse_guess`, `test_difficulty`, `test_scoring`, `test_game_logic` | Each rule in isolation. |
| State transitions | `test_game_state` | Attempts, status, history, immutability. |
| UI flow | `test_app_flow`, `test_regressions`, `test_phase1_review` | Real Streamlit runs via `AppTest`. |

Bugs found during development each got a regression test written **before** the
fix, and the test was confirmed failing first. That is why `test_regressions.py`
and `test_phase1_review.py` exist as separate files: they document defects, not
just behavior.
