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

---

## Phase 3 components: the LLM layer

```
config.py        Settings from environment variables. Never holds defaults
                 that require credentials.
schemas.py       Pydantic models for validated model output.
prompts/         Versioned prompt templates. Data, not logic.
providers/
  base.py        LLMProvider protocol, request/response types, error taxonomy.
  mock.py        MockProvider + ScriptedProvider. No network, no key.
  anthropic_provider.py  The real Claude API, SDK imported lazily.
  client.py      ReliableLLMClient: bounded retries, structured validation.
  __init__.py    build_provider() / build_client() factories.
assistant.py     The boundary service. Enforces the three guarantees below.
ui/              Streamlit pages: game.py, assistant_page.py.
```

### The provider abstraction

Application code imports from `providers`, never from `anthropic`. A provider
implements one method:

```python
def complete(self, request: LLMRequest) -> LLMResponse: ...
```

Vendor exceptions are translated into this project's own error taxonomy
(`LLMTimeoutError`, `LLMRateLimitError`, `LLMAuthError`, …), so retry logic and
the UI never branch on a vendor class. Adding a provider means adding one class.

The `anthropic` SDK is imported *inside* the constructor, so the package is
optional and the entire test suite runs without it.

### Request path for a hint

```
Assistant.hint(state, last_outcome)
   │
   ├── build_facts(state)          redacted: no secret while status == "playing"
   ├── get_prompt("hint")          versioned template
   ├── ReliableLLMClient.complete_structured(request, AssistantHint)
   │      ├── provider.complete()         bounded retries on transient errors
   │      ├── extract_json()              tolerates fences and surrounding prose
   │      └── AssistantHint.model_validate()   retried on near-miss output
   │
   ├── leaks_secret(message, state)       -> fallback if the secret appears
   ├── category cross-check               -> deterministic outcome wins
   └── AssistantReply(source="llm" | "fallback")
```

### Three enforced guarantees

1. **The secret never enters a prompt mid-game.** `build_facts` omits the key
   entirely while `status == "playing"`. This is structural: a prompt cannot
   leak a value it never received.
2. **The model never decides an outcome.** `check_guess` decides. The model's
   `hint_category` is compared against it and overridden on disagreement, with
   the correction recorded in `AssistantReply.warnings`.
3. **The game always works.** Provider failure, timeout, retry exhaustion,
   malformed JSON, schema violation and secret leakage all fall back to a
   deterministic hint. `source` is shown in the UI so the user can tell which
   they are reading.

### Reliability

| Failure | Handling |
|---|---|
| Timeout | Retried, bounded by `LLM_MAX_RETRIES`, exponential backoff |
| Rate limit | Retried, honouring the provider's `retry-after` when present |
| 5xx / connection | Retried |
| Auth / bad request | **Not** retried — the same request fails the same way |
| Malformed JSON | Retried, then fallback |
| Schema violation | Retried, then fallback |
| Model refusal | Translated to a bad request, then fallback |

Token usage is reported as `None` when the provider does not supply it. It is
never estimated for a real provider, because a fabricated count would corrupt
cost reporting. The mock provider labels its counts as estimates.
