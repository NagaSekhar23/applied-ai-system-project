# Design decisions

Each entry records a decision, the problem it solves, the alternatives that
were considered, and the trade-off accepted. Entries are append-only.

---

## D1 — Deterministic core, interpretive outer layer

**Problem.** An LLM is non-deterministic and can fail or be unavailable. If it
decides whether a guess is correct, the game becomes untestable and unreliable.

**Decision.** Python owns all truth: outcomes, scores, attempts, state. The LLM
may only interpret language and explain results the core already produced.

**Alternatives.**
- *LLM decides outcomes.* Rejected: unreliable, costly, impossible to unit test.
- *LLM with a validation pass.* Rejected: still pays latency and cost for a
  comparison that is one line of Python.

**Trade-off.** The AI features are narrower than a "the model does everything"
demo. In exchange, every rule is testable offline and a provider outage
degrades the app instead of breaking it.

---

## D2 — Frozen `GameState` dataclass

**Problem.** The original code mutated several loose `st.session_state` keys in
different places. Attempts, score and status could disagree with each other.

**Decision.** One frozen dataclass holds all game state. Transitions return a
new instance via `dataclasses.replace`.

**Alternatives.**
- *Plain dict.* Rejected: typos fail silently, no type checking.
- *Mutable class.* Rejected: a transition could half-apply and leave an
  inconsistent state.
- *Pydantic model.* Rejected for the core: no parsing or coercion is needed
  here, and a dataclass has no runtime dependency. Pydantic is used later where
  untrusted external data genuinely needs validation (see D8).

**Trade-off.** Slightly more verbose than mutation. In exchange, state is
always internally consistent and tests assert on whole values.

---

## D3 — Streamlit `on_click` callbacks for state changes

**Problem.** Reading `if st.button(...)` in the script body means the state
change happens *after* widgets have already rendered. The guess box stayed
enabled for one extra run after the game ended.

**Decision.** State changes happen in `on_click` callbacks, which Streamlit runs
before the script body.

**Alternatives.**
- *Call `st.rerun()` after each change.* Works, but costs an extra full rerun
  and makes "show this message once" logic harder.

**Trade-off.** Callbacks are slightly less obvious to read than top-to-bottom
code, so the reasoning is documented in `app.py`'s module docstring.

---

## D4 — Invalid input does not consume an attempt

**Problem.** Typing `abc` used up one of the player's limited guesses.

**Decision.** `submit_guess` returns the *unchanged* state with an error when
parsing fails. Only a valid, in-range guess advances the game.

**Trade-off.** None identified. This matches player expectation.

---

## D5 — Fail loudly on unknown difficulty

**Decision.** `get_difficulty_config` raises `ValueError` rather than falling
back to Normal.

**Rationale.** A silent fallback is how the original "New Game ignores
difficulty" bug hid for so long. A raised error surfaces programming mistakes
immediately. User-facing input is validated separately and never reaches this
function unchecked.

---

## D6 — 18-significant-digit input cap

**Problem.** CPython refuses `int()` on strings longer than 4300 digits and
raises `ValueError`. Pasting a long digit string crashed the app.

**Decision.** Reject anything over 18 significant digits as out of range,
before conversion. Leading zeros are stripped first, so `000...042` still works.

**Alternatives.**
- *Catch the `ValueError`.* Rejected: converting a 4000-digit number just to
  discard it is wasted work.
- *Raise `sys.set_int_max_str_digits`.* Rejected: solves nothing; any value that
  long is out of range anyway.

**Trade-off.** The cap is tied to the assumption that difficulty ranges stay far
below 10^18. Documented in the module.

---

## D7 — ASCII digits only

**Problem.** `\d` in Python regex matches Unicode digits, so `٤٢`
(Arabic-Indic) parsed as 42.

**Decision.** Match `[0-9]` explicitly.

**Trade-off.** Players using non-ASCII numerals must switch keyboards. Accepted
for predictability; the alternative is input that behaves differently depending
on the keyboard in use.

---

## D8 — Score floored at zero

**Decision.** `update_score` returns `max(0, score - 5)` for a wrong guess.

**Rationale.** Product decision by the repository owner. A negative score reads
as punishment rather than feedback.

**Note.** The floor is applied *per penalty*, not to the running total at the
end. A win after two wrong guesses from zero scores `100 - 10*3 = 70`, because
the penalties floored at 0 rather than accumulating to -10.

---

## D9 — Flat project layout

**Decision.** Keep `app.py`, `game.py`, `logic_utils.py` at the repository root
and add focused packages (`evaluation/`, `providers/`, …) only when a phase
genuinely needs one.

**Alternatives.**
- *`src/application/domain/...` hierarchy.* Rejected: for a project this size it
  adds import ceremony without improving navigation.

**Trade-off.** The root directory grows over time. Mitigated by keeping each
package single-purpose and documenting responsibilities in `architecture.md`.
