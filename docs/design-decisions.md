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

---

## D10 — Mock provider as the default

**Problem.** A portfolio project that needs a paid API key to start is a
project nobody runs, and a test suite that costs money per run is a test suite
nobody runs.

**Decision.** `LLM_PROVIDER` defaults to `mock`. The mock reads the same
structured facts block the real provider receives and returns schema-valid
output, so validation, fallback, cross-checking and evaluation are all
genuinely exercised offline.

**Alternatives.**
- *Skip LLM tests without a key.* Rejected: the interesting logic (validation,
  fallback, safety) would then be untested in CI.
- *Record/replay real responses.* Rejected: more machinery than this needs, and
  fixtures go stale silently.

**Trade-off.** Mock output cannot show whether phrasing is good. That is stated
explicitly in the evaluation dataset rather than papered over.

---

## D11 — Retries live in the client, not the SDK

**Decision.** `AnthropicProvider` is constructed with `max_retries=0`;
`ReliableLLMClient` owns retrying.

**Rationale.** Retry behavior is then identical across providers, testable
against a scripted provider with an injected `sleep`, and countable — the
retry and timeout counts feed the observability layer in Phase 6. Two layers
of retry would also silently multiply (3 × 3 = 9 real calls).

---

## D12 — Validation failures are retried like transient errors

**Decision.** Output that parses but fails schema validation is retried,
bounded by the same `max_retries`.

**Rationale.** Models occasionally emit a near-miss. One cheap retry often
fixes it, and the bound means a persistently broken response still terminates
and falls back.

**Trade-off.** A deterministic schema bug costs `max_retries` wasted calls
before surfacing. Accepted: `validation_failures` is counted separately in
`CallStats` precisely so that pattern is visible.

---

## D13 — Structural secret redaction, not prompt instruction

**Decision.** The secret is omitted from the facts dict while a game is in
progress, in addition to a prompt restriction and an output leak check.

**Rationale.** A prompt instruction is a request, not a guarantee. Defence in
depth: the model cannot reveal what it never received (structural), is told not
to (prompt), and output is checked anyway (`leaks_secret`) in case a future
change weakens the first two.

**Trade-off.** The assistant cannot give proximity hints like "very close".
Accepted: leaking the answer would break the game.

---

## D14 — `claude-opus-5-5` as the default model

**Decision.** Default to `claude-opus-5-5`, overridable with `LLM_MODEL`, at
`LLM_EFFORT=low`.

**Rationale.** Model choice is a cost decision that belongs to whoever runs the
project, so the default is the current general-purpose model rather than a
quietly cheaper one. Effort defaults to `low` because generating a one-sentence
hint is a simple task and effort is the right lever for that, not model
downgrade. Both are one environment variable away from changing.

---

## D15 — A closed command enum instead of free-form instructions

**Problem.** If model output is executed as an instruction, the model's output
space becomes the application's action space.

**Decision.** The parser emits one of seven `CommandType` values with typed
arguments. The executor is a dispatch table over that enum.

**Alternatives.**
- *Tool/function calling with arbitrary arguments.* Used in Phase 5, where the
  registry provides the same closure property.
- *Natural-language instruction executed by code-gen.* Rejected outright: this
  is the shape of injection vulnerabilities.

**Trade-off.** Adding a capability means adding an enum member, a validator
branch and a handler. That friction is the feature.

---

## D16 — A rule-based parser alongside the LLM one

**Decision.** `rule_based_parse` handles common phrasings deterministically and
is used as the mock behavior, the offline evaluation path, and the fallback.

**Rationale.** It makes the feature free and testable offline, and it degrades
gracefully: a provider outage costs unusual-phrasing support, not the feature.
Writing it also forced an explicit list of the phrasings that must work, which
became evaluation cases.

**Trade-off.** Two parsers to maintain. Accepted because the rule parser is
~60 lines and is pinned by its own tests.

---

## D17 — Validation re-uses the game's own `parse_guess`

**Decision.** `_validate_guess` calls `logic_utils.parse_guess` rather than
range-checking itself.

**Rationale.** Two validators drift. If the conversational path had its own
bounds check, a future change to the typed path would silently create a hole
where chat accepts what the UI rejects.

---

## D18 — The difficulty control follows game state

**Problem.** The sidebar selectbox and `GameState.difficulty` were two sources
of truth. Starting a Hard game from the chat page set the state correctly, and
the next run's stale "Normal" selectbox immediately replaced it with a new
Normal game. Caught by an AppTest, not by reasoning.

**Decision.** The selectbox initialises its index from the current game's
difficulty, so it reflects state rather than competing with it.

**Alternatives.**
- *Write the widget's `key` in session state from the chat handler.* Rejected:
  Streamlit restricts mutating a widget key after instantiation, so it would
  work only from a callback and would be fragile.

**Trade-off.** The selectbox is now derived state. That is the correct
direction of flow here: one owner of difficulty, which is the game.

---

## D19 — Tools as a registry, not a dispatch on names

**Decision.** `ToolRegistry` holds explicitly constructed `Tool` objects.
Lookup is a dict access that raises `UnknownToolError` on a miss.

**Alternatives.**
- *`getattr(module, tool_name)`.* Rejected: the model's output space becomes
  the module's namespace.
- *Decorator-based auto-registration.* Rejected as implicit — the point is
  that the reachable set is visible in one list.

**Trade-off.** Registering a tool is ~10 lines of boilerplate. That is the
cost of being able to state exactly what the agent can do.

---

## D20 — Output schemas as well as input schemas

**Decision.** Each tool declares `output_model`, and `validate_output` runs
after every successful call.

**Rationale.** Input validation protects the app from the model. Output
validation protects the agent from the app: if a tool drifts from its
contract, that surfaces as a clear schema error rather than as confusing
downstream behavior.

**Trade-off.** A little duplication between the tool's return and its schema.
Worth it — it caught contract mistakes while this phase was being written.

---

## D21 — Recoverable tool errors, terminal guard breaches

**Decision.** A failed tool call is fed back to the planner as an observation
and the loop continues. Guard breaches end the run.

**Rationale.** Self-correction is the main practical benefit of an agent loop:
"that guess was out of range" is enough for the next decision to be better.
But self-correction must not become an excuse to never stop, so the error
budget is itself bounded.

---

## D22 — Repeated-call protection by argument signature

**Decision.** `tool:{sorted json of arguments}` is recorded; an exact repeat
ends the run.

**Rationale.** The common agent failure is calling the same tool with the same
arguments forever. Keying on arguments rather than the tool name alone still
allows legitimate repeats — two different guesses — while catching the loop.

**Trade-off.** A genuine identical repeat (checking status twice) is refused.
Acceptable, and `allow_repeat_calls` exists for tests that need it.

---

## D23 — Trace structure built in Phase 5, consumed in Phase 6

**Decision.** `Span`/`AgentTrace` with parent ids and durations exist now,
before there is a dashboard to read them.

**Rationale.** Retro-fitting tracing means touching every call site. Building
the structure while writing the loop costs almost nothing; adding it later
would be a refactor of the thing most at risk of regression.

---

## D24 — Observability depends on services, never the reverse

**Problem.** The usual way instrumentation rots a codebase is logging calls
scattered through business logic, which then cannot be tested or reused
without the logging stack.

**Decision.** Services return data (`CallStats`, `AgentTrace`,
`AssistantReply`); the UI passes that data to the tracker. No business-logic
module imports `observability`.

**Alternatives.**
- *Decorators on service functions.* Rejected: couples every service to the
  tracker and makes the pure functions impure to test.
- *A logger injected into every constructor.* Rejected as ceremony for a
  project this size.

**Trade-off.** Call sites must remember to record. Acceptable because there are
only three (the three AI pages), and the AppTest suite asserts that each one
produces events.

---

## D25 — `contextvars` for request correlation

**Decision.** `request_id` and `trace_id` live in context variables set by
`tracker.request(...)`.

**Rationale.** Correlation without threading an id through every signature.
Streamlit may run callbacks on other threads, and `contextvars` are
thread-local-ish by design, so each request's id stays its own.

**Trade-off.** Implicit state, which is normally worth avoiding. Contained by
resetting the token in a `finally`, with a test asserting the id is cleared.

---

## D26 — Refuse to estimate cost rather than guess

**Decision.** `estimate_cost` returns `None` plus a human-readable `reason`
for mock providers, missing usage, or unpriced models.

**Rationale.** A cost figure is used to make decisions. A number inferred from
a "similar" model, or from character counts, looks identical to a real one and
is wrong in a way nobody notices. The dashboard shows how many calls were
excluded so a partial total is never read as complete.

**Trade-off.** Mock-mode runs show no cost at all. Correct: nothing was spent.

---

## D27 — `degraded` as a third status

**Decision.** Status is `ok`, `error` or `degraded`, and `degraded` counts as
success in `success_rate` while also being reported on its own.

**Rationale.** A fallback hint is a *successful* user outcome and a
*concerning* system signal. Collapsing it into either bucket loses one of
those. Keeping `fallback_rate` separate means a provider outage shows up as a
rising fallback rate rather than a flat success rate hiding the problem.

---

## D28 — A bounded store, with evictions counted

**Decision.** `EventStore` is a `deque(maxlen=500)`, and `dropped_count` is
shown in the UI.

**Rationale.** An unbounded list in a long-lived Streamlit session is a slow
memory leak. Showing the eviction count means the window is visible rather
than being a silent reason the numbers look wrong.
