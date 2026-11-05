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

---

## D29 — A lexical embedder, honestly labelled

**Problem.** Anthropic does not provide an embeddings endpoint, and a trained
embedding model means another vendor, another key, and a paid dependency for
every test run.

**Decision.** Ship `HashingEmbedder`, a hashing vectoriser over word n-grams,
behind an `Embedder` protocol. Label it `semantic = False`, and say plainly in
the code, docs and UI that it matches lexically, not semantically.

**Alternatives.**
- *Call it "semantic retrieval" anyway.* Rejected. It is the single easiest
  thing to overclaim in a RAG project and the easiest for an interviewer to
  catch.
- *Require a paid embeddings key.* Rejected: the project must run offline.

**Trade-off.** Paraphrase without word overlap will miss. That limitation is
documented and is visible in the evaluation results rather than hidden.

---

## D30 — Fusion on rank (RRF), not on score

**Decision.** Combine BM25 and cosine results with Reciprocal Rank Fusion,
`k = 60`.

**Rationale.** BM25 scores are unbounded and corpus-dependent; cosine is
bounded in [0, 1]. Summing them lets the larger scale win silently, and
min-max normalising is unstable on a handful of results. RRF uses only
ordering, so no normalisation is needed and a chunk both retrievers liked
beats one that only one retriever ranked first.

**Trade-off.** Magnitude information is discarded — a runaway-best match
ranked first scores the same as any other first place.

---

## D31 — Document-level relevance labels, stated as such

**Decision.** RAG evaluation labels relevance by source document, not by
passage.

**Rationale.** Passage-level ground truth over ~100 chunks would mean hand-
labelling every chunk against every query, and would need re-labelling each
time the docs change. Document-level labels are maintainable and still catch
the failure that matters most: retrieving from entirely the wrong document.

**Trade-off.** Recall and precision are an *upper bound* on true passage
relevance — a chunk from the right document about the wrong topic counts as a
hit. The metric docstring and the architecture doc both say so.

---

## D32 — Abstention before generation, not after

**Decision.** If no retrieved passage clears `MIN_EVIDENCE_SCORE`, abstain
without calling the model at all.

**Rationale.** Cheaper, faster, and it removes the opportunity for an
unsupported answer entirely rather than relying on the model to decline. Two
further gates (schema, citations) catch what gets past the first.

**Trade-off.** A fixed threshold is crude and corpus-dependent. It is a named
constant with the tuning trade-off documented: raising it trades recall for
safety.

---

## D33 — The tokenizer emits identifiers *and* their parts

**Problem.** `[a-z0-9_]+` made `p95_latency_ms` a single token, so the query
"how is P95 latency calculated" matched nothing. The evaluation case
`retrieve-observability-metrics` failed and exposed it.

**Decision.** Emit the whole token and its underscore-separated parts.

**Rationale.** Keeps BM25's advantage on exact identifiers (high-IDF rare
term) while letting natural-language queries match the same chunk.

**Note.** This is the clearest example in the project of evaluation earning
its keep: the bug was invisible in unit tests, which tested the tokenizer's
stated behavior rather than whether retrieval actually worked.


---

## D34 — Stopwords removed before embedding, kept for BM25

**Problem.** Rewriting the README changed the RAG corpus and two retrieval
evaluation cases regressed. Root cause: the hashing embedder counted function
words, so "What chunking strategy is used for documents?" matched generic
prose more strongly than the section titled "Chunking strategy" — which did
not reach the candidate set at all.

**Decision.** Filter stopwords in the embedding path. Keep them for BM25,
where IDF already discounts them and dropping them would lose literal-phrase
matching.

**Rationale.** In a count-based vector, high-frequency function words carry no
topical signal but plenty of magnitude. Removing them is standard lexical-
retrieval practice.

**Note.** The regression was caught by the evaluation suite, not by any unit
test — the unit tests asserted that the retrievers behaved as specified, which
they did. Only a dataset measuring end-to-end retrieval quality could notice
that *adding documentation* had degraded it.

---

## D35 — Documenting a retrieval limitation instead of tuning it away

**Problem.** After the stopword fix, two cases still failed: queries where the
right document ranks second because another chunk *mentions* the same term.

**Decision.** Correct the expectations to what the system genuinely delivers
(the document is found, at rank 2), pin that rank so a further regression
still fails, tag the cases `known-limitation`, and document the weakness.

**Alternatives.**
- *Raise the reranker's title weight until both pass.* Rejected: it would not
  have fixed the P95 case anyway (no title overlap), and tuning weights until
  two named queries pass is overfitting to the test set.
- *Leave the suite failing.* Rejected: a permanently red suite stops being a
  usable CI gate, and the information would be in a failure message rather
  than in the dataset.

**Rationale.** The honest statement is "hybrid lexical retrieval finds the
right document but cannot always rank it first", not "retrieval is perfect".
Distinguishing a passage that mentions a term from one that explains it needs
semantics, which is precisely what the lexical embedder lacks. The fix is a
trained embedding model, and the `Embedder` protocol is already in place.

---

## D36 — Rank-sensitive evaluation runs against a frozen corpus

**Problem.** D35 relaxed two retrieval cases after the README rewrite moved
their rankings. That fix did not hold: the very commit that documented the
limitation added a design-decision entry mentioning `p95_latency_ms` and
`chunking strategy`, which shifted the rankings again and re-failed the cases.

Twice is a pattern, not an accident. Retrieval rank is a property of the
*whole corpus*. The live corpus is this project's documentation, which changes
on most commits, so any rank assertion against it is a tripwire on unrelated
edits.

**Decision.** Split the RAG dataset across two corpora:

- Cases asserting only robust properties ("the relevant document was
  retrieved") run against the **live** docs.
- Cases asserting **rank** (`reciprocal_rank`, `top_is_relevant`) run against
  a **frozen fixture corpus** in `evaluation/fixtures/corpus/`, four small
  documents marked do-not-edit.

**Alternatives.**
- *Keep relaxing expectations after each break.* Rejected: the measurement
  would erode until it measured nothing.
- *Stop asserting rank anywhere.* Rejected: MRR and Precision@K are the
  metrics that distinguish good retrieval from retrieval that merely returns
  the right document somewhere in the top k.
- *Tune the reranker until the live cases pass.* Rejected as overfitting to
  named queries, and it would not have fixed the P95 case anyway.

**Rationale.** Ground truth needs a stable substrate. This is why retrieval
benchmarks ship fixed corpora rather than pointing at a living wiki.

**Verification.** `test_rank_metrics_survive_an_edit_to_the_live_corpus` pins
it, and appending an adversarial paragraph (mentioning chunking, p95, RRF and
BM25) to the live docs now leaves every fingerprint unchanged.
