# Interview preparation

Twenty topics this project actually demonstrates. Each has a plain
explanation, the technical version, the concrete example from this codebase, a
likely question, and a sample answer.

**Two ground rules.** Only claim what the repository does — every example here
is a real file you can open. And when a sample answer names a limitation, keep
it: naming the limits of your own work is the single strongest signal in a
technical interview.

---

## 1. Why the architecture is shaped this way

**Simple.** The Python code decides what is true. The AI only explains and
interprets.

**Technical.** A deterministic core (`game.py`, `logic_utils.py`) owns all
state transitions and outcomes. An interpretive outer layer (assistant,
natural-language parser, agent, RAG) may read redacted state and produce text,
but reaches the core only through validated interfaces.

**In this project.** `check_guess` is the only function that decides Win /
Too High / Too Low. The assistant receives that outcome as an input and
explains it.

**Question.** *"Walk me through your architecture."*

**Answer.** "There's a deterministic core and an interpretive outer layer. The
core — the game rules and state machine — decides every outcome. The LLM layer
explains results, parses language and picks tools, but never decides anything.
Concretely: `check_guess` returns Too High, and the assistant's job is to turn
that into a sentence. If the model claims a different outcome, my code
overrides it and logs the disagreement. That boundary is why the whole system
is testable without an API key — 520 tests, no network — and why a provider
outage degrades the product instead of breaking it."

---

## 2. Why deterministic logic stays separate from the LLM

**Simple.** Models are non-deterministic and sometimes unavailable. Game rules
are neither.

**Technical.** Putting a non-deterministic component on the critical path for
a decision that is one line of Python costs latency, money and testability,
and buys nothing.

**In this project.** `Assistant.hint` cross-checks the model's
`hint_category` against the deterministic outcome and overrides on
disagreement, recording it in `reply.warnings`.

**Question.** *"Why not let the model handle the game logic too?"*

**Answer.** "Three reasons. Correctness: comparing two integers is exact in
Python and probabilistic in a model. Testability: I can assert on
`check_guess` in a unit test; I can't assert on a sampled token sequence.
Availability: if the provider is down, the game still works — only the
explanations fall back. There's a test where the model is scripted to claim a
win on a losing guess, and the assertion is that the user still sees the
correct outcome."

---

## 3. Provider abstraction

**Simple.** Application code never imports the vendor's SDK.

**Technical.** An `LLMProvider` protocol with one method,
`complete(LLMRequest) -> LLMResponse`. Vendor exceptions are translated into a
local error taxonomy. The SDK is imported lazily inside the provider's
constructor.

**In this project.** `providers/base.py` defines the protocol;
`anthropic_provider.py` is the only file that imports `anthropic`, inside
`__init__`, so the package is optional and the whole suite runs without it.

**Question.** *"How hard would it be to switch providers?"*

**Answer.** "One new class. Everything upstream depends on `LLMProvider`, not
on a vendor type, and vendor exceptions are mapped to my own — `LLMTimeoutError`,
`LLMRateLimitError` and so on — so my retry logic and UI never branch on a
vendor class. The SDK import is lazy, which is also what lets the test suite
run with the package uninstalled. The mock provider is just another
implementation of the same protocol."

---

## 4. Structured output validation

**Simple.** Never trust what the model returns. Parse it, check it, and have a
plan when it's wrong.

**Technical.** Request a JSON schema; parse tolerantly (the model may wrap it
in prose or fences); validate with Pydantic; treat a validation failure as a
retryable error, bounded, then fall back.

**In this project.** `AssistantHint` in `schemas.py`, validated in
`ReliableLLMClient.complete_structured`, which counts `validation_failures`
separately from transport errors.

**Question.** *"What happens if the model returns malformed JSON?"*

**Answer.** "It's caught in three stages. `extract_json` handles the common
case of valid JSON wrapped in prose or a code fence. If it still won't parse,
or parses but fails Pydantic validation, that counts as a validation failure
and is retried — models do emit near-misses, and one retry often fixes it.
Retries are bounded by the same `max_retries`, and after that the assistant
falls back to a deterministic hint. I count validation failures separately
from timeouts, because a persistent validation failure means my schema or
prompt is wrong, while a timeout means the network is."

---

## 5. Natural-language intent parsing

**Simple.** Turn "I think it's 45" into a typed command the program can check.

**Technical.** `message → CommandParser → Command → validate → executor`. The
parser only interprets; a separate validator decides whether the command is
permitted; the executor dispatches over a closed enum.

**In this project.** `natural_language/`. The validator re-uses the game's own
`parse_guess`, so the chat path can't accept a guess the typed UI would reject.

**Question.** *"How do you stop the model from doing something unintended?"*

**Answer.** "The model's output space is a seven-value enum, not a string I
execute. If it emits `DELETE_EVERYTHING`, Pydantic rejects it and the parser
falls back to the rule-based one — there's a test for exactly that. Then
validation runs separately: is there a game, is it still playing, is the guess
in range. The range check calls the same `parse_guess` the UI uses, so there's
no second implementation to drift. Three gates, all of which must pass before
state changes."

---

## 6. Tool registration and execution

**Simple.** The agent has a fixed menu. It cannot order off-menu.

**Technical.** Each `Tool` has a name, description, Pydantic input schema with
`extra="forbid"`, output schema, and an execution function delegating to an
existing service. The registry raises `UnknownToolError` on anything else.

**In this project.** `agents/tools.py`, seven tools. Output is validated too,
so a tool drifting from its contract surfaces as a schema error.

**Question.** *"How does the agent decide what it's allowed to do?"*

**Answer.** "It doesn't decide — the registry does. Lookup is a dict access
that raises on a miss. There's no `getattr`, no string-to-function dispatch,
no `eval`, no shell access anywhere in the agents package. Arguments are
validated against a Pydantic model with `extra='forbid'`, so a tool can't be
passed anything it didn't declare. I also validate tool *output* against a
schema, which caught two contract mistakes while I was writing the phase."

---

## 7. Controlling the agent loop

**Simple.** Give the agent a step limit, a clock, and a rule against repeating
itself.

**Technical.** Four terminal guards — `max_iterations`, wall-clock timeout,
repeated-call detection by `tool + sorted(arguments)` signature, and a tool
error budget. Individual tool errors are *recoverable*: fed back as
observations so the agent can self-correct.

**In this project.** `agents/controller.py`, with a test for each guard.

**Question.** *"How do you prevent infinite agent loops?"*

**Answer.** "Four independent guards, because any one can be defeated. An
iteration ceiling, a wall-clock timeout checked before each pass, repeated-call
detection keyed on the tool *and* its arguments, and an error budget. The
distinction that matters is recoverable versus terminal: one bad tool call is
useful feedback — 'that guess was out of range' lets the next decision be
better — but a *pattern* of bad calls is terminal. The repeat check keys on
arguments rather than just the tool name, so two different guesses are fine
but the same guess twice stops the run."

---

## 8. Designing evaluation datasets

**Simple.** Write down what good behavior looks like, as data, before you
measure it.

**Technical.** JSON cases with id, category, target executor, input, initial
state, expected output and a plain-English `expected_behavior`. Validated by
Pydantic before any case runs. Comparison is a *subset* match, so a case
asserts only what it's about.

**In this project.** 174 cases across five datasets. Cases tagged `regression`
reproduce defects that were actually found and fixed here.

**Question.** *"How do you know your AI system works?"*

**Answer.** "Datasets, not vibes. Each case has an input, an initial state and
an expected output, plus a sentence saying what behavior it's protecting.
Comparison is a subset match — a case asserts the one thing it cares about, so
adding a field doesn't break fifty cases. I built the framework in phase 2
against purely deterministic behavior, before any LLM existed, so the
measurement machinery was trustworthy before I pointed it at anything
non-deterministic. And it earns its keep: it found a retrieval bug my unit
tests missed."

---

## 9. Calculating evaluation metrics

**Simple.** Count what passed, group it usefully, and be clear what the number
does not mean.

**Technical.** `pass_rate = passed / total`; `failure_rate = (failed +
errored) / total`, with failures and errors counted separately. Every metric
block carries a `MetricKind` — `deterministic_correctness`, `model_quality`,
or `retrieval_quality` — so they are never averaged together.

**In this project.** `evaluation/metrics.py`. Each run emits a fingerprint, a
SHA-256 over sorted `(case_id, status)` pairs, excluding timing.

**Question.** *"Your evaluation says 100%. What does that actually mean?"*

**Answer.** "Less than it sounds, and I'd say so unprompted. It means the
implementation matches my dataset on a mock provider. It says nothing about
scenarios the dataset doesn't cover, and nothing about whether a real model's
phrasing is good. That's why every metric block is tagged with what it
measures — I never average a deterministic pass rate with a subjective quality
score, because the result would mean nothing. For real model quality I'd need
an LLM-as-judge pass with a rubric, reported separately."

---

## 10. Observability

**Simple.** Record what happened so you can answer "why was that slow?"
afterwards.

**Technical.** One `Event` shape for every component. `request_id` and
`trace_id` in `contextvars`, so correlation needs no parameter threading.
Metrics computed from recorded events; nothing hardcoded.

**In this project.** `observability/`. No business-logic module imports it —
services return data, the caller records it.

**Question.** *"How would you debug a slow request in production?"*

**Answer.** "Every event carries a request id, so I'd pull all the events for
one request and look at the per-stage durations — in the agent case that's
each planning decision and each tool call, linked by trace id with parent
spans. The dashboard groups by request id and shows exactly that. The honest
caveat: this is in-process, a bounded deque plus JSONL. Real production would
ship these to a time-series backend, and the `Event` shape is deliberately
flat so that's an exporter, not a rewrite."

---

## 11. Hybrid retrieval

**Simple.** Two search methods that fail differently, combined.

**Technical.** BM25 for exact terms, a vector index for fuzzier overlap, fused
by rank.

**In this project.** `retrieval/keyword_search.py` and `embeddings.py`, fused
in `hybrid.py`.

**Question.** *"Why not just use embeddings?"*

**Answer.** "Because they fail differently, which is the whole argument. BM25
nails exact technical terms — a query for `GAME_DEBUG` pins the one chunk
containing it, because rare terms get high IDF. Vector search handles looser
overlap. Running both and fusing means a weakness in one is covered by the
other. I should be precise about my default embedder though: it's a hashing
vectoriser over word n-grams, which is lexical, not semantic. I labelled it
`semantic = False` in code and said so in the docs, because calling it
semantic retrieval would be the easiest thing in this project to overclaim."

---

## 12. Reciprocal Rank Fusion

**Simple.** Combine two ranked lists by position, not by score.

**Technical.** `RRF(d) = Σ 1 / (k + rank_r(d))`, `k = 60`.

**In this project.** `retrieval/hybrid.py`, with a test asserting a huge BM25
score cannot dominate a bounded cosine score.

**Question.** *"Why RRF instead of just adding the scores?"*

**Answer.** "Because the scores aren't comparable. BM25 is unbounded and
corpus-dependent — scores of 5 or 50 depending on the corpus — while cosine is
bounded in zero to one. Add them and BM25 always wins, not because it's more
relevant but because its numbers are bigger. Normalising first helps but is
unstable on a handful of results. RRF throws away magnitude and uses only
ordering, so no normalisation is needed. The nice property is that agreement
wins: a chunk both retrievers ranked second beats one that was first in only
one list. `k` damps the top ranks — larger `k` favours broad agreement."

---

## 13. Citation validation

**Simple.** Check that every source the model cites is one it was actually
given.

**Technical.** Evidence is numbered `[1]`, `[2]`; the model cites indexes, not
names. `validate_citations` resolves each marker against the retrieved list. An
out-of-range marker invalidates the answer.

**In this project.** `retrieval/citations.py`. `format_evidence` and
`validate_citations` are two halves of one contract.

**Question.** *"How do you stop the model inventing sources?"*

**Answer.** "By making citation an index rather than a name. The model gets
numbered passages and cites `[2]`. A number either refers to something I
retrieved or it doesn't — that's mechanically checkable, unlike a free-form
source name which I'd have to fuzzy-match. If any marker is out of range, I
reject the whole answer and abstain rather than showing it with a broken
footnote. There's a test where the model is scripted to cite `[9]` out of four
passages, and the assertion is that the user sees an abstention."

---

## 14. Reducing hallucination risk

**Simple.** Give the model only what it should know, check what it says, and
have a safe default.

**Technical.** Structural redaction, schema validation, output cross-checks,
citation validation, evidence thresholds and deterministic fallbacks.

**In this project.** The clearest case: the game's secret is **omitted from
the facts dict** while a game is in progress.

**Question.** *"How do you handle hallucination?"*

**Answer.** "Layered, and the most effective layer is structural rather than
corrective. The model can't reveal the secret number because it never receives
it — `build_facts` omits the key entirely while the game is live. Then
defence in depth: the prompt forbids it, and I scan the output for the number
anyway in case a future change weakens the first two. The same pattern in RAG:
abstain before calling the model if no passage clears the evidence threshold,
so there's no opportunity for an unsupported answer rather than a correction
afterwards."

---

## 15. Abstention

**Simple.** Saying "I don't know" is a correct answer, not a failure.

**Technical.** Three gates: an evidence-score threshold checked *before*
calling the model; schema validation including the model's own
`sufficient_evidence` flag; citation validation.

**In this project.** `retrieval/answering.py`. Six evaluation cases cover
abstention, including correct abstention on off-topic questions.

**Question.** *"What happens when your RAG system can't answer?"*

**Answer.** "It says so, and the first gate fires before any model call — if
nothing clears the evidence threshold I abstain immediately, which is cheaper
and removes the chance of an unsupported answer entirely. The model can also
set `sufficient_evidence: false` and I honour it. I treat abstention as a
first-class outcome with its own evaluation category, because a confident
wrong answer about your own architecture is much worse than 'I couldn't find
that.' The threshold is a tuned constant and I'd flag that as the crude part —
it's corpus-dependent."

---

## 16. Managing latency and cost

**Simple.** Measure both, pick the cheapest setting that works, never guess a
cost.

**Technical.** Per-call latency and token usage recorded; cost computed only
from a configured price table and reported usage. Effort defaults to `low` for
simple tasks.

**In this project.** `observability/pricing.py` returns `usd = None` with a
reason for mock providers, missing usage, or unpriced models.

**Question.** *"How do you manage LLM costs?"*

**Answer.** "First by measuring: token usage and latency are recorded per call
and rolled up. Then by using the cheapest lever that holds quality — effort
level before model downgrade, since generating a one-sentence hint doesn't
need deep reasoning. The part I'd emphasise is that my cost figure refuses to
lie: if the provider didn't report usage, or the model isn't in my price
table, it returns null with a reason instead of estimating. The dashboard
shows how many calls were excluded, so a partial total is never mistaken for a
complete one. An invented cost looks identical to a real one and that's how
bad decisions get made."

---

## 17. Handling provider failures

**Simple.** Retry the things worth retrying, give up quickly on the rest, and
always have a fallback.

**Technical.** Retryable (rate limits, timeouts, 5xx) versus non-retryable
(auth, bad request). Exponential backoff honouring `retry-after`. Bounded
retries in the client, with the SDK's own retries disabled so they can't
multiply.

**In this project.** `providers/client.py`, tested against a scripted provider
with an injected `sleep`.

**Question.** *"What happens when the API goes down?"*

**Answer.** "The game keeps working; only explanations degrade. Retries are
bounded and classified — a rate limit is worth retrying and I honour
`retry-after`, but an auth error means the same request fails the same way, so
retrying just burns time. Importantly I disabled the SDK's own retries and own
that logic in one place, otherwise two layers multiply: three by three is nine
real calls. After exhaustion the assistant returns a deterministic hint, the
UI labels it as a fallback, and the event is recorded as `degraded` rather
than `error` — because the user got a usable answer, but I still want the
fallback rate visible."

---

## 18. Scaling this system

**Simple.** Name what breaks first.

**Technical.** Current bottlenecks: in-memory indexes rebuilt per process,
in-process event store, Streamlit session state, synchronous LLM calls.

**Question.** *"How would this scale to a thousand users?"*

**Answer.** "It wouldn't, as written, and I can tell you exactly where it
breaks. Game state lives in Streamlit session state, so it's single-process —
that'd move to Redis or a database. The RAG index is rebuilt in memory at
startup; at scale it's a real vector store built offline. The event store is a
bounded in-process deque; that becomes an exporter to a time-series backend,
and the `Event` shape is flat precisely so that's an exporter rather than a
rewrite. LLM calls are synchronous, so I'd move to async with a connection
pool. The parts that *would* survive are the deterministic core, the provider
abstraction, the tool registry and the evaluation suite — they're all
stateless."

---

## 19. Trade-offs made

**Question.** *"What would you do differently?"*

**Answer.** "Four I'd call out. First, the lexical embedder — a trained model
would genuinely improve retrieval, and I built the `Embedder` protocol so
that's a swap, but today paraphrase without word overlap misses. Second,
document-level relevance labels in the RAG evaluation: maintainable, but an
upper bound on true passage relevance, and I say so in the metric docstring.
Third, I wrote a rule-based parser *and* an LLM parser — more code, but it
makes the feature free, offline-testable, and degradable, and writing it
forced me to enumerate the phrasings that must work, which became evaluation
cases. Fourth, no persistence anywhere; deliberate scope control, but it's the
first thing a real deployment needs."

---

## 20. Production readiness

**Question.** *"What would you add before shipping this?"*

**Answer.** "In order: persistence for game state and events, because
everything is in-memory today. Then CI running the test suite and the
evaluation suite on every push — the evaluation CLI already exits non-zero, so
that's configuration, not code. Then an exporter from my event store to a real
observability backend. Then authentication, which doesn't exist at all. Then
an LLM-as-judge evaluation for hint quality, reported under `MODEL_QUALITY` so
it's never mixed with correctness. And I'd swap the embedder and re-measure
Recall@K before claiming the RAG system is good — right now I can only claim
the pipeline is correct, which isn't the same thing."

---

## Things to say, and things not to

**Worth saying:**
- "The model proposes, deterministic code disposes."
- "Evaluation found a bug my unit tests missed." (The tokenizer bug — a real,
  specific story with a root cause and a fix.)
- "I labelled the embedder non-semantic because calling it semantic would be
  overclaiming."
- "Abstention is a feature, not a failure."
- "That metric is an upper bound, and here's why."

**Not worth saying:**
- Anything about production traffic, uptime or users. There are none.
- "100% evaluation pass rate" without immediately explaining what it measures.
- "Semantic search" for the default embedder.
- That the heuristic reranker is comparable to a cross-encoder.

The strongest thing this project demonstrates is not that every number is
green. It's that the numbers are defined precisely enough to be worth
believing, and that the limitations are written down before anyone asks.
