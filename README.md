# 🎮 Applied AI Engineering System

A deterministic number-guessing game with an AI layer built around it — an
assistant, a natural-language interface, a tool-using agent, a knowledge
assistant over the project's own docs, an evaluation framework and an
observability dashboard.

**The organising idea:** deterministic Python decides everything that matters;
the LLM interprets language and explains results. The model never decides
whether a guess is correct, never mutates state, and never has the last word
over game logic. Every AI feature degrades to a working deterministic path
when the provider is unavailable.

**It runs with no API key.** The default provider is a mock, so the whole
application, its 520 tests and its 174 evaluation cases run offline, free, and
deterministically.

---

## Quick start

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

That's it — mock mode needs no configuration. For a real model:

```bash
cp .env.example .env          # then add your key
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python -m streamlit run app.py
```

---

## What it does

| Page | What it demonstrates |
|---|---|
| **Game** | The deterministic core: pure rules, frozen state, validated input |
| **AI Assistant** | Structured LLM output, validation, fallback, a secret the model never sees |
| **Natural Language** | Intent → typed command → validation → execution, with each stage shown |
| **Agent** | A bounded tool-calling loop over a closed registry, with its trace |
| **Knowledge Assistant** | Hybrid RAG over the project docs, with verified citations and abstention |
| **Observability** | Real recorded metrics: latency, retries, tool usage, cost (when honest) |

---

## Architecture

```
                      Streamlit UI (ui/)
                            │
      ┌─────────────────────┼─────────────────────┐
      │                     │                     │
  assistant.py      natural_language/          agents/
  explain only      parse→validate→execute     bounded tool loop
      │                     │                     │
      └─────────────────────┼─────────────────────┘
                            │ validated calls only
              ┌─────────────▼──────────────┐
              │   DETERMINISTIC CORE       │
              │   game.py, logic_utils.py  │  ← source of truth
              └────────────────────────────┘

  providers/      vendor-neutral LLM interface + mock + retries
  retrieval/      ingest → chunk → hybrid search → rerank → cited answer
  evaluation/     dataset-driven measurement with a CLI
  observability/  structured events, metrics, redaction (one-way dependency)
```

Full detail: [docs/architecture.md](docs/architecture.md).

### The boundary, concretely

```python
# assistant.py — the LLM proposes, deterministic code disposes
outcome, hint = check_guess(guess, secret)       # Python decides

if leaks_secret(model_message, state):           # model disclosed the answer?
    return fallback(...)                         # suppress it
if model_category != true_category:              # model disagreed on outcome?
    category = true_category                     # Python wins, disagreement logged
```

The secret is **structurally** unavailable to the model: `build_facts` omits it
entirely while a game is in progress, so no prompt can leak a value it never
received.

---

## Technology

| | |
|---|---|
| Language | Python 3.12 |
| UI | Streamlit |
| Validation | Pydantic v2 |
| LLM | Anthropic (optional) behind a provider interface; mock by default |
| Retrieval | BM25 + hashing vectoriser + Reciprocal Rank Fusion — no vector DB |
| Testing | pytest, including Streamlit `AppTest` |

Four runtime dependencies. No LangChain, no vector database, no agent
framework — the point is to show the mechanics, not to configure someone
else's.

---

## Configuration

Every variable is optional. With none set, the app runs in mock mode.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` or `anthropic` |
| `LLM_MODEL` | `claude-opus-5-5` | Model id |
| `ANTHROPIC_API_KEY` | — | Required only for `anthropic` |
| `LLM_TIMEOUT_SECONDS` | `20` | Per-request timeout |
| `LLM_MAX_RETRIES` | `2` | Bounded retries |
| `LLM_MAX_TOKENS` | `1024` | Output cap |
| `LLM_EFFORT` | `low` | `low`…`max` |
| `LLM_PRICE_INPUT_PER_MTOK` | — | Enables cost estimates for an unlisted model |
| `LLM_PRICE_OUTPUT_PER_MTOK` | — | As above |
| `GAME_DEBUG` | unset | `true` reveals the debug panel (and the secret) |

See [`.env.example`](.env.example). `.env` is git-ignored; no key is ever
logged, displayed, or written to a report.

---

## Running things

```bash
python -m streamlit run app.py     # the app
python -m pytest                   # 520 tests (includes the evaluation suite), no network
python -m evaluation.run           # 174 evaluation cases across 5 datasets
python -m evaluation.run --quiet   # one line per dataset
```

`evaluation.run` exits non-zero when a case fails, so it works as a CI gate.

### Current measured results

```
520 passed                     (519 + 1 skip if the optional anthropic package is absent)

agent:            27/27  (100%)
game-logic:       56/56  (100%)
llm-assistant:    21/21  (100%)
natural-language: 36/36  (100%)
rag:              34/34  (100%)
```

These are deterministic-correctness results measured in mock mode. They say
the plumbing is right; they do not measure the phrasing quality of a real
model. See [docs/evaluation.md](docs/evaluation.md) for what each metric
can and cannot tell you.

---

## Game rules

- **Difficulty:** Easy 1–20 / 6 attempts · Normal 1–100 / 8 · Hard 1–50 / 5.
  Changing difficulty starts a fresh game.
- **Attempts:** only valid guesses count. Invalid or out-of-range input is
  rejected with a message and costs nothing.
- **Winning:** `max(10, 100 − 10 × guess number)` points.
- **Wrong guess:** −5, floored at 0.
- **Game over:** the input is disabled; New Game stays available.

---

## Documentation

| Document | Contents |
|---|---|
| [architecture.md](docs/architecture.md) | Components, data flow, the deterministic boundary |
| [design-decisions.md](docs/design-decisions.md) | 33 decisions with alternatives and trade-offs |
| [evaluation.md](docs/evaluation.md) | Dataset format, metrics, reproducibility |
| [agent-architecture.md](docs/agent-architecture.md) | Tool registry, the loop, guardrails |
| [rag-architecture.md](docs/rag-architecture.md) | Ingestion, hybrid retrieval, RRF, citations |
| [observability.md](docs/observability.md) | Events, metric formulas, privacy controls |
| [security.md](docs/security.md) | Threat model and controls |
| [interview-preparation.md](docs/interview-preparation.md) | 20 topics with sample answers |

---

## Known limitations

Stated plainly, because a portfolio project that hides its limits is less
convincing than one that names them.

- **The default embedder is lexical, not semantic.** `HashingEmbedder` matches
  word overlap. A paraphrase sharing no vocabulary will miss. The `Embedder`
  protocol exists so a trained model can replace it.
- **The reranker is heuristic, not a trained cross-encoder.**
- **RAG relevance labels are document-level**, so Recall@K and Precision@K are
  an upper bound on true passage relevance.
- **Mock mode cannot measure answer quality.** It exercises validation,
  fallback and citation checking; it says nothing about phrasing.
- **Observability is in-process.** A bounded deque plus optional JSONL — no
  time-series database, no alerting, no cross-process aggregation.
- **Cost figures are estimates** from a dated price table, and are shown only
  when they can be computed honestly. Unpriced calls are reported, not guessed.
- **The corpus is six documents.** RAG metrics on a corpus this small
  demonstrate method, not stable measurement.
- **No persistence.** Game state, events and the index live in memory.
- **No authentication or multi-user support.** It is a local demo.

## Possible next steps

- Swap `HashingEmbedder` for a trained embedding model and re-measure
  Recall@K — the interface is already in place.
- Add an LLM-as-judge evaluation for hint quality, reported under
  `MetricKind.MODEL_QUALITY` so it is never mixed with correctness.
- Persist events to SQLite for metrics across restarts.
- Run the evaluation suite in CI on every push.
