# Observability

## Scope, stated honestly

This is **local, in-process observability**: a bounded event store in the
running Python process, optionally mirrored to a JSONL file. It is not a
production monitoring stack, there is no time-series database, no alerting and
no cross-process aggregation. Everything the dashboard shows comes from events
this session recorded.

That constraint is deliberate. The point is to demonstrate the *discipline* —
request correlation, structured events, honest metrics, redaction — without
pretending to operate infrastructure this project does not have.

## Architecture

```
observability/
  redaction.py  scrub secrets and minimise user content (runs on write)
  events.py     the Event record; Component / Status enums; TokenCost
  pricing.py    configured price table -> cost estimate, or None with a reason
  store.py      bounded thread-safe deque + optional JSONL
  metrics.py    rates, latency percentiles, per-component rollups
  tracker.py    request/trace context (contextvars) + the recording API
ui/observability_page.py  the dashboard
```

**Dependency direction is one-way.** No module in `game.py`,
`logic_utils.py`, `assistant.py`, `natural_language/` or `agents/` imports
`observability`. Those services *return* data (`CallStats`, `AgentTrace`,
`AssistantReply`); the UI hands that data to the tracker. Observability can be
removed entirely without touching business logic.

## Request correlation

`request_id` and `trace_id` live in `contextvars`, so everything recorded
inside `with tracker.request(...)` shares an id without threading a parameter
through every signature.

```
tracker.request("agent.request")   -> request_id
  └── agent.run()                  -> trace_id from AgentTrace
        ├── tool.start_game        parent_id = span parent
        ├── tool.explain_rules
        ├── agent.plan (LLM)
        └── agent.run              iterations, stop_reason, tools_called
```

The dashboard groups by `request_id` and links tool events by `trace_id`.

## Metrics and their formulas

| Metric | Formula | What it cannot tell you |
|---|---|---|
| `success_rate` | ok / total, where ok includes `degraded` | A degraded request succeeded via fallback — good for users, worth watching separately |
| `failure_rate` | errors / total | — |
| `avg_latency_ms` | mean `duration_ms` | Hides bimodal distributions |
| `p95_latency_ms` | nearest-rank 95th percentile | Meaningless below ~20 samples; the UI says so |
| `llm_failure_rate` | failed LLM events / LLM events | Counts failed *calls*, after the client's own retries |
| `retry_rate` | retries / LLM calls | Per call, so it can exceed 1.0 |
| `tool_failure_rate` | failed tool events / tool events | A failure the agent recovered from still counts |
| `fallback_rate` | degraded / total | — |
| `avg_agent_iterations` | total iterations / agent runs | — |
| `evaluation_pass_rate` | read from `reports/latest-*.json` | Produced by `python -m evaluation.run`, not computed here |

Nearest-rank is used for percentiles so the reported value is one that
actually occurred, rather than an interpolation between two samples.

## Cost: only when it can be honest

`estimate_cost` returns `usd = None`, with a `reason`, whenever:

- the provider is `mock` — nothing was billed, and its token counts are
  estimates;
- the provider reported no token usage;
- the model has no configured price.

It never falls back to a "similar" model's price and never infers tokens from
text length. The dashboard shows how many calls were excluded, so a partial
figure is never mistaken for a total.

The price table is a snapshot with its date recorded in
`PRICING_SNAPSHOT_DATE`. For an unlisted model, set
`LLM_PRICE_INPUT_PER_MTOK` and `LLM_PRICE_OUTPUT_PER_MTOK`.

## Privacy

| Risk | Control |
|---|---|
| API keys in attributes | Field names in `SENSITIVE_FIELDS` are dropped |
| API keys pasted into text | Regex matching on key *shape*, anywhere in a string |
| Keys in exception messages | `error_message` is scrubbed too |
| User question content | Stored as `{present, chars}`, not the text |
| The game's secret | `secret` fields are redacted; tool snapshots omit it mid-game |
| Unbounded growth | `max_events` deque; evictions are counted and displayed |
| Debug detail for end users | The debug panel stays behind `GAME_DEBUG` |

Redaction runs when the event is constructed, so an unsafe value is never
written to the store or the log file.

## Reliability

Error handling is consistent across layers and never silent:

- `tracker.operation()` records the exception and **re-raises** it.
- A failure to write the JSONL log records `log_write_error` on the event
  rather than breaking the request.
- `load_events` skips malformed lines instead of refusing to load.
- Retries, timeouts and validation failures are counted per call in
  `CallStats` and surfaced as metrics.

## Testing

56 tests across `test_observability.py` and `test_observability_page.py`,
covering redaction (including keys inside free text), store bounds and JSONL
round-trip, every metric formula including empty input, pricing refusals, and
request-id propagation through a real agent run.

One of these caught a flaw in its own first version: a whole-payload
substring search for the secret false-positived on a timestamp. The test now
inspects attributes rather than the serialised envelope.
