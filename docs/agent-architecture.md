# Agent architecture

## What makes this an agent

The natural language layer (Phase 4) maps one message to one action. The agent
differs in three ways: it decides *whether* a tool is needed, it can take
several steps toward one request, and it inspects each result before deciding
what to do next.

That extra freedom is exactly what has to be bounded, which is most of what
this document is about.

## Components

```
agents/
  tools.py       ToolRegistry + 7 registered tools with typed schemas
  planner.py     AgentDecision schema, LLM planner, rule-based planner
  controller.py  the bounded loop and its guardrails
  trace.py       Span / AgentTrace, the observability hooks Phase 6 consumes
```

Each is separately testable: the registry without a planner, the planner
without a loop, the loop with a scripted planner.

## The loop

```
receive request
   │
   ▼
┌─ for iteration in 1..max_iterations ──────────────────────┐
│  timeout check                                            │
│  planner.plan(message, state, completed, observations)    │
│     ├── "respond" / "clarify"  -> leave the loop          │
│     └── "call_tool"                                       │
│  repeated-call check (tool + arguments signature)         │
│  registry.get(tool)            unknown -> recoverable     │
│  tool.validate_arguments()     invalid -> recoverable     │
│  tool.run(args, context)                                  │
│  tool.validate_output()        contract drift -> error    │
│  record observation; update state                         │
└───────────────────────────────────────────────────────────┘
   │
   ▼
compose the final response
```

## Guardrails

| Guard | Setting | Behaviour |
|---|---|---|
| Iteration ceiling | `max_iterations` (5) | Terminal. Loop cannot run forever. |
| Wall-clock budget | `timeout_seconds` (30) | Terminal. Checked before each pass. |
| Repeated calls | `allow_repeat_calls` (False) | Terminal. Same tool + same arguments is refused. |
| Tool error budget | `max_tool_errors` (3) | Terminal after N. Individual errors are recoverable. |
| Unknown tool | always | Recoverable. Planner falls back to rules. |
| Invalid arguments | always | Recoverable. Reported back as an observation. |

**Recoverable vs terminal** is the key distinction. A single bad tool call is
useful feedback — the agent is told what went wrong and can correct itself. A
*pattern* of bad calls, or a loop that will not end, is terminal.

The rule-based planner gets termination for free: each branch fires only if
its tool is not already in `completed`, so a multi-step request drains its
sub-tasks and then responds.

## What the agent cannot do

- **Call unregistered code.** `ToolRegistry.get` raises on anything not
  registered. There is no `getattr`, no string-to-function lookup, no `eval`,
  no shell or filesystem access in `agents/`.
- **Pass unexpected arguments.** Every input model sets
  `extra="forbid"`, so a tool receives only what it declared.
- **Bypass game validation.** `submit_guess` calls the same
  `game.submit_guess` the UI does. An out-of-range guess fails there.
- **See the secret mid-game.** `snapshot_of` sets `secret` to `None` while
  `status == "playing"`.
- **Act without being asked.** State-changing tools are marked
  `mutates_state`, and the planner prompt forbids calling them unprompted.
  `test_agent_does_not_start_a_game_unprompted` pins this.

## Tracing

Every decision and tool call becomes a `Span` with a `parent_id`, an
iteration number, a duration and a status. `AgentTrace.to_dict()` is
JSON-serialisable, which is the contract Phase 6 builds on.

Tool arguments are recorded; the secret is not, because no tool emits it
while a game is live.

## Measured results

`python -m evaluation.run --dataset evaluation/datasets/agent.json`

27 cases, 27 passing: tool_selection 7, multi_step 3, guardrails 9,
observability 2, tool_contract 6.

These measure orchestration under the deterministic planner. They do **not**
measure how well a real model plans — that needs a live provider and is not
claimed here.
