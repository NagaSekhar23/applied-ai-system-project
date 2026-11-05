# Security

This is a local demo application, not a deployed service. The threat model
below is scoped to what the project actually faces: untrusted model output,
credentials on a developer machine, and the risk of a system that acts on
language.

## Threats and controls

### 1. The model producing an action it should not

| Risk | Control | Where |
|---|---|---|
| Model "decides" a guess is correct | `check_guess` is the only authority; the model's category is cross-checked and overridden | `assistant.py` |
| Model-emitted text executed as an instruction | Output is a closed `CommandType` enum, never a string to dispatch on | `natural_language/commands.py` |
| Model calls arbitrary code | `ToolRegistry` lookup raises on anything unregistered; no `getattr`, no `eval`, no shell | `agents/tools.py` |
| Model passes unexpected arguments | Every tool input model sets `extra="forbid"` | `agents/tools.py` |
| Model bypasses game validation | Tools call `game.submit_guess`, the same path the UI uses | `agents/tools.py` |
| Model loops forever | Four terminal guards: iterations, timeout, repeated calls, error budget | `agents/controller.py` |
| Model fabricates a citation | Markers are validated against retrieved evidence; an invalid one rejects the answer | `retrieval/citations.py` |
| Low-confidence output triggers an action | Confidence can only *withhold* an action, never authorise one | `natural_language/parser.py` |

**There is no code path from model output to code execution.** The model can
choose among enumerated options; it cannot name a function.

### 2. Secret disclosure (the game's answer)

Defence in depth, three independent layers:

1. **Structural** — `build_facts` omits the `secret` key entirely while
   `status == "playing"`. The prompt cannot leak a value it never received.
2. **Instructional** — the prompt forbids revealing it.
3. **Output check** — `leaks_secret` scans the reply for the number as a
   standalone token; a match forces the deterministic fallback.

Tool snapshots apply the same rule: `snapshot_of` sets `secret = None` while a
game is live. Observability redacts `secret` fields before storing.

### 3. Credential handling

| Risk | Control |
|---|---|
| Key committed to git | `.gitignore` blocks `.env`, `.env.*`, `*.env`; `.env.example` holds placeholders only and is explicitly un-ignored |
| Key hardcoded | Read from the environment only; `config.py` has no default key |
| Key logged | `SENSITIVE_FIELDS` drops key-named fields; regex scrubbing catches `sk-ant-…`, `sk-…`, `Bearer …` **anywhere in a string**, including inside error messages |
| Key shown in the UI | `Settings.redacted()` returns `api_key_configured: bool`, never the value; an AppTest asserts the key never appears in rendered output |
| Key in a report | Reports are built from redacted events |

Verified by `test_api_key_is_never_rendered` and the redaction test suite.

### 4. Privacy of user content

- Questions and messages are recorded as `{present, chars}` — the length, not
  the text.
- Free text that *is* stored is truncated to 200 characters and scrubbed.
- The debug panel, which shows the secret, is off unless `GAME_DEBUG=true`.
- The event store is bounded (500 events), so data does not accumulate
  indefinitely; evictions are counted and displayed.
- Logs are local JSONL. Retention is the operator deleting the file; there is
  no shipping to a third party.

### 5. Ingestion of sensitive files

RAG ingests an **explicit allow-list** of documentation paths, never a
directory walk. `BLOCKED_PATTERNS` additionally refuses any path matching
`.env`, `secrets`, `credentials` or `.venv`, even if someone adds it to the
source list. Tested in `test_sensitive_paths_are_blocked`.

### 6. Prompt injection via retrieved content

The corpus is this repository's own documentation, so the realistic risk is
low. The structural mitigations that matter if the corpus were opened up:

- Retrieved text is placed in the **user** turn, never the system prompt.
- The system prompt states the model is a reader of passages, not an actor.
- Retrieved content cannot reach the agent's tool registry — RAG and the agent
  are separate paths with no shared execution surface.
- Citations are validated against retrieved chunk ids, so injected text
  claiming to be a source cannot produce a valid citation.

**Not mitigated:** a document in the corpus containing adversarial
instructions could still influence an answer's wording. With a trusted,
version-controlled corpus this is acceptable; with user-uploaded documents it
would not be, and would need output filtering and provenance marking.

## Testing

Security-relevant behavior is pinned by tests rather than asserted in prose:

- `test_api_key_is_never_rendered`
- `test_redaction_*` — field names, key shapes, nesting, error messages
- `test_facts_omit_the_secret_while_playing`
- `test_assistant_falls_back_when_the_model_leaks_the_secret`
- `test_agent_never_executes_an_unregistered_tool`
- `test_pipeline_survives_a_hostile_command_string`
- `test_sensitive_paths_are_blocked`
- `test_rejects_a_fabricated_citation`

## Out of scope

Honest boundaries for a local demo:

- No authentication, authorisation or multi-tenancy.
- No rate limiting on the app itself (only on provider calls).
- No encryption at rest for the event log.
- No supply-chain verification beyond pinned minimum versions.
- Streamlit is bound to localhost by default and is not hardened for
  public deployment.
