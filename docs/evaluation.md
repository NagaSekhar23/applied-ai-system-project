# Evaluation

## Why this exists

Tests answer "is this function correct?". Evaluation answers "how well does the
system behave across a representative set of scenarios, and is that getting
better or worse?"

They overlap for deterministic code. They stop overlapping once an LLM is
involved, because a model's output is not a fixed value you can assert on. The
framework is built in Phase 2, against deterministic behavior, so the
measurement machinery is already trustworthy before anything non-deterministic
is plugged into it.

## Running it

```bash
python -m evaluation.run                      # every dataset in evaluation/datasets/
python -m evaluation.run --quiet              # one summary line per dataset
python -m evaluation.run --dataset path.json  # a specific dataset (repeatable)
python -m evaluation.run --no-report          # don't write report files
python -m evaluation.run --fail-under 0.95    # tolerate a 95% pass rate
```

**Exit codes:** `0` success, `1` a case failed or errored, `2` the dataset could
not be loaded. The non-zero codes make this usable as a CI gate.

## Dataset format

Datasets are JSON, validated by Pydantic before anything runs. A malformed
dataset fails immediately with a readable message instead of halfway through.

```json
{
  "name": "game-logic",
  "version": 1,
  "description": "...",
  "cases": [
    {
      "id": "cmp-too-high",
      "category": "comparison",
      "target": "check_guess",
      "input": { "guess": 60, "secret": 50 },
      "initial_state": null,
      "expected": { "outcome": "Too High", "hint": "📉 Go LOWER!" },
      "expected_behavior": "A guess above the secret tells the player to go lower.",
      "metadata": { "tags": ["regression"] }
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `id` | Unique within the dataset. Enforced. |
| `category` | Groups cases for per-category metrics. |
| `target` | Which executor runs this case. |
| `input` | Arguments for the executor. |
| `initial_state` | Starting state for multi-step cases. |
| `expected` | Asserted values. Must be non-empty. |
| `expected_behavior` | Plain-English intent, shown in failure output. |
| `metadata` | Free-form tags. Not used for pass/fail. |

### Subset comparison

`expected` is compared as a **subset** of the executor's output: every key in
`expected` must match, extra keys in the actual result are ignored. A case
therefore asserts only what it is about, and adding a new output field does not
break every existing case.

Tuples and lists compare equal, because JSON cannot express a tuple but the
application returns one for `history`.

## Executors

An executor turns a case into a call against real application code. It never
re-implements logic — that would mean evaluating the test harness rather than
the system.

| Target | Calls | Asserts on |
|---|---|---|
| `check_guess` | `logic_utils.check_guess` | `outcome`, `hint` |
| `parse_guess` | `logic_utils.parse_guess` | `ok`, `value`, `error` |
| `update_score` | `logic_utils.update_score` | `score` |
| `difficulty_config` | `logic_utils.get_difficulty_config` | `low`, `high`, `max_attempts` |
| `game_flow` | `game.new_game` + `game.submit_guess` | `status`, `attempts_used`, `score`, `history`, `errors`, … |

`game_flow` is the multi-step executor: it replays a list of raw strings exactly
as a player would type them, against the real game service, with a fixed secret
so the run is deterministic.

Executors convert *expected* exceptions (`ValueError`, `TypeError`, `KeyError`)
into comparable fields (`raised`, `error_message`), so a case can assert that
something fails loudly. Any other exception is recorded as an `error` result.

## Metrics, and what they do not mean

| Metric | Formula |
|---|---|
| `pass_rate` | `passed / total_cases` |
| `failure_rate` | `(failed + errored) / total_cases` |
| `accuracy` | Alias of `pass_rate` |
| per-category `pass_rate` | `passed / total` within that category |
| `duration_seconds` | Wall clock for the run |

Every metric block carries a `kind`:

- `deterministic_correctness` — Phase 2. Did the code produce exactly the
  specified output?
- `model_quality` — later phases, for model-dependent judgements.
- `retrieval_quality` — Phase 7 retrieval metrics.

These are kept separate on purpose. Averaging a deterministic pass rate
together with a subjective model score would produce a number that means
nothing. **A 100% pass rate here means the implementation matches the dataset.**
It says nothing about scenarios the dataset does not cover, and nothing about
whether an LLM's phrasing is good.

`failed` and `errored` are tracked separately: a failure is a wrong answer, an
error is a crash. They need different follow-up.

## Reports

Each run writes three files to `reports/`:

- `<dataset>-<timestamp>.json` — full machine-readable results
- `<dataset>-<timestamp>.txt` — the human-readable summary
- `latest-<dataset>.json` — a stable path for tooling and the dashboard

`reports/` is git-ignored; it is run output, not source.

## Reproducibility

Each run produces a **fingerprint**: a SHA-256 over the sorted
`(case_id, status)` pairs, truncated to 16 characters. Durations and timestamps
are deliberately excluded, because they vary between runs.

Same dataset + same code ⇒ same fingerprint. A changed fingerprint means
behavior changed. This is asserted by `test_runs_are_reproducible`.

## Current dataset

`evaluation/datasets/game_logic.json` — 56 cases across 9 categories:
comparison, input_validation, boundary, edge_case, scoring, difficulty,
game_flow, game_over, error_handling.

Cases tagged `regression` in metadata reproduce defects that were actually found
and fixed in this repository.

### Latest measured result

Command: `python -m evaluation.run`

```
Total cases : 56
Passed      : 56
Failed      : 0
Errored     : 0
Pass rate   : 100.0%
Duration    : 0.001s
```

Per category: boundary 7/7, comparison 8/8, difficulty 6/6, edge_case 2/2,
error_handling 2/2, game_flow 4/4, game_over 5/5, input_validation 12/12,
scoring 10/10.

This is a deterministic-correctness result on a mock-free, API-free code path.

## Extending it

To evaluate something new:

1. Write an executor: `(EvalCase) -> dict[str, Any]`.
2. Register it in a registry dict.
3. Write a dataset whose cases use that `target`.
4. Pass the registry to `run_dataset(dataset, registry=...)`.

The runner, metrics, reports and CLI need no changes. That is the point of the
executor indirection.
