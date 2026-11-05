"""Tests for the evaluation framework itself.

A framework that reports "all passed" is only trustworthy if it can be shown to
detect failures, so several tests feed it deliberately wrong expectations.
"""

import json

import pytest

from evaluation.cases import Dataset, DatasetError, EvalCase, discover_datasets, load_dataset
from evaluation.executors import UnknownExecutorError, default_registry, resolve
from evaluation.metrics import MetricKind, compute_metrics
from evaluation.report import load_latest_reports, render_text, write_reports
from evaluation.runner import CaseResult, compare, fingerprint_results, run_case, run_dataset

DATASET_DIR = "evaluation/datasets"


def make_case(**overrides) -> EvalCase:
    payload = {
        "id": "case-1",
        "category": "comparison",
        "target": "check_guess",
        "input": {"guess": 60, "secret": 50},
        "expected": {"outcome": "Too High"},
    }
    payload.update(overrides)
    return EvalCase(**payload)


def write_json(tmp_path, payload, name="ds.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_payload(**overrides):
    payload = {
        "name": "tiny",
        "version": 1,
        "cases": [
            {
                "id": "a",
                "category": "comparison",
                "target": "check_guess",
                "input": {"guess": 60, "secret": 50},
                "expected": {"outcome": "Too High"},
            }
        ],
    }
    payload.update(overrides)
    return payload


# --- Dataset loading and validation -----------------------------------------


def test_loads_a_valid_dataset(tmp_path):
    dataset = load_dataset(write_json(tmp_path, valid_payload()))
    assert dataset.name == "tiny"
    assert len(dataset.cases) == 1
    assert dataset.categories == ["comparison"]


def test_missing_file_raises_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="Cannot read dataset"):
        load_dataset(tmp_path / "nope.json")


def test_malformed_json_raises_dataset_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_dataset(path)


def test_non_object_json_raises_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="must be a JSON object"):
        load_dataset(write_json(tmp_path, ["a", "list"]))


def test_duplicate_case_ids_are_rejected(tmp_path):
    payload = valid_payload()
    payload["cases"] = payload["cases"] + [dict(payload["cases"][0])]
    with pytest.raises(DatasetError, match="duplicate case ids"):
        load_dataset(write_json(tmp_path, payload))


def test_case_without_expected_is_rejected(tmp_path):
    payload = valid_payload()
    payload["cases"][0]["expected"] = {}
    with pytest.raises(DatasetError, match="at least one key"):
        load_dataset(write_json(tmp_path, payload))


def test_empty_case_list_is_rejected(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(write_json(tmp_path, valid_payload(cases=[])))


def test_unknown_field_is_rejected(tmp_path):
    payload = valid_payload()
    payload["cases"][0]["typo_field"] = 1
    with pytest.raises(DatasetError):
        load_dataset(write_json(tmp_path, payload))


def test_case_is_immutable():
    case = make_case()
    with pytest.raises(Exception):
        case.id = "changed"


def test_shipped_datasets_are_valid():
    paths = discover_datasets(DATASET_DIR)
    assert paths, "expected at least one shipped dataset"
    for path in paths:
        dataset = load_dataset(path)
        assert dataset.cases


# --- Comparison --------------------------------------------------------------


def test_compare_passes_on_subset_match():
    assert compare({"a": 1}, {"a": 1, "b": 2}) == []


def test_compare_reports_mismatch():
    diffs = compare({"a": 1}, {"a": 2})
    assert len(diffs) == 1
    assert diffs[0].field_name == "a"
    assert "expected 1" in diffs[0].describe()


def test_compare_reports_missing_key():
    diffs = compare({"a": 1}, {})
    assert diffs[0].actual == "<missing>"


def test_compare_treats_tuples_and_lists_as_equal():
    # history comes back as a tuple; JSON can only express a list.
    assert compare({"history": [1, 2]}, {"history": (1, 2)}) == []


def test_compare_distinguishes_false_from_zero():
    assert compare({"ok": False}, {"ok": False}) == []
    assert compare({"value": None}, {"value": None}) == []


# --- Running cases -----------------------------------------------------------


def test_run_case_passes_for_correct_expectation():
    result = run_case(make_case(), default_registry())
    assert result.status == "pass"
    assert result.passed


def test_run_case_fails_for_wrong_expectation():
    case = make_case(id="wrong", expected={"outcome": "Too Low"})
    result = run_case(case, default_registry())
    assert result.status == "fail"
    assert result.diffs[0].expected == "Too Low"
    assert result.diffs[0].actual == "Too High"


def test_run_case_records_unknown_target_as_error():
    case = make_case(target="does_not_exist")
    result = run_case(case, default_registry())
    assert result.status == "error"
    assert "UnknownExecutorError" in result.error


def test_run_case_records_executor_exception_as_error():
    def boom(case):
        raise RuntimeError("provider exploded")

    registry = default_registry()
    registry["boom"] = boom
    result = run_case(make_case(target="boom"), registry)
    assert result.status == "error"
    assert "provider exploded" in result.error


def test_expected_exceptions_become_comparable_fields():
    case = make_case(
        id="raises",
        target="difficulty_config",
        input={"difficulty": "Nope"},
        expected={"raised": "ValueError"},
    )
    assert run_case(case, default_registry()).status == "pass"


def test_resolve_lists_available_targets():
    with pytest.raises(UnknownExecutorError, match="check_guess"):
        resolve(default_registry(), "missing")


def test_default_registry_returns_independent_copies():
    first = default_registry()
    first["extra"] = lambda case: {}
    assert "extra" not in default_registry()


# --- Metrics -----------------------------------------------------------------


def results_fixture():
    return [
        CaseResult("a", "cat1", "t", "pass", {}, {}),
        CaseResult("b", "cat1", "t", "fail", {}, {}),
        CaseResult("c", "cat2", "t", "pass", {}, {}),
        CaseResult("d", "cat2", "t", "error", {}, {}),
    ]


def test_metrics_counts_and_rates():
    m = compute_metrics(results_fixture(), duration_seconds=1.5)
    assert (m.total_cases, m.passed, m.failed, m.errored) == (4, 2, 1, 1)
    assert m.pass_rate == 0.5
    assert m.failure_rate == 0.5
    assert m.accuracy == m.pass_rate
    assert m.duration_seconds == 1.5


def test_metrics_rates_sum_to_one():
    m = compute_metrics(results_fixture(), 0.1)
    assert m.pass_rate + m.failure_rate == pytest.approx(1.0)


def test_metrics_by_category():
    m = compute_metrics(results_fixture(), 0.1)
    by_name = {c.category: c for c in m.by_category}
    assert by_name["cat1"].passed == 1
    assert by_name["cat1"].pass_rate == 0.5
    assert by_name["cat2"].errored == 1


def test_metrics_handles_empty_run_without_dividing_by_zero():
    m = compute_metrics([], 0.0)
    assert m.total_cases == 0
    assert m.pass_rate == 0.0
    assert m.failure_rate == 0.0


def test_metrics_label_their_kind():
    m = compute_metrics([], 0.0)
    assert m.kind is MetricKind.DETERMINISTIC_CORRECTNESS
    assert m.to_dict()["kind"] == "deterministic_correctness"

    model = compute_metrics([], 0.0, kind=MetricKind.MODEL_QUALITY)
    assert model.to_dict()["kind"] == "model_quality"


# --- Full runs, reproducibility and reports ----------------------------------


def shipped_dataset() -> Dataset:
    return load_dataset(f"{DATASET_DIR}/game_logic.json")


def test_shipped_dataset_passes_completely():
    run = run_dataset(shipped_dataset())
    assert run.ok, [r.case_id for r in run.failed_results]
    assert run.metrics.pass_rate == 1.0


def test_runs_are_reproducible():
    first = run_dataset(shipped_dataset())
    second = run_dataset(shipped_dataset())
    assert first.fingerprint == second.fingerprint
    assert [r.status for r in first.results] == [r.status for r in second.results]


def test_fingerprint_changes_when_a_status_changes():
    base = [CaseResult("a", "c", "t", "pass", {}, {})]
    changed = [CaseResult("a", "c", "t", "fail", {}, {})]
    assert fingerprint_results(base) != fingerprint_results(changed)


def test_fingerprint_ignores_duration():
    slow = [CaseResult("a", "c", "t", "pass", {}, {}, duration_seconds=9.9)]
    fast = [CaseResult("a", "c", "t", "pass", {}, {}, duration_seconds=0.1)]
    assert fingerprint_results(slow) == fingerprint_results(fast)


def test_failed_results_are_listed(tmp_path):
    payload = valid_payload()
    payload["cases"][0]["expected"] = {"outcome": "Win"}  # deliberately wrong
    run = run_dataset(load_dataset(write_json(tmp_path, payload)))
    assert not run.ok
    assert run.failed_results[0].case_id == "a"


def test_render_text_includes_summary_and_failures(tmp_path):
    payload = valid_payload()
    payload["cases"][0]["expected"] = {"outcome": "Win"}
    run = run_dataset(load_dataset(write_json(tmp_path, payload)))
    text = render_text(run)
    assert "Total cases : 1" in text
    assert "Failures (1)" in text
    assert "outcome: expected 'Win', got 'Too High'" in text


def test_render_text_reports_clean_run():
    assert "All cases passed." in render_text(run_dataset(shipped_dataset()))


def test_write_reports_creates_json_text_and_latest(tmp_path):
    run = run_dataset(shipped_dataset())
    written = write_reports(run, tmp_path)

    assert written["json"].exists() and written["text"].exists()
    assert written["latest_json"].exists()

    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert payload["dataset"]["name"] == "game-logic"
    assert payload["metrics"]["total_cases"] == len(run.results)
    assert payload["fingerprint"] == run.fingerprint
    assert len(payload["results"]) == len(run.results)


def test_load_latest_reports_round_trip(tmp_path):
    write_reports(run_dataset(shipped_dataset()), tmp_path)
    reports = load_latest_reports(tmp_path)
    assert len(reports) == 1
    assert reports[0]["dataset"]["name"] == "game-logic"


def test_load_latest_reports_skips_unreadable_files(tmp_path):
    (tmp_path / "latest-broken.json").write_text("{bad", encoding="utf-8")
    assert load_latest_reports(tmp_path) == []


def test_load_latest_reports_on_missing_directory(tmp_path):
    assert load_latest_reports(tmp_path / "absent") == []


# --- CLI ---------------------------------------------------------------------


def test_cli_exits_zero_on_success(tmp_path, capsys):
    from evaluation.run import main

    code = main(
        [
            "--dataset",
            f"{DATASET_DIR}/game_logic.json",
            "--report-dir",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 0
    assert "passed" in capsys.readouterr().out


def test_cli_exits_one_when_a_case_fails(tmp_path, capsys):
    from evaluation.run import main

    payload = valid_payload()
    payload["cases"][0]["expected"] = {"outcome": "Win"}
    path = write_json(tmp_path, payload, "failing.json")

    code = main(["--dataset", str(path), "--no-report", "--quiet"])
    assert code == 1


def test_cli_exits_two_on_invalid_dataset(tmp_path, capsys):
    from evaluation.run import main

    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    code = main(["--dataset", str(bad), "--no-report"])
    assert code == 2
    assert "ERROR" in capsys.readouterr().err


def test_cli_fail_under_threshold_allows_partial_pass(tmp_path):
    from evaluation.run import main

    payload = valid_payload()
    payload["cases"].append(
        {
            "id": "b",
            "category": "comparison",
            "target": "check_guess",
            "input": {"guess": 1, "secret": 2},
            "expected": {"outcome": "Win"},  # wrong on purpose -> 50% pass rate
        }
    )
    path = write_json(tmp_path, payload, "half.json")

    assert main(["--dataset", str(path), "--no-report", "--quiet"]) == 1
    assert main(["--dataset", str(path), "--no-report", "--quiet", "--fail-under", "0.5"]) == 0


def test_cli_no_report_writes_nothing(tmp_path):
    from evaluation.run import main

    main(["--dataset", f"{DATASET_DIR}/game_logic.json", "--report-dir", str(tmp_path), "--no-report", "--quiet"])
    assert list(tmp_path.iterdir()) == []
