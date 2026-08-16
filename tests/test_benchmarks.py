"""Tests for the decoupled benchmark abstraction.

All unit tests build small synthetic fixture directories under ``tmp_path``.
Only the final test touches ``datasets/gaia`` and is skipped when absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.benchmarks import (
    BenchmarkGrading,
    BenchmarkTask,
    GaiaBenchmark,
    GradingUnavailableError,
    LeakageError,
    RunObservations,
    TaskOutcome,
    UnknownGraderError,
    UnknownTaskError,
    compare_runs,
    compute_run_statistics,
    discover_gaia_runs,
    outcomes_disagree,
)

REAL_GAIA_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "gaia"


# --------------------------------------------------------------------------
# fixture helpers
# --------------------------------------------------------------------------


def write_run(
    root: Path,
    run_name: str,
    *,
    records: list[dict],
    evaluations: list[list[dict]] | None = None,
    failed_batches: int = 0,
    config: dict | None = None,
    empty_task_dirs: tuple[str, ...] = (),
) -> Path:
    """Create a synthetic Gaia-shaped run directory."""
    run_dir = root / run_name
    (run_dir / "tasks").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config or {"model": "fake/model"}))
    for record in records:
        task_dir = run_dir / "tasks" / str(record["task_id"]).replace("-", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "result.json").write_text(json.dumps(record))
    for name in empty_task_dirs:
        (run_dir / "tasks" / name).mkdir(parents=True, exist_ok=True)
    if evaluations is not None or failed_batches:
        eval_dir = run_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for index, batch in enumerate(evaluations or [], start=1):
            (eval_dir / f"batch_{index:04d}.json").write_text(json.dumps(batch))
        start = len(evaluations or []) + 1
        for index in range(start, start + failed_batches):
            (eval_dir / f"batch_{index:04d}_error.json").write_text(
                json.dumps(
                    {
                        "batch_number": index,
                        "error": "URLError: connection reset",
                        "created_at": "2026-08-12T14:11:46+00:00",
                    }
                )
            )
    return run_dir


def record(
    task_id: str,
    *,
    question: str = "q?",
    answer: str = "a",
    expected_regex: str | None = "(?i)a",
    regex_passed: bool | None = True,
    verdict: str | None = "correct",
    status: str = "passed_direct",
    **extra: object,
) -> dict:
    out: dict = {
        "task_id": task_id,
        "question": question,
        "answer": answer,
        "category": "gaia-level-1",
        "difficulty": "easy",
        "status": status,
        "timed_out": False,
        "error": None,
        "elapsed_seconds": 1.0,
    }
    if expected_regex is not None:
        out["expected_regex"] = expected_regex
        out["direct_regex"] = {
            "available": True,
            "passed": regex_passed,
            "pattern": expected_regex,
            "error": None,
        }
    if verdict is not None:
        out["llm_verdict"] = {
            "task_id": task_id,
            "verdict": verdict,
            "reason": "because the gold answer is Rockhopper penguin",
            "answer_span": answer,
        }
    out.update(extra)
    return out


@pytest.fixture()
def simple_run(tmp_path: Path) -> Path:
    return write_run(
        tmp_path,
        "run_a",
        records=[
            record("gaia-001", question="one?", answer="alpha", expected_regex="(?i)alpha"),
            record(
                "gaia-002",
                question="two?",
                answer="beta",
                expected_regex="(?i)gamma",
                regex_passed=False,
                verdict="wrong",
                status="failed_llm",
            ),
        ],
        evaluations=[
            [
                {
                    "task_id": "gaia-001",
                    "verdict": "correct",
                    "reason": "matches gold",
                    "answer_span": "alpha",
                },
                {
                    "task_id": "gaia-002",
                    "verdict": "wrong",
                    "reason": "gold is gamma",
                    "answer_span": "beta",
                },
            ]
        ],
    )


# --------------------------------------------------------------------------
# task loading
# --------------------------------------------------------------------------


def test_load_tasks_returns_task_facing_records(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    tasks = bench.load_tasks()
    assert bench.name == "gaia"
    assert bench.run_name == "run_a"
    assert [t.task_id for t in tasks] == ["gaia-001", "gaia-002"]
    assert tasks[0].question == "one?"
    assert tasks[0].metadata["category"] == "gaia-level-1"
    assert tasks[0].metadata["difficulty"] == "easy"


def test_load_tasks_tolerates_missing_keys_and_reports_coverage(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "sparse",
        records=[
            record("gaia-001"),
            {"task_id": "gaia-002"},  # question/answer/category/... all absent
            {"task_id": "gaia-003", "question": "three?"},
        ],
        empty_task_dirs=("gaia_004",),
    )
    bench = GaiaBenchmark.from_run_dir(run)
    tasks = bench.load_tasks()
    assert len(tasks) == 3
    sparse = {t.task_id: t for t in tasks}["gaia-002"]
    assert sparse.question == ""
    assert sparse.metadata["question_present"] is False
    coverage = bench.key_coverage()
    assert coverage["record_count"] == 3
    assert coverage["task_dirs_without_record"] == 1
    assert coverage["missing"]["question"] == 1
    assert coverage["missing"]["expected_regex"] == 2
    assert coverage["empty"]["answer"] >= 1


def test_task_dir_without_result_json_does_not_crash(tmp_path: Path) -> None:
    run = write_run(tmp_path, "holes", records=[record("gaia-001")], empty_task_dirs=("gaia_002",))
    bench = GaiaBenchmark.from_run_dir(run)
    assert len(bench.load_tasks()) == 1


def test_unreadable_result_json_is_reported_not_fatal(tmp_path: Path) -> None:
    run = write_run(tmp_path, "broken", records=[record("gaia-001")])
    bad = run / "tasks" / "gaia_002"
    bad.mkdir()
    (bad / "result.json").write_text("{not json")
    bench = GaiaBenchmark.from_run_dir(run)
    assert len(bench.load_tasks()) == 1
    assert bench.key_coverage()["unreadable_records"] == 1


# --------------------------------------------------------------------------
# evaluation batch handling
# --------------------------------------------------------------------------


def test_error_batches_excluded_from_results_and_counted(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "partial_eval",
        records=[
            record("gaia-001"),
            record("gaia-002", verdict=None),
        ],
        evaluations=[
            [{"task_id": "gaia-001", "verdict": "correct", "reason": "ok", "answer_span": "a"}]
        ],
        failed_batches=2,
    )
    bench = GaiaBenchmark.from_run_dir(run)
    obs = bench.observations()
    assert obs.failed_eval_batches == 2
    assert bench.recorded_verdict_count() == 1
    # the failed batch's task has no verdict -> grader is unavailable, not failing
    with pytest.raises(GradingUnavailableError):
        bench.score("gaia-002", "a", grader="recorded_llm_verdict")


def test_zero_evaluations_does_not_crash(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "no_eval",
        records=[record("gaia-001", verdict=None), record("gaia-002", verdict=None)],
        failed_batches=1,
    )
    bench = GaiaBenchmark.from_run_dir(run)
    stats = compute_run_statistics(bench, bench.observations())
    llm = stats.grader_stats["recorded_llm_verdict"]
    assert llm.evaluated == 0
    assert llm.passed == 0
    assert llm.pass_rate is None
    assert llm.unavailable == 2
    assert llm.is_partial is True
    assert stats.failed_eval_batches == 1
    assert stats.grader_stats["expected_regex"].evaluated == 2


# --------------------------------------------------------------------------
# leakage prevention
# --------------------------------------------------------------------------


def test_benchmark_task_repr_contains_no_grading_material(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    for task in bench.load_tasks():
        text = repr(task)
        for secret in ("expected_regex", "direct_regex", "llm_verdict", "verdict",
                       "answer_span", "reason", "gamma", "matches gold", "gold is gamma"):
            assert secret not in text, f"{secret!r} leaked into BenchmarkTask repr: {text}"
        assert not hasattr(task, "expected_regex")
        assert "expected_regex" not in task.metadata


def test_benchmark_task_rejects_grading_metadata() -> None:
    with pytest.raises(LeakageError):
        BenchmarkTask(task_id="t1", question="q", metadata={"expected_regex": "(?i)x"})
    with pytest.raises(LeakageError):
        BenchmarkTask(task_id="t1", question="q", metadata={"llm_verdict": "correct"})
    with pytest.raises(LeakageError):
        BenchmarkTask(task_id="t1", question="q", metadata={"ground_truth": "42"})


def test_task_outcome_detail_rejects_grading_material() -> None:
    with pytest.raises(LeakageError):
        TaskOutcome(
            task_id="t1",
            score=1.0,
            passed=True,
            grader_name="expected_regex",
            detail={"pattern": "(?i)rockhopper"},
        )
    with pytest.raises(LeakageError):
        TaskOutcome(
            task_id="t1",
            score=0.0,
            passed=False,
            grader_name="recorded_llm_verdict",
            detail={"reason": "gold answer was Guatemala"},
        )


def test_grading_repr_is_redacted(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    grading = bench.grading_for("gaia-002")
    assert grading is not None
    assert isinstance(grading, BenchmarkGrading)
    text = repr(grading)
    assert "gaia-002" in text
    assert "redacted" in text
    for secret in ("gamma", "gold is gamma", "wrong"):
        assert secret not in text
    # scorer-only access still works explicitly
    assert grading.expected_regex == "(?i)gamma"
    assert grading.recorded_verdict == "wrong"


def test_grading_for_unknown_task_returns_none(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    assert bench.grading_for("nope") is None


# --------------------------------------------------------------------------
# grading / scoring
# --------------------------------------------------------------------------


def test_graders_are_named_and_scored_independently(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    assert bench.graders() == ("expected_regex", "recorded_llm_verdict")

    regex = bench.score("gaia-001", "alpha", grader="expected_regex")
    assert regex.grader_name == "expected_regex"
    assert regex.passed is True
    assert regex.score == 1.0

    llm = bench.score("gaia-001", "alpha", grader="recorded_llm_verdict")
    assert llm.grader_name == "recorded_llm_verdict"
    assert llm.passed is True

    fail = bench.score("gaia-002", "beta", grader="expected_regex")
    assert fail.passed is False
    assert fail.score == 0.0


def test_score_all_surfaces_grader_disagreement(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "disagree",
        records=[
            record(
                "gaia-001",
                answer="Guatemala City",
                expected_regex="(?i)^Guatemala$",
                regex_passed=False,
                verdict="correct",
            )
        ],
    )
    bench = GaiaBenchmark.from_run_dir(run)
    outcomes = bench.score_all("gaia-001", "Guatemala City")
    assert outcomes["expected_regex"].passed is False
    assert outcomes["recorded_llm_verdict"].passed is True
    assert outcomes_disagree(outcomes) is True

    agreeing = {
        "a": TaskOutcome(task_id="t", score=1.0, passed=True, grader_name="a", detail={}),
        "b": TaskOutcome(task_id="t", score=1.0, passed=True, grader_name="b", detail={}),
    }
    assert outcomes_disagree(agreeing) is False


def test_unknown_grader_raises(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    with pytest.raises(UnknownGraderError):
        bench.score("gaia-001", "alpha", grader="live_llm_judge")


def test_unknown_task_raises(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    with pytest.raises(UnknownTaskError):
        bench.score("gaia-999", "alpha", grader="expected_regex")


def test_recorded_verdict_grader_refuses_a_different_answer(simple_run: Path) -> None:
    """The recorded verdict is a replayed historical judgment, not a live judge."""
    bench = GaiaBenchmark.from_run_dir(simple_run)
    with pytest.raises(GradingUnavailableError) as exc:
        bench.score("gaia-001", "a brand new answer", grader="recorded_llm_verdict")
    assert "recorded" in str(exc.value).lower()
    # ...while the live regex grader happily scores a new answer
    assert bench.score("gaia-001", "ALPHA and more", grader="expected_regex").passed is True


def test_missing_expected_regex_is_unavailable_not_failing(tmp_path: Path) -> None:
    run = write_run(tmp_path, "nore", records=[record("gaia-001", expected_regex=None)])
    bench = GaiaBenchmark.from_run_dir(run)
    with pytest.raises(GradingUnavailableError):
        bench.score("gaia-001", "anything", grader="expected_regex")
    assert bench.try_score("gaia-001", "anything", grader="expected_regex") is None


def test_invalid_regex_is_unavailable_not_failing(tmp_path: Path) -> None:
    run = write_run(tmp_path, "badre", records=[record("gaia-001", expected_regex="(unclosed")])
    bench = GaiaBenchmark.from_run_dir(run)
    with pytest.raises(GradingUnavailableError):
        bench.score("gaia-001", "anything", grader="expected_regex")


def test_unknown_recorded_verdict_value_is_unavailable(tmp_path: Path) -> None:
    run = write_run(tmp_path, "weird", records=[record("gaia-001", verdict="maybe")])
    bench = GaiaBenchmark.from_run_dir(run)
    with pytest.raises(GradingUnavailableError):
        bench.score("gaia-001", "a", grader="recorded_llm_verdict")


def test_scores_are_within_unit_interval(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    for task in bench.load_tasks():
        for outcome in bench.score_all(task.task_id, "alpha").values():
            assert 0.0 <= outcome.score <= 1.0
    with pytest.raises(ValueError):
        TaskOutcome(task_id="t", score=1.5, passed=True, grader_name="g", detail={})
    with pytest.raises(ValueError):
        TaskOutcome(task_id="t", score=-0.1, passed=False, grader_name="g", detail={})


def test_outcome_requires_a_grader_name() -> None:
    with pytest.raises(ValueError):
        TaskOutcome(task_id="t", score=1.0, passed=True, grader_name="", detail={})


# --------------------------------------------------------------------------
# run statistics + comparison
# --------------------------------------------------------------------------


def test_run_statistics_counts_denominators_timeouts_and_errors(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        "stats",
        records=[
            record("gaia-001"),
            record("gaia-002", regex_passed=False, expected_regex="(?i)zzz", verdict="wrong"),
            record("gaia-003", verdict=None, status="errored", error="exited -2", timed_out=False),
            record("gaia-004", verdict=None, status="timeout", timed_out=True),
        ],
        failed_batches=1,
    )
    bench = GaiaBenchmark.from_run_dir(run)
    stats = compute_run_statistics(bench, bench.observations())
    assert stats.task_count == 4
    assert stats.errored == 1
    assert stats.timed_out == 1
    assert stats.failed_eval_batches == 1

    regex = stats.grader_stats["expected_regex"]
    assert (regex.passed, regex.evaluated, regex.total_tasks) == (3, 4, 4)
    assert regex.is_partial is False

    llm = stats.grader_stats["recorded_llm_verdict"]
    assert (llm.passed, llm.evaluated, llm.total_tasks) == (1, 2, 4)
    assert llm.is_partial is True
    assert abs(llm.pass_rate - 0.5) < 1e-9


def test_compare_runs_flags_partial_denominators(tmp_path: Path) -> None:
    full = write_run(
        tmp_path,
        "full",
        records=[record(f"gaia-{i:03d}") for i in range(1, 5)],
    )
    partial = write_run(
        tmp_path,
        "partial",
        records=[
            record("gaia-001"),
            record("gaia-002"),
            record("gaia-003", verdict=None),
            record("gaia-004", verdict=None),
        ],
    )
    a = compute_run_statistics(*_bench_and_obs(full))
    b = compute_run_statistics(*_bench_and_obs(partial))
    comparison = compare_runs(a, b)

    assert comparison.shared_task_count == 4
    regex_delta = comparison.deltas["expected_regex"]
    assert regex_delta.comparable is True
    assert regex_delta.passed_delta == 0

    llm_delta = comparison.deltas["recorded_llm_verdict"]
    assert llm_delta.comparable is False
    assert "denominator" in " ".join(llm_delta.notes).lower()
    assert llm_delta.pass_rate_delta is None


def test_compare_runs_flags_different_task_sets(tmp_path: Path) -> None:
    a_dir = write_run(tmp_path, "a", records=[record("gaia-001"), record("gaia-002")])
    b_dir = write_run(tmp_path, "b", records=[record("gaia-001"), record("gaia-003")])
    a = compute_run_statistics(*_bench_and_obs(a_dir))
    b = compute_run_statistics(*_bench_and_obs(b_dir))
    comparison = compare_runs(a, b)
    assert comparison.same_task_set is False
    assert comparison.shared_task_count == 1
    assert any("task set" in note.lower() for note in comparison.notes)


def test_compare_runs_reports_real_delta(tmp_path: Path) -> None:
    a_dir = write_run(
        tmp_path,
        "worse",
        records=[
            record("gaia-001", expected_regex="(?i)zzz", regex_passed=False, verdict="wrong"),
            record("gaia-002", expected_regex="(?i)zzz", regex_passed=False, verdict="wrong"),
        ],
    )
    b_dir = write_run(tmp_path, "better", records=[record("gaia-001"), record("gaia-002")])
    a = compute_run_statistics(*_bench_and_obs(a_dir))
    b = compute_run_statistics(*_bench_and_obs(b_dir))
    delta = compare_runs(a, b).deltas["expected_regex"]
    assert delta.comparable is True
    assert delta.passed_delta == 2
    assert abs(delta.pass_rate_delta - 1.0) < 1e-9


def test_observations_is_a_plain_transport_object(simple_run: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(simple_run)
    obs = bench.observations()
    assert isinstance(obs, RunObservations)
    assert obs.run_name == "run_a"
    assert set(obs.answers) == {"gaia-001", "gaia-002"}


def test_discover_gaia_runs(tmp_path: Path) -> None:
    write_run(tmp_path, "run_b", records=[record("gaia-001")])
    write_run(tmp_path, "run_a", records=[record("gaia-001")])
    (tmp_path / "not_a_run").mkdir()
    (tmp_path / "stray.txt").write_text("x")
    found = discover_gaia_runs(tmp_path)
    assert [p.name for p in found] == ["run_a", "run_b"]


def test_discover_gaia_runs_on_missing_root(tmp_path: Path) -> None:
    assert discover_gaia_runs(tmp_path / "absent") == ()


def _bench_and_obs(run_dir: Path):
    bench = GaiaBenchmark.from_run_dir(run_dir)
    return bench, bench.observations()


# --------------------------------------------------------------------------
# real-data schema-drift guard (skipped when datasets/ is absent)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_GAIA_ROOT.exists(), reason="datasets/gaia not present")
def test_real_gaia_runs_load_and_score() -> None:
    runs = discover_gaia_runs(REAL_GAIA_ROOT)
    assert runs, "expected at least one run under datasets/gaia"
    for run_dir in runs:
        bench = GaiaBenchmark.from_run_dir(run_dir)
        tasks = bench.load_tasks()
        assert tasks, f"{run_dir.name} produced no tasks"
        stats = compute_run_statistics(bench, bench.observations())
        assert stats.task_count == len(tasks)
        for grader in bench.graders():
            gs = stats.grader_stats[grader]
            assert gs.evaluated <= gs.total_tasks
            assert gs.passed <= gs.evaluated
            if gs.evaluated:
                assert 0.0 <= gs.pass_rate <= 1.0
        for task in tasks:
            assert repr(task).count("expected_regex") == 0
            for outcome in bench.score_all(task.task_id, bench.recorded_answer(task.task_id) or "").values():
                assert 0.0 <= outcome.score <= 1.0
