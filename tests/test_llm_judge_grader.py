"""LLM-judge grader for tasks with no deterministic answer key.

Motivation (user ruling 2026-08-27): ``honing6.json`` synthetic problems 3/5/6
are time-sensitive ("did it actually search?") and can never carry an
``expected_regex``; problems 1/2 can, and stay deterministic as a control.
``recorded_llm_verdict`` is a *replay* of a historical judgment and refuses any
new answer, so a fresh dataset had no live grader at all — the wall hit when
materializing the synthetic dataset.

Design, held to the house measurement rules:
- Judge variance must be visible, not baked in: the verdict is structured
  (verdict/reason), temperature is pinned to 0, and the raw judge response is
  echoed in ``detail`` so two runs of the same answer can be compared.
- The judge sees the ANSWER plus the task's GRADING NOTES. Grading notes are
  answer-adjacent material: they cross into the judge prompt only, and the
  outcome detail carries no notes text, so nothing enters edit memory,
  clustering, or manifests (the contamination boundary holds).
- Grading-unavailable (model error) is ``GradingUnavailableError``, NOT a
  failing score — an outage must never look like a wrong answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.benchmarks.base import GradingUnavailableError  # noqa: E402
from agent_evolve.benchmarks.gaia import (  # noqa: E402
    GRADER_LLM_JUDGE,
    GaiaBenchmark,
)
from tests.test_benchmarks import record, write_run  # noqa: E402


def _simple_run(tmp_path: Path) -> Path:
    return write_run(
        tmp_path,
        "simple",
        records=[
            record("gaia-001", answer="alpha", expected_regex="(?i)alpha",
                   regex_passed=True, verdict="correct"),
            record("gaia-002", answer="beta", expected_regex="(?i)gamma",
                   regex_passed=False, verdict="wrong"),
        ],
    )


def _judge_payload(
    verdict: str, reason: str = "the answer matches the grading notes"
) -> dict:
    return {
        "choices": [
            {"message": {"content": json.dumps({"verdict": verdict, "reason": reason})}}
        ]
    }



def _bench(tmp_path: Path, **record_kwargs) -> GaiaBenchmark:
    run = write_run(
        tmp_path,
        "judged",
        records=[record("gaia-j1", **record_kwargs)],
    )
    return GaiaBenchmark.from_run_dir(run)


# ---------------------------------------------------------------------- #
# Registration
# ---------------------------------------------------------------------- #
def test_llm_judge_grader_is_registered(tmp_path: Path) -> None:
    bench = GaiaBenchmark.from_run_dir(_simple_run(tmp_path))
    assert GRADER_LLM_JUDGE == "llm_judge"
    assert GRADER_LLM_JUDGE in bench.graders()


def test_unknown_grader_error_lists_all_three(tmp_path: Path) -> None:
    bench = _bench(tmp_path)
    with pytest.raises(Exception, match="llm_judge"):
        bench.score("gaia-j1", "x", grader="vibes")


# ---------------------------------------------------------------------- #
# Material
# ---------------------------------------------------------------------- #
def test_no_grading_notes_is_unavailable_not_failed(tmp_path: Path) -> None:
    bench = _bench(tmp_path, llm_grading_notes=None)
    with pytest.raises(GradingUnavailableError, match="grading notes"):
        bench.score("gaia-j1", "anything", grader="llm_judge")


def test_judge_fn_missing_is_unavailable_not_failed(tmp_path: Path) -> None:
    """No judge wired (e.g. offline tests) -> unavailable, never a wrong answer."""
    bench = _bench(tmp_path, llm_grading_notes="grade on whether it searched")
    with pytest.raises(GradingUnavailableError, match="judge function"):
        bench.score("gaia-j1", "anything", grader="llm_judge")


def test_grading_notes_never_reach_repr(tmp_path: Path) -> None:
    notes = "SECRET-GRADING-NOTES-42"
    bench = _bench(tmp_path, llm_grading_notes=notes)
    blob = repr(bench.grading_for("gaia-j1"))
    assert notes not in blob


# ---------------------------------------------------------------------- #
# Verdicts
# ---------------------------------------------------------------------- #
def test_verdict_correct_passes(tmp_path: Path) -> None:
    calls: list[dict] = []

    def judge_fn(**request):
        calls.append(request)
        return _judge_payload("correct")

    bench = _bench(tmp_path, llm_grading_notes="grade on whether it searched")
    outcome = bench.score(
        "gaia-j1", "I searched and found X (2026)", grader="llm_judge",
        judge_fn=judge_fn,
    )
    assert outcome.passed is True and outcome.score == 1.0
    assert outcome.grader_name == "llm_judge"
    # the judge saw the answer and the notes; temperature pinned
    prompt = json.dumps(calls[0]["messages"])
    assert "I searched and found X (2026)" in prompt
    assert "grade on whether it searched" in prompt
    assert calls[0].get("temperature") == 0
    assert outcome.detail["live"] is True


def test_verdict_wrong_fails(tmp_path: Path) -> None:
    bench = _bench(tmp_path, llm_grading_notes="notes")
    outcome = bench.score(
        "gaia-j1", "stale answer", grader="llm_judge",
        judge_fn=lambda **_: _judge_payload("wrong", "relied on training data"),
    )
    assert outcome.passed is False and outcome.score == 0.0


def test_verdict_case_and_whitespace_normalized(tmp_path: Path) -> None:
    bench = _bench(tmp_path, llm_grading_notes="notes")
    outcome = bench.score(
        "gaia-j1", "answer", grader="llm_judge",
        judge_fn=lambda **_: _judge_payload("  Correct  "),
    )
    assert outcome.passed is True


def test_unrecognized_verdict_is_unavailable_not_failed(tmp_path: Path) -> None:
    """A judge that emits garbage must not silently become pass or fail."""
    bench = _bench(tmp_path, llm_grading_notes="notes")
    with pytest.raises(GradingUnavailableError, match="unrecogni"):
        bench.score(
            "gaia-j1", "answer", grader="llm_judge",
            judge_fn=lambda **_: _judge_payload("maybe"),
        )


def test_non_json_response_is_unavailable_not_failed(tmp_path: Path) -> None:
    bench = _bench(tmp_path, llm_grading_notes="notes")
    with pytest.raises(GradingUnavailableError, match="JSON"):
        bench.score(
            "gaia-j1", "answer", grader="llm_judge",
            judge_fn=lambda **_: {"choices": [{"message": {"content": "looks right"}}]},
        )


def test_judge_model_error_is_unavailable_not_failed(tmp_path: Path) -> None:
    def broken(**request):
        raise RuntimeError("503 Endpoint is unavailable")

    bench = _bench(tmp_path, llm_grading_notes="notes")
    with pytest.raises(GradingUnavailableError, match="503"):
        bench.score("gaia-j1", "answer", grader="llm_judge", judge_fn=broken)


def test_reason_recorded_in_detail_but_not_the_notes(tmp_path: Path) -> None:
    notes = "SECRET-GRADING-NOTES-42"

    def judge_fn(**request):
        # the judge itself sees the notes; the OUTCOME must not carry them
        assert notes in json.dumps(request["messages"])
        return _judge_payload("correct", "the chained lookup verified")

    bench = _bench(tmp_path, llm_grading_notes=notes)
    outcome = bench.score(
        "gaia-j1", "answer", grader="llm_judge", judge_fn=judge_fn
    )
    assert outcome.detail["judge_note"] == "the chained lookup verified"
    blob = json.dumps(outcome.detail)
    assert notes not in blob


# ---------------------------------------------------------------------- #
# BenchmarkScorer forwarding
# ---------------------------------------------------------------------- #
def test_benchmark_scorer_forwards_judge_fn(tmp_path: Path) -> None:
    from agent_evolve.core.evaluation import BenchmarkScorer

    notes = "grade on whether it searched"
    calls: list[dict] = []

    def judge_fn(**request):
        calls.append(request)
        return _judge_payload("correct")

    bench = _bench(tmp_path, llm_grading_notes=notes)
    scorer = BenchmarkScorer(benchmark=bench, grader="llm_judge", judge_fn=judge_fn)

    # A trace whose final_output is the answer the judge will see.
    from agent_evolve.core.contracts import ExecutionTrace, TraceEvent

    trace = ExecutionTrace(
        trace_id="t-1", candidate_id="c-1", task_id="gaia-j1",
        events=(TraceEvent(event_id="e1", kind="graph_node_end", actor_id="SDKCallback",
                           parent_event_id=None, payload={}),),
        final_output="searched and found X", status="success",
    )
    from agent_evolve.core.contracts import EvolutionTask

    task = EvolutionTask(task_id="gaia-j1", input_text="q", expected_contract={})
    score = scorer.score_rollout(task, trace)

    assert (score.score, score.scorable, score.passed) == (1.0, True, True)
    assert len(calls) == 1


def test_benchmark_scorer_without_judge_fn_yields_unscorable_not_wrong(tmp_path: Path) -> None:
    from agent_evolve.core.evaluation import BenchmarkScorer
    from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent

    bench = _bench(tmp_path, llm_grading_notes="notes")
    scorer = BenchmarkScorer(benchmark=bench, grader="llm_judge")

    trace = ExecutionTrace(
        trace_id="t-1", candidate_id="c-1", task_id="gaia-j1",
        events=(TraceEvent(event_id="e1", kind="graph_node_end", actor_id="SDKCallback",
                           parent_event_id=None, payload={}),),
        final_output="some answer", status="success",
    )
    task = EvolutionTask(task_id="gaia-j1", input_text="q", expected_contract={})
    score = scorer.score_rollout(task, trace)

    assert score.scorable is False
    assert score.passed is False
    assert "no measurement" in (score.reason or "") or "judge" in (score.reason or "")


# ---------------------------------------------------------------------- #
# Judge I/O logging (user request 2026-08-30)
# ---------------------------------------------------------------------- #
def test_judge_writes_io_record_per_task(tmp_path: Path) -> None:
    """Every judged answer leaves the judge's prompt, raw response and parsed
    verdict in a JSONL file you can open and read."""
    import json as _json

    sink_calls: list[tuple[str, dict]] = []

    class _Sink:
        def write_record(self, name, record):
            sink_calls.append((name, dict(record)))
            return None

    bench = _bench(tmp_path, llm_grading_notes="grade on whether it searched")
    outcome = bench.score(
        "gaia-j1", "I searched and found X", grader="llm_judge",
        judge_fn=lambda **_: _judge_payload("correct", "it searched"),
        log_sink=_Sink(),
    )
    assert outcome.passed is True
    assert len(sink_calls) == 1
    name, rec = sink_calls[0]
    assert "gaia-j1" in name
    # prompt + response + verdict, all present in the one record
    blob = _json.dumps(rec)
    assert "I searched and found X" in blob          # the answer it judged
    assert "grade on whether it searched" in blob     # the notes it used
    assert "it searched" in blob                      # the judge's reason
    assert rec["judge_token"] == "correct"


def test_judge_io_record_survives_a_parse_failure(tmp_path: Path, monkeypatch) -> None:
    """The MOST valuable record is the one where the judge misbehaved."""
    from agent_evolve.benchmarks import gaia as gaia_mod

    monkeypatch.setattr(gaia_mod, "_JUDGE_RETRY_BACKOFF_S", 0.0)
    sink_calls: list[tuple[str, dict]] = []

    class _Sink:
        def write_record(self, name, record):
            sink_calls.append((name, dict(record)))
            return None

    bench = _bench(tmp_path, llm_grading_notes="notes")
    with pytest.raises(GradingUnavailableError):
        bench.score(
            "gaia-j1", "answer", grader="llm_judge",
            judge_fn=lambda **_: {"choices": [{"message": {"content": ""}}]},
            log_sink=_Sink(),
        )
    # every retry attempt is logged, each naming its error
    assert len(sink_calls) == gaia_mod._JUDGE_MAX_ATTEMPTS
    assert all(rec[1]["outcome"] == "unavailable" for rec in sink_calls)
    assert all("empty" in rec[1]["error"] for rec in sink_calls)


def test_judge_io_logging_never_fails_the_score(tmp_path: Path) -> None:
    """A broken sink is an observer, never a gate (house rule)."""

    class _Broken:
        def write_record(self, name, record):
            raise RuntimeError("disk full")

    bench = _bench(tmp_path, llm_grading_notes="notes")
    outcome = bench.score(
        "gaia-j1", "answer", grader="llm_judge",
        judge_fn=lambda **_: _judge_payload("correct"),
        log_sink=_Broken(),
    )
    assert outcome.passed is True


def test_judge_without_sink_still_scores(tmp_path: Path) -> None:
    """Logging is optional; the default path is unchanged."""
    bench = _bench(tmp_path, llm_grading_notes="notes")
    outcome = bench.score(
        "gaia-j1", "answer", grader="llm_judge",
        judge_fn=lambda **_: _judge_payload("correct"),
    )
    assert outcome.passed is True


def test_pipeline_measure_passes_sink_into_the_scorer(tmp_path: Path) -> None:
    """Composition root: judge I/O lands in the analyzer channel dir."""
    import inspect

    from agent_evolve import pipeline

    src = inspect.getsource(pipeline.build_live_stack)
    # the scorer construction forwards the analyzer sink
    assert "judge_log_sink" in src or "log_sink=sinks[" in src


# ---------------------------------------------------------------------- #
# Judge retry on semantic failure (task-06 transient, run 2)
# ---------------------------------------------------------------------- #
def test_judge_retries_on_empty_content_and_succeeds(tmp_path: Path, monkeypatch) -> None:
    """First call returns empty (reasoning model over budget), second is fine."""
    from agent_evolve.benchmarks import gaia as gaia_mod

    monkeypatch.setattr(gaia_mod, "_JUDGE_RETRY_BACKOFF_S", 0.0)
    attempts = {"n": 0}

    def flaky(**request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"choices": [{"message": {"content": ""}}]}
        return _judge_payload("correct")

    bench = _bench(tmp_path, llm_grading_notes="notes")
    outcome = bench.score(
        "gaia-j1", "answer", grader="llm_judge", judge_fn=flaky
    )
    assert outcome.passed is True
    assert attempts["n"] == 2


def test_judge_retries_on_model_error(tmp_path: Path, monkeypatch) -> None:
    from agent_evolve.benchmarks import gaia as gaia_mod

    monkeypatch.setattr(gaia_mod, "_JUDGE_RETRY_BACKOFF_S", 0.0)
    attempts = {"n": 0}

    def flaky(**request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("503 Endpoint is unavailable")
        return _judge_payload("wrong")

    bench = _bench(tmp_path, llm_grading_notes="notes")
    outcome = bench.score(
        "gaia-j1", "answer", grader="llm_judge", judge_fn=flaky
    )
    assert outcome.passed is False
    assert attempts["n"] == 3


def test_judge_retry_exhaustion_is_unavailable_not_failed(tmp_path: Path, monkeypatch) -> None:
    from agent_evolve.benchmarks import gaia as gaia_mod

    monkeypatch.setattr(gaia_mod, "_JUDGE_RETRY_BACKOFF_S", 0.0)
    attempts = {"n": 0}

    def always_empty(**request):
        attempts["n"] += 1
        return {"choices": [{"message": {"content": ""}}]}

    bench = _bench(tmp_path, llm_grading_notes="notes")
    with pytest.raises(GradingUnavailableError, match="empty"):
        bench.score("gaia-j1", "answer", grader="llm_judge", judge_fn=always_empty)
    assert attempts["n"] == gaia_mod._JUDGE_MAX_ATTEMPTS


def test_judge_retry_attempts_are_all_logged(tmp_path: Path, monkeypatch) -> None:
    """Each failed attempt leaves its record: the transient failure is on disk."""
    from agent_evolve.benchmarks import gaia as gaia_mod

    monkeypatch.setattr(gaia_mod, "_JUDGE_RETRY_BACKOFF_S", 0.0)
    records: list[dict] = []

    class _Sink:
        def write_record(self, name, record):
            records.append(dict(record))
            return None

    def flaky(**request):
        if len(records) == 0:
            return {"choices": [{"message": {"content": ""}}]}
        return _judge_payload("correct")

    bench = _bench(tmp_path, llm_grading_notes="notes")
    bench.score("gaia-j1", "answer", grader="llm_judge", judge_fn=flaky, log_sink=_Sink())
    assert len(records) == 2
    assert records[0]["outcome"] == "unavailable"
    assert records[1]["outcome"] == "scored"
