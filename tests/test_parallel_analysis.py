"""Parallel analyzer fan-out.

Trajectory analysis of distinct (candidate, task) rollout groups is independent
work, so it parallelizes. These tests pin the properties that must hold no
matter whether the analyzer is a single LLM call today or a stateful CUGA agent
later:

* results are ordered by input position, never by completion order;
* one analyzer instance per worker thread, because a CUGA agent carries
  conversation state and must never be shared across threads;
* a failing analyzer isolates to its own work item;
* concurrency is bounded by ``max_workers``.
"""
from __future__ import annotations

import threading
import time

import pytest

from agent_evolve.core.analysis import RolloutGroupReport
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalFinding
from agent_evolve.core.parallel_analysis import (
    AnalysisOutcome,
    ParallelAnalysisRunner,
)


def _report(candidate: str, task: str) -> RolloutGroupReport:
    return RolloutGroupReport(
        candidate_id=candidate,
        task_id=task,
        trace_refs=(f"trace-{candidate}-{task}",),
        rollout_ids=(f"rollout-{candidate}-{task}",),
        sanitized_evidence=(),
    )


def _finding(report: RolloutGroupReport, mechanism: str) -> CausalFinding:
    return CausalFinding(
        verdict_id=f"{report.task_id}:{mechanism}",
        candidate_id=report.candidate_id,
        task_id=report.task_id,
        trace_id=report.trace_refs[0],
        status="observed",
        mechanism_description=mechanism,
        mechanism_cluster_id="c0",
        severity=0.5,
        confidence=0.5,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/a",)),)
        ),
        evidence_refs=("skills/a",),
        rationale="test finding",
    )


class _RecordingAnalyzer:
    """Analyzer that records which thread and instance handled each report."""

    instances: list["_RecordingAnalyzer"] = []

    def __init__(self) -> None:
        self.seen: list[str] = []
        _RecordingAnalyzer.instances.append(self)

    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]:
        self.seen.append(report.task_id)
        return (_finding(report, f"mech-{report.task_id}"),)


def test_results_are_ordered_by_input_position_not_completion_order():
    """A slow first item must still come back first."""

    class StaggeredAnalyzer:
        def analyze(self, report):
            # Earlier tasks sleep longer, so completion order inverts input order.
            delay = {"t0": 0.06, "t1": 0.03, "t2": 0.0}[report.task_id]
            time.sleep(delay)
            return (_finding(report, f"mech-{report.task_id}"),)

    reports = [_report("cand-1", f"t{i}") for i in range(3)]
    runner = ParallelAnalysisRunner(
        analyzer_factory=StaggeredAnalyzer, max_workers=3
    )

    outcomes = runner.run(reports)

    assert [o.report.task_id for o in outcomes] == ["t0", "t1", "t2"]
    assert all(o.ok for o in outcomes)
    assert [o.findings[0].mechanism_description for o in outcomes] == [
        "mech-t0",
        "mech-t1",
        "mech-t2",
    ]


def test_one_analyzer_instance_per_worker_thread_not_shared():
    """A stateful CUGA analyzer must never be shared across threads."""
    _RecordingAnalyzer.instances = []
    seen_by_thread: dict[int, set[int]] = {}
    lock = threading.Lock()

    class ThreadTrackingAnalyzer(_RecordingAnalyzer):
        def analyze(self, report):
            with lock:
                seen_by_thread.setdefault(threading.get_ident(), set()).add(id(self))
            time.sleep(0.02)
            return super().analyze(report)

    reports = [_report("cand-1", f"t{i}") for i in range(8)]
    runner = ParallelAnalysisRunner(
        analyzer_factory=ThreadTrackingAnalyzer, max_workers=4
    )

    outcomes = runner.run(reports)

    assert all(o.ok for o in outcomes)
    # Each thread used exactly one analyzer instance.
    for instance_ids in seen_by_thread.values():
        assert len(instance_ids) == 1
    # Instances were reused across items, not constructed per item.
    assert len(_RecordingAnalyzer.instances) <= 4


def test_failing_analyzer_isolates_to_its_own_work_item():
    """One bad trajectory must not lose the other findings."""

    class FlakyAnalyzer:
        def analyze(self, report):
            if report.task_id == "t1":
                raise RuntimeError("analyzer exploded")
            return (_finding(report, f"mech-{report.task_id}"),)

    reports = [_report("cand-1", f"t{i}") for i in range(3)]
    runner = ParallelAnalysisRunner(analyzer_factory=FlakyAnalyzer, max_workers=3)

    outcomes = runner.run(reports)

    assert [o.ok for o in outcomes] == [True, False, True]
    failed = outcomes[1]
    assert failed.findings == ()
    assert "analyzer exploded" in failed.error
    # The successful neighbours still carry their findings.
    assert outcomes[0].findings[0].mechanism_description == "mech-t0"
    assert outcomes[2].findings[0].mechanism_description == "mech-t2"


def test_concurrency_is_bounded_by_max_workers():
    peak = 0
    active = 0
    lock = threading.Lock()

    class ConcurrencyProbe:
        def analyze(self, report):
            nonlocal peak, active
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return (_finding(report, f"mech-{report.task_id}"),)

    reports = [_report("cand-1", f"t{i}") for i in range(10)]
    runner = ParallelAnalysisRunner(analyzer_factory=ConcurrencyProbe, max_workers=3)

    outcomes = runner.run(reports)

    assert all(o.ok for o in outcomes)
    assert peak <= 3


def test_max_workers_one_runs_inline_without_spawning_threads():
    """Sequential mode must stay debuggable: no worker threads at all."""
    main_thread = threading.get_ident()
    observed: list[int] = []

    class ThreadNotingAnalyzer:
        def analyze(self, report):
            observed.append(threading.get_ident())
            return (_finding(report, f"mech-{report.task_id}"),)

    reports = [_report("cand-1", f"t{i}") for i in range(3)]
    runner = ParallelAnalysisRunner(
        analyzer_factory=ThreadNotingAnalyzer, max_workers=1
    )

    outcomes = runner.run(reports)

    assert all(o.ok for o in outcomes)
    assert observed == [main_thread, main_thread, main_thread]


def test_empty_input_does_no_work_and_builds_no_analyzer():
    built = 0

    def factory():
        nonlocal built
        built += 1
        return _RecordingAnalyzer()

    runner = ParallelAnalysisRunner(analyzer_factory=factory, max_workers=4)

    assert runner.run([]) == ()
    assert built == 0


def test_max_workers_must_be_at_least_one():
    with pytest.raises(ValueError, match="max_workers"):
        ParallelAnalysisRunner(analyzer_factory=_RecordingAnalyzer, max_workers=0)


def test_analyzer_construction_failure_is_reported_not_raised():
    """A factory that cannot build (e.g. missing model env) must not crash the batch."""

    def factory():
        raise RuntimeError("no model configured")

    reports = [_report("cand-1", "t0")]
    runner = ParallelAnalysisRunner(analyzer_factory=factory, max_workers=2)

    outcomes = runner.run(reports)

    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert "no model configured" in outcomes[0].error


def test_outcome_exposes_findings_flattened_across_reports():
    """Callers need one ordered stream of findings for clustering."""
    reports = [_report("cand-1", f"t{i}") for i in range(3)]
    runner = ParallelAnalysisRunner(
        analyzer_factory=_RecordingAnalyzer, max_workers=2
    )

    outcomes = runner.run(reports)
    findings = ParallelAnalysisRunner.flatten(outcomes)

    assert [f.mechanism_description for f in findings] == [
        "mech-t0",
        "mech-t1",
        "mech-t2",
    ]


def test_order_is_stable_under_randomized_delays_over_many_items():
    """Ordering must be structural, not a lucky timing outcome.

    The staggered test above uses only 3 items; with random delays across 40
    items, any reliance on completion order would surface.
    """
    import random

    rng = random.Random(1234)
    delays = {f"t{i}": rng.uniform(0.0, 0.02) for i in range(40)}

    class RandomDelayAnalyzer:
        def analyze(self, report):
            time.sleep(delays[report.task_id])
            return (_finding(report, f"mech-{report.task_id}"),)

    reports = [_report("cand-1", f"t{i}") for i in range(40)]
    runner = ParallelAnalysisRunner(
        analyzer_factory=RandomDelayAnalyzer, max_workers=8
    )

    outcomes = runner.run(reports)

    assert [o.report.task_id for o in outcomes] == [f"t{i}" for i in range(40)]
    assert ParallelAnalysisRunner.flatten(outcomes) != ()
    assert [f.mechanism_description for f in ParallelAnalysisRunner.flatten(outcomes)] == [
        f"mech-t{i}" for i in range(40)
    ]


def test_failures_helper_returns_only_failed_outcomes():
    class HalfFailingAnalyzer:
        def analyze(self, report):
            if report.task_id in {"t1", "t3"}:
                raise ValueError(f"bad {report.task_id}")
            return (_finding(report, f"mech-{report.task_id}"),)

    reports = [_report("cand-1", f"t{i}") for i in range(4)]
    runner = ParallelAnalysisRunner(
        analyzer_factory=HalfFailingAnalyzer, max_workers=4
    )

    outcomes = runner.run(reports)
    failures = ParallelAnalysisRunner.failures(outcomes)

    assert [f.report.task_id for f in failures] == ["t1", "t3"]
    # flatten() skips failures rather than raising.
    assert [f.mechanism_description for f in ParallelAnalysisRunner.flatten(outcomes)] == [
        "mech-t0",
        "mech-t2",
    ]


def test_workers_never_exceed_item_count():
    """10 configured workers for 2 items must not spawn 10 CUGA agents."""
    built = 0
    lock = threading.Lock()

    class CountingAnalyzer:
        def __init__(self) -> None:
            nonlocal built
            with lock:
                built += 1

        def analyze(self, report):
            time.sleep(0.01)
            return (_finding(report, f"mech-{report.task_id}"),)

    reports = [_report("cand-1", f"t{i}") for i in range(2)]
    runner = ParallelAnalysisRunner(
        analyzer_factory=CountingAnalyzer, max_workers=10
    )

    outcomes = runner.run(reports)

    assert all(o.ok for o in outcomes)
    assert built <= 2


def test_core_parallel_analysis_imports_no_agent_implementation():
    """The boundary rule: core must never import cuga or an adapter."""
    import ast

    import agent_evolve.core.parallel_analysis as mod

    tree = ast.parse(open(str(mod.__file__), encoding="utf-8").read())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    assert imported, "expected at least one import to be checked"
    for name in imported:
        assert not name.startswith("cuga"), f"core imported {name}"
        assert "adapters" not in name, f"core imported {name}"


class TestAnalysisOutcome:
    def test_ok_outcome_rejects_an_error_string(self):
        with pytest.raises(ValueError):
            AnalysisOutcome(
                report=_report("c", "t"),
                findings=(),
                error="boom",
                ok=True,
            )

    def test_failed_outcome_requires_an_error(self):
        with pytest.raises(ValueError):
            AnalysisOutcome(
                report=_report("c", "t"),
                findings=(),
                error="",
                ok=False,
            )
