"""The CausalFinding -> CausalAnalysis bridge and the dual-protocol shim.

Two incompatible ``AnalyzerJudge`` protocols exist in the core:

* :mod:`agent_evolve.core.analyzer` -- ``analyze(task, trace) -> CausalAnalysis``.
  What the orchestrator's call sites actually invoke. ``CausalAnalysis`` carries
  a ``score``.
* :mod:`agent_evolve.core.analysis` -- ``analyze(report) -> tuple[CausalFinding, ...]``.
  Rollout-group aware, carries ``status``/``confidence`` so abstention is
  expressible, and deliberately carries no score.

These tests pin the boundary between them: a finding is a *diagnosis*, a score is
a *measurement*, so the converter must never invent one from the other, and an
abstention must never be laundered into a confident blame graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analysis import RolloutGroupReport  # noqa: E402
from agent_evolve.core.analyzer import (  # noqa: E402
    FakeAnalyzerJudge,
    ReportAnalyzerShim,
    analyze_groups,
    as_legacy_analyzer,
    contract_score,
    is_report_analyzer,
)
from agent_evolve.core.blame import (  # noqa: E402
    ABSTAINED_MECHANISM_PREFIX,
    UNANALYZED_MECHANISM,
    BlameGraph,
    BlameNode,
    CausalFinding,
    analysis_from_finding,
    is_placeholder_mechanism,
    unanalyzed_analysis,
)
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def _task(task_id: str = "task-1", expected: str = "widget") -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"do {task_id}",
        expected_contract={"expected_substring": expected},
    )


def _trace(
    *,
    trace_id: str = "tr-1",
    candidate_id: str = "cand-1",
    task_id: str = "task-1",
    final_output: str = "nope",
    actors: tuple[str, ...] = ("planner", "tool"),
) -> ExecutionTrace:
    events = tuple(
        TraceEvent(
            event_id=f"e{i}",
            kind="tool_call",
            actor_id=actor,
            parent_event_id=None,
            payload={"name": actor},
        )
        for i, actor in enumerate(actors)
    )
    return ExecutionTrace(
        trace_id=trace_id,
        candidate_id=candidate_id,
        task_id=task_id,
        events=events,
        final_output=final_output,
        status="completed",
    )


def _observed_finding(**overrides: object) -> CausalFinding:
    kwargs: dict[str, object] = dict(
        verdict_id="v-1",
        candidate_id="cand-1",
        task_id="task-1",
        trace_id="tr-1",
        status="observed",
        mechanism_description="retriever returned stale documents",
        mechanism_cluster_id="c0",
        severity=0.75,
        confidence=0.9,
        blame_graph=BlameGraph(
            nodes=(
                BlameNode(actor_id="retriever", blame=0.8, artifacts=("skills/retrieval",)),
                BlameNode(actor_id="planner", blame=0.2),
            )
        ),
        evidence_refs=("skills/retrieval", "tr-1#e0"),
        rationale="the retrieval tool call returned documents from a stale index",
        counterfactual_notes=("a fresh index would have returned the target doc",),
    )
    kwargs.update(overrides)
    return CausalFinding(**kwargs)  # type: ignore[arg-type]


def _abstained_finding(status: str, **overrides: object) -> CausalFinding:
    kwargs: dict[str, object] = dict(
        verdict_id=f"v-{status}",
        candidate_id="cand-1",
        task_id="task-1",
        trace_id="tr-1",
        status=status,
        rationale=f"analyzer could not conclude: {status}",
    )
    kwargs.update(overrides)
    return CausalFinding(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# analysis_from_finding: the score must be supplied, never invented
# ---------------------------------------------------------------------- #
def test_analysis_from_finding_requires_a_score():
    """A finding carries no measurement; the converter must not fabricate one."""
    with pytest.raises(TypeError):
        analysis_from_finding(_observed_finding())  # type: ignore[call-arg]


def test_analysis_from_finding_score_is_keyword_only():
    with pytest.raises(TypeError):
        analysis_from_finding(_observed_finding(), 0.0)  # type: ignore[misc]


def test_analysis_from_finding_carries_the_supplied_score_verbatim():
    analysis = analysis_from_finding(_observed_finding(), score=0.25)
    assert analysis.score == 0.25


def test_analysis_from_finding_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        analysis_from_finding(_observed_finding(), score=1.5)


def test_analysis_from_finding_does_not_derive_score_from_severity():
    """severity is a judged impact, score is a measured outcome: independent."""
    high_severity = analysis_from_finding(
        _observed_finding(severity=1.0), score=1.0
    )
    assert high_severity.severity == 1.0
    assert high_severity.score == 1.0


# ---------------------------------------------------------------------- #
# status == observed
# ---------------------------------------------------------------------- #
def test_observed_finding_maps_mechanism_severity_and_blame_graph():
    finding = _observed_finding()
    analysis = analysis_from_finding(finding, score=0.0)
    assert analysis.mechanism == "retriever returned stale documents"
    assert analysis.severity == 0.75
    assert analysis.blame_graph == finding.blame_graph
    assert analysis.counterfactual_evidence == finding.counterfactual_notes


def test_observed_mechanism_is_not_a_placeholder():
    analysis = analysis_from_finding(_observed_finding(), score=0.0)
    assert not is_placeholder_mechanism(analysis.mechanism)


def test_observed_finding_preserves_blamed_artifacts():
    analysis = analysis_from_finding(_observed_finding(), score=0.0)
    assert analysis.artifact_ids == ("skills/retrieval",)


def test_observed_finding_forwards_model_ids():
    analysis = analysis_from_finding(
        _observed_finding(),
        score=0.0,
        analyzer_model_id="real-analyzer",
        judge_model_id="real-judge",
    )
    assert analysis.analyzer_model_id == "real-analyzer"
    assert analysis.judge_model_id == "real-judge"


# ---------------------------------------------------------------------- #
# Abstention statuses must stay visibly abstentions
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status", ["insufficient_evidence", "uncertain", "malformed"]
)
def test_abstention_yields_a_reserved_placeholder_mechanism(status: str):
    analysis = analysis_from_finding(_abstained_finding(status), score=0.0)
    assert analysis.mechanism.startswith(ABSTAINED_MECHANISM_PREFIX)
    assert status in analysis.mechanism
    assert is_placeholder_mechanism(analysis.mechanism)


@pytest.mark.parametrize(
    "status", ["insufficient_evidence", "uncertain", "malformed"]
)
def test_abstention_never_becomes_a_confident_blame_graph(status: str):
    """No blame may be manufactured from an absence of evidence."""
    finding = _abstained_finding(
        status,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="retriever", blame=1.0, artifacts=("a",)),)
        ),
    )
    analysis = analysis_from_finding(finding, score=0.0)
    assert analysis.blame_graph.nodes == ()
    assert analysis.blame_graph.total_blame() == 0.0
    assert analysis.artifact_ids == ()


@pytest.mark.parametrize(
    "status", ["insufficient_evidence", "uncertain", "malformed"]
)
def test_abstention_has_zero_severity(status: str):
    """Judged impact is unclaimable without a conclusion."""
    analysis = analysis_from_finding(_abstained_finding(status), score=0.0)
    assert analysis.severity == 0.0


@pytest.mark.parametrize(
    "status", ["insufficient_evidence", "uncertain", "malformed"]
)
def test_abstention_preserves_the_rationale_as_evidence_prose(status: str):
    finding = _abstained_finding(status)
    analysis = analysis_from_finding(finding, score=0.0)
    assert finding.rationale in analysis.counterfactual_evidence


def test_abstention_ignores_a_mechanism_description_it_cannot_stand_behind():
    """A non-observed finding may still name a guess; it must not be promoted."""
    finding = _abstained_finding(
        "uncertain", mechanism_description="maybe the retriever"
    )
    analysis = analysis_from_finding(finding, score=0.0)
    assert "maybe the retriever" not in analysis.mechanism
    assert is_placeholder_mechanism(analysis.mechanism)


def test_abstention_still_carries_the_measured_score():
    """Abstaining on the diagnosis does not discard the measurement."""
    analysis = analysis_from_finding(
        _abstained_finding("insufficient_evidence"), score=0.5
    )
    assert analysis.score == 0.5


# ---------------------------------------------------------------------- #
# The minimal-profile placeholder
# ---------------------------------------------------------------------- #
def test_unanalyzed_analysis_uses_the_reserved_sentinel():
    analysis = unanalyzed_analysis(score=0.0, actor_id="planner")
    assert analysis.mechanism == UNANALYZED_MECHANISM
    assert is_placeholder_mechanism(analysis.mechanism)


def test_unanalyzed_sentinel_is_distinguishable_from_a_real_mechanism():
    real = analysis_from_finding(_observed_finding(), score=0.0)
    placeholder = unanalyzed_analysis(score=0.0, actor_id="planner")
    assert is_placeholder_mechanism(placeholder.mechanism)
    assert not is_placeholder_mechanism(real.mechanism)
    assert placeholder.mechanism != real.mechanism


def test_unanalyzed_sentinel_does_not_vary_with_task():
    """The old template embedded task_id, implying task-specific diagnosis.

    With no analyzer there is no diagnosis to be task-specific about, so the
    sentinel is constant: one obviously-not-a-mechanism value rather than one
    fake mechanism per task.
    """
    a = unanalyzed_analysis(score=0.0, actor_id="planner")
    b = unanalyzed_analysis(score=0.0, actor_id="tool")
    assert a.mechanism == b.mechanism


def test_unanalyzed_analysis_keeps_the_blamed_actor():
    analysis = unanalyzed_analysis(score=0.0, actor_id="planner")
    assert analysis.actor_ids == ("planner",)
    assert analysis.artifact_ids == ()


def test_unanalyzed_analysis_carries_the_supplied_score():
    assert unanalyzed_analysis(score=0.0, actor_id="x").score == 0.0


def test_placeholder_predicate_rejects_ordinary_mechanisms():
    assert not is_placeholder_mechanism("none")
    assert not is_placeholder_mechanism("retriever returned stale documents")


# ---------------------------------------------------------------------- #
# Explicit protocol detection
# ---------------------------------------------------------------------- #
class _ReportAnalyzer:
    """Report-based analyzer: one finding per trace in the report."""

    analyzer_model_id = "report-analyzer"

    def __init__(self, findings_for=None) -> None:
        self.seen: list[RolloutGroupReport] = []
        self._findings_for = findings_for

    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]:
        self.seen.append(report)
        if self._findings_for is not None:
            return self._findings_for(report)
        return (
            _observed_finding(
                candidate_id=report.candidate_id,
                task_id=report.task_id,
                trace_id=report.trace_refs[0],
            ),
        )


def test_is_report_analyzer_detects_the_report_protocol():
    assert is_report_analyzer(_ReportAnalyzer()) is True


def test_is_report_analyzer_detects_the_legacy_protocol():
    assert is_report_analyzer(FakeAnalyzerJudge()) is False


def test_is_report_analyzer_rejects_an_unrecognisable_signature():
    class _Weird:
        def analyze(self, a, b, c):  # noqa: ANN001
            return ()

    with pytest.raises(TypeError):
        is_report_analyzer(_Weird())


def test_is_report_analyzer_rejects_an_object_without_analyze():
    with pytest.raises(TypeError):
        is_report_analyzer(object())


def test_detection_does_not_call_analyze():
    """Detection must be structural, not a try/except around a real call."""
    analyzer = _ReportAnalyzer()
    is_report_analyzer(analyzer)
    assert analyzer.seen == []


# ---------------------------------------------------------------------- #
# as_legacy_analyzer: legacy passthrough must be byte-identical
# ---------------------------------------------------------------------- #
def test_as_legacy_analyzer_returns_a_legacy_analyzer_unchanged():
    legacy = FakeAnalyzerJudge()
    assert as_legacy_analyzer(legacy) is legacy


def test_as_legacy_analyzer_preserves_legacy_verdicts_exactly():
    legacy = FakeAnalyzerJudge()
    task, trace = _task(), _trace()
    assert as_legacy_analyzer(legacy).analyze(task, trace) == legacy.analyze(
        task, trace
    )


def test_as_legacy_analyzer_wraps_a_report_analyzer():
    wrapped = as_legacy_analyzer(_ReportAnalyzer())
    assert isinstance(wrapped, ReportAnalyzerShim)


# ---------------------------------------------------------------------- #
# The shim
# ---------------------------------------------------------------------- #
def test_shim_satisfies_the_legacy_call_signature():
    shim = as_legacy_analyzer(_ReportAnalyzer())
    analysis = shim.analyze(_task(), _trace())
    assert analysis.mechanism == "retriever returned stale documents"


def test_shim_builds_a_sanitized_report_from_the_trace():
    analyzer = _ReportAnalyzer()
    as_legacy_analyzer(analyzer).analyze(_task(), _trace())
    assert len(analyzer.seen) == 1
    report = analyzer.seen[0]
    assert report.candidate_id == "cand-1"
    assert report.task_id == "task-1"
    assert report.trace_refs == ("tr-1",)


def test_shim_never_forwards_the_final_output_or_the_answer_key():
    analyzer = _ReportAnalyzer()
    task = _task(expected="SECRET-ANSWER")
    trace = _trace(final_output="SECRET-ANSWER is here")
    as_legacy_analyzer(analyzer).analyze(task, trace)
    blob = repr(analyzer.seen[0])
    assert "SECRET-ANSWER" not in blob


def test_shim_scores_via_the_injected_scorer_not_the_analyzer():
    shim = as_legacy_analyzer(_ReportAnalyzer(), score_fn=lambda task, trace: 0.5)
    assert shim.analyze(_task(), _trace()).score == 0.5


def test_shim_default_scorer_is_the_contract_check():
    shim = as_legacy_analyzer(_ReportAnalyzer())
    miss = shim.analyze(_task(expected="widget"), _trace(final_output="nope"))
    hit = shim.analyze(_task(expected="widget"), _trace(final_output="a widget"))
    assert miss.score == 0.0
    assert hit.score == 1.0


def test_shim_abstains_visibly_when_the_analyzer_returns_no_finding():
    shim = as_legacy_analyzer(_ReportAnalyzer(findings_for=lambda r: ()))
    analysis = shim.analyze(_task(), _trace())
    assert is_placeholder_mechanism(analysis.mechanism)
    assert analysis.blame_graph.nodes == ()
    assert analysis.severity == 0.0


def test_shim_abstention_still_reports_the_measured_score():
    shim = as_legacy_analyzer(
        _ReportAnalyzer(findings_for=lambda r: ()),
        score_fn=lambda task, trace: 1.0,
    )
    assert shim.analyze(_task(), _trace()).score == 1.0


def test_shim_propagates_an_abstained_finding_as_an_abstention():
    shim = as_legacy_analyzer(
        _ReportAnalyzer(
            findings_for=lambda r: (_abstained_finding("insufficient_evidence"),)
        )
    )
    analysis = shim.analyze(_task(), _trace())
    assert is_placeholder_mechanism(analysis.mechanism)
    assert "insufficient_evidence" in analysis.mechanism


def test_shim_rejects_multiple_findings_for_a_single_trace():
    """A one-trace report yields one finding; more is a contract violation.

    Silently keeping the first would discard a verdict the analyzer produced.
    """
    shim = as_legacy_analyzer(
        _ReportAnalyzer(
            findings_for=lambda r: (_observed_finding(), _observed_finding())
        )
    )
    with pytest.raises(ValueError, match="one finding"):
        shim.analyze(_task(), _trace())


def test_shim_exposes_model_ids_for_score_provenance():
    shim = as_legacy_analyzer(_ReportAnalyzer())
    assert shim.analyzer_model_id == "report-analyzer"
    assert isinstance(shim.judge_model_id, str)


def test_shim_stamps_model_ids_onto_the_analysis():
    shim = as_legacy_analyzer(_ReportAnalyzer())
    analysis = shim.analyze(_task(), _trace())
    assert analysis.analyzer_model_id == "report-analyzer"


def test_shim_does_not_swallow_analyzer_errors():
    """A real failure must surface, not be converted into an abstention."""

    def boom(report: RolloutGroupReport):
        raise RuntimeError("model unreachable")

    shim = as_legacy_analyzer(_ReportAnalyzer(findings_for=boom))
    with pytest.raises(RuntimeError, match="model unreachable"):
        shim.analyze(_task(), _trace())


# ---------------------------------------------------------------------- #
# contract_score
# ---------------------------------------------------------------------- #
def test_contract_score_matches_substring():
    assert contract_score(_task(expected="widget"), _trace(final_output="a widget")) == 1.0


def test_contract_score_misses_substring():
    assert contract_score(_task(expected="widget"), _trace(final_output="nope")) == 0.0


def test_contract_score_matches_regex():
    task = EvolutionTask(
        task_id="t",
        input_text="i",
        expected_contract={"expected_regex": r"wid\w+"},
    )
    assert contract_score(task, _trace(final_output="widget")) == 1.0


def test_contract_score_with_no_contract_is_a_miss():
    task = EvolutionTask(task_id="t", input_text="i", expected_contract={})
    assert contract_score(task, _trace(final_output="anything")) == 0.0


def test_contract_score_agrees_with_the_fake_analyzer():
    """The extracted scorer must not change the fake analyzer's verdicts."""
    fake = FakeAnalyzerJudge()
    for output in ("a widget", "nope", "", "widget widget"):
        task, trace = _task(expected="widget"), _trace(final_output=output)
        assert contract_score(task, trace) == fake.analyze(task, trace).score


# ---------------------------------------------------------------------- #
# Group analysis entry point
# ---------------------------------------------------------------------- #
def test_analyze_groups_returns_one_outcome_per_group_in_input_order():
    groups = [
        (_task("task-1"), [_trace(trace_id="tr-1", task_id="task-1")]),
        (_task("task-2"), [_trace(trace_id="tr-2", task_id="task-2")]),
    ]
    outcomes = analyze_groups(_ReportAnalyzer, groups)
    assert [o.report.task_id for o in outcomes] == ["task-1", "task-2"]
    assert all(o.ok for o in outcomes)


def test_analyze_groups_accepts_a_multi_rollout_group():
    traces = [
        _trace(trace_id="tr-1"),
        _trace(trace_id="tr-2"),
        _trace(trace_id="tr-3"),
    ]
    outcomes = analyze_groups(_ReportAnalyzer, [(_task(), traces)])
    assert outcomes[0].report.trace_refs == ("tr-1", "tr-2", "tr-3")


def test_analyze_groups_isolates_a_per_group_failure():
    def maybe_boom(report: RolloutGroupReport):
        if report.task_id == "task-1":
            raise RuntimeError("boom")
        return (
            _observed_finding(task_id=report.task_id, trace_id=report.trace_refs[0]),
        )

    groups = [
        (_task("task-1"), [_trace(trace_id="tr-1", task_id="task-1")]),
        (_task("task-2"), [_trace(trace_id="tr-2", task_id="task-2")]),
    ]
    outcomes = analyze_groups(
        lambda: _ReportAnalyzer(findings_for=maybe_boom), groups
    )
    assert outcomes[0].ok is False
    assert "boom" in outcomes[0].error
    assert outcomes[1].ok is True


def test_analyze_groups_honors_max_workers():
    groups = [
        (_task(f"task-{i}"), [_trace(trace_id=f"tr-{i}", task_id=f"task-{i}")])
        for i in range(4)
    ]
    outcomes = analyze_groups(_ReportAnalyzer, groups, max_workers=3)
    assert [o.report.task_id for o in outcomes] == [f"task-{i}" for i in range(4)]


def test_analyze_groups_rejects_a_legacy_analyzer_factory():
    with pytest.raises(TypeError):
        analyze_groups(FakeAnalyzerJudge, [(_task(), [_trace()])])


def test_analyze_groups_with_no_groups_returns_nothing():
    assert analyze_groups(_ReportAnalyzer, []) == ()


def test_analyze_groups_sanitizes_every_report():
    seen: list[RolloutGroupReport] = []
    task = _task(expected="SECRET")
    groups = [(task, [_trace(final_output="SECRET leaked")])]

    def capture(report: RolloutGroupReport):
        seen.append(report)
        return ()

    analyze_groups(lambda: _ReportAnalyzer(findings_for=capture), groups)
    assert "SECRET" not in repr(seen[0])


# ---------------------------------------------------------------------- #
# Orchestrator wiring: no degenerate mechanism templates
# ---------------------------------------------------------------------- #
def _orchestrator(profile, analyzer):
    from agent_evolve.core.contracts import EvolutionCandidate
    from agent_evolve.core.fake_editor import FakeEditor
    from agent_evolve.core.orchestrator import Orchestrator
    from agent_evolve.core.pool import PersistentPool
    from examples.fake_adapter import FakeAdapter

    orch = Orchestrator(
        adapter=FakeAdapter(),
        analyzer_judge=analyzer,
        editor=FakeEditor(),
        pool=PersistentPool(),
        profile=profile,
    )
    orch.initialize_base(
        EvolutionCandidate(
            candidate_id="base",
            version="base-v0",
            artifact_hashes={
                "skills/retrieval": "h1",
                "policies/execution": "h2",
                "prompts/system": "h3",
            },
        )
    )
    return orch


def _recorded_mechanisms(orch) -> list[str]:
    """Every mechanism the orchestrator sent into clustering, via the clusterer."""
    return list(orch._observed_mechanisms)


def test_minimal_profile_records_the_unanalyzed_sentinel_not_a_task_template():
    """The old ``failed-to-match-{task_id}`` template must be gone."""
    from agent_evolve.core.orchestrator import MINIMAL

    orch = _orchestrator(MINIMAL, FakeAnalyzerJudge())
    orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
    mechanisms = _recorded_mechanisms(orch)
    assert mechanisms, "the iteration recorded no mechanism at all"
    assert not any("failed-to-match-task-1" == m for m in mechanisms)
    assert UNANALYZED_MECHANISM in mechanisms


def test_minimal_profile_mechanisms_are_all_flagged_as_placeholders():
    """A profile with no analyzer must never look like it has real mechanisms."""
    from agent_evolve.core.orchestrator import MINIMAL

    orch = _orchestrator(MINIMAL, FakeAnalyzerJudge())
    orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
    for mechanism in _recorded_mechanisms(orch):
        assert mechanism == "none" or is_placeholder_mechanism(mechanism), mechanism


def test_minimal_profile_sentinel_is_shared_across_tasks():
    """Two different failing tasks must not fabricate two distinct mechanisms."""
    from agent_evolve.core.orchestrator import MINIMAL

    orch = _orchestrator(MINIMAL, FakeAnalyzerJudge())
    orch.run_iteration(
        [
            _task("task-1", expected="graphrag-retrieval"),
            _task("task-2", expected="semantic-cache"),
        ]
    )
    placeholders = {
        m for m in _recorded_mechanisms(orch) if is_placeholder_mechanism(m)
    }
    assert placeholders == {UNANALYZED_MECHANISM}


def test_causal_blame_profile_records_the_real_analyzer_mechanism():
    """With an analyzer wired, real mechanisms must reach clustering unchanged."""
    from agent_evolve.core.orchestrator import RESEARCH_SEQUENTIAL

    orch = _orchestrator(RESEARCH_SEQUENTIAL, FakeAnalyzerJudge())
    orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
    mechanisms = _recorded_mechanisms(orch)
    assert any(not is_placeholder_mechanism(m) and m != "none" for m in mechanisms)


def test_orchestrator_accepts_a_report_based_analyzer():
    """The dual-protocol shim must let a report analyzer drive an iteration."""
    from agent_evolve.core.orchestrator import RESEARCH_SEQUENTIAL

    orch = _orchestrator(RESEARCH_SEQUENTIAL, _ReportAnalyzer())
    result = orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
    assert result.iteration == 1
    assert "retriever returned stale documents" in _recorded_mechanisms(orch)


def test_orchestrator_keeps_a_legacy_analyzer_by_identity():
    """Wrapping a legacy analyzer would change behaviour; it must not happen."""
    from agent_evolve.core.orchestrator import RESEARCH_SEQUENTIAL

    legacy = FakeAnalyzerJudge()
    orch = _orchestrator(RESEARCH_SEQUENTIAL, legacy)
    assert orch.resolved_analyzer is legacy


def test_orchestrator_wraps_a_report_analyzer_in_the_shim():
    from agent_evolve.core.orchestrator import RESEARCH_SEQUENTIAL

    orch = _orchestrator(RESEARCH_SEQUENTIAL, _ReportAnalyzer())
    assert isinstance(orch.resolved_analyzer, ReportAnalyzerShim)


def test_runner_accepts_a_report_based_analyzer():
    """SequentialGepaRunner shares the same resolution path."""
    from agent_evolve.core.orchestrator import SequentialGepaRunner
    from agent_evolve.core.pool import PersistentPool
    from agent_evolve.core.fake_editor import FakeEditor
    from examples.fake_adapter import FakeAdapter

    runner = SequentialGepaRunner(
        adapter=FakeAdapter(),
        pool=PersistentPool(),
        analyzer_judge=_ReportAnalyzer(),
        editor=FakeEditor(),
    )
    assert isinstance(runner.resolved_analyzer, ReportAnalyzerShim)


def test_runner_keeps_a_legacy_analyzer_by_identity():
    from agent_evolve.core.orchestrator import SequentialGepaRunner
    from agent_evolve.core.pool import PersistentPool
    from agent_evolve.core.fake_editor import FakeEditor
    from examples.fake_adapter import FakeAdapter

    legacy = FakeAnalyzerJudge()
    runner = SequentialGepaRunner(
        adapter=FakeAdapter(),
        pool=PersistentPool(),
        analyzer_judge=legacy,
        editor=FakeEditor(),
    )
    assert runner.resolved_analyzer is legacy


def test_base_issue_synthesis_no_longer_fabricates_a_mechanism():
    """The ``base-failed-{task}-{cluster}`` template was equally degenerate."""
    from agent_evolve.core.orchestrator import MINIMAL

    orch = _orchestrator(MINIMAL, FakeAnalyzerJudge())
    orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
    for mechanism in orch._issue_mechanisms:
        assert "base-failed-task-1" not in mechanism


def test_issue_synthesis_reuses_the_real_analysis_from_the_base_rollout():
    """Step 2 must reuse step 1's analysis, not fabricate a replacement.

    ``run_iteration`` rolls the base out and analyzes it, then builds issues from
    the score tensor. The analysis is already in hand at that point, so
    discarding it and synthesizing a stand-in is pure information loss.
    """
    from agent_evolve.core.orchestrator import RESEARCH_SEQUENTIAL

    orch = _orchestrator(RESEARCH_SEQUENTIAL, _ReportAnalyzer())
    orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
    assert orch._issue_mechanisms
    assert "retriever returned stale documents" in orch._issue_mechanisms


def test_issue_synthesis_keeps_the_real_blame_graph():
    """The editor selects its target from the blame graph; it must be the real one."""
    from agent_evolve.core.orchestrator import RESEARCH_SEQUENTIAL

    orch = _orchestrator(RESEARCH_SEQUENTIAL, FakeAnalyzerJudge())
    orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
    assert orch._issue_blame_actors
    assert orch._issue_blame_actors != [()], "blame graph was dropped"


def test_issue_synthesis_abstains_when_no_analysis_was_retained():
    """A cell with no retained analysis must abstain, not fabricate one."""
    from agent_evolve.core.orchestrator import MINIMAL

    orch = _orchestrator(MINIMAL, FakeAnalyzerJudge())
    orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
    # Simulate a stale cell from a prior process with no retained analysis.
    orch._base_analyses.clear()
    orch._issue_mechanisms.clear()
    orch.run_iteration([_task("task-1", expected="graphrag-retrieval")])
