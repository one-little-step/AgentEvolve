"""Tests for the analysis exchange record and analyzer protocol."""
from __future__ import annotations

import pytest

from agent_evolve.core.analysis import AnalyzerJudge, RolloutGroupReport
from agent_evolve.core.blame import CausalFinding


def _report(**overrides) -> RolloutGroupReport:
    kwargs = dict(
        candidate_id="cand-1",
        task_id="task-1",
        trace_refs=("trace-1",),
        rollout_ids=("rollout-1",),
        sanitized_evidence=({"final_output": "ok"},),
    )
    kwargs.update(overrides)
    return RolloutGroupReport(**kwargs)


def test_rollout_group_report_holds_references() -> None:
    r = _report()
    assert r.candidate_id == "cand-1"
    assert r.task_id == "task-1"
    assert r.trace_refs == ("trace-1",)
    assert r.rollout_ids == ("rollout-1",)
    assert r.sanitized_evidence == ({"final_output": "ok"},)


def test_rollout_group_report_is_frozen() -> None:
    r = _report()
    with pytest.raises(AttributeError):
        r.candidate_id = "other"  # type: ignore[misc]


class _FakeAnalyzer:
    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]:
        return (
            CausalFinding(
                verdict_id="v-1",
                candidate_id=report.candidate_id,
                task_id=report.task_id,
                trace_id=report.trace_refs[0],
                status="observed",
                mechanism_description="bad retrieval",
                mechanism_cluster_id="cluster-1",
                severity=0.8,
                confidence=0.9,
                evidence_refs=report.trace_refs,
                rationale="trace-backed",
            ),
        )


def test_analyzer_protocol_returns_trace_backed_findings() -> None:
    analyzer: AnalyzerJudge = _FakeAnalyzer()
    findings = analyzer.analyze(_report())
    assert len(findings) == 1
    assert findings[0].status == "observed"
    assert findings[0].evidence_refs == ("trace-1",)
