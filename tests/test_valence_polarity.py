"""VAL: the valence field with structural polarity isolation (D5 step VAL).

Governing contracts:
* ``docs/design/issue-lifecycle.md`` D5.2/D5.3 -- direction lives in a dedicated
  ``valence`` field (+1 fault / -1 strength); magnitude stays in ``severity``,
  always fractional [0, 1], for both polarities. Negative severity is forbidden
  twice over: guards reject it, and ``w_severity * issue.severity``
  (``core/issues.py``) would make a strength subtract from issue quality.
* Decision made with the user 2026-08-23: **the model never picks its own
  sign** -- magnitude is the judge's opinion, polarity is added by code.
* D5.3: each judge sees exactly ONE polarity, enforced structurally:
    - ``CausalFinding``'s validator rejects any valence outside {-1, +1}
      (floor: garbage impossible anywhere);
    - ``finding_from_analysis`` constructs valence=+1 explicitly (wall: Judge 1
      findings are faults by construction);
    - the parallel-analysis receive site REFUSES a non-fault finding loudly
      rather than flipping it silently (an adapter bug must surface).
* Q6 answered here too: the sign is carried onto ``Issue`` so downstream
  ranking can read it; a propagation test pins that it survives the journey.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analysis import RolloutGroupReport  # noqa: E402
from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.blame import (  # noqa: E402
    BlameGraph,
    BlameNode,
    CausalFinding,
    analysis_from_finding,
)
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    ArtifactDescriptor,
    EvolutionCandidate,
    EvolutionTask,
)
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.issues import Issue, build_issue  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"


def _task(task_id: str = "task-a", expected: str = _TOKEN) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": expected},
    )


def _finding(**overrides: Any) -> CausalFinding:
    """A minimal valid finding (uncertain status skips evidence checks)."""
    fields: dict[str, Any] = dict(
        verdict_id="v-1",
        candidate_id="cand",
        task_id="task-a",
        trace_id="tr-1",
        status="uncertain",
        rationale="test rationale",
    )
    fields.update(overrides)
    return CausalFinding(**fields)


def _runner(analyzer_judge: FakeAnalyzerJudge) -> SequentialGepaRunner:
    adapter = FakeAdapter()
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base",
            version="base-v0",
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in adapter.artifact_inventory("base-v0")
            },
        )
    )
    return SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=analyzer_judge,
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
    )


# ---------------------------------------------------------------------- #
# The floor: the validator rejects anything that is not a polarity
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, 2, -2, 1.5, "+1"])
def test_causal_finding_rejects_anything_that_is_not_a_polarity(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        _finding(valence=bad)


def test_default_valence_is_fault_the_historical_polarity() -> None:
    assert _finding().valence == 1


def test_negative_one_is_a_valid_strength() -> None:
    assert _finding(valence=-1).valence == -1


def test_severity_stays_fractional_for_both_polarities() -> None:
    strong_strength = _finding(valence=-1, severity=0.9)
    strong_fault = _finding(valence=1, severity=0.9)
    assert strong_strength.severity == strong_fault.severity == 0.9


# ---------------------------------------------------------------------- #
# Wall 1: Judge 1's seam produces faults by construction
# ---------------------------------------------------------------------- #
def test_finding_from_analysis_always_emits_faults() -> None:
    runner = _runner(FakeAnalyzerJudge())
    analysis = FakeAnalyzerJudge().analyze(_task(), _failing_trace())

    finding = runner.finding_from_analysis(
        analysis,
        task=_task(),
        candidate_id="cand",
        trace_id="tr-1",
        verdict_id="v-x",
        writable_artifact_ids=("skills/retrieval",),
    )

    assert finding.status == "observed"
    assert finding.valence == 1


def _failing_trace():
    from agent_evolve.core.contracts import ExecutionTrace

    return ExecutionTrace(
        trace_id="tr-fail",
        candidate_id="cand",
        task_id="task-a",
        events=(),
        final_output="wrong answer",
        status="success",
    )


# ---------------------------------------------------------------------- #
# Wall 2: the parallel path refuses a smuggled strength, loudly
# ---------------------------------------------------------------------- #
class _StrengthSmuggler:
    """Speaks the parallel path's REPORT protocol, but reports a strength.

    The parallel analyzer receives ``RolloutGroupReport`` objects and returns
    findings -- this is the door an adapter-backed Judge 1 walks through, so
    it is the door a mis-wired adapter would smuggle strengths through.
    """

    analyzer_model_id = "rogue-strength"

    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]:
        return (
            _finding(
                verdict_id="v-smuggled",
                candidate_id="base",
                task_id="task-a",
                trace_id=report.trace_refs[0],
                status="observed",
                mechanism_description="everything was fine actually",
                mechanism_cluster_id=_CLUSTER,
                severity=0.8,
                confidence=0.9,
                blame_graph=BlameGraph(
                    nodes=(
                        BlameNode(
                            actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)
                        ),
                    )
                ),
                evidence_refs=("skills/retrieval",),
                valence=-1,
            ),
        )


def test_parallel_path_refuses_a_strength_from_the_failure_judge() -> None:
    from agent_evolve import pipeline

    stack = pipeline.build_offline_stack(
        task_count=1,
        analyzer_factory=_StrengthSmuggler,
        analyzer_workers=2,
    )

    issues = stack.runner.build_issues(stack.tasks)

    assert not issues, "a strength must not become a Judge-1 issue"
    failures = list(stack.runner._analysis_failures)
    assert failures, "the refusal was silent; it must be recorded"
    assert any("polarity" in (f.error or "") for f in failures)


# ---------------------------------------------------------------------- #
# Q6: the sign survives onto Issue
# ---------------------------------------------------------------------- #
def _observed_finding(verdict_id: str, description: str, valence: int) -> CausalFinding:
    return _finding(
        verdict_id=verdict_id,
        task_id="task-a",
        status="observed",
        mechanism_description=description,
        mechanism_cluster_id=_CLUSTER,
        severity=0.7,
        confidence=0.9,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)),)
        ),
        evidence_refs=("skills/retrieval",),
        valence=valence,
    )


def _writable_inventory() -> tuple[ArtifactDescriptor, ...]:
    return (
        ArtifactDescriptor(
            artifact_id="skills/retrieval",
            kind="skill",
            format="text/plain",
            version_hash="sha256:" + "0" * 64,
            readable=True,
            writable=True,
            merge_strategy="replace-overwrites",
            bindings=(),
        ),
    )


def test_issue_carries_valence_from_its_finding() -> None:
    inventory = _writable_inventory()

    strength_issue = build_issue(
        _observed_finding("v-s", "retrieval worked", -1), inventory, embedding=(0.0,)
    )
    assert strength_issue is not None
    assert isinstance(strength_issue, Issue)
    assert strength_issue.valence == -1

    fault_issue = build_issue(
        _observed_finding("v-f", "retrieval broke", 1), inventory, embedding=(0.0,)
    )
    assert fault_issue is not None
    assert fault_issue.valence == 1


def test_issue_rejects_an_invalid_valence_too() -> None:
    with pytest.raises(ValueError):
        build_issue(_observed_finding("v-bad", "broken", 2), _writable_inventory(), embedding=(0.0,))
