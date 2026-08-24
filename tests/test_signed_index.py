"""IDX2: the signed mechanism index (D5 step, core side).

Governing contracts:
* ``docs/design/issue-lifecycle.md`` D5.4/D5.6 -- the index maps a mechanism
  cluster to its members RANKED: strongest solvers first (``valence=-1``, high
  magnitude), then least-bad failures (``valence=+1``, low magnitude).
* D5.1 -- ONE shared cluster namespace: a strength whose mechanism text is
  nearly identical to a fault's MUST land in that fault's cluster. This is the
  measured property (fault-vs-fix cosine 0.963) that makes complementary
  parenthood findable at all.
* Feeding comes from the TS2 cross-attempt store; clustering reuses the
  existing ``MechanismClusterer`` UNCHANGED (its ``assign`` already accepts
  ``CausalFinding``).
* Honesty: unscorable rollouts, rollouts without analyses/strengths, and
  refused cluster assignments contribute NOTHING -- no invented cells. A
  runner without a cluster registry raises rather than returning a silent
  empty index.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import FakeAnalyzerJudge, FakePositivityJudge  # noqa: E402
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import EvolutionCandidate, EvolutionTask, ExecutionTrace  # noqa: E402
from agent_evolve.core.evaluation import ObservedRollout, RolloutScore  # noqa: E402
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.mechanism_index import SignedMechanismIndex  # noqa: E402
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


def _runner(positivity_judge: object = None) -> SequentialGepaRunner:
    """``positivity_judge`` left loosely typed: frozen fake vs Protocol variance."""
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
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        positivity_judge=positivity_judge,
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
    )


def _score(passed: bool) -> RolloutScore:
    return RolloutScore(
        task_id="task-a", grader_name="g", score=1.0 if passed else 0.0,
        scorable=True, passed=passed,
    )


def _trace(candidate_id: str, trace_id: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id, candidate_id=candidate_id, task_id="task-a",
        events=(), final_output="x", status="success",
    )


def _strength_finding(text: str, candidate_id: str, trace_id: str):
    from agent_evolve.core.blame import CausalFinding

    return CausalFinding(
        verdict_id=f"strength-{trace_id}",
        candidate_id=candidate_id,
        task_id="task-a",
        trace_id=trace_id,
        valence=-1,
        status="observed",
        mechanism_description=text,
        mechanism_cluster_id="mechanism-cluster-unassigned",
        severity=0.9,
        confidence=0.9,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)),)
        ),
        evidence_refs=("skills/retrieval",),
        rationale="test",
    )


_FAULT_TEXT = (
    "the planner hit a retrieval timeout so the context held no documents "
    "and the model answered from memory"
)
_STRENGTH_TEXT = (
    "the planner avoided a retrieval timeout because the context held fresh "
    "documents so the model answered with grounded citations"
)


def _store_pair(runner: SequentialGepaRunner, *, fault_severity: float = 0.7) -> None:
    """One diagnosed fault (parent) + one near-identical strength (child)."""
    fault_analysis = CausalAnalysis(
        mechanism=_FAULT_TEXT,
        severity=fault_severity,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)),)
        ),
    )
    runner._record_stored_trace(
        "parent",
        ObservedRollout(
            task=_task(), trace=_trace("parent-v", "tr-fault"),
            score=_score(False), analysis=fault_analysis,
        ),
    )
    runner._record_stored_trace(
        "child",
        ObservedRollout(
            task=_task(), trace=_trace("child-v", "tr-strength"),
            score=_score(True),
            strengths=(
                _strength_finding(_STRENGTH_TEXT, "child", "tr-strength"),
            ),
        ),
    )


# ---------------------------------------------------------------------- #
# Basics
# ---------------------------------------------------------------------- #
def test_empty_store_builds_an_empty_index() -> None:
    runner = _runner()
    index = runner.signed_mechanism_index()
    assert isinstance(index, SignedMechanismIndex)
    assert len(index) == 0


def test_no_cluster_registry_raises_rather_than_returning_silence() -> None:
    runner = _runner()
    runner.cluster_registry = None
    with pytest.raises(ValueError, match="registry"):
        runner.signed_mechanism_index()


# ---------------------------------------------------------------------- #
# The crown property: one shared namespace
# ---------------------------------------------------------------------- #
def test_near_identical_strength_joins_the_faults_cluster_and_ranks_first() -> None:
    runner = _runner()
    _store_pair(runner)

    index = runner.signed_mechanism_index()

    # Exactly ONE cluster owns both members -- not two parallel universes.
    assert len(index) == 1
    task_id, cluster_id = index.clusters()[0]
    members = index.members_for(task_id, cluster_id)

    assert [m.valence for m in members] == [-1, 1], (
        "the solver must rank before the fault in the shared cluster"
    )
    assert members[0].candidate_id == "child"
    assert members[1].candidate_id == "parent"
    assert members[1].artifact_ids == ("skills/retrieval",)
    assert members[0].artifact_ids == ("skills/retrieval",)


# ---------------------------------------------------------------------- #
# Ranking details
# ---------------------------------------------------------------------- #
def test_solvers_rank_by_severity_desc_faults_asc_least_bad_last() -> None:
    runner = _runner()
    _store_pair(runner, fault_severity=0.7)
    # A second, milder fault on another candidate: least-bad ordering target.
    mild = CausalAnalysis(
        mechanism=_FAULT_TEXT,
        severity=0.3,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="api_agent", blame=1.0, artifacts=("policies/x",)),)
        ),
    )
    runner._record_stored_trace(
        "cousin",
        ObservedRollout(
            task=_task(), trace=_trace("cousin-v", "tr-mild"),
            score=_score(False), analysis=mild,
        ),
    )

    index = runner.signed_mechanism_index()
    task_id, cluster_id = index.clusters()[0]
    members = index.members_for(task_id, cluster_id)

    assert [m.candidate_id for m in members] == ["child", "cousin", "parent"]
    limited = index.members_for(task_id, cluster_id, limit=2)
    assert [m.candidate_id for m in limited] == ["child", "cousin"]


def test_unscored_or_undiagnosed_rollouts_contribute_nothing() -> None:
    runner = _runner()
    runner._record_stored_trace(
        "ghost",
        ObservedRollout(
            task=_task(), trace=_trace("ghost-v", "tr-g"),
            score=_score(False), analysis=None,
        ),
    )
    index = runner.signed_mechanism_index()
    assert len(index) == 0


# ---------------------------------------------------------------------- #
# End-to-end: an accepted attempt feeds the index cross-candidate
# ---------------------------------------------------------------------- #
def test_accepted_attempt_puts_child_strength_and_parent_fault_in_one_index() -> None:
    runner = _runner(FakePositivityJudge())

    outcome = runner.run_attempt([_task("task-a")])
    assert outcome.accepted, outcome.reason

    index = runner.signed_mechanism_index()
    candidates = {
        m.candidate_id
        for key in index.clusters()
        for m in index.members_for(*key)
    }
    assert runner.pool.base.candidate_id in candidates
    assert outcome.result_candidate_id in candidates