"""?13: a passing offspring's strengths must make it entropy-visible.

Measured gap (tools/probes/entropy_availability_probe.py, offline): after an
accepted attempt, the tracker cell held ONLY the base candidate's score even
though TS2 stored both candidates and the child carried positivity strengths.
``_entropy_cluster_id`` keyed off ``rollout.analysis`` alone -- a field that is
None for every passing rollout -- so D5 gave strengths TS2/IDX2/TL visibility
but never cross-candidate entropy.

Policy under test (mirrors the fault side's one-analysis-one-cell rule):
the rollout's measured score files into EVERY cluster its evidence actually
assigned -- one cluster for a diagnosed fault, k clusters for k assigned
strengths, nothing where the clusterer refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.blame import (
    BlameGraph,
    BlameNode,
    CausalAnalysis,
    CausalFinding,
)  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import EvolutionCandidate, EvolutionTask  # noqa: E402
from agent_evolve.core.entropy import EntropyTracker  # noqa: E402
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"

# Measured pair (fault-vs-fix cosine 0.963 on this embedder): the strength
# text JOINS the fault's cluster instead of opening a parallel universe.
_FAULT_TEXT = (
    "the planner hit a retrieval timeout so the context held no documents "
    "and the model answered from memory"
)
_STRENGTH_TEXT = (
    "the planner avoided a retrieval timeout because the context held fresh "
    "documents so the model answered with grounded citations"
)


def _task(task_id: str) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": _TOKEN},
    )


def _strength(text: str, trace_id: str, task_id: str) -> CausalFinding:
    return CausalFinding(
        verdict_id=f"strength-{trace_id}",
        candidate_id="child",
        task_id=task_id,
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


class _JoiningPositivityJudge:
    """One observed strength per passing rollout, worded to join the fault."""

    analyzer_model_id = "fake-positivity"

    def analyze_success(self, task, trace, *, clusters=(), stored_traces=()):
        return (_strength(_STRENGTH_TEXT, trace.trace_id, task.task_id),)


class _StableFaultAnalyzerJudge:
    """Like ``FakeAnalyzerJudge`` but with a STABLE failure mechanism text.

    ``FakeAnalyzerJudge`` embeds the trace id into its mechanism string
    (``trace-{id}-failed-to-match``), so no fixed strength wording could ever
    join it -- the join under test needs a deterministic fault text.
    """

    analyzer_model_id = "fake-analyzer"
    judge_model_id = "fake-judge"

    def analyze(self, task, trace):
        from agent_evolve.core.analyzer import contract_score

        if contract_score(task, trace) == 1.0:
            return CausalAnalysis(
                mechanism="none",
                severity=0.0,
                score=1.0,
                blame_graph=BlameGraph(nodes=()),
            )
        return CausalAnalysis(
            mechanism=_FAULT_TEXT,
            severity=1.0,
            score=0.0,
            blame_graph=BlameGraph(
                nodes=(BlameNode(actor_id="planner", blame=1.0,
                                 artifacts=("skills/retrieval",)),)
            ),
            analyzer_model_id=self.analyzer_model_id,
            judge_model_id=self.judge_model_id,
        )


def _runner(positivity_judge=None, floors=(2, 1), analyzer_judge=None) -> SequentialGepaRunner:
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
    # Floors are governed BY CONFIG (__post_init__ stamps config values onto
    # the tracker), so relaxation must go through resolve_profile overrides.
    config = resolve_profile(
        "research_sequential",
        seed=0,
        entropy_min_comparable_candidates=floors[0],
        entropy_min_rollouts_per_candidate=floors[1],
    )
    return SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        # Stable fault text: FakeAnalyzerJudge embeds the trace id into its
        # mechanism string, which would make the join under test impossible.
        analyzer_judge=analyzer_judge or _StableFaultAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        positivity_judge=positivity_judge,
        config=config,
        mechanism_cluster_id=_CLUSTER,
        seed=0,
        entropy=EntropyTracker(),
    )


def _cells_by_candidate(runner: SequentialGepaRunner) -> dict:
    out = {}
    for key in runner.entropy.all_cells():
        cell = runner.entropy._cells[key]
        out[(key.task_id, key.mechanism_cluster_id)] = {
            c: list(v) for c, v in cell.scores.items()
        }
    return out


def test_passing_child_with_strengths_enters_the_mechanism_cell() -> None:
    runner = _runner(positivity_judge=_JoiningPositivityJudge())
    outcome = runner.run_attempt([_task("task-a")])
    assert outcome.accepted, outcome.reason

    cells = _cells_by_candidate(runner)
    joined = [
        scores for scores in cells.values()
        if outcome.result_candidate_id in scores
    ]
    assert joined, (
        f"passing child filed into NO entropy cell; cells={cells} -- ?13 gap"
    )
    # The child JOINS the base fault's cluster (shared namespace), making the
    # cell comparable and its entropy measurable.
    target = next(
        (key, scores) for key, scores in cells.items()
        if outcome.result_candidate_id in scores
    )
    scores = target[1]
    assert set(scores) == {"base", outcome.result_candidate_id}
    assert scores["base"] == [0.0]
    h = runner.entropy.entropy(*target[0])
    assert h is not None, "a two-candidate comparable cell must yield real entropy"


def test_without_a_judge_a_passing_child_stays_absent() -> None:
    runner = _runner(positivity_judge=None)
    outcome = runner.run_attempt([_task("task-a")])
    assert outcome.accepted, outcome.reason

    cells = _cells_by_candidate(runner)
    assert cells, "base observation must still file"
    assert all(outcome.result_candidate_id not in s for s in cells.values()), (
        "without any diagnosis the honest skip must keep the child absent"
    )


def test_refused_strength_assignment_invents_no_cell() -> None:
    runner = _runner(positivity_judge=_JoiningPositivityJudge())

    class _RefusingClusterer:
        def assign(self, analysis):
            from agent_evolve.core.clustering import ClusterAssignment

            return ClusterAssignment(
                cluster_id="", similarity=0.0, is_new_cluster=False,
                unassigned_reason="stub refusal",
            )

        def assign_finding(self, finding):
            from agent_evolve.core.clustering import ClusterAssignment

            return ClusterAssignment(
                cluster_id="", similarity=0.0, is_new_cluster=False,
                unassigned_reason="stub refusal",
            )

        def cluster_exemplars(self):
            return ()

    class _Registry:
        def __init__(self):
            self.calls = 0

        def clusterer_for(self, task_id):
            self.calls += 1
            return _RefusingClusterer()

    runner.cluster_registry = _Registry()
    outcome = runner.run_attempt([_task("task-a")])
    assert outcome.accepted, outcome.reason

    cells = _cells_by_candidate(runner)
    assert all(outcome.result_candidate_id not in s for s in cells.values()), (
        "a refused assignment must never invent a cell (variance over unrelated "
        "faults reads as reachability for a mechanism that does not exist)"
    )


def test_multiple_strengths_file_into_each_assigned_cluster() -> None:
    runner = _runner(positivity_judge=_JoiningPositivityJudge())

    from agent_evolve.core.clustering import ClusterAssignment

    class _TwoWayClusterer:
        def __init__(self, counter):
            self._counter = counter

        def assign(self, analysis):
            return ClusterAssignment(
                cluster_id="c0", similarity=0.9, is_new_cluster=False,
            )

        def assign_finding(self, finding):
            # Distinct findings land in distinct clusters deterministically;
            # the counter is SHARED across every clusterer_for() call.
            self._counter[0] += 1
            return ClusterAssignment(
                cluster_id=f"c{self._counter[0]}",
                similarity=0.8,
                is_new_cluster=True,
            )

        def cluster_exemplars(self):
            return ()

    class _Registry:
        def __init__(self):
            self._counter = [0]

        def clusterer_for(self, task_id):
            return _TwoWayClusterer(self._counter)

    runner.cluster_registry = _Registry()
    outcome = runner.run_attempt([_task("task-a"), _task("task-b")])
    assert outcome.accepted, outcome.reason

    child_cells = [
        key for key in runner.entropy.all_cells()
        if outcome.result_candidate_id in runner.entropy._cells[key].scores
    ]
    assert len(child_cells) >= 2, (
        f"k assigned strengths must file into k cells; got {child_cells}"
    )
