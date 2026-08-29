"""Behavioral tests for the Phase 6 sequential GEPA runner.

Governing contracts:
* ``docs/architecture/orchestration-lifecycle.md:40-66`` — attempt lifecycle:
  select work item -> retrieve history -> editor -> materialize -> apply edits ->
  validate -> stage -> commit or record rejection.
* ``docs/architecture/selection-algorithms.md:131-142`` — hard constraints: an
  issue without an attributable inventory-declared writable artifact is rejected
  before ranking.
* ``docs/architecture/selection-algorithms.md:306-315`` — parent sampling is
  proportional to ``frequency(c)`` under a seeded RNG.

Everything here is offline and deterministic: FakeAdapter, FakeAnalyzerJudge,
FakeEditor, and a lexical embedder. No LLM, no CUGA, no network, and no
merge/parallel service is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import AnalyzerJudge, FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.blame import (  # noqa: E402
    BlameGraph,
    CausalAnalysis,
    CausalFinding,
)
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionCandidate,
    EvolutionTask,
)
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.issues import Issue  # noqa: E402
from agent_evolve.core.memory import AttemptStatus  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool, ScoreProvenance  # noqa: E402
from agent_evolve.core.storage import JSONFileStorage  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"


def _task(task_id: str = "task-a", expected: str = _TOKEN) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": expected},
    )


def _base_candidate(adapter: FakeAdapter) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id="base",
        version="base-v0",
        artifact_hashes={
            d.artifact_id: d.version_hash
            for d in adapter.artifact_inventory("base-v0")
        },
    )


def _runner(
    *,
    seed: int = 0,
    storage: JSONFileStorage | None = None,
    min_comparable_rollouts: int = 1,
    adapter: FakeAdapter | None = None,
    analyzer_judge: AnalyzerJudge | None = None,
) -> SequentialGepaRunner:
    adapter = adapter or FakeAdapter()
    pool = PersistentPool(min_comparable_rollouts=min_comparable_rollouts)
    pool.add_base(_base_candidate(adapter))
    return SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=analyzer_judge if analyzer_judge is not None else FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        storage=storage,
        config=resolve_profile("research_sequential", seed=seed),
        mechanism_cluster_id=_CLUSTER,
        seed=seed,
    )


def _record(pool: PersistentPool, candidate_id: str, task_id: str, score: float) -> None:
    entry = pool.get(candidate_id)
    cell = entry.cell(task_id, _CLUSTER)
    pool.record_score(
        candidate_id,
        score,
        ScoreProvenance(
            task_id=task_id,
            mechanism_cluster_id=_CLUSTER,
            trace_id=f"trace-{candidate_id}-{task_id}",
            rollout_seq=cell.rollout_count,
            analyzer_model_id="fake-analyzer",
            judge_model_id="fake-judge",
            blame_confidence=1.0,
            blame_stability=1.0,
        ),
    )


# ---------------------------------------------------------------------- #
# build_issues: trace-backed writable attribution
# ---------------------------------------------------------------------- #
def test_build_issues_produces_a_trace_backed_issue_for_a_failing_task() -> None:
    """A failing base rollout yields an issue whose write set is writable."""
    runner = _runner()

    issues = runner.build_issues([_task()])

    assert len(issues) == 1
    issue = issues[0]
    assert isinstance(issue, Issue)
    assert issue.task_id == "task-a"
    assert issue.mechanism_cluster_id == _CLUSTER
    writable = {
        d.artifact_id
        for d in runner.adapter.artifact_inventory("base-v0")
        if d.writable
    }
    assert set(issue.writable_artifact_ids) <= writable
    assert issue.writable_artifact_ids  # never empty, else it cannot rank


def test_build_issues_skips_tasks_the_base_already_satisfies() -> None:
    """A passing rollout is not an issue; no work item is emitted."""
    adapter = FakeAdapter(
        base_artifacts=(
            ("skills/retrieval", "skill", f"retrieve(query): use {_TOKEN}"),
            ("policies/execution", "policy", "execute(tool, args)"),
            ("prompts/system", "prompt", "You are a helpful assistant."),
        )
    )
    runner = _runner(adapter=adapter)

    assert runner.build_issues([_task()]) == ()


def test_build_issues_carries_a_nonempty_embedding_for_the_dpp_kernel() -> None:
    """An empty embedding would force a permanent degenerate-kernel fallback."""
    runner = _runner()

    issue = runner.build_issues([_task()])[0]

    assert len(issue.embedding) == 32
    assert any(value != 0.0 for value in issue.embedding)


def test_build_issues_emits_one_issue_per_failing_task() -> None:
    runner = _runner()

    issues = runner.build_issues([_task("task-a"), _task("task-b", "semantic-cache")])

    assert {issue.task_id for issue in issues} == {"task-a", "task-b"}


def test_build_issues_does_not_mutate_base_artifacts() -> None:
    """Evidence gathering must never write through to the read-only base."""
    runner = _runner()
    before = runner.adapter.read_artifacts("base-v0", ("skills/retrieval",))

    runner.build_issues([_task()])

    assert runner.adapter.read_artifacts("base-v0", ("skills/retrieval",)) == before


# ---------------------------------------------------------------------- #
# Synthesized finding: trace-backed, contract-valid
# ---------------------------------------------------------------------- #
def test_synthesized_finding_is_observed_and_trace_backed() -> None:
    """The finding must satisfy the observed-status data contract."""
    runner = _runner()
    task = _task()
    trace, analysis = runner.observe(runner.pool.base, task)

    finding = runner.finding_from_analysis(
        analysis,
        task=task,
        candidate_id="base",
        trace_id=trace.trace_id,
        verdict_id="v-1",
        writable_artifact_ids=("skills/retrieval",),
    )

    assert isinstance(finding, CausalFinding)
    assert finding.status == "observed"
    assert finding.trace_id == trace.trace_id
    assert finding.evidence_refs == ("skills/retrieval",)
    attributed = {
        aid for node in finding.blame_graph.nodes for aid in node.artifacts
    }
    assert attributed <= set(finding.evidence_refs)


def test_synthesized_finding_never_carries_the_expected_substring() -> None:
    """Evaluator internals must not leak into a persisted finding."""
    runner = _runner()
    task = _task()
    trace, analysis = runner.observe(runner.pool.base, task)

    finding = runner.finding_from_analysis(
        analysis,
        task=task,
        candidate_id="base",
        trace_id=trace.trace_id,
        verdict_id="v-1",
        writable_artifact_ids=("skills/retrieval",),
    )

    blob = json.dumps(finding.model_dump(mode="json"), sort_keys=True)
    assert _TOKEN not in blob


def test_finding_without_blame_nodes_is_insufficient_evidence() -> None:
    """Absence of evidence must be expressed, never a synthetic placeholder node."""
    runner = _runner()
    analysis = CausalAnalysis(
        mechanism="failed-to-match",
        severity=1.0,
        score=0.0,
        blame_graph=BlameGraph(nodes=()),
    )

    finding = runner.finding_from_analysis(
        analysis,
        task=_task(),
        candidate_id="base",
        trace_id="tr-1",
        verdict_id="v-1",
        writable_artifact_ids=("skills/retrieval",),
    )

    assert finding.status == "insufficient_evidence"
    assert finding.blame_graph.nodes == ()


class _NoBlameAnalyzer:
    """An analyzer+judge that reports a failure with zero blamed actors."""

    analyzer_model_id = "no-blame"
    judge_model_id = "no-blame"

    def analyze(self, task: EvolutionTask, trace: object) -> CausalAnalysis:
        return CausalAnalysis(
            mechanism="failure-with-no-blame",
            severity=1.0,
            score=0.0,
            blame_graph=BlameGraph(nodes=()),
        )


def test_build_issues_skips_tasks_with_insufficient_evidence() -> None:
    """No issue is built from absent evidence; the write set stays unattributed."""
    runner = _runner(analyzer_judge=_NoBlameAnalyzer())

    assert runner.build_issues([_task()]) == ()


class _AbsenceAnalyzer:
    """S4-9: a failure with no blamed actors but MEASURED surface absence."""

    analyzer_model_id = "absence"
    judge_model_id = "absence"

    def analyze(self, task: EvolutionTask, trace: object) -> CausalAnalysis:
        return CausalAnalysis(
            mechanism="no guidance was ever loaded to steer the run",
            severity=1.0,
            score=0.0,
            blame_graph=BlameGraph(nodes=()),
            absent_surfaces=("skills",),
        )


def test_absence_finding_forwards_absent_surfaces() -> None:
    """The synthesized finding carries the analysis's measured absence."""
    runner = _runner(analyzer_judge=_AbsenceAnalyzer())
    task = _task()
    trace, analysis = runner.observe(runner.pool.base, task)

    finding = runner.finding_from_analysis(
        analysis,
        task=task,
        candidate_id="base",
        trace_id=trace.trace_id,
        verdict_id="v-1",
        writable_artifact_ids=("skills/retrieval",),
    )

    assert finding.absent_surfaces == ("skills",)


def test_build_issues_turns_absence_into_a_work_item() -> None:
    """THE S4-9 live defect: absence with no blamed actor must still be editable."""
    runner = _runner(analyzer_judge=_AbsenceAnalyzer())

    issues = runner.build_issues([_task()])

    assert len(issues) == 1
    assert issues[0].absent_surfaces == ("skills",)
    # Attributed to the declared-but-unused writable artifacts of the
    # absent surface.
    assert issues[0].writable_artifact_ids == ("skills/retrieval",)


def test_run_attempt_with_no_blame_produces_no_work_item() -> None:
    """A failure with no blamed actor yields a PENDING no-issue outcome."""
    runner = _runner(analyzer_judge=_NoBlameAnalyzer())

    outcome = runner.run_attempt([_task()])

    assert outcome.accepted is False
    assert outcome.status is AttemptStatus.PENDING
    assert outcome.issue_id == ""
    assert outcome.result_candidate_id is None
    assert len(runner.pool) == 1


# ---------------------------------------------------------------------- #
# select_issues
# ---------------------------------------------------------------------- #
def test_select_issues_returns_a_report_with_recorded_configuration() -> None:
    runner = _runner()
    issues = runner.build_issues([_task("task-a"), _task("task-b", "semantic-cache")])

    report = runner.select_issues(issues, k=1)

    assert len(report.items) == 1
    assert report.mode == "dpp"
    assert 0.0 <= report.theta <= 1.0
    assert report.weights and sum(report.weights) == pytest.approx(1.0)


def test_select_issues_on_an_empty_set_selects_nothing() -> None:
    runner = _runner()

    assert runner.select_issues((), k=1).items == ()


def test_select_issues_is_deterministic() -> None:
    runner = _runner()
    issues = runner.build_issues([_task("task-a"), _task("task-b", "semantic-cache")])

    assert runner.select_issues(issues, k=1) == runner.select_issues(issues, k=1)


# ---------------------------------------------------------------------- #
# select_parent
# ---------------------------------------------------------------------- #
def test_select_parent_falls_back_to_base_without_evidence() -> None:
    """No winning cell means no frequency mass; the base is the only parent."""
    runner = _runner()

    assert runner.select_parent().candidate_id == "base"


def test_select_parent_only_returns_candidates_with_frequency_mass() -> None:
    """A candidate that wins no cell must never be sampled as a parent."""
    runner = _runner()
    winner = EvolutionCandidate(
        candidate_id="c1", version="base-v0", artifact_hashes={}
    )
    loser = EvolutionCandidate(
        candidate_id="c2", version="base-v0", artifact_hashes={}
    )
    runner.pool.add_candidate(winner)
    runner.pool.add_candidate(loser)
    _record(runner.pool, "base", "task-a", 0.0)
    _record(runner.pool, "c1", "task-a", 1.0)
    _record(runner.pool, "c2", "task-a", 0.0)

    parents = {runner.select_parent().candidate_id for _ in range(25)}

    assert parents == {"c1"}


def test_select_parent_is_reproducible_for_a_fixed_seed() -> None:
    """Two identically seeded runners draw the same parent sequence."""

    def draw(seed: int) -> list[str]:
        runner = _runner(seed=seed)
        for candidate_id, task_id in (("c1", "task-a"), ("c2", "task-b")):
            runner.pool.add_candidate(
                EvolutionCandidate(
                    candidate_id=candidate_id, version="base-v0", artifact_hashes={}
                )
            )
            _record(runner.pool, candidate_id, task_id, 1.0)
        return [runner.select_parent().candidate_id for _ in range(12)]

    assert draw(11) == draw(11)


def test_select_parent_sampling_is_seed_sensitive() -> None:
    """Different seeds must be able to produce different parent sequences."""

    def draw(seed: int) -> list[str]:
        runner = _runner(seed=seed)
        for candidate_id, task_id in (("c1", "task-a"), ("c2", "task-b")):
            runner.pool.add_candidate(
                EvolutionCandidate(
                    candidate_id=candidate_id, version="base-v0", artifact_hashes={}
                )
            )
            _record(runner.pool, candidate_id, task_id, 1.0)
        return [runner.select_parent().candidate_id for _ in range(20)]

    assert draw(1) != draw(9999)


# ---------------------------------------------------------------------- #
# run_attempt: the full lifecycle
# ---------------------------------------------------------------------- #
def test_run_attempt_accepts_an_edit_that_repairs_the_origin_task() -> None:
    """FakeEditor injects the expected token, so validation must pass."""
    runner = _runner()

    outcome = runner.run_attempt([_task()])

    assert outcome.accepted is True
    assert outcome.status is AttemptStatus.ACCEPTED
    assert outcome.result_candidate_id is not None
    assert outcome.parent_candidate_id == "base"
    assert outcome.weighted_net_gain > 0.0


def test_run_attempt_commits_the_accepted_candidate_to_the_pool() -> None:
    runner = _runner()

    outcome = runner.run_attempt([_task()])

    assert len(runner.pool) == 2
    assert outcome.result_candidate_id is not None
    committed = runner.pool.get(outcome.result_candidate_id)
    assert committed.candidate.parent_ids == ("base",)
    assert committed.origin_attempt_ids == (outcome.attempt_id,)
    assert committed.score_tensor, "accepted candidate must carry score evidence"


def test_run_attempt_records_no_issue_when_every_task_passes() -> None:
    adapter = FakeAdapter(
        base_artifacts=(
            ("skills/retrieval", "skill", f"retrieve(query): use {_TOKEN}"),
            ("policies/execution", "policy", "execute(tool, args)"),
            ("prompts/system", "prompt", "You are a helpful assistant."),
        )
    )
    runner = _runner(adapter=adapter)

    outcome = runner.run_attempt([_task()])

    assert outcome.accepted is False
    assert outcome.status is AttemptStatus.PENDING
    assert outcome.issue_id == ""
    assert outcome.result_candidate_id is None
    assert len(runner.pool) == 1  # nothing committed


def test_run_attempt_leaves_the_pool_unchanged_when_nothing_is_selected() -> None:
    adapter = FakeAdapter(
        base_artifacts=(
            ("skills/retrieval", "skill", f"retrieve(query): use {_TOKEN}"),
            ("policies/execution", "policy", "execute(tool, args)"),
            ("prompts/system", "prompt", "You are a helpful assistant."),
        )
    )
    runner = _runner(adapter=adapter)
    before = runner.pool.candidate_ids()

    runner.run_attempt([_task()])

    assert runner.pool.candidate_ids() == before


def test_attempt_ids_are_unique_across_attempts() -> None:
    runner = _runner()

    ids = [runner.run_attempt([_task()]).attempt_id for _ in range(3)]

    assert len(set(ids)) == 3


# ---------------------------------------------------------------------- #
# run: N sequential attempts
# ---------------------------------------------------------------------- #
def test_run_executes_the_requested_number_of_attempts() -> None:
    runner = _runner()

    result = runner.run([_task()], n_attempts=3)

    assert len(result.attempts) == 3
    assert result.attempts_run == 3


def test_run_reports_the_final_champion_and_pool_state() -> None:
    runner = _runner()

    result = runner.run([_task()], n_attempts=2)

    assert result.champion is not None
    assert result.champion.candidate_id in runner.pool.candidate_ids()
    assert result.pool_size == len(runner.pool)
    assert set(result.pareto_frontier) == set(runner.pool.pareto_frontier())


def test_run_rejects_a_non_positive_attempt_count() -> None:
    runner = _runner()

    with pytest.raises(ValueError):
        runner.run([_task()], n_attempts=0)


def test_run_is_deterministic_for_a_fixed_seed() -> None:
    """Identical seed and inputs reproduce the same accepted/rejected counts."""

    def summary(seed: int) -> tuple[int, int, int]:
        runner = _runner(seed=seed)
        result = runner.run([_task("task-a"), _task("task-b", "semantic-cache")], n_attempts=3)
        return (result.accepted_count, result.rejected_count, result.pool_size)

    assert summary(5) == summary(5)


# ---------------------------------------------------------------------- #
# Storage: redacted persistence
# ---------------------------------------------------------------------- #
def test_run_persists_attempt_records_when_storage_is_configured(
    tmp_path: Path,
) -> None:
    """Every attempt that actually ran leaves a record.

    Asserted as "one record per attempted edit" rather than a hardcoded 2:
    since SV-11 made ``build_issues`` diagnose the selected parent, an accepted
    candidate that now passes every task correctly yields no further work item,
    so the second call is a planned non-attempt rather than an edit. Requiring 2
    records would force the loop to keep re-diagnosing failures it had already
    fixed, which was the defect.
    """
    storage = JSONFileStorage(tmp_path)
    runner = _runner(storage=storage)

    result = runner.run([_task()], n_attempts=2)

    attempted = [o for o in result.attempts if o.status is not AttemptStatus.PENDING]
    assert attempted, "no attempt ran at all"
    assert len(storage.list_records("attempts")) == len(attempted)


def test_persisted_records_never_contain_the_expected_substring(
    tmp_path: Path,
) -> None:
    """Evaluator internals must not reach disk from any runner record."""
    storage = JSONFileStorage(tmp_path)
    runner = _runner(storage=storage)

    runner.run([_task()], n_attempts=2)

    blob = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(tmp_path.rglob("*.json"))
    )
    assert blob  # something was written
    assert _TOKEN not in blob


def test_runner_works_without_storage(tmp_path: Path) -> None:
    """Storage is optional; the loop must not require a backend."""
    runner = _runner(storage=None)

    result = runner.run([_task()], n_attempts=1)

    assert result.attempts_run == 1
