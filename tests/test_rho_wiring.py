"""Composition-root wiring for the RHO round.

``core/rho/rounds.py`` is deliberately hook-shaped: it may not import ``cuga``,
``litellm``, or ``agent_evolve.adapters``, so every model call, agent invocation
and rollout arrives as an injected callable on :class:`RhoHooks`. This file pins
the one place allowed to bind those callables to live adapters --
:func:`agent_evolve.pipeline.build_rho_hooks` -- and the CLI call site that
actually invokes ``run_rounds``.

Every test here is offline. No test constructs a real comprehender, judge,
diagnoser, optimizer or preference judge; each is injected as a duck-typed fake,
which is exactly the seam those adapters were built with.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.entropy import EntropyTracker  # noqa: E402
from agent_evolve.core.rho.history import HistoryLoadReport  # noqa: E402
from agent_evolve.core.rho.rounds import (  # noqa: E402
    BASE_VERSION,
    RhoHooks,
    RoundConfig,
    rho_cluster_id,
    run_rounds,
)
from agent_evolve.core.rho.scheduler import ConcurrencyPlan  # noqa: E402
from agent_evolve.pipeline import (  # noqa: E402
    build_offline_stack,
    build_rho_hooks,
)
from scripts.run_evolution import build_parser  # noqa: E402


# --------------------------------------------------------------------------- #
# Duck-typed adapter fakes. Each mirrors only the attributes rounds.py reads.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Summary:
    task_id: str
    observed: bool = True
    embedding_text: str = "a summary"


@dataclass(frozen=True, slots=True)
class _Verdict:
    task_id: str
    difficulty: float = 7.0
    abstract_fingerprint: str = "fingerprint"
    observed: bool = True


@dataclass(frozen=True, slots=True)
class _Diagnosis:
    task_id: str
    observed: bool = True
    severity: float = 0.8
    status: str = "OK"


@dataclass(frozen=True, slots=True)
class _Proposed:
    candidate_index: int
    artifacts: Mapping[str, str]
    observed: bool = True
    rationale: str = "improve retrieval"


@dataclass(frozen=True, slots=True)
class _ProposalReport:
    candidates: tuple[_Proposed, ...]
    requested: int
    discarded: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _Preference:
    task_id: str
    score: float = 0.5
    available: bool = True


@dataclass(slots=True)
class _FakeComprehender:
    calls: list[str] = field(default_factory=list)

    def comprehend(self, record: object) -> _Summary:
        self.calls.append(record.task_id)  # type: ignore[attr-defined]
        return _Summary(task_id=record.task_id)  # type: ignore[attr-defined]


@dataclass(slots=True)
class _FakeDifficultyJudge:
    calls: list[tuple[str, str]] = field(default_factory=list)
    expected_answers_seen: list[str | None] = field(default_factory=list)

    def judge(
        self,
        record: object,
        summary_text: str,
        *,
        expected_answer: str | None = None,
    ) -> _Verdict:
        self.calls.append((record.task_id, summary_text))  # type: ignore[attr-defined]
        self.expected_answers_seen.append(expected_answer)
        return _Verdict(task_id=record.task_id)  # type: ignore[attr-defined]


@dataclass(slots=True)
class _FakeDiagnoser:
    calls: list[str] = field(default_factory=list)

    def diagnose(
        self, task_id: str, task_input: str, traces: Sequence[ExecutionTrace]
    ) -> _Diagnosis:
        self.calls.append(task_id)
        return _Diagnosis(task_id=task_id)


@dataclass(slots=True)
class _FakeOptimizer:
    n_seen: list[int] = field(default_factory=list)
    #: artifact id every candidate additionally creates, to prove a created id
    #: survives registration.
    created_id: str = "skills/generated-rho"

    def propose(
        self,
        base_artifacts: Mapping[str, str],
        diagnoses: Sequence[object],
        n: int,
    ) -> _ProposalReport:
        self.n_seen.append(n)
        candidates = []
        for index in range(n):
            artifacts = dict(base_artifacts)
            artifacts[self.created_id] = f"generated body {index}"
            candidates.append(
                _Proposed(candidate_index=index, artifacts=artifacts)
            )
        return _ProposalReport(candidates=tuple(candidates), requested=n)


@dataclass(slots=True)
class _FakePreferenceJudge:
    calls: list[tuple[str, str]] = field(default_factory=list)
    available: bool = True

    def compare_symmetric(
        self,
        task: EvolutionTask,
        baseline: ExecutionTrace,
        candidate: ExecutionTrace,
    ) -> _Preference:
        self.calls.append((task.task_id, candidate.candidate_id))
        return _Preference(task_id=task.task_id, available=self.available)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _offline_stack(task_count: int = 2):
    return build_offline_stack(task_count=task_count, task_token="tok")


def _components() -> dict[str, object]:
    return {
        "comprehender": _FakeComprehender(),
        "difficulty_judge": _FakeDifficultyJudge(),
        "diagnoser": _FakeDiagnoser(),
        "optimizer": _FakeOptimizer(),
        "preference_judge": _FakePreferenceJudge(),
    }


def _config(**overrides: object) -> RoundConfig:
    kwargs: dict[str, object] = dict(
        mode="rho",
        rounds=1,
        coreset_size=2,
        group_rollouts=2,
        candidates=2,
        candidate_rollouts=2,
        concurrency=ConcurrencyPlan.validated(1, 1, 1),
    )
    kwargs.update(overrides)
    return RoundConfig(**kwargs)  # type: ignore[arg-type]


def _write_history(root: Path, task_ids: Sequence[str]) -> None:
    """Write one current-format causal trace per task id."""
    for task_id in task_ids:
        run_dir = root / f"run-{task_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "causal-trace.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "input_text": f"produce result for {task_id}",
                    "final_output": "an answer",
                    "harness_version": "base",
                    "tool_observations": [{"tool": "search"}],
                    "events": [
                        {
                            "event_id": "e0",
                            "kind": "tool_call",
                            "actor_id": "agent",
                            "payload": {},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )


def _trace(task_id: str, version: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=f"t-{version}-{task_id}",
        candidate_id=version,
        task_id=task_id,
        events=(
            TraceEvent(
                event_id="e0",
                kind="tool",
                actor_id="agent",
                parent_event_id=None,
                payload={},
            ),
        ),
        final_output="tok",
        status="success",
    )


# --------------------------------------------------------------------------- #
# The factory binds every hook the round needs
# --------------------------------------------------------------------------- #
def test_build_rho_hooks_returns_rho_hooks() -> None:
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
    finally:
        stack.close()
    assert isinstance(hooks, RhoHooks)


def test_every_required_hook_is_bound() -> None:
    """A missing hook is a wiring error the round discovers after 90 rollouts.

    Binding all of them at the composition root is what makes ``require`` a
    dead branch in a real run rather than a latent failure.
    """
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
    finally:
        stack.close()
    for name in (
        "load_history",
        "comprehend",
        "judge",
        "task_for",
        "rollout",
        "diagnose",
        "base_artifacts",
        "propose",
        "register_candidate",
        "compare",
        "commit",
        "pool_size",
        "score",
        "run_genetic",
        "cache_hits",
    ):
        assert getattr(hooks, name) is not None, f"{name} was left unbound"


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def test_no_history_root_is_a_cold_start_not_an_error() -> None:
    """``--rho-history`` is optional; its absence must be reported as data."""
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack, history_root=None, **_components())  # type: ignore[arg-type]
        report = hooks.load_history()  # type: ignore[misc]
    finally:
        stack.close()
    assert isinstance(report, HistoryLoadReport)
    assert report.is_cold_start


def test_history_root_is_actually_read(tmp_path: Path) -> None:
    """``--rho-history`` was parsed but unread before this wiring existed."""
    _write_history(tmp_path, ["task-1"])
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack, history_root=tmp_path, **_components())  # type: ignore[arg-type]
        report = hooks.load_history()  # type: ignore[misc]
    finally:
        stack.close()
    assert not report.is_cold_start
    assert [r.task_id for r in report.records] == ["task-1"]


# --------------------------------------------------------------------------- #
# Rollout: batch-shaped adapter API adapted to per-index
# --------------------------------------------------------------------------- #
def test_rollout_resolves_the_rounds_base_version_to_the_stacks_own() -> None:
    """``rounds.py`` hardcodes ``BASE_VERSION='base'``; the stack may not use it.

    The offline stack's base is ``base-v0``. Passing ``'base'`` straight through
    would roll out a version the adapter has never heard of.
    """
    stack = _offline_stack(task_count=1)
    try:
        assert stack.base_version != BASE_VERSION
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        outcome = hooks.rollout(BASE_VERSION, stack.tasks[0], 0)  # type: ignore[misc]
    finally:
        stack.close()
    assert outcome.trace is not None, outcome.error
    assert outcome.trace.candidate_id == stack.base_version


def test_rollout_never_raises_even_when_the_adapter_does() -> None:
    """``RhoHooks.rollout`` must return failure as data.

    A raised exception here discards an entire group's evidence for one broken
    rollout, which is the failure ``_rollout_grid`` exists to avoid.
    """

    class _Exploding:
        adapter_name = "exploding"

        def run_rollouts(self, version, tasks, *, prefix):  # noqa: ANN001
            raise RuntimeError("worker pool is gone")

    stack = _offline_stack(task_count=1)
    try:
        stack.runner.rollout_batch = _Exploding()  # type: ignore[assignment]
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        outcome = hooks.rollout(BASE_VERSION, stack.tasks[0], 0)  # type: ignore[misc]
    finally:
        stack.close()
    assert outcome.trace is None
    assert "worker pool is gone" in outcome.error


def test_rollout_uses_the_batch_runner_when_one_is_present() -> None:
    """A live stack owns a ``CugaRolloutRunner``; the hook must go through it."""
    seen: list[tuple[str, tuple[str, ...], str]] = []

    class _Batch:
        def run_rollouts(self, version, tasks, *, prefix):  # noqa: ANN001
            seen.append((version, tuple(t.task_id for t in tasks), prefix))
            from agent_evolve.core.evaluation import RolloutOutcome

            return tuple(
                RolloutOutcome(task=t, trace=_trace(t.task_id, version))
                for t in tasks
            )

    stack = _offline_stack(task_count=1)
    try:
        stack.runner.rollout_batch = _Batch()  # type: ignore[assignment]
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        outcome = hooks.rollout("cand-x", stack.tasks[0], 3)  # type: ignore[misc]
    finally:
        stack.close()
    assert len(seen) == 1
    assert seen[0][0] == "cand-x"
    assert seen[0][1] == (stack.tasks[0].task_id,)
    assert outcome.trace is not None


# --------------------------------------------------------------------------- #
# task_for, base_artifacts
# --------------------------------------------------------------------------- #
def test_task_for_resolves_a_benchmark_task_and_reports_an_absent_one() -> None:
    stack = _offline_stack(task_count=2)
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        found = hooks.task_for(stack.tasks[0].task_id)  # type: ignore[misc]
        missing = hooks.task_for("no-such-task")  # type: ignore[misc]
    finally:
        stack.close()
    assert found == stack.tasks[0]
    assert missing is None


def test_base_artifacts_are_the_incumbents_complete_set() -> None:
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        artifacts = hooks.base_artifacts()  # type: ignore[misc]
        expected = {
            d.artifact_id
            for d in stack.adapter.artifact_inventory(stack.base_version)  # type: ignore[attr-defined]
        }
    finally:
        stack.close()
    assert set(artifacts) == expected
    assert all(isinstance(v, str) for v in artifacts.values())


# --------------------------------------------------------------------------- #
# register_candidate
# --------------------------------------------------------------------------- #
def test_register_candidate_returns_a_distinct_version_per_proposal() -> None:
    """Two candidates sharing a version would collide in the pool and the
    adapter, and the second would silently overwrite the first's workspace."""
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        base = dict(hooks.base_artifacts())  # type: ignore[misc]
        versions = [
            hooks.register_candidate(  # type: ignore[misc]
                _Proposed(candidate_index=i, artifacts=dict(base))
            )
            for i in range(3)
        ]
    finally:
        stack.close()
    assert len(set(versions)) == 3


def test_a_registered_candidate_is_rollout_able() -> None:
    """Registration exists so the rollout hook accepts the returned version."""
    stack = _offline_stack(task_count=1)
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        base = dict(hooks.base_artifacts())  # type: ignore[misc]
        base["skills/generated-rho"] = "new body"
        version = hooks.register_candidate(  # type: ignore[misc]
            _Proposed(candidate_index=0, artifacts=base)
        )
        outcome = hooks.rollout(version, stack.tasks[0], 0)  # type: ignore[misc]
    finally:
        stack.close()
    assert outcome.trace is not None, outcome.error
    assert outcome.trace.candidate_id == version


# --------------------------------------------------------------------------- #
# commit + score: pool provenance
# --------------------------------------------------------------------------- #
def test_commit_adds_the_candidate_to_the_persistent_pool() -> None:
    from agent_evolve.core.rho.rounds import CandidateEvidence

    stack = _offline_stack(task_count=1)
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        before = stack.pool_size()
        base = dict(hooks.base_artifacts())  # type: ignore[misc]
        version = hooks.register_candidate(  # type: ignore[misc]
            _Proposed(candidate_index=0, artifacts=base)
        )
        hooks.commit(  # type: ignore[misc]
            CandidateEvidence(
                candidate_index=0,
                version=version,
                artifacts=base,
                rollouts=2,
                mean_preference=0.4,
                preferences_available=1,
                task_scores={stack.tasks[0].task_id: 0.5},
            )
        )
        after = stack.pool_size()
        entry = stack.pool.get(version)
    finally:
        stack.close()
    assert after == before + 1
    assert entry.candidate.parent_ids == (stack.pool.base_id,)


def test_commit_propagates_the_pairwise_preference_into_the_pool() -> None:
    """The judge's verdict must survive the commit boundary (SV-4).

    This is the assertion that the old code could not satisfy. ``mean_preference``
    was computed in phase 9, printed by the CLI, and dropped at ``commit``: the
    pool had no field for it, so the paper's acceptance signal was purchased on
    every round and then discarded. Asserting on the *pool entry* rather than on
    the evidence object is the point -- evidence carrying the number proves
    nothing about whether selection can ever see it.
    """
    from agent_evolve.core.rho.rounds import CandidateEvidence

    stack = _offline_stack(task_count=1)
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        base = dict(hooks.base_artifacts())  # type: ignore[misc]
        version = hooks.register_candidate(  # type: ignore[misc]
            _Proposed(candidate_index=0, artifacts=base)
        )
        hooks.commit(  # type: ignore[misc]
            CandidateEvidence(
                candidate_index=0,
                version=version,
                artifacts=base,
                rollouts=2,
                mean_preference=0.4,
                preferences_available=3,
                preferences_unavailable=1,
                task_scores={stack.tasks[0].task_id: 0.5},
            )
        )
        entry = stack.pool.get(version)
    finally:
        stack.close()
    assert entry.preference == pytest.approx(0.4)
    assert entry.preference_available == 3
    assert entry.preference_unavailable == 1


def test_commit_without_verdicts_leaves_the_preference_absent() -> None:
    """Zero available verdicts must store ``None``, not ``0.0``.

    An undecided candidate and a candidate the judge scored as a dead tie are
    different facts, and the ``S_j > 0`` gate rejects both -- but for different
    stated reasons. Storing 0.0 here would make a judging failure permanently
    indistinguishable from a measured tie in the exported manifest.
    """
    from agent_evolve.core.rho.rounds import CandidateEvidence

    stack = _offline_stack(task_count=1)
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        base = dict(hooks.base_artifacts())  # type: ignore[misc]
        version = hooks.register_candidate(  # type: ignore[misc]
            _Proposed(candidate_index=0, artifacts=base)
        )
        hooks.commit(  # type: ignore[misc]
            CandidateEvidence(
                candidate_index=0,
                version=version,
                artifacts=base,
                rollouts=2,
                mean_preference=0.0,
                preferences_available=0,
                preferences_unavailable=2,
                task_scores={stack.tasks[0].task_id: 0.9},
            )
        )
        entry = stack.pool.get(version)
    finally:
        stack.close()
    assert entry.preference is None
    assert entry.preference_unavailable == 2


def test_gated_candidate_does_not_become_the_exported_champion() -> None:
    """End-to-end: a dispreferred candidate must not reach ``champion_version``.

    The full path -- commit, preference, pool, ``select_champion``,
    ``champion_version`` -- with the candidate scoring *higher* than the base, so
    only the gate can produce the correct answer. Without the gate the aggregate
    would promote it on score alone, which is the exported-harness defect that
    seeds the next run.
    """
    from agent_evolve.core.rho.rounds import CandidateEvidence

    stack = _offline_stack(task_count=1)
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        task_id = stack.tasks[0].task_id
        base = dict(hooks.base_artifacts())  # type: ignore[misc]

        # Give the incumbent comparable evidence so it is a valid fallback.
        hooks.score(stack.tasks[0], _trace(task_id, stack.base_version))  # type: ignore[misc]

        version = hooks.register_candidate(  # type: ignore[misc]
            _Proposed(candidate_index=0, artifacts=base)
        )
        hooks.commit(  # type: ignore[misc]
            CandidateEvidence(
                candidate_index=0,
                version=version,
                artifacts=base,
                rollouts=2,
                mean_preference=-0.5,  # judge dispreferred it
                preferences_available=2,
                task_scores={task_id: 1.0},  # but it scores better
            )
        )
        champion = stack.champion_version()
    finally:
        stack.close()
    assert champion == stack.base_version


def test_commit_records_score_provenance_into_the_candidates_cell() -> None:
    """A pool entry with an empty tensor cannot be selected or compared.

    The cell is keyed by ``rho_cluster_id`` so the pool and the entropy tracker
    file the same evidence in the same cell.
    """
    from agent_evolve.core.rho.rounds import CandidateEvidence

    stack = _offline_stack(task_count=1)
    task_id = stack.tasks[0].task_id
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        base = dict(hooks.base_artifacts())  # type: ignore[misc]
        version = hooks.register_candidate(  # type: ignore[misc]
            _Proposed(candidate_index=0, artifacts=base)
        )
        # The score hook is what buffers the value; commit flushes it.
        outcome = hooks.rollout(version, stack.tasks[0], 0)  # type: ignore[misc]
        assert outcome.trace is not None
        hooks.score(stack.tasks[0], outcome.trace)  # type: ignore[misc]
        hooks.commit(  # type: ignore[misc]
            CandidateEvidence(
                candidate_index=0, version=version, artifacts=base
            )
        )
        cell = stack.pool.get(version).cell(task_id, rho_cluster_id(task_id))
    finally:
        stack.close()
    assert cell.rollout_count == 1
    assert cell.provenance[0].mechanism_cluster_id == rho_cluster_id(task_id)
    assert cell.provenance[0].task_id == task_id


def test_scoring_the_base_populates_the_bases_own_pool_cell() -> None:
    """Without base cells, champion coverage is zero for the incumbent and a
    candidate wins selection on coverage alone rather than on outcome."""
    stack = _offline_stack(task_count=1)
    task_id = stack.tasks[0].task_id
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        outcome = hooks.rollout(BASE_VERSION, stack.tasks[0], 0)  # type: ignore[misc]
        assert outcome.trace is not None, outcome.error
        hooks.score(stack.tasks[0], outcome.trace)  # type: ignore[misc]
        cell = stack.pool.base.cell(task_id, rho_cluster_id(task_id))
    finally:
        stack.close()
    assert cell.rollout_count == 1


def test_score_returns_a_value_in_the_unit_interval() -> None:
    stack = _offline_stack(task_count=1)
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        outcome = hooks.rollout(BASE_VERSION, stack.tasks[0], 0)  # type: ignore[misc]
        assert outcome.trace is not None
        value = hooks.score(stack.tasks[0], outcome.trace)  # type: ignore[misc]
    finally:
        stack.close()
    assert 0.0 <= float(value) <= 1.0


# --------------------------------------------------------------------------- #
# run_genetic: coreset restriction
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _RecordingBatch:
    """A rollout batch that records which tasks each call received.

    ``EvolutionStack`` and ``SequentialGepaRunner`` are ``slots=True``
    dataclasses, so a method cannot be monkeypatched onto them. ``rollout_batch``
    is a real field and is the runner's own documented seam, so observing there
    exercises the production path rather than a substituted one.
    """

    seen: list[tuple[str, ...]] = field(default_factory=list)
    explode: bool = False

    def run_rollouts(self, version, tasks, *, prefix):  # noqa: ANN001, ANN201
        self.seen.append(tuple(t.task_id for t in tasks))
        if self.explode:
            raise RuntimeError("worker pool is gone")
        from agent_evolve.core.evaluation import RolloutOutcome

        return tuple(
            RolloutOutcome(task=t, trace=_trace(t.task_id, version))
            for t in tasks
        )


def test_run_genetic_restricts_the_stack_to_the_coreset_and_restores_it() -> None:
    """In ``rho-genetic`` the genetic phase runs on coreset tasks only: after a
    RHO round, ``(task, mechanism)`` cells exist only there, so off the coreset
    cross-candidate variance is undefined rather than low."""
    stack = _offline_stack(task_count=3)
    original = stack.tasks
    batch = _RecordingBatch()
    try:
        stack.runner.rollout_batch = batch  # type: ignore[assignment]
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        hooks.run_genetic((original[0],), 1)  # type: ignore[misc]
    finally:
        stack.close()
    assert batch.seen, "the genetic loop never rolled anything out"
    # Every batch the genetic loop asked for saw the coreset task only.
    assert {ids for ids in batch.seen} == {(original[0].task_id,)}
    assert stack.tasks == original


def test_run_genetic_restores_the_task_set_even_when_the_loop_raises() -> None:
    """A narrowed stack left behind would silently shrink the final champion
    measurement to the coreset, reporting a number for the wrong task set."""
    stack = _offline_stack(task_count=3)
    original = stack.tasks
    try:
        stack.runner.rollout_batch = _RecordingBatch(explode=True)  # type: ignore[assignment]
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        with pytest.raises(RuntimeError):
            hooks.run_genetic((original[0],), 1)  # type: ignore[misc]
    finally:
        stack.close()
    assert stack.tasks == original


def test_run_genetic_is_a_no_op_at_zero_iterations() -> None:
    stack = _offline_stack(task_count=2)
    batch = _RecordingBatch()
    try:
        stack.runner.rollout_batch = batch  # type: ignore[assignment]
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        hooks.run_genetic(stack.tasks, 0)  # type: ignore[misc]
    finally:
        stack.close()
    assert batch.seen == []


# --------------------------------------------------------------------------- #
# cache_hits
# --------------------------------------------------------------------------- #
def test_cache_hits_reports_every_cache_the_cli_can_configure() -> None:
    """``--rho-summary-cache``/``--rho-difficulty-cache``/``--rho-embedding-cache``
    were parsed but unread; a silently ignored cache flag makes an operator
    believe a round was cheap when it was not."""
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack, **_components())  # type: ignore[arg-type]
        counts = hooks.cache_hits()  # type: ignore[misc]
    finally:
        stack.close()
    assert set(counts) >= {"summary", "difficulty", "embedding"}


def test_cache_roots_are_handed_to_the_components(tmp_path: Path) -> None:
    """A cache root the caller supplied must reach the adapter that reads it."""
    from agent_evolve.core.rho.cache import JsonDiskCache

    @dataclass(slots=True)
    class _Cached:
        cache: JsonDiskCache = field(default_factory=lambda: JsonDiskCache(None))

        def comprehend(self, record: object) -> _Summary:
            return _Summary(task_id=record.task_id)  # type: ignore[attr-defined]

    comprehender = _Cached()
    components = _components()
    components["comprehender"] = comprehender
    stack = _offline_stack()
    try:
        build_rho_hooks(
            stack,
            summary_cache_root=tmp_path / "summary",
            **components,  # type: ignore[arg-type]
        )
    finally:
        stack.close()
    assert comprehender.cache.root == tmp_path / "summary"


def test_the_live_adapters_receive_the_cli_cache_roots(tmp_path: Path) -> None:
    """``--rho-summary-cache`` and ``--rho-difficulty-cache`` were parsed but
    nothing read them. They only pay for themselves if the two adapters that
    re-run over the same corpus every round actually see them."""
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(
            stack,
            summary_cache_root=tmp_path / "s",
            difficulty_cache_root=tmp_path / "d",
        )
        comprehender = hooks.comprehend.__self__  # type: ignore[union-attr]
        judge_cache_root = _judge_cache_root(hooks)
    finally:
        stack.close()
    assert comprehender.cache.root == tmp_path / "s"
    assert judge_cache_root == tmp_path / "d"


def _judge_cache_root(hooks: RhoHooks) -> Path | None:
    """The difficulty judge's cache root, reached through the judge closure.

    ``judge`` is wrapped (it forwards ``expected_answer``), so the adapter is not
    reachable as ``__self__``; it is read out of the closure instead.
    """
    cells = getattr(hooks.judge, "__closure__", ()) or ()
    for cell in cells:
        cache = getattr(cell.cell_contents, "cache", None)
        if cache is not None:
            return cache.root
    return None


def test_the_live_optimizer_never_receives_a_zero_temperature() -> None:
    """0.0 is rejected by the endpoint outright ("does not support 0.0"), and
    ``RhoOptimizer`` raises on it. Unset is the default: diversity comes from N
    independent invocations, not from sampling."""
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack)
        optimizer = hooks.propose.__self__  # type: ignore[union-attr]
    finally:
        stack.close()
    assert optimizer.temperature is None


# --------------------------------------------------------------------------- #
# End to end through run_rounds
# --------------------------------------------------------------------------- #
def test_a_full_round_runs_end_to_end_over_the_offline_stack(
    tmp_path: Path,
) -> None:
    stack = _offline_stack(task_count=2)
    _write_history(tmp_path, [t.task_id for t in stack.tasks])
    tracker = EntropyTracker()
    try:
        hooks = build_rho_hooks(stack, history_root=tmp_path, **_components())  # type: ignore[arg-type]
        summaries = run_rounds(_config(), hooks, tracker=tracker)
    finally:
        stack.close()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.coreset_ids  # a coreset resolved to runnable tasks
    assert summary.rollouts_spent == _config().rollouts_per_round
    assert summary.diagnoses_observed == 2
    assert summary.candidates_distinct == 2
    assert summary.preferences_available == 4  # N=2 candidates x k=2 tasks


def test_a_full_round_retains_all_n_candidates_in_the_pool(
    tmp_path: Path,
) -> None:
    """The paper takes best-of-N and discards the rest; we keep every proposal.

    N distinct harness hypotheses are the parents whose disagreement the genetic
    stage exploits, so preference rank must never decide survival.
    """
    stack = _offline_stack(task_count=2)
    _write_history(tmp_path, [t.task_id for t in stack.tasks])
    try:
        hooks = build_rho_hooks(stack, history_root=tmp_path, **_components())  # type: ignore[arg-type]
        run_rounds(_config(candidates=3), hooks)
        size = stack.pool_size()
    finally:
        stack.close()
    assert size == 1 + 3  # base plus all three proposals


def test_a_full_round_populates_the_entropy_tracker(tmp_path: Path) -> None:
    """``mark_comparable`` is the wiring gotcha: rollout count alone leaves
    entropy ``None``. R=2 clears ``min_rollouts_per_candidate`` naturally."""
    stack = _offline_stack(task_count=2)
    _write_history(tmp_path, [t.task_id for t in stack.tasks])
    tracker = EntropyTracker()
    try:
        hooks = build_rho_hooks(stack, history_root=tmp_path, **_components())  # type: ignore[arg-type]
        run_rounds(_config(candidates=3), hooks, tracker=tracker)
        task_id = stack.tasks[0].task_id
        entropy = tracker.cell_entropy(task_id, rho_cluster_id(task_id))
    finally:
        stack.close()
    # base + 3 candidates = 4 comparable candidates, each with >= 2 rollouts.
    assert entropy is not None


def test_a_cold_start_round_runs_without_a_single_model_call() -> None:
    """``--mode rho`` with no history must complete and say what it skipped."""
    stack = _offline_stack(task_count=2)
    try:
        hooks = build_rho_hooks(stack, history_root=None, **_components())  # type: ignore[arg-type]
        summaries = run_rounds(_config(), hooks)
    finally:
        stack.close()
    assert len(summaries) == 1
    assert summaries[0].rollouts_spent == 0
    assert any("cold start" in note for note in summaries[0].notes)


def test_rho_genetic_hands_the_genetic_phase_the_coreset(tmp_path: Path) -> None:
    """k=2 coreset out of a 3-task benchmark: the genetic phase must see 2."""
    stack = _offline_stack(task_count=3)
    _write_history(tmp_path, [t.task_id for t in stack.tasks])
    try:
        hooks = build_rho_hooks(stack, history_root=tmp_path, **_components())  # type: ignore[arg-type]
        summaries = run_rounds(
            _config(
                mode="rho-genetic",
                coreset_size=2,
                genetic_iterations_per_round=1,
            ),
            hooks,
        )
        # The stack is restored, so the champion measurement still spans all 3.
        assert len(stack.tasks) == 3
    finally:
        stack.close()
    assert summaries[0].genetic_iterations == 1
    assert len(summaries[0].coreset_ids) == 2


# --------------------------------------------------------------------------- #
# The import boundary the whole hook shape exists to protect
# --------------------------------------------------------------------------- #
def test_core_rho_never_imports_an_adapter() -> None:
    """``pipeline.py`` is the composition root precisely so ``core/`` stays
    agent-neutral. If this ever fails, the binding leaked downward."""
    import ast

    core_rho = ROOT / "src" / "agent_evolve" / "core" / "rho"
    forbidden = ("cuga", "litellm", "agent_evolve.adapters")
    offenders: list[str] = []
    for path in sorted(core_rho.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(
                    name == bad or name.startswith(f"{bad}.")
                    for bad in forbidden
                ):
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []


# --------------------------------------------------------------------------- #
# The CLI call site
# --------------------------------------------------------------------------- #
def test_rho_mode_no_longer_claims_it_did_not_execute(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The loud ``NOT YET EXECUTED`` line existed because falling through to the
    genetic loop under ``--mode rho`` would attribute genetic results to RHO.
    Now that the round actually runs, that line would itself be the lie."""
    from scripts.run_evolution import main as run_evolution_main

    code = run_evolution_main(
        ["--dry-run", "--mode", "rho", "--tasks", "2", "--max-workers", "1"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "NOT YET EXECUTED" not in out


def test_rho_mode_reports_a_round_rather_than_genetic_iterations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.run_evolution import main as run_evolution_main

    code = run_evolution_main(
        ["--dry-run", "--mode", "rho", "--tasks", "2", "--max-workers", "1"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "round 1" in out
    # The genetic iteration line must not appear: no genetic iteration ran.
    assert "iteration 1: attempts=" not in out


def test_genetic_mode_still_runs_the_genetic_loop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every existing invocation is unchanged; ``--mode genetic`` is default."""
    from scripts.run_evolution import main as run_evolution_main

    code = run_evolution_main(["--dry-run", "--tasks", "1", "--iterations", "1"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "iteration 1: attempts=" in out
    assert "round 1" not in out


def test_rho_genetic_mode_runs_both_phases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.run_evolution import main as run_evolution_main

    code = run_evolution_main(
        [
            "--dry-run",
            "--mode", "rho-genetic",
            "--tasks", "2",
            "--max-workers", "1",
            "--genetic-iterations-per-round", "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "round 1" in out


# --------------------------------------------------------------------------- #
# --dry-run promises no model endpoint and no network. That promise has to
# survive --mode rho, or the offline rehearsal is not one.
# --------------------------------------------------------------------------- #
def test_dry_run_rho_builds_no_model_calling_component() -> None:
    """``--dry-run`` documents "no CUGA process, no model endpoint, no network".

    With a history corpus present, the RHO phases reach comprehension and
    difficulty judging, whose default adapters call ``litellm``. A dry run that
    constructed them would make a real network call while claiming not to -- and
    would then report the failed call as "summaries unavailable", which reads
    like a data problem rather than a wiring one.
    """
    from scripts.run_evolution import _rho_components_for

    args = build_parser().parse_args(["--dry-run", "--mode", "rho"])
    components = _rho_components_for(args)

    # Every component must be an offline one: no litellm, no CUGA agent.
    for name, component in components.items():
        module = type(component).__module__
        assert not module.startswith("agent_evolve.adapters"), (
            f"{name} is a live adapter ({module}); a dry run must not build one"
        )


def test_live_mode_supplies_no_offline_substitute() -> None:
    """The inverse: a live run must not silently get the offline fakes, which
    would report a fabricated round as a real one.

    ``{}`` is the correct answer rather than a dict of real adapters --
    :func:`build_rho_hooks` constructs those itself, lazily, so importing this
    module never requires the CUGA SDK.
    """
    from scripts.run_evolution import _rho_components_for

    args = build_parser().parse_args(["--mode", "rho"])
    assert _rho_components_for(args) == {}


def test_build_rho_hooks_defaults_to_the_live_adapters() -> None:
    """With no components injected, the real ones must be what gets bound."""
    stack = _offline_stack()
    try:
        hooks = build_rho_hooks(stack)
        bound = {
            "comprehend": hooks.comprehend,
            "diagnose": hooks.diagnose,
            "propose": hooks.propose,
            "compare": hooks.compare,
        }
    finally:
        stack.close()
    for name, hook in bound.items():
        module = type(hook.__self__).__module__  # type: ignore[union-attr]
        assert module.startswith("agent_evolve.adapters"), (
            f"{name} bound to {module}, not a live adapter"
        )


def test_dry_run_rho_with_history_makes_no_network_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end-to-end version: a dry run with a corpus must complete offline."""
    from scripts.run_evolution import main as run_evolution_main

    stack = _offline_stack(task_count=3)
    task_ids = [t.task_id for t in stack.tasks]
    stack.close()
    _write_history(tmp_path, task_ids)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("a dry run attempted a model call")

    import agent_evolve.adapters.cuga_rho_comprehender as comp_mod
    import agent_evolve.adapters.cuga_rho_judge as judge_mod

    original = (comp_mod._litellm_completion, judge_mod._litellm_completion)
    comp_mod._litellm_completion = _forbidden  # type: ignore[assignment]
    judge_mod._litellm_completion = _forbidden  # type: ignore[assignment]
    try:
        code = run_evolution_main(
            [
                "--dry-run",
                "--mode", "rho",
                "--tasks", "3",
                "--rho-history", str(tmp_path),
                "--rho-coreset-size", "2",
                "--rho-candidates", "2",
                "--max-workers", "1",
                "--rho-group-workers", "1",
                "--rho-rollout-workers", "1",
            ]
        )
    finally:
        comp_mod._litellm_completion, judge_mod._litellm_completion = original  # type: ignore[assignment]

    out = capsys.readouterr().out
    assert code == 0, out
    # The round did real work rather than degrading to "everything unavailable".
    assert "round 1: 2 coreset tasks" in out
    assert "candidates 2 of 2 distinct" in out


def test_dry_run_rho_retains_all_candidates_in_the_pool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The offline rehearsal must exercise all-N retention, not just plumbing."""
    from scripts.run_evolution import main as run_evolution_main

    stack = _offline_stack(task_count=3)
    task_ids = [t.task_id for t in stack.tasks]
    stack.close()
    _write_history(tmp_path, task_ids)

    code = run_evolution_main(
        [
            "--dry-run",
            "--mode", "rho",
            "--tasks", "3",
            "--rho-history", str(tmp_path),
            "--rho-coreset-size", "2",
            "--rho-candidates", "3",
            "--max-workers", "1",
            "--rho-group-workers", "1",
            "--rho-rollout-workers", "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "note   : 4 candidate(s) in the pool" in out  # base + all 3


# --------------------------------------------------------------------------- #
# The header describes the run that is about to happen, so it must not carry a
# claim the selected mode falsifies.
# --------------------------------------------------------------------------- #
def test_the_header_no_longer_claims_no_rho_seeder_exists_under_mode_rho(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The header's "no RHO seeder exists" line predates this wiring.

    Under ``--mode rho`` a seeder now runs and produces N candidates, so the
    claim is false -- and it is exactly the kind of false reassurance that makes
    an operator discount a real cross-candidate result as inert.
    """
    from scripts.run_evolution import main as run_evolution_main

    code = run_evolution_main(
        ["--dry-run", "--mode", "rho", "--tasks", "2", "--max-workers", "1"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "no RHO seeder exists" not in out


def test_the_header_still_says_so_under_mode_genetic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Under ``--mode genetic`` the claim is still true and must stay loud."""
    from scripts.run_evolution import main as run_evolution_main

    code = run_evolution_main(["--dry-run", "--tasks", "1", "--iterations", "1"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "no RHO seeder" in out
