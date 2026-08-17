"""Tests for RHO mode dispatch, round summaries, and the round executor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pytest

from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace
from agent_evolve.core.evaluation import RolloutOutcome
from agent_evolve.core.rho.history import HistoricalRecord, HistoryLoadReport
from agent_evolve.core.rho.rounds import (
    PHASES,
    CandidateEvidence,
    RhoHooks,
    RoundConfig,
    RoundSummary,
    phases_for,
    rho_cluster_id,
    run_round,
    run_rounds,
)
from agent_evolve.core.rho.scheduler import ConcurrencyPlan


def _plan() -> ConcurrencyPlan:
    return ConcurrencyPlan.validated(
        group_workers=4, rollout_workers=3, global_cap=6
    )


# ---------------------------------------------------------------------- #
# Phase sequencing (plan Task 13 Step 1)
# ---------------------------------------------------------------------- #
def test_rho_mode_runs_the_ten_rho_phases() -> None:
    phases = phases_for("rho")

    assert phases[0] == "history_load"
    assert "trajectory_comprehension" in phases
    assert "difficulty_fingerprint" in phases
    assert "coreset_selection" in phases
    assert "group_rollouts" in phases
    assert "group_diagnosis" in phases
    assert "candidate_proposal" in phases
    assert "candidate_rollouts" in phases
    assert "preference_judging" in phases
    assert phases[-1] == "pool_commit"
    assert "genetic_iterations" not in phases


def test_genetic_mode_runs_only_the_genetic_phase() -> None:
    phases = phases_for("genetic")

    assert phases == ("genetic_iterations",)


def test_rho_genetic_runs_rho_then_genetic() -> None:
    phases = phases_for("rho-genetic")

    assert phases[0] == "history_load"
    assert phases[-1] == "genetic_iterations"
    assert phases.index("pool_commit") < phases.index("genetic_iterations")


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        phases_for("nonsense")


def test_every_declared_mode_has_a_phase_sequence() -> None:
    assert set(PHASES) == {"rho", "genetic", "rho-genetic"}
    assert all(seq for seq in PHASES.values())


def test_config_rejects_non_positive_rounds() -> None:
    with pytest.raises(ValueError, match="rounds"):
        RoundConfig(
            mode="rho", rounds=0, coreset_size=10, group_rollouts=3,
            candidates=3, selector="dpp", genetic_iterations_per_round=1,
            concurrency=_plan(),
        )


def test_config_rejects_non_positive_candidates() -> None:
    with pytest.raises(ValueError, match="candidates"):
        RoundConfig(
            mode="rho", rounds=1, coreset_size=10, group_rollouts=3,
            candidates=0, selector="dpp", genetic_iterations_per_round=1,
            concurrency=_plan(),
        )


def test_config_defaults_match_the_paper() -> None:
    config = RoundConfig(
        mode="rho", rounds=1, coreset_size=10, group_rollouts=3,
        candidates=3, candidate_rollouts=2, selector="dpp",
        genetic_iterations_per_round=1, concurrency=_plan(),
    )

    assert config.coreset_size == 10
    assert config.group_rollouts == 3
    assert config.candidates == 3


def test_candidate_rollouts_default_clears_the_entropy_floor() -> None:
    """R=2 satisfies min_rollouts_per_candidate=2 without touching EntropyTracker.

    See the Task 11 note: the floor is met by spending a second rollout, not by
    deleting the guard. Informative because CUGA rollouts are stochastic (tiny5
    gave 3/5 then 1/5 on the same harness).
    """
    from agent_evolve.core.entropy import EntropyTracker

    config = RoundConfig(
        mode="rho", rounds=1, coreset_size=10, group_rollouts=3,
        candidates=3, candidate_rollouts=2, selector="dpp",
        genetic_iterations_per_round=1, concurrency=_plan(),
    )
    tracker = EntropyTracker()

    assert config.candidate_rollouts >= tracker.min_rollouts_per_candidate
    # base + N candidates must clear the comparable-candidate floor.
    assert 1 + config.candidates >= tracker.min_comparable_candidates


def test_candidate_rollouts_defaults_to_two_when_unspecified() -> None:
    """The paper default is explicit: an unspecified R must not silently be 1."""
    config = RoundConfig(
        mode="rho", rounds=1, coreset_size=10, group_rollouts=3,
        candidates=3, selector="dpp", genetic_iterations_per_round=1,
        concurrency=_plan(),
    )

    assert config.candidate_rollouts == 2


def test_entropy_floor_needs_mark_comparable_not_just_rollouts() -> None:
    """Pins the wiring gotcha: rollout count alone does not satisfy the floor.

    EntropyTracker._comparable_candidates counts only candidates promoted via
    mark_comparable(). Without that call `comp` is empty and entropy is None no
    matter how many rollouts were spent.
    """
    from agent_evolve.core.entropy import EntropyTracker

    tracker = EntropyTracker()
    for candidate_id in ("base", "cand-0", "cand-1", "cand-2"):
        tracker.record_score("task-1", "cluster-1", candidate_id, 0.5)
        tracker.record_score("task-1", "cluster-1", candidate_id, 1.0)

    # Scores recorded but nobody promoted: still unavailable.
    assert tracker.entropy("task-1", "cluster-1") is None

    for candidate_id in ("base", "cand-0", "cand-1", "cand-2"):
        tracker.mark_comparable("task-1", "cluster-1", candidate_id)

    assert tracker.entropy("task-1", "cluster-1") is not None


def test_rollout_cost_is_reported() -> None:
    config = RoundConfig(
        mode="rho", rounds=1, coreset_size=10, group_rollouts=3,
        candidates=3, candidate_rollouts=2, selector="dpp",
        genetic_iterations_per_round=1, concurrency=_plan(),
    )

    # k*G baseline + k*N*R candidate = 30 + 60
    assert config.rollouts_per_round == 90


def test_candidate_rollouts_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="candidate_rollouts"):
        RoundConfig(
            mode="rho", rounds=1, coreset_size=10, group_rollouts=3,
            candidates=3, candidate_rollouts=0, selector="dpp",
            genetic_iterations_per_round=1, concurrency=_plan(),
        )


def test_summary_reports_a_candidate_collapse() -> None:
    summary = RoundSummary(
        round_index=1,
        selection_method="dpp",
        coreset_ids=("a", "b"),
        candidates_requested=3,
        candidates_distinct=1,
        discarded=((1, "duplicate"), (2, "no-op")),
        cache_hits={"summary": 4, "difficulty": 4},
        pool_size=2,
    )

    assert summary.collapsed is True
    assert "1 of 3" in summary.line()


def test_summary_without_collapse() -> None:
    summary = RoundSummary(
        round_index=1,
        selection_method="dpp",
        coreset_ids=("a",),
        candidates_requested=3,
        candidates_distinct=3,
        discarded=(),
        cache_hits={},
        pool_size=4,
    )

    assert summary.collapsed is False


def test_unknown_mode_in_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        RoundConfig(mode="nope", rounds=1, coreset_size=1, group_rollouts=1,
                    candidates=1, concurrency=_plan())


def test_config_exposes_its_phase_sequence() -> None:
    config = RoundConfig(mode="rho-genetic", rounds=1, coreset_size=1,
                         group_rollouts=1, candidates=1, concurrency=_plan())

    assert config.phases == phases_for("rho-genetic")


# ---------------------------------------------------------------------- #
# Agent neutrality
# ---------------------------------------------------------------------- #
def test_rounds_module_imports_no_adapter_or_agent_code() -> None:
    """core/ must stay agent-neutral: every adapter arrives as an injected hook."""
    import ast
    import pathlib

    import agent_evolve.core.rho.rounds as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    forbidden = ("cuga", "litellm", "agent_evolve.adapters")
    offenders = [
        name
        for name in imported
        for bad in forbidden
        if name == bad or name.startswith(bad + ".")
    ]
    assert offenders == []


# ---------------------------------------------------------------------- #
# Fakes for the round executor. Everything is duck-typed on purpose: the
# real objects live in adapters/ which core/ may not import.
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _FakeSummary:
    task_id: str
    observed: bool = True
    embedding_text: str = "summary text"


@dataclass(frozen=True, slots=True)
class _FakeVerdict:
    task_id: str
    difficulty: float = 5.0
    abstract_fingerprint: str = "fingerprint"
    observed: bool = True


@dataclass(frozen=True, slots=True)
class _FakeDiagnosis:
    task_id: str
    observed: bool = True
    recurring_failure_mode: str = "stops before verifying"
    severity: float = 0.8
    candidate_surfaces: tuple[str, ...] = ("instructions",)
    status: str = "OK"


@dataclass(frozen=True, slots=True)
class _FakeProposed:
    candidate_index: int
    artifacts: Mapping[str, str] = field(default_factory=lambda: {"instructions": "x"})
    observed: bool = True
    fingerprint: str = ""
    edited_ids: tuple[str, ...] = ("instructions",)
    created_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _FakeProposalReport:
    candidates: tuple[_FakeProposed, ...]
    requested: int
    discarded: tuple[tuple[int, str], ...] = ()

    @property
    def distinct(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class _FakePreference:
    task_id: str
    score: float = 0.5
    available: bool = True
    winner: str = "candidate"


def _task(task_id: str) -> EvolutionTask:
    return EvolutionTask(task_id=task_id, input_text=f"question {task_id}")


def _record(task_id: str) -> HistoricalRecord:
    return HistoricalRecord(
        task_id=task_id,
        input_text=f"question {task_id}",
        trace_path=f"/tmp/{task_id}/causal-trace.json",
        raw_trace={},
        final_output="answer",
        tool_observation_count=4,
        harness_version="v0",
        content_hash=f"hash-{task_id}",
    )


def _trace(task_id: str, version: str, index: int) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=f"{version}-{task_id}-{index}",
        candidate_id=version,
        task_id=task_id,
        events=(),
        final_output="answer",
        status="success",
    )


@dataclass
class _Recorder:
    """A complete, deterministic, offline hook implementation."""

    task_ids: tuple[str, ...]
    candidates: int = 2
    committed: list[CandidateEvidence] = field(default_factory=list)
    rollouts: list[tuple[str, str, int]] = field(default_factory=list)
    diagnosed: list[str] = field(default_factory=list)
    compared: list[tuple[str, str]] = field(default_factory=list)
    genetic_calls: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    proposal_diagnoses: list[int] = field(default_factory=list)
    cold_start: bool = False
    unobserved_summaries: frozenset[str] = frozenset()
    scores: Mapping[str, float] = field(default_factory=dict)

    # -- phase 1
    def load_history(self) -> HistoryLoadReport:
        if self.cold_start:
            return HistoryLoadReport(records=(), rejected=())
        return HistoryLoadReport(
            records=tuple(_record(t) for t in self.task_ids), rejected=()
        )

    # -- phase 2
    def comprehend(self, record: HistoricalRecord) -> _FakeSummary:
        return _FakeSummary(
            task_id=record.task_id,
            observed=record.task_id not in self.unobserved_summaries,
        )

    # -- phase 3
    def judge(self, record: HistoricalRecord, summary_text: str) -> _FakeVerdict:
        return _FakeVerdict(
            task_id=record.task_id, difficulty=1.0 + len(record.task_id)
        )

    # -- phase 4
    def task_for(self, task_id: str) -> EvolutionTask | None:
        return _task(task_id) if task_id in self.task_ids else None

    # -- phases 5 and 8
    def rollout(
        self, version: str, task: EvolutionTask, index: int
    ) -> RolloutOutcome:
        self.rollouts.append((version, task.task_id, index))
        return RolloutOutcome(
            task=task, trace=_trace(task.task_id, version, index)
        )

    # -- phase 6
    def diagnose(
        self, task_id: str, task_input: str, traces: Sequence[ExecutionTrace]
    ) -> _FakeDiagnosis:
        self.diagnosed.append(task_id)
        return _FakeDiagnosis(task_id=task_id)

    # -- phase 7
    def base_artifacts(self) -> Mapping[str, str]:
        return {"instructions": "base instructions"}

    def propose(
        self,
        base_artifacts: Mapping[str, str],
        diagnoses: Sequence[object],
        n: int,
    ) -> _FakeProposalReport:
        self.proposal_diagnoses.append(len(diagnoses))
        return _FakeProposalReport(
            candidates=tuple(_FakeProposed(candidate_index=i) for i in range(n)),
            requested=n,
        )

    def register_candidate(self, candidate: _FakeProposed) -> str:
        return f"cand-{candidate.candidate_index}"

    # -- phase 9
    def compare(
        self,
        task: EvolutionTask,
        baseline: ExecutionTrace,
        candidate: ExecutionTrace,
    ) -> _FakePreference:
        self.compared.append((task.task_id, candidate.candidate_id))
        return _FakePreference(task_id=task.task_id)

    # -- phase 10
    def commit(self, evidence: CandidateEvidence) -> None:
        self.committed.append(evidence)

    def pool_size(self) -> int:
        return 1 + len(self.committed)

    # -- optional
    def score(self, task: EvolutionTask, trace: ExecutionTrace) -> float:
        return self.scores.get(f"{trace.candidate_id}:{task.task_id}", 0.5)

    def run_genetic(self, tasks: Sequence[EvolutionTask], iterations: int) -> None:
        self.genetic_calls.append(
            (tuple(t.task_id for t in tasks), iterations)
        )

    def cache_hits(self) -> Mapping[str, int]:
        return {"summary": 3, "difficulty": 2}

    def hooks(self) -> RhoHooks:
        return RhoHooks(
            load_history=self.load_history,
            comprehend=self.comprehend,
            judge=self.judge,
            task_for=self.task_for,
            rollout=self.rollout,
            diagnose=self.diagnose,
            base_artifacts=self.base_artifacts,
            propose=self.propose,
            register_candidate=self.register_candidate,
            compare=self.compare,
            commit=self.commit,
            pool_size=self.pool_size,
            score=self.score,
            run_genetic=self.run_genetic,
            cache_hits=self.cache_hits,
        )


def _config(**overrides: object) -> RoundConfig:
    kwargs: dict[str, object] = dict(
        mode="rho",
        rounds=1,
        coreset_size=2,
        group_rollouts=3,
        candidates=2,
        candidate_rollouts=2,
        selector="difficulty_rank",
        genetic_iterations_per_round=1,
        concurrency=ConcurrencyPlan.validated(1, 1, 1),
    )
    kwargs.update(overrides)
    return RoundConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# Round executor
# ---------------------------------------------------------------------- #
def test_run_round_executes_every_rho_phase_in_order() -> None:
    recorder = _Recorder(task_ids=("t1", "t2", "t3"))

    summary = run_round(_config(), recorder.hooks(), round_index=1)

    assert summary.phases_run == phases_for("rho")


def test_run_round_selects_the_coreset_and_reports_the_method() -> None:
    recorder = _Recorder(task_ids=("t1", "t2", "t3"))

    summary = run_round(_config(), recorder.hooks())

    assert len(summary.coreset_ids) == 2
    assert summary.selection_method == "difficulty_rank"


def test_group_rollouts_spend_k_times_g() -> None:
    recorder = _Recorder(task_ids=("t1", "t2", "t3"))

    run_round(_config(), recorder.hooks())

    base = [r for r in recorder.rollouts if r[0] == "base"]
    assert len(base) == 2 * 3  # k=2 coreset tasks x G=3


def test_candidate_rollouts_spend_k_times_n_times_r() -> None:
    recorder = _Recorder(task_ids=("t1", "t2", "t3"))

    run_round(_config(), recorder.hooks())

    candidate = [r for r in recorder.rollouts if r[0] != "base"]
    assert len(candidate) == 2 * 2 * 2  # k=2 x N=2 x R=2


def test_all_n_candidates_are_retained_never_pruned_to_best_of_n() -> None:
    """The one deliberate deviation from the paper. Best-of-N is forbidden."""
    recorder = _Recorder(task_ids=("t1", "t2", "t3"))

    summary = run_round(_config(candidates=3), recorder.hooks())

    assert len(recorder.committed) == 3
    assert summary.candidates_distinct == 3
    assert {e.candidate_index for e in recorder.committed} == {0, 1, 2}


def test_a_losing_candidate_is_still_committed() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))
    hooks = recorder.hooks()

    def losing(task, baseline, candidate):
        return _FakePreference(
            task_id=task.task_id, score=-1.0, winner="baseline"
        )

    summary = run_round(_config(), _replace_hook(hooks, compare=losing))

    assert len(recorder.committed) == 2
    assert summary.preference_mean < 0


def test_unavailable_preferences_are_excluded_never_counted_as_ties() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))
    hooks = recorder.hooks()
    seen: list[str] = []

    def half_unavailable(task, baseline, candidate):
        seen.append(task.task_id)
        available = task.task_id == "t1"
        return _FakePreference(
            task_id=task.task_id,
            score=1.0 if available else 0.0,
            available=available,
            winner="candidate" if available else "unavailable",
        )

    summary = run_round(_config(), _replace_hook(hooks, compare=half_unavailable))

    # Only the available verdict contributes: a tie-defaulted mean would be 0.5.
    assert summary.preference_mean == pytest.approx(1.0)
    assert summary.preferences_unavailable == 2  # one per candidate


def test_preference_judging_costs_one_verdict_per_candidate_task_not_per_rollout() -> None:
    recorder = _Recorder(task_ids=("t1", "t2", "t3"))

    run_round(_config(), recorder.hooks())

    assert len(recorder.compared) == 2 * 2  # k=2 x N=2, not x R


def test_unobserved_summaries_are_excluded_from_the_coreset() -> None:
    recorder = _Recorder(
        task_ids=("t1", "t2", "t3"), unobserved_summaries=frozenset({"t3"})
    )

    summary = run_round(_config(coreset_size=3), recorder.hooks())

    assert "t3" not in summary.coreset_ids


def test_cold_start_history_short_circuits_the_round() -> None:
    recorder = _Recorder(task_ids=(), cold_start=True)

    summary = run_round(_config(), recorder.hooks())

    assert summary.coreset_ids == ()
    assert recorder.rollouts == []
    assert any("cold start" in note for note in summary.notes)


def test_entropy_cells_meet_the_floor_after_one_round() -> None:
    """The wiring gotcha, end to end.

    Base plus N=3 candidates each get >= 2 scored rollouts in the same
    task-local cell, and the round must promote each of them via
    mark_comparable(). Without the promotion entropy stays None regardless of R.
    """
    from agent_evolve.core.entropy import EntropyTracker

    recorder = _Recorder(task_ids=("t1", "t2"))
    tracker = EntropyTracker()
    scores = {
        "base:t1": 0.2, "cand-0:t1": 0.9, "cand-1:t1": 0.4, "cand-2:t1": 0.7,
    }
    recorder.scores = scores

    run_round(_config(candidates=3), recorder.hooks(), tracker=tracker)

    entropy = tracker.entropy("t1", rho_cluster_id("t1"))
    assert entropy is not None
    assert tracker.classify("t1", rho_cluster_id("t1")) != "skip"


def test_base_and_candidates_share_one_task_local_cell() -> None:
    """A cluster id derived from diagnosis text would split base from candidates.

    Base rollouts happen in phase 5, before any diagnosis exists in phase 6, so
    a diagnosis-derived cluster id would put base evidence in a different cell
    and the comparable-candidate floor could never be met.
    """
    from agent_evolve.core.entropy import EntropyTracker

    recorder = _Recorder(task_ids=("t1",))
    tracker = EntropyTracker()

    run_round(_config(coreset_size=1, candidates=3), recorder.hooks(), tracker=tracker)

    cells = tracker.all_cells()
    assert len(cells) == 1
    assert cells[0].mechanism_cluster_id == rho_cluster_id("t1")


def test_no_tracker_writes_without_a_score_hook() -> None:
    from agent_evolve.core.entropy import EntropyTracker

    recorder = _Recorder(task_ids=("t1",))
    hooks = _replace_hook(recorder.hooks(), score=None)
    tracker = EntropyTracker()

    run_round(_config(coreset_size=1), hooks, tracker=tracker)

    assert tracker.all_cells() == ()


def test_genetic_mode_runs_only_the_genetic_hook() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))

    summary = run_round(
        _config(mode="genetic", genetic_iterations_per_round=4), recorder.hooks()
    )

    assert summary.phases_run == ("genetic_iterations",)
    assert recorder.rollouts == []
    assert recorder.genetic_calls == [((), 4)]


def test_rho_genetic_restricts_the_genetic_phase_to_the_coreset() -> None:
    """Cross-candidate variance is undefined off the coreset, not low.

    Cells are created by rollouts; after a RHO round they exist only for the k
    coreset tasks. Running genetic work on all 42 would be 336 rollouts a round.
    """
    recorder = _Recorder(task_ids=("t1", "t2", "t3", "t4"))

    summary = run_round(_config(mode="rho-genetic", coreset_size=2), recorder.hooks())

    assert len(recorder.genetic_calls) == 1
    genetic_tasks, iterations = recorder.genetic_calls[0]
    assert set(genetic_tasks) == set(summary.coreset_ids)
    assert len(genetic_tasks) == 2
    assert iterations == 1


def test_zero_genetic_iterations_skips_the_hook() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))

    run_round(
        _config(mode="rho-genetic", genetic_iterations_per_round=0), recorder.hooks()
    )

    assert recorder.genetic_calls == []


def test_run_rounds_returns_one_summary_per_round_with_increasing_index() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))

    summaries = run_rounds(_config(rounds=3), recorder.hooks())

    assert [s.round_index for s in summaries] == [1, 2, 3]


def test_summary_carries_cache_hits_and_pool_size() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))

    summary = run_round(_config(), recorder.hooks())

    assert summary.cache_hits == {"summary": 3, "difficulty": 2}
    assert summary.pool_size == 3  # base + 2 committed candidates


def test_a_missing_required_hook_names_itself() -> None:
    recorder = _Recorder(task_ids=("t1",))
    hooks = _replace_hook(recorder.hooks(), diagnose=None)

    with pytest.raises(ValueError, match="diagnose"):
        run_round(_config(coreset_size=1), hooks)


def test_only_observed_diagnoses_reach_the_optimizer() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))
    hooks = recorder.hooks()

    def unobserved(task_id, task_input, traces):
        recorder.diagnosed.append(task_id)
        return _FakeDiagnosis(task_id=task_id, observed=False, status="NO_OP")

    run_round(_config(), _replace_hook(hooks, diagnose=unobserved))

    assert recorder.diagnosed == ["t1", "t2"]
    assert recorder.proposal_diagnoses == [0]


def test_failed_rollouts_do_not_discard_a_group() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))
    hooks = recorder.hooks()

    def flaky(version: str, task: EvolutionTask, index: int) -> RolloutOutcome:
        recorder.rollouts.append((version, task.task_id, index))
        if index == 0:
            return RolloutOutcome(task=task, trace=None, error="agent crashed")
        return RolloutOutcome(task=task, trace=_trace(task.task_id, version, index))

    summary = run_round(_config(), _replace_hook(hooks, rollout=flaky))

    assert summary.rollout_failures > 0
    assert recorder.diagnosed == ["t1", "t2"]  # groups survived


def test_a_task_with_no_usable_trace_is_not_diagnosed() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))
    hooks = recorder.hooks()

    def always_fails(version: str, task: EvolutionTask, index: int) -> RolloutOutcome:
        return RolloutOutcome(task=task, trace=None, error="agent crashed")

    run_round(_config(), _replace_hook(hooks, rollout=always_fails))

    assert recorder.diagnosed == []


_SECRET_LITERAL = "Rosetta Stone 1799"


def _memorizing_hooks(recorder: _Recorder) -> RhoHooks:
    """A proposal that smuggled an answer key into an artifact."""

    def propose(base_artifacts, diagnoses, n):
        recorder.proposal_diagnoses.append(len(diagnoses))
        return _FakeProposalReport(
            candidates=tuple(
                _FakeProposed(
                    candidate_index=i,
                    artifacts={"instructions": f"answer is {_SECRET_LITERAL}"},
                )
                for i in range(n)
            ),
            requested=n,
        )

    return _replace_hook(
        recorder.hooks(),
        propose=propose,
        contamination_literals=(_SECRET_LITERAL,),
    )


def test_contamination_scan_is_observational_and_reported() -> None:
    recorder = _Recorder(task_ids=("t1",))

    summary = run_round(_config(coreset_size=1), _memorizing_hooks(recorder))

    # Reported, and nothing was blocked: all candidates still committed.
    assert summary.contamination
    assert len(recorder.committed) == 2
    assert any("contamination" in note for note in summary.notes)


def test_contamination_report_names_artifacts_not_literals() -> None:
    recorder = _Recorder(task_ids=("t1",))

    summary = run_round(_config(coreset_size=1), _memorizing_hooks(recorder))

    for candidate_index, artifact_id, confidence in summary.contamination:
        assert isinstance(candidate_index, int)
        assert artifact_id == "instructions"
        assert confidence == "high"
        # The literal itself must never appear in the report.
        assert _SECRET_LITERAL not in artifact_id


def test_no_contamination_scan_without_literals() -> None:
    recorder = _Recorder(task_ids=("t1",))

    summary = run_round(_config(coreset_size=1), recorder.hooks())

    assert summary.contamination == ()


def test_candidate_evidence_reports_mean_preference_and_rollouts() -> None:
    recorder = _Recorder(task_ids=("t1", "t2"))

    run_round(_config(), recorder.hooks())

    evidence = recorder.committed[0]
    assert evidence.version == "cand-0"
    assert evidence.rollouts == 2 * 2  # k=2 x R=2
    assert evidence.mean_preference == pytest.approx(0.5)
    assert evidence.artifacts == {"instructions": "x"}


def _replace_hook(hooks: RhoHooks, **overrides: object) -> RhoHooks:
    import dataclasses

    return dataclasses.replace(hooks, **overrides)  # type: ignore[arg-type]
