"""SV-13d — resolving the surviving pool to one winner.

Two paths, in this order:

1. **Sole survivor.** If generational retirement has reduced the live pool to one
   candidate, that candidate is the winner outright. Nothing to compare it
   against, so no judge call is spent.
2. **Symmetric ladder.** Otherwise the survivors are resolved by pairwise
   preference over the coreset: a linear king-of-the-hill, ``N-1`` comparisons for
   ``N`` survivors, each comparison being the symmetric two-call form. ``2(N-1)``
   model calls, not ``N choose 2``.

**Why a ladder replaces the score ranking for final selection.** When this module
was written the aggregate ranking was measurably wrong. Reproduced against the real
pool:

    base     outcome=0.5000 coverage=1.0000 aggregate=0.6250
    cand-A   outcome=0.9000 coverage=0.5000 aggregate=0.7450   <- exported champion

``cand-A`` never ran the ``hard`` task at all. It won because ``outcome`` averaged
over whatever each candidate happened to measure (SV-2), and because ``coverage``
was scored as if breadth of measurement were quality (SV-3).

Both have since been fixed in ``PersistentPool.select_champion``, which now ranks by
pairwise comparison over the cells two candidates share -- so that exported champion
is unreachable by either route. The ladder is still preferred because it compares
*trajectories*: it can prefer a candidate whose scores tie but whose reasoning is
sound, which no score comparison can express.

**Ladder order is deterministic.** Insertion order, base first. An LLM judge is
not perfectly transitive, so the outcome can depend on comparison order; fixing the
order makes the result reproducible from the pool alone rather than from whatever
order a dict happened to yield.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import PersistentPool, ScoreProvenance
from agent_evolve.core.resolution import (
    FinalResolution,
    resolve_final_candidate,
)

_MECH = "m0"


class _Verdict:
    def __init__(self, score: float, available: bool = True) -> None:
        self.score = score
        self.available = available


class _Task:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


class _Trace:
    def __init__(self, cid: str, task_id: str) -> None:
        self.candidate_id = cid
        self.trace_id = f"tr-{cid}-{task_id}"


_TASKS = (_Task("task-a"), _Task("task-b"))


class _Ladder:
    """Judge that always prefers whichever candidate id is in ``winner``."""

    def __init__(self, winner: str, available: bool = True) -> None:
        self.winner = winner
        self._available = available
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, task, baseline, candidate):  # noqa: ANN001
        self.calls.append(
            (task.task_id, baseline.candidate_id, candidate.candidate_id)
        )
        if not self._available:
            return _Verdict(0.0, False)
        if candidate.candidate_id == self.winner:
            return _Verdict(0.9)
        if baseline.candidate_id == self.winner:
            return _Verdict(-0.9)
        return _Verdict(0.0)


def _cand(cid: str, parents: tuple[str, ...] = ()) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=cid,
        version=cid,
        artifact_hashes={},
        parent_ids=parents,
        ancestor_ids=parents,
        attempt_ids=(),
    )


def _pool(*extra: str) -> PersistentPool:
    """Base plus ``extra`` candidates, all measured on both tasks.

    Every non-base candidate is given a *positive* recorded preference, because
    resolution enforces the SV-4 acceptance gate: a candidate the RHO judge
    dispreferred or never judged cannot be promoted, and re-judging it during
    final resolution would let a second opinion overturn that gate. A fixture
    without preferences would therefore leave only the base eligible and every
    ladder assertion would be vacuous.
    """
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    for cid in extra:
        pool.add_candidate(_cand(cid, ("base",)))
        pool.record_preference(cid, 0.5, available=2)
    for cid in ("base", *extra):
        for task in _TASKS:
            seq = pool.get(cid).cell(task.task_id, _MECH).rollout_count
            pool.record_score(
                cid,
                0.5,
                ScoreProvenance(
                    task_id=task.task_id,
                    mechanism_cluster_id=_MECH,
                    trace_id=f"tr-{cid}-{task.task_id}",
                    rollout_seq=seq,
                    analyzer_model_id="a",
                    judge_model_id="j",
                    blame_confidence=0.0,
                    blame_stability=0.0,
                ),
            )
    return pool


def _traces(pool: PersistentPool) -> dict[str, dict[str, object]]:
    return {
        e.candidate_id: {t.task_id: _Trace(e.candidate_id, t.task_id) for t in _TASKS}
        for e in pool.all_entries()
    }


# --------------------------------------------------------------------------- #
# 1. Sole survivor short-circuits
# --------------------------------------------------------------------------- #


def test_a_sole_survivor_wins_without_any_judge_call() -> None:
    """"If the pool shrinks to 1, declare that the winner." No rollouts, no calls."""
    pool = _pool("child")
    pool.retire("base", superseded_by="child")
    judge = _Ladder("base")

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=judge
    )

    assert result.candidate_id == "child"
    assert result.method == "sole_survivor"
    assert judge.calls == []


def test_a_sole_survivor_is_reported_as_such() -> None:
    pool = _pool("child")
    pool.retire("base", superseded_by="child")

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=_Ladder("child")
    )

    assert isinstance(result, FinalResolution)
    assert result.judge_calls == 0
    assert result.method == "sole_survivor"
    assert result.comparisons == 0
    assert result.reason  # a resolution must always state its basis


# --------------------------------------------------------------------------- #
# 2. The ladder
# --------------------------------------------------------------------------- #


def test_the_ladder_finds_the_preferred_candidate() -> None:
    pool = _pool("c1", "c2", "c3")
    judge = _Ladder("c2")

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=judge
    )

    assert result.candidate_id == "c2"
    assert result.method == "pairwise_ladder"


def test_the_ladder_costs_n_minus_one_comparisons() -> None:
    """Linear, not quadratic. 4 survivors -> 3 comparisons, each over k tasks."""
    pool = _pool("c1", "c2", "c3")  # base + 3 = 4 live
    judge = _Ladder("c2")

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=judge
    )

    survivors = len(pool.live_candidate_ids())
    assert result.comparisons == survivors - 1
    # Each comparison is one call per coreset task.
    assert len(judge.calls) == (survivors - 1) * len(_TASKS)


def test_a_retired_candidate_never_enters_the_ladder() -> None:
    """Retirement must actually reduce the judged set, or it saved nothing."""
    pool = _pool("c1", "c2")
    pool.retire("base", superseded_by="c1")
    judge = _Ladder("c2")

    resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=judge
    )

    judged = {c[1] for c in judge.calls} | {c[2] for c in judge.calls}
    assert "base" not in judged


def test_the_ladder_order_is_deterministic() -> None:
    """An LLM judge is not perfectly transitive, so order must not vary."""
    first = _Ladder("c2")
    second = _Ladder("c2")

    pool_a = _pool("c1", "c2", "c3")
    resolve_final_candidate(
        pool_a, tasks=_TASKS, traces=_traces(pool_a), compare=first
    )
    pool_b = _pool("c1", "c2", "c3")
    resolve_final_candidate(
        pool_b, tasks=_TASKS, traces=_traces(pool_b), compare=second
    )

    assert first.calls == second.calls


def test_the_incumbent_holds_when_no_challenger_is_preferred() -> None:
    """A tie or a loss leaves the incumbent standing, so the first survivor wins
    by default rather than the last one compared."""
    pool = _pool("c1", "c2")
    judge = _Ladder("nobody")  # every verdict is 0.0

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=judge
    )

    assert result.candidate_id == "base"


# --------------------------------------------------------------------------- #
# 3. It replaces the aggregate where the aggregate is wrong
# --------------------------------------------------------------------------- #


def test_the_ladder_can_reject_the_candidate_the_aggregate_would_export() -> None:
    """The SV-2/SV-3 consequence, removed -- now at both layers.

    ``cand-A`` measures only the easy cell. When this test was written it therefore
    out-ranked base on the aggregate, and the ladder was the only thing standing
    between that and ``champion.json``.

    SV-2 has since fixed the ranking itself: ``select_champion`` compares on the
    cells both entries measured, where ``cand-A`` only ties. So both assertions
    below now say ``base``, by two independent routes -- which is stronger than the
    original claim, not weaker. The ladder is no longer the sole guard.
    """
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("cand-A", ("base",)))
    for task, value in (("task-a", 0.9), ("task-b", 0.1)):
        pool.record_score(
            "base",
            value,
            ScoreProvenance(
                task_id=task,
                mechanism_cluster_id=_MECH,
                trace_id=f"tr-base-{task}",
                rollout_seq=0,
                analyzer_model_id="a",
                judge_model_id="j",
                blame_confidence=0.0,
                blame_stability=0.0,
            ),
        )
    pool.record_score(
        "cand-A",
        0.9,
        ScoreProvenance(
            task_id="task-a",
            mechanism_cluster_id=_MECH,
            trace_id="tr-candA-task-a",
            rollout_seq=0,
            analyzer_model_id="a",
            judge_model_id="j",
            blame_confidence=0.0,
            blame_stability=0.0,
        ),
    )
    pool.record_preference("cand-A", 0.5, available=2)

    # SV-2 has since removed the aggregate's half of this defect: ranking is now a
    # pairwise comparison on shared cells, and cand-A merely *ties* base on the one
    # cell both measured, so the incumbent holds here too.
    assert pool.select_champion().entry.candidate_id == "base"

    # The judge, comparing on shared tasks, reaches the same answer by its own
    # route. Two independent instruments agreeing is the point: the aggregate can no
    # longer express the skip-the-hard-task win, and the judge never could.
    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=_Ladder("base")
    )

    assert result.candidate_id == "base"


# --------------------------------------------------------------------------- #
# 3b. SV-4 survives SV-13d
# --------------------------------------------------------------------------- #


def test_a_dispreferred_candidate_cannot_win_the_ladder() -> None:
    """Regression guard for a real defect this module introduced.

    A candidate the RHO judge dispreferred (``preference = -0.5``) was promoted by
    the ladder, because a *fresh* comparison overturned the recorded verdict. That
    is exactly the exported-harness defect the SV-4 gate exists to prevent, and it
    reached ``champion_version`` end to end before being caught.

    Resolution now filters to promotable candidates first, so no second opinion can
    overturn acceptance.
    """
    pool = _pool()
    pool.add_candidate(_cand("bad", ("base",)))
    pool.record_preference("bad", -0.5, available=2)
    for task in _TASKS:
        pool.record_score(
            "bad",
            1.0,  # scores *better* than base, so only the gate gives the right answer
            ScoreProvenance(
                task_id=task.task_id,
                mechanism_cluster_id=_MECH,
                trace_id=f"tr-bad-{task.task_id}",
                rollout_seq=0,
                analyzer_model_id="a",
                judge_model_id="j",
                blame_confidence=0.0,
                blame_stability=0.0,
            ),
        )

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=_Ladder("bad")
    )

    assert result.candidate_id == "base"


def test_an_unjudged_candidate_cannot_win_the_ladder() -> None:
    """``preference is None`` means no evidence of improvement, which is not the
    same as a tie and must not be promotable."""
    pool = _pool()
    pool.add_candidate(_cand("unjudged", ("base",)))
    for task in _TASKS:
        pool.record_score(
            "unjudged",
            1.0,
            ScoreProvenance(
                task_id=task.task_id,
                mechanism_cluster_id=_MECH,
                trace_id=f"tr-unjudged-{task.task_id}",
                rollout_seq=0,
                analyzer_model_id="a",
                judge_model_id="j",
                blame_confidence=0.0,
                blame_stability=0.0,
            ),
        )

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=_Ladder("unjudged")
    )

    assert result.candidate_id == "base"


def test_an_ineligible_sole_survivor_does_not_win_by_default() -> None:
    """Ordering matters: the gate is applied *before* the sole-survivor
    short-circuit, or retiring a parent would hand the win to a candidate that
    could not have been promoted on its own evidence.
    """
    pool = _pool()
    pool.add_candidate(_cand("bad", ("base",)))
    pool.record_preference("bad", -0.5, available=2)
    for task in _TASKS:
        pool.record_score(
            "bad",
            1.0,
            ScoreProvenance(
                task_id=task.task_id,
                mechanism_cluster_id=_MECH,
                trace_id=f"tr-bad2-{task.task_id}",
                rollout_seq=0,
                analyzer_model_id="a",
                judge_model_id="j",
                blame_confidence=0.0,
                blame_stability=0.0,
            ),
        )
    pool.retire("base", superseded_by="bad")
    assert pool.live_candidate_ids() == ("bad",)

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=_Ladder("bad")
    )

    assert result.candidate_id != "bad"
    assert result.method == "aggregate_fallback"


# --------------------------------------------------------------------------- #
# 4. Degenerate and failure cases
# --------------------------------------------------------------------------- #


def test_an_empty_pool_raises() -> None:
    pool = PersistentPool(min_comparable_rollouts=1)

    with pytest.raises(ValueError, match="no candidates"):
        resolve_final_candidate(
            pool, tasks=_TASKS, traces={}, compare=_Ladder("x")
        )


def test_no_judge_falls_back_to_the_aggregate() -> None:
    """Without a judge there is no preference evidence, so the documented
    aggregate remains the only available opinion. Reported honestly as such."""
    pool = _pool("c1")
    pool.record_preference("c1", 0.5, available=2)

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=None
    )

    assert result.method == "aggregate_fallback"
    assert result.judge_calls == 0


def test_an_unavailable_judge_falls_back_and_says_so() -> None:
    """A judge outage must not silently pick the first candidate in a list."""
    pool = _pool("c1")
    pool.record_preference("c1", 0.5, available=2)

    result = resolve_final_candidate(
        pool,
        tasks=_TASKS,
        traces=_traces(pool),
        compare=_Ladder("c1", available=False),
    )

    assert result.method == "aggregate_fallback"
    assert "unavailable" in result.reason


def test_a_missing_trace_pair_is_skipped_not_fatal() -> None:
    """One unrollable survivor must not abort resolution for everyone else."""
    pool = _pool("c1", "c2")
    traces = _traces(pool)
    del traces["c1"]  # c1 was never rolled out
    judge = _Ladder("c2")

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=traces, compare=judge
    )

    assert result.candidate_id == "c2"
    judged = {c[1] for c in judge.calls} | {c[2] for c in judge.calls}
    assert "c1" not in judged


def test_a_raising_judge_falls_back_to_the_aggregate() -> None:
    def boom(task, baseline, candidate):  # noqa: ANN001
        raise RuntimeError("judge exploded")

    pool = _pool("c1")
    pool.record_preference("c1", 0.5, available=2)

    result = resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=boom
    )

    assert result.method == "aggregate_fallback"
    assert "error" in result.reason


def test_resolution_mutates_nothing() -> None:
    """Pure: a caller may resolve twice, or dry-run, without side effects."""
    pool = _pool("c1", "c2")
    before = pool.live_candidate_ids()

    resolve_final_candidate(
        pool, tasks=_TASKS, traces=_traces(pool), compare=_Ladder("c1")
    )

    assert pool.live_candidate_ids() == before
    assert all(not e.retired for e in pool.all_entries())
