"""Behavioural tests for the RHO pairwise preference gate (SV-4).

What this pins down, and why each assertion is behavioural rather than
structural:

The RHO paper accepts a candidate only when its symmetric pairwise preference
score ``S_j`` is positive. ``PreferenceJudge.compare_symmetric`` computes exactly
that and RHO already pays for it, but before this change the value was written to
``CandidateEvidence.mean_preference``, printed to the console, and read by
nothing: two writes and zero reads across ``src/``. Champion selection instead
ranked on ``0.55*outcome + 0.20*coverage``, a rule the paper does not specify.

So these tests assert on the *selected champion*, not on the presence of a field.
A test that only checked ``entry.preference is not None`` would pass against a
wiring that stores the number and still ignores it -- which is precisely the bug.

Ablation direction matters. The gate is ON by default (paper behaviour) and the
``experimental_candidate_promotion`` flag turns it OFF to recover the old
aggregate-only ranking for comparison. Defaulting the flag to ``True`` would mean
a plain run silently keeps the defect, so the default is ``False``.
"""
from __future__ import annotations

import pytest

from agent_evolve.core.config import resolve_profile
from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import PersistentPool, ScoreProvenance


def _candidate(cid: str) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=cid, version=cid, artifact_hashes={}, parent_ids=(), ancestor_ids=()
    )


def _record(pool: PersistentPool, cid: str, task: str, value: float, seq: int) -> None:
    pool.record_score(
        cid,
        value,
        ScoreProvenance(
            task_id=task,
            mechanism_cluster_id=f"rho-task:{task}",
            trace_id=f"t:{cid}:{task}:{seq}",
            rollout_seq=seq,
            analyzer_model_id="a",
            judge_model_id="j",
            blame_confidence=0.0,
            blame_stability=0.0,
        ),
    )


@pytest.fixture()
def pool() -> PersistentPool:
    p = PersistentPool()
    p.add_base(_candidate("base-v0"))
    return p


# ---------------------------------------------------------------------- #
# Storage: preference must survive the commit boundary at all.
# ---------------------------------------------------------------------- #
def test_preference_defaults_to_absent_not_zero(pool: PersistentPool) -> None:
    """A candidate with no verdict must be distinguishable from one scoring 0.0.

    Absent evidence and a measured tie are different facts. Collapsing both to
    0.0 would make ``S_j > 0`` silently reject an unjudged candidate for the
    wrong reason.
    """
    entry = pool.add_candidate(_candidate("cand-A"))
    assert entry.preference is None
    assert entry.preference_available == 0


def test_record_preference_round_trips(pool: PersistentPool) -> None:
    pool.add_candidate(_candidate("cand-A"))
    pool.record_preference("cand-A", 0.42, available=3, unavailable=1)
    entry = pool.get("cand-A")
    assert entry.preference == pytest.approx(0.42)
    assert entry.preference_available == 3
    assert entry.preference_unavailable == 1


def test_record_preference_rejects_unknown_candidate(pool: PersistentPool) -> None:
    with pytest.raises(KeyError):
        pool.record_preference("nope", 0.5, available=1)


def test_record_preference_rejects_out_of_range(pool: PersistentPool) -> None:
    """S_j is an antisymmetric preference in [-1, 1]; anything else is a bug."""
    pool.add_candidate(_candidate("cand-A"))
    with pytest.raises(ValueError):
        pool.record_preference("cand-A", 1.5, available=1)
    with pytest.raises(ValueError):
        pool.record_preference("cand-A", -2.0, available=1)


def test_available_zero_forces_preference_absent(pool: PersistentPool) -> None:
    """Zero available verdicts cannot back a preference number."""
    pool.add_candidate(_candidate("cand-A"))
    pool.record_preference("cand-A", 0.9, available=0, unavailable=4)
    assert pool.get("cand-A").preference is None


# ---------------------------------------------------------------------- #
# The gate: this is the behaviour SV-4 is about.
# ---------------------------------------------------------------------- #
def test_negative_preference_candidate_cannot_be_champion(pool: PersistentPool) -> None:
    """A candidate the judge dispreferred must not be exported as champion.

    The aggregate is deliberately rigged to favour ``cand-A`` (higher outcome,
    equal coverage) so that only the gate can produce the correct answer.
    """
    pool.add_candidate(_candidate("cand-A"))
    for task in ("t1", "t2"):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.50, seq)
            _record(pool, "cand-A", task, 0.95, seq)
    pool.record_preference("cand-A", -0.60, available=2)

    report = pool.select_champion()
    assert report.candidate_id == "base-v0"
    assert "cand-A" in report.disqualifications


def test_positive_preference_candidate_wins(pool: PersistentPool) -> None:
    pool.add_candidate(_candidate("cand-A"))
    for task in ("t1", "t2"):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.50, seq)
            _record(pool, "cand-A", task, 0.95, seq)
    pool.record_preference("cand-A", 0.60, available=2)

    report = pool.select_champion()
    assert report.candidate_id == "cand-A"
    assert report.preference == pytest.approx(0.60)


def test_zero_preference_is_not_positive_and_is_gated(pool: PersistentPool) -> None:
    """``S_j > 0`` is strict: a measured tie is not an improvement."""
    pool.add_candidate(_candidate("cand-A"))
    for task in ("t1", "t2"):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.50, seq)
            _record(pool, "cand-A", task, 0.95, seq)
    pool.record_preference("cand-A", 0.0, available=2)

    assert pool.select_champion().candidate_id == "base-v0"


def test_unjudged_candidate_is_gated_out(pool: PersistentPool) -> None:
    """No verdict means no evidence of improvement, so no promotion.

    This is the conservative reading and it matters: the alternative would let a
    candidate whose judging failed inherit a promotion it never earned.
    """
    pool.add_candidate(_candidate("cand-A"))
    for task in ("t1", "t2"):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.50, seq)
            _record(pool, "cand-A", task, 0.95, seq)

    report = pool.select_champion()
    assert report.candidate_id == "base-v0"
    assert "cand-A" in report.disqualifications


def test_base_is_never_gated_by_preference(pool: PersistentPool) -> None:
    """The incumbent is not a candidate for promotion; it is the fallback.

    Gating the base would empty the eligible set and raise, turning "nothing
    improved" into a crash.
    """
    for task in ("t1", "t2"):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.50, seq)
    report = pool.select_champion()
    assert report.candidate_id == "base-v0"


def test_gate_picks_best_among_positive_candidates(pool: PersistentPool) -> None:
    """Among gate survivors the aggregate still ranks -- the gate is eligibility."""
    for cid in ("cand-A", "cand-B", "cand-C"):
        pool.add_candidate(_candidate(cid))
    for task in ("t1", "t2"):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.50, seq)
            _record(pool, "cand-A", task, 0.70, seq)
            _record(pool, "cand-B", task, 0.90, seq)
            _record(pool, "cand-C", task, 0.99, seq)
    pool.record_preference("cand-A", 0.20, available=2)
    pool.record_preference("cand-B", 0.10, available=2)
    pool.record_preference("cand-C", -0.90, available=2)  # best score, dispreferred

    report = pool.select_champion()
    assert report.candidate_id == "cand-B"
    assert "cand-C" in report.disqualifications


# ---------------------------------------------------------------------- #
# Ablation flag: OFF by default, and it must genuinely change the outcome.
# ---------------------------------------------------------------------- #
def test_flag_defaults_to_false_so_gate_is_active() -> None:
    assert resolve_profile("minimal").experimental_candidate_promotion is False


def test_flag_true_restores_ungated_aggregate_ranking(pool: PersistentPool) -> None:
    """The ablation arm must actually reproduce the pre-gate behaviour."""
    pool.add_candidate(_candidate("cand-A"))
    for task in ("t1", "t2"):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.50, seq)
            _record(pool, "cand-A", task, 0.95, seq)
    pool.record_preference("cand-A", -0.60, available=2)

    ablation = resolve_profile("minimal", experimental_candidate_promotion=True)
    assert pool.select_champion(config=ablation).candidate_id == "cand-A"

    paper = resolve_profile("minimal", experimental_candidate_promotion=False)
    assert pool.select_champion(config=paper).candidate_id == "base-v0"


def test_report_exposes_gate_state_for_audit(pool: PersistentPool) -> None:
    """A champion manifest must say whether the gate was applied.

    Without this an exported champion.json is ambiguous between a paper run and
    an ablation run, which would make the two indistinguishable after the fact.
    """
    pool.add_candidate(_candidate("cand-A"))
    for task in ("t1",):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.50, seq)
            _record(pool, "cand-A", task, 0.95, seq)
    pool.record_preference("cand-A", 0.30, available=1)

    gated = pool.select_champion()
    assert gated.preference_gate_applied is True

    ungated = pool.select_champion(
        config=resolve_profile("minimal", experimental_candidate_promotion=True)
    )
    assert ungated.preference_gate_applied is False


def test_all_candidates_gated_falls_back_to_base_not_crash(pool: PersistentPool) -> None:
    for cid in ("cand-A", "cand-B"):
        pool.add_candidate(_candidate(cid))
    for task in ("t1",):
        for seq in range(2):
            _record(pool, "base-v0", task, 0.40, seq)
            _record(pool, "cand-A", task, 0.99, seq)
            _record(pool, "cand-B", task, 0.99, seq)
    pool.record_preference("cand-A", -0.10, available=1)
    pool.record_preference("cand-B", -0.20, available=1)

    report = pool.select_champion()
    assert report.candidate_id == "base-v0"
    assert set(report.disqualifications) == {"cand-A", "cand-B"}


def test_pool_retention_is_unaffected_by_the_gate(pool: PersistentPool) -> None:
    """The gate governs promotion, never survival.

    AGENTS.md requires base plus every proposal to be retained; a gate that
    dropped candidates from the pool would violate the persistent-pool design
    and destroy the negative evidence a later analysis needs.
    """
    for cid in ("cand-A", "cand-B"):
        pool.add_candidate(_candidate(cid))
    for seq in range(2):
        _record(pool, "base-v0", "t1", 0.40, seq)
        _record(pool, "cand-A", "t1", 0.99, seq)
        _record(pool, "cand-B", "t1", 0.99, seq)
    pool.record_preference("cand-A", -0.10, available=1)
    pool.record_preference("cand-B", 0.10, available=1)

    pool.select_champion()
    assert len(pool) == 3
    assert set(pool.candidate_ids()) == {"base-v0", "cand-A", "cand-B"}
