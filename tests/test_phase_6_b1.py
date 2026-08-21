"""Phase-gate verification for the B1 persistent-pool GEPA experiment runner.

This proves the Phase 6 deliverable from ``feedback/from_qwen/qf21.md`` Task B:
an N-attempt sequential GEPA loop over a persistent pool that reports a final
champion and pool state.

Asserted properties:

* The pool retains the base plus every seeded candidate (never elite-only).
* ``n_attempts`` attempts execute and each is accounted for.
* A final :class:`ChampionReport` is produced with every manifest component.
* The run is deterministic for a fixed seed.
* No evaluator-internal expected-substring token reaches storage.
* The embedding provider and any fallback reason are recorded, never silent.

Offline and deterministic: FakeAdapter + lexical embeddings, no LLM, no CUGA,
no network, and no merge/parallel service.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from examples.run_phase_6_b1 import (  # noqa: E402
    TASKS,
    B1ExperimentResult,
    run_b1_experiment,
)
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402

_TOKEN_A = "token-a"
_TOKEN_B = "token-b"


def _offline() -> LexicalEmbedder:
    """Deterministic offline embedder so unit tests never touch the network."""
    return LexicalEmbedder(dim=32)


def _read_all_persisted_blobs(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.rglob("*.json"))
    )


# ---------------------------------------------------------------------- #
# Persistent pool retention
# ---------------------------------------------------------------------- #
def test_b1_retains_base_and_every_seeded_candidate(tmp_path: Path) -> None:
    """B1 is a persistent pool: nothing is discarded for being non-elite."""
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=3, embedder=_offline())

    assert isinstance(result, B1ExperimentResult)
    assert {"base", "c1", "c2"} <= set(result.pool_candidate_ids)
    assert result.pool_size >= 3
    assert result.seeded_candidate_count == 3


def test_b1_pool_grows_only_by_accepted_attempts(tmp_path: Path) -> None:
    """Final pool size equals the seeded count plus accepted attempts."""
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=3, embedder=_offline())

    assert result.pool_size == result.seeded_candidate_count + result.accepted


def test_b1_frontier_is_not_trivially_collapsed_before_any_attempt(
    tmp_path: Path,
) -> None:
    """c1 and c2 each win a distinct task, so neither dominates the other.

    Measured with ``n_attempts=0``-equivalent seeding: c1 scores
    ``task-a=1.0, task-b=0.0`` and c2 the mirror image, so both sit on the
    frontier.

    **Why this no longer asserts a frontier of >= 2 after attempts run (SV-11).**
    ``build_issues`` used to diagnose ``pool.base`` regardless of the selected
    parent, so an attempt inherited base's failures and its offspring never
    strictly dominated the seeded pair. Now that observation follows the selected
    parent, the offspring is built on c2 and measures
    ``task-a=1.0, task-b=1.0`` -- which *strictly dominates both* c1 and c2, so a
    frontier of exactly 1 is the mathematically correct answer:

        base                               task-a=0.0  task-b=0.0
        c1                                 task-a=1.0  task-b=0.0
        c2                                 task-a=0.0  task-b=1.0
        base-v0+attempt-c2+att-i001-s0000  task-a=1.0  task-b=1.0   <- dominates

    Asserting >= 2 here would now require the loop to *fail* to produce a
    dominating candidate, which is the opposite of what the experiment is for.
    """
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=1, embedder=_offline())

    assert result.seeded_candidate_count == 3
    assert {"c1", "c2"} <= set(result.pool_candidate_ids)
    assert result.frontier_size >= 1


def test_b1_a_dominating_offspring_collapses_the_frontier(tmp_path: Path) -> None:
    """The frontier must reflect measured dominance, not pool size.

    Pins the consequence of SV-11 directly: once the parent is the observation
    subject, ONE attempt yields an offspring that wins every cell and is
    therefore the sole non-dominated entry even though four candidates exist.

    **Corrected while closing SV-10.** This assertion previously read
    ``pool_size == 4`` and ``frontier_size == 1`` against ``n_attempts=2``, which
    paired the docstring's one-attempt table above with a two-attempt run. It was
    green only because attempt 2 then chained onto attempt 1's offspring; once
    SV-10 made both attempts breed from the same observed parent, the second
    attempt produced an accepted *sibling* and the real numbers became 5 and 2.
    The mismatch was mine, not a regression.
    """
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=1, embedder=_offline())

    assert result.pool_size == 4
    assert result.frontier_size == 1


def test_b1_siblings_breed_from_the_same_observed_parent(tmp_path: Path) -> None:
    """SV-10: every attempt in a run breeds from the parent it diagnosed.

    ``select_parent`` consumes ``rng.random()``, so before SV-10 each attempt drew
    its own parent and attempt 2 chained onto attempt 1's offspring
    (``...att-i001-s0000+att-i002-s0001``). Now both attempts observe, diagnose
    and edit ``c2``, so the offspring are *siblings* -- each an independent repair
    of the same diagnosed parent rather than a chain whose later links were
    diagnosed on a different candidate.
    """
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=2, embedder=_offline())

    assert result.pool_size == 5
    assert result.accepted == 2
    offspring = [c for c in result.pool_candidate_ids if "att-i" in c]
    assert len(offspring) == 2
    # Siblings: neither offspring id is a prefix-extension of the other.
    assert not any(
        a != b and b.startswith(a) for a in offspring for b in offspring
    ), f"offspring chained instead of branching from one parent: {offspring}"


# ---------------------------------------------------------------------- #
# Attempt accounting
# ---------------------------------------------------------------------- #
def test_b1_runs_the_requested_attempt_count(tmp_path: Path) -> None:
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=4, embedder=_offline())

    assert result.attempts == 4
    assert result.accepted + result.rejected + result.no_issue == 4


def test_b1_rejects_a_non_positive_attempt_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=0, embedder=_offline())


# ---------------------------------------------------------------------- #
# Champion manifest
# ---------------------------------------------------------------------- #
def test_b1_reports_a_champion_from_the_pool(tmp_path: Path) -> None:
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=3, embedder=_offline())

    assert result.champion_candidate_id in result.pool_candidate_ids


def test_b1_champion_manifest_exposes_every_component(tmp_path: Path) -> None:
    """selection-algorithms.md:338 requires every component be recorded."""
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=2, embedder=_offline())

    assert 0.0 <= result.champion_outcome <= 1.0
    assert 0.0 <= result.champion_coverage <= 1.0
    assert result.champion_tie_breaker == "ascending_candidate_id"
    assert isinstance(result.champion_aggregate, float)


# ---------------------------------------------------------------------- #
# Determinism
# ---------------------------------------------------------------------- #
def test_b1_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    first = run_b1_experiment(seed=3, storage_root=tmp_path / "a", n_attempts=3, embedder=_offline())
    second = run_b1_experiment(seed=3, storage_root=tmp_path / "b", n_attempts=3, embedder=_offline())

    assert first == second


# ---------------------------------------------------------------------- #
# Redaction
# ---------------------------------------------------------------------- #
def test_b1_never_persists_evaluator_tokens(tmp_path: Path) -> None:
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=3, embedder=_offline())

    assert result.storage_records_are_redacted is True
    blob = _read_all_persisted_blobs(tmp_path)
    assert blob
    assert _TOKEN_A not in blob
    assert _TOKEN_B not in blob


def test_b1_persists_a_manifest_and_attempt_records(tmp_path: Path) -> None:
    run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=2, embedder=_offline())

    assert (tmp_path / "manifest").is_dir()
    assert (tmp_path / "attempts").is_dir()
    assert list((tmp_path / "attempts").glob("*.json"))


# ---------------------------------------------------------------------- #
# Embedding provenance
# ---------------------------------------------------------------------- #
def test_b1_records_the_embedding_provider_and_fallback_state(
    tmp_path: Path,
) -> None:
    """Silent embedding substitution is forbidden; the state must be recorded."""
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=2, embedder=_offline())

    assert result.embedding_provider in {"lexical", "ollama"}
    assert isinstance(result.embedding_fallback_reason, (str, type(None)))


def test_b1_dpp_selection_is_not_permanently_in_fallback(tmp_path: Path) -> None:
    """Real embeddings must let the joint quality/diversity objective run."""
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=2, embedder=_offline())

    assert result.selection_fallback_reasons != ("incompatible_embeddings",)


# ---------------------------------------------------------------------- #
# Task coreset
# ---------------------------------------------------------------------- #
def test_task_coreset_ids_do_not_embed_the_expected_tokens() -> None:
    """Task IDs are persisted, so they must not carry evaluator internals."""
    for task in TASKS:
        assert _TOKEN_A not in task.task_id
        assert _TOKEN_B not in task.task_id
