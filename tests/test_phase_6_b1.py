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


def test_b1_frontier_keeps_more_than_one_non_dominated_candidate(
    tmp_path: Path,
) -> None:
    """c1 and c2 each win a distinct task, so no single candidate dominates."""
    result = run_b1_experiment(seed=0, storage_root=tmp_path, n_attempts=2, embedder=_offline())

    assert result.frontier_size >= 2


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
