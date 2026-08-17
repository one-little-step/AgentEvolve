"""Tests for RHO coreset selection.

Quality is judge-assigned difficulty; diversity is cosine over fingerprint
embeddings. This is NOT the genetic issue-selection quality function (which is
cross-candidate variance) -- only the DPP primitives are shared.
"""
from __future__ import annotations

from agent_evolve.core.clustering import EmbeddingProviderUnavailable
from agent_evolve.core.rho.coreset import (
    CoresetCandidate,
    select_coreset,
)


class _FixedEmbedder:
    """Maps a marker token to an orthogonal-ish unit vector."""

    dim = 4

    def embed(self, text: str) -> tuple[float, ...]:
        if "ALPHA" in text:
            return (1.0, 0.0, 0.0, 0.0)
        if "BETA" in text:
            return (0.0, 1.0, 0.0, 0.0)
        return (0.0, 0.0, 1.0, 0.0)


class _BrokenEmbedder:
    dim = 4

    def embed(self, text: str) -> tuple[float, ...]:
        raise EmbeddingProviderUnavailable("ollama down")


def _c(task_id: str, difficulty: float, marker: str) -> CoresetCandidate:
    return CoresetCandidate(
        task_id=task_id,
        difficulty=difficulty,
        fingerprint=f"{marker} shaped failure",
        embedding_text=f"{marker} shaped failure",
    )


def test_prefers_higher_difficulty() -> None:
    candidates = (
        _c("easy", 1.0, "ALPHA"),
        _c("hard", 9.0, "ALPHA"),
    )

    report = select_coreset(candidates, 1, embedder=_FixedEmbedder())

    assert report.selected_ids == ("hard",)


def test_penalizes_a_near_duplicate_fingerprint() -> None:
    # Two ALPHA candidates are near-identical; BETA is distinct. With k=2 the
    # selector should take the best ALPHA and then BETA, not both ALPHAs.
    candidates = (
        _c("alpha-hi", 9.0, "ALPHA"),
        _c("alpha-lo", 8.5, "ALPHA"),
        _c("beta", 7.0, "BETA"),
    )

    report = select_coreset(candidates, 2, embedder=_FixedEmbedder())

    assert "alpha-hi" in report.selected_ids
    assert "beta" in report.selected_ids
    assert "alpha-lo" not in report.selected_ids


def test_selects_at_most_k() -> None:
    candidates = tuple(_c(f"t{i}", 5.0, "ALPHA") for i in range(10))

    report = select_coreset(candidates, 3, embedder=_FixedEmbedder())

    assert len(report.selected_ids) <= 3


def test_k_larger_than_corpus_returns_everything() -> None:
    candidates = (_c("a", 5.0, "ALPHA"), _c("b", 6.0, "BETA"))

    report = select_coreset(candidates, 10, embedder=_FixedEmbedder())

    assert set(report.selected_ids) == {"a", "b"}


def test_difficulty_rank_mode_is_pure_ordering() -> None:
    candidates = (
        _c("mid", 5.0, "ALPHA"),
        _c("top", 9.0, "ALPHA"),
        _c("low", 1.0, "BETA"),
    )

    report = select_coreset(
        candidates, 2, selector="difficulty_rank", embedder=_FixedEmbedder()
    )

    assert report.selected_ids == ("top", "mid")
    assert report.selection_method == "difficulty_rank"


def test_random_mode_is_seeded_and_reproducible() -> None:
    candidates = tuple(_c(f"t{i}", 5.0, "ALPHA") for i in range(8))

    first = select_coreset(
        candidates, 3, selector="random", seed=7, embedder=_FixedEmbedder()
    )
    second = select_coreset(
        candidates, 3, selector="random", seed=7, embedder=_FixedEmbedder()
    )

    assert first.selected_ids == second.selected_ids
    assert first.selection_method == "random"


def test_cold_start_with_no_difficulty_records_the_method() -> None:
    # Cold start: every difficulty is 0.0 because no history was judged.
    candidates = tuple(_c(f"t{i}", 0.0, "ALPHA") for i in range(5))

    report = select_coreset(
        candidates, 2, selector="random", seed=1, embedder=_FixedEmbedder()
    )

    assert len(report.selected_ids) == 2
    assert report.selection_method == "random"


def test_embedder_outage_degrades_to_quality_only_with_a_reason() -> None:
    candidates = (
        _c("hard", 9.0, "ALPHA"),
        _c("easy", 2.0, "BETA"),
    )

    report = select_coreset(candidates, 1, embedder=_BrokenEmbedder())

    assert report.selected_ids == ("hard",)
    assert report.fallback_reason
    assert "ollama down" in report.fallback_reason


def test_empty_corpus_yields_empty_selection() -> None:
    report = select_coreset((), 3, embedder=_FixedEmbedder())

    assert report.selected_ids == ()


def test_unknown_selector_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown selector"):
        select_coreset((_c("a", 1.0, "ALPHA"),), 1, selector="nope")


def test_summaries_discriminate_better_than_raw_traces() -> None:
    """Pins the reason trajectory comprehension exists (spec 4.2).

    Raw traces share ~60% schema vocabulary, so a lexical embedder sees them as
    near-identical and the DPP diversity term cannot discriminate. Comprehended
    summaries differ in behaviour words, so diversity works.
    """
    from agent_evolve.core.clustering import LexicalEmbedder

    raw_a = '{"event_id": "3f57289b-1e31-4a36", "kind": "graph_node_start"}'
    raw_b = '{"event_id": "7b10295b-5287-41d7", "kind": "graph_node_start"}'
    summary_a = "planned in prose but never executed a retrieval step"
    summary_b = "computed arithmetic incorrectly after a successful lookup"

    embedder = LexicalEmbedder(dim=64)
    raw_similarity = _cosine_pub(embedder.embed(raw_a), embedder.embed(raw_b))
    summary_similarity = _cosine_pub(
        embedder.embed(summary_a), embedder.embed(summary_b)
    )

    assert raw_similarity > summary_similarity


def _cosine_pub(a, b):
    from agent_evolve.core.rho.coreset import _cosine

    return _cosine(a, b)


# ------------------------------------------------------------------ #
# The `observed` gate (Task 4 contract) and the Task 13 input seam.
# ------------------------------------------------------------------ #
def test_unobserved_candidates_are_excluded_not_treated_as_easy() -> None:
    """A rejected verdict is `observed=False`; its 0.0 difficulty is not data.

    A legitimate difficulty of 0.0 and a rejected verdict are deliberately
    indistinguishable by value, so selection MUST gate on `observed`.
    """
    candidates = (
        CoresetCandidate("kept", 4.0, "ALPHA f", "ALPHA f", observed=True),
        CoresetCandidate("dropped", 9.9, "BETA f", "BETA f", observed=False),
    )

    report = select_coreset(candidates, 2, embedder=_FixedEmbedder())

    assert report.selected_ids == ("kept",)
    assert report.excluded_ids == ("dropped",)


def test_all_unobserved_yields_empty_selection() -> None:
    candidates = (
        CoresetCandidate("a", 9.0, "ALPHA f", "ALPHA f", observed=False),
        CoresetCandidate("b", 8.0, "BETA f", "BETA f", observed=False),
    )

    report = select_coreset(candidates, 2, embedder=_FixedEmbedder())

    assert report.selected_ids == ()
    assert set(report.excluded_ids) == {"a", "b"}


def test_observed_gate_applies_to_ablation_selectors_too() -> None:
    candidates = (
        CoresetCandidate("bad", 10.0, "ALPHA f", "ALPHA f", observed=False),
        CoresetCandidate("good", 1.0, "BETA f", "BETA f", observed=True),
    )

    for selector in ("difficulty_rank", "random"):
        report = select_coreset(
            candidates, 2, selector=selector, seed=3, embedder=_FixedEmbedder()
        )
        assert report.selected_ids == ("good",), selector


def test_candidates_from_verdicts_duck_types_the_judge_shape() -> None:
    """Task 13 seam: `core/` never imports the adapter's verdict type."""
    from dataclasses import dataclass

    from agent_evolve.core.rho.coreset import candidates_from_verdicts

    @dataclass(frozen=True)
    class _Verdict:  # structurally the adapter's DifficultyVerdict
        task_id: str
        difficulty: float
        abstract_fingerprint: str
        observed: bool

    verdicts = (
        _Verdict("t1", 7.5, "task shape: multi-hop retrieval", True),
        _Verdict("t2", 0.0, "", False),
    )

    built = candidates_from_verdicts(
        verdicts, summaries={"t1": "wandered without retrieving"}
    )

    assert tuple(c.task_id for c in built) == ("t1", "t2")
    assert built[0].observed is True
    assert built[0].difficulty == 7.5
    assert built[0].fingerprint == "task shape: multi-hop retrieval"
    # Comprehended summary is appended to the fingerprint for embedding, since
    # spec 4.2 says the summary -- never the raw trace -- is the embedded text.
    assert "wandered without retrieving" in built[0].embedding_text
    assert "task shape: multi-hop retrieval" in built[0].embedding_text
    assert built[1].observed is False


def test_selection_is_deterministic_across_repeated_calls() -> None:
    candidates = tuple(
        _c(f"t{i}", 9.0 - (i * 0.1), "ALPHA" if i % 2 else "BETA")
        for i in range(12)
    )

    runs = {
        select_coreset(candidates, 5, embedder=_FixedEmbedder()).selected_ids
        for _ in range(5)
    }

    assert len(runs) == 1


def test_dpp_is_not_degenerate_top_k_by_difficulty() -> None:
    """Guards the diversity half of the design.

    Top-3-by-difficulty is all-ALPHA. A real DPP must reach for the lower
    difficulty BETA instead of a third redundant ALPHA.
    """
    candidates = (
        _c("a1", 9.0, "ALPHA"),
        _c("a2", 8.9, "ALPHA"),
        _c("a3", 8.8, "ALPHA"),
        _c("b1", 5.0, "BETA"),
    )

    dpp = select_coreset(candidates, 2, embedder=_FixedEmbedder())
    rank = select_coreset(
        candidates, 2, selector="difficulty_rank", embedder=_FixedEmbedder()
    )

    assert dpp.selected_ids == ("a1", "b1")
    assert rank.selected_ids == ("a1", "a2")
    assert dpp.selected_ids != rank.selected_ids


def test_no_embedder_records_a_degradation_reason() -> None:
    report = select_coreset((_c("a", 5.0, "ALPHA"), _c("b", 6.0, "BETA")), 1)

    assert report.selection_method == "dpp_quality_only"
    assert report.fallback_reason
