"""Behavioral tests for trace-backed issues and hierarchical DPP selection.

Governing contract: docs/architecture/selection-algorithms.md:67-280.
"""
from __future__ import annotations

import numpy as np

from agent_evolve.core.blame import BlameGraph, BlameNode, CausalFinding
from agent_evolve.core.contracts import ArtifactDescriptor
from agent_evolve.core.issues import (
    HierarchicalDPPSelector,
    Issue,
    build_issue,
    greedy_map,
)


def _desc(artifact_id: str, *, writable: bool = True) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=artifact_id,
        kind="memory",
        format="text",
        version_hash="h",
        readable=True,
        writable=writable,
        merge_strategy="overwrite",
    )


def _issue(
    issue_id: str,
    *,
    severity: float = 0.5,
    confidence: float = 0.5,
    entropy: float = 0.0,
    embedding: tuple[float, ...] = (1.0, 0.0),
    task_id: str = "t0",
    mechanism: str = "m0",
    writable: tuple[str, ...] = ("w",),
) -> Issue:
    return Issue(
        issue_id=issue_id,
        task_id=task_id,
        mechanism_cluster_id=mechanism,
        severity=severity,
        confidence=confidence,
        entropy=entropy,
        coverage_need=0.0,
        pareto_relevance=0.0,
        raw_quality=severity,
        embedding=embedding,
        writable_artifact_ids=writable,
        evidence_refs=writable,
        lineage="",
    )


def _ids(selected: object) -> set[str]:
    return {issue.issue_id for issue in selected.items}  # type: ignore[attr-defined]


def test_dpp_penalizes_similarity_and_promotes_diversity() -> None:
    """Two near-duplicate issues lose to one dissimilar issue, k=2."""
    near_duplicate_a = _issue("a", embedding=(1.0, 0.0), writable=("wa",))
    near_duplicate_b = _issue("b", embedding=(0.99, 0.141), writable=("wb",))
    dissimilar = _issue("c", embedding=(0.0, 1.0), writable=("wc",))

    selector = HierarchicalDPPSelector()
    selected = selector.select((near_duplicate_a, near_duplicate_b, dissimilar), k=2)

    assert dissimilar.issue_id in _ids(selected)
    assert not {near_duplicate_a.issue_id, near_duplicate_b.issue_id} <= _ids(selected)


def test_dpp_prefers_quality_among_equally_diverse_items() -> None:
    """Three mutually dissimilar issues with distinct quality, k=1: highest wins."""
    low = _issue("low", severity=0.2, embedding=(1.0, 0.0, 0.0), writable=("wl",))
    mid = _issue("mid", severity=0.5, embedding=(0.0, 1.0, 0.0), writable=("wm",))
    high = _issue("high", severity=0.9, embedding=(0.0, 0.0, 1.0), writable=("wh",))

    selector = HierarchicalDPPSelector()
    selected = selector.select((low, mid, high), k=1)

    assert selected.items == (high,)


def test_dpp_theta_shifts_quality_diversity_balance() -> None:
    """Low theta favors distinct families; high theta favors high-quality dups."""
    dup_a = _issue("a", severity=0.9, embedding=(1.0, 0.0), writable=("wa",))
    dup_b = _issue("b", severity=0.9, embedding=(0.99, 0.141), writable=("wb",))
    distinct = _issue("c", severity=0.3, embedding=(0.0, 1.0), writable=("wc",))
    issues = (dup_a, dup_b, distinct)

    low_theta = HierarchicalDPPSelector(theta=0.0).select(issues, k=2)
    high_theta = HierarchicalDPPSelector(theta=0.9).select(issues, k=2)

    low_ids = _ids(low_theta)
    high_ids = _ids(high_theta)

    # High theta must select both near-duplicates; low theta must spread.
    assert {"a", "b"} <= high_ids
    assert "c" in low_ids
    # Direction: the near-duplicate family count increases with theta.
    assert len({"a", "b"} & high_ids) > len({"a", "b"} & low_ids)


def test_dpp_is_deterministic() -> None:
    """Identical input and configuration produce identical selections."""
    issues = (
        _issue("a", severity=0.9, embedding=(1.0, 0.0), writable=("wa",)),
        _issue("b", severity=0.8, embedding=(0.99, 0.141), writable=("wb",)),
        _issue("c", severity=0.5, embedding=(0.0, 1.0), writable=("wc",)),
        _issue("d", severity=0.4, embedding=(0.0, 0.0, 1.0), writable=("wd",)),
    )
    selector = HierarchicalDPPSelector()

    assert selector.select(issues, k=2) == selector.select(issues, k=2)


def test_dpp_full_id_tie_break_is_ascending() -> None:
    """Equal marginal gains break ties by ascending full string ID, not index."""
    kernel = np.eye(3)
    # ids are out of index order to prove the string sort, not the index sort.
    selected = greedy_map(kernel, ("c", "a", "b"), k=2, min_gain=1e-12)

    assert selected == (1, 2)


def test_flat_dpp_tie_break_uses_issue_id() -> None:
    """The flat-DPP selector ties on ``issue_id``, not the dataclass repr."""
    a = _issue("a", severity=0.5, embedding=(1.0, 0.0), writable=("wa",))
    a_bang = _issue("a!", severity=0.5, embedding=(1.0, 0.0), writable=("wb",))
    c = _issue("c", severity=0.5, embedding=(1.0, 0.0), writable=("wc",))

    selector = HierarchicalDPPSelector()
    selected = selector.select((a, a_bang, c), k=2)

    assert [i.issue_id for i in selected.items] == ["a", "a!"]


def test_greedy_map_caps_k_to_ids_length() -> None:
    """A request for more items than exist is capped, not an IndexError."""
    kernel = np.eye(2)
    selected = greedy_map(kernel, ("a", "b"), k=10, min_gain=1e-12)

    assert selected == (0, 1)


def test_empty_writeset_does_not_misalign_hard_constraints() -> None:
    """An empty-write-set issue must not shift the raw-quality index alignment."""
    empty = _issue("e", severity=0.0, writable=())
    high = _issue("a", severity=0.9, writable=("w",))
    low = _issue("b", severity=0.1, writable=("w",))

    selector = HierarchicalDPPSelector(mode="severity_rank")
    report = selector.select((empty, high, low), k=2)

    assert _ids(report) == {"a"}


def test_no_silent_fallback_on_incompatible_embeddings() -> None:
    """A degenerate kernel falls back with a recorded reason, never silently."""
    a = _issue("a", embedding=(1.0, 0.0), writable=("wa",))
    b = _issue("b", embedding=(0.0, 1.0, 0.0), writable=("wb",))  # dim mismatch

    selector = HierarchicalDPPSelector()
    report = selector.select((a, b), k=1)

    assert report.fallback_reason is not None
    assert report.fallback_reason == "incompatible_embeddings"
    assert report.items == (a,)  # deterministic quality-ordered fallback


def test_issue_without_trace_backed_writable_artifact_is_rejected() -> None:
    """A finding with no trace-backed writable attribution yields None."""
    unattributed_finding = CausalFinding(
        verdict_id="v1",
        candidate_id="c1",
        task_id="t1",
        trace_id="tr1",
        status="uncertain",
        rationale="no trace evidence",
        evidence_refs=(),
        blame_graph=BlameGraph(nodes=()),
    )
    inventory = (_desc("w0"),)

    assert build_issue(unattributed_finding, inventory) is None


def test_issue_with_trace_backed_writable_artifact_is_built() -> None:
    """A finding attributing a writable, trace-backed artifact yields an Issue."""
    attributed_finding = CausalFinding(
        verdict_id="v2",
        candidate_id="c1",
        task_id="t1",
        trace_id="tr2",
        status="observed",
        mechanism_description="bad tool call",
        mechanism_cluster_id="m1",
        severity=0.8,
        confidence=0.7,
        rationale="ok",
        evidence_refs=("w0",),
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="agent", blame=1.0, artifacts=("w0",)),)
        ),
    )
    inventory = (_desc("w0", writable=True), _desc("r1", writable=False))

    issue = build_issue(attributed_finding, inventory)

    assert issue is not None
    assert issue.writable_artifact_ids == ("w0",)
    assert issue.evidence_refs == ("w0",)
    assert issue.task_id == "t1"
    assert issue.mechanism_cluster_id == "m1"


def test_hierarchical_dpp_selects_within_tasks() -> None:
    """Hierarchical select picks tasks first, then mechanisms within tasks."""
    issues = (
        _issue("t0_m0", task_id="t0", mechanism="m0", severity=0.9, writable=("w0",)),
        _issue("t0_m1", task_id="t0", mechanism="m1", severity=0.8, writable=("w1",)),
        _issue("t1_m0", task_id="t1", mechanism="m0", severity=0.2, writable=("w2",)),
        _issue("t1_m1", task_id="t1", mechanism="m1", severity=0.1, writable=("w3",)),
    )

    selector = HierarchicalDPPSelector()
    report = selector.select(issues, k_tasks=1, k_mechanisms_per_task=2)

    assert {i.task_id for i in report.items} == {"t0"}
    assert len(report.items) == 2
