"""SV-12 step 2: the mechanism-merge adjudicator.

Embedding cosine is reliable at the extremes and unreliable in the middle. These
tests pin the three measured failure modes of the current single-pass
nearest-centroid clusterer and the contract that fixes them.

Measured against unmodified source before these tests were written:

1. **Centroid drift splits one fault.** The running-mean centroid moves as
   members join, so later phrasings of the same fault fall below threshold:
   ``"date filter missing"`` -> c0, ``"date filter absent"`` -> c0,
   ``"filter for dates omitted"`` -> c1, ``"dates unfiltered entirely"`` -> c2.
2. **Order dependence.** The same four mechanisms assign differently depending on
   arrival order.
3. **``at_cap`` forces false merges.** ``if best_sim >= join_threshold or at_cap``
   short-circuits the similarity check entirely at the cluster cap, so an
   unrelated mechanism joined at cosine ``0.822``. Two unrelated faults in one
   cell yield a *high* variance reading -- "a fix is reachable here" -- for a
   mechanism that does not exist. Measured: forced merge gives entropy 0.07200
   and ``classify == "recombination_target"``; refusing gives ``None`` / ``skip``.

The contract under test: an injected adjudicator is consulted **only** inside the
ambiguous band and on an ``at_cap`` decision, never for the clear cases, and its
unavailability never silently changes a clustering decision.

``core/clustering.py`` is agent-neutral, so the adjudicator enters through a
protocol like ``MechanismEmbedder`` does -- never a model import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis  # noqa: E402
from agent_evolve.core.clustering import (  # noqa: E402
    LexicalEmbedder,
    MechanismClusterer,
)


def _analysis(mechanism: str, artifact: str = "skills/a.md") -> CausalAnalysis:
    return CausalAnalysis(
        mechanism=mechanism,
        severity=0.6,
        score=0.2,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="agent", artifacts=(artifact,), blame=0.9),)
        ),
    )


class _RecordingAdjudicator:
    """Records every pair it is asked about and answers from a fixed script."""

    def __init__(self, verdict: bool | None = True) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    def same_mechanism(self, left: str, right: str) -> bool | None:
        self.calls.append((left, right))
        return self.verdict


class _RaisingAdjudicator:
    """An outage. Must never be allowed to change a clustering decision."""

    def __init__(self) -> None:
        self.calls = 0

    def same_mechanism(self, left: str, right: str) -> bool | None:
        self.calls += 1
        raise RuntimeError("dedup endpoint unreachable")


# ---------------------------------------------------------------------- #
# 1. The band: clear cases must not cost a model call
# ---------------------------------------------------------------------- #
def test_clearly_identical_mechanisms_do_not_consult_the_adjudicator():
    """Above band_high, cosine is trusted. Paying a model call here is waste."""
    adj = _RecordingAdjudicator()
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        adjudicator=adj,
        band_low=0.60,
        band_high=0.85,
    )
    cl.begin_iteration(1)
    text = "the agent did not verify the units before reporting"
    cl.assign(_analysis(text))
    cl.assign(_analysis(text))  # identical -> cosine 1.0
    assert adj.calls == [], (
        f"adjudicator consulted for an unambiguous pair: {adj.calls}"
    )


def test_clearly_distinct_mechanisms_do_not_consult_the_adjudicator():
    """Below band_low, cosine is trusted."""
    adj = _RecordingAdjudicator()
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        adjudicator=adj,
        band_low=0.60,
        band_high=0.85,
    )
    cl.begin_iteration(1)
    cl.assign(_analysis("alpha beta gamma delta epsilon"))
    cl.assign(_analysis("zeta eta theta iota kappa lambda mu nu"))
    assert adj.calls == [], (
        f"adjudicator consulted for an unambiguous pair: {adj.calls}"
    )


def test_ambiguous_pair_consults_the_adjudicator_and_its_verdict_wins():
    """Inside the band, the model decides -- not the raw cosine.

    The band is set wide here so the pair lands inside it deterministically,
    rather than depending on a specific embedder's similarity value.
    """
    adj = _RecordingAdjudicator(verdict=True)
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        adjudicator=adj,
        band_low=0.0,
        band_high=1.0,
    )
    cl.begin_iteration(1)
    a = cl.assign(_analysis("retrieval step omitted the date filter"))
    b = cl.assign(_analysis("the query lacked a temporal constraint"))
    assert adj.calls, "an ambiguous pair must reach the adjudicator"
    assert a.cluster_id == b.cluster_id, (
        "the adjudicator said same-mechanism, so the two must share a cluster"
    )


def test_adjudicator_can_split_a_pair_cosine_would_have_joined():
    """A ``False`` verdict inside the band must force a new cluster."""
    adj = _RecordingAdjudicator(verdict=False)
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        adjudicator=adj,
        band_low=0.0,
        band_high=1.0,
    )
    cl.begin_iteration(1)
    a = cl.assign(_analysis("date filter missing from retrieval"))
    b = cl.assign(_analysis("date filter missing from retrieval step"))
    assert adj.calls, "expected the adjudicator to be consulted"
    assert a.cluster_id != b.cluster_id, (
        "the adjudicator said different mechanisms, so they must not share a cell"
    )


# ---------------------------------------------------------------------- #
# 2. Drift: the same fault must not fragment
# ---------------------------------------------------------------------- #
def test_adjudicator_prevents_centroid_drift_from_splitting_one_fault():
    """The measured drift case: four phrasings of one fault -> one cluster.

    Without an adjudicator this produced three clusters (c0, c0, c1, c2).
    """
    adj = _RecordingAdjudicator(verdict=True)
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        adjudicator=adj,
        band_low=0.0,
        band_high=1.0,
    )
    cl.begin_iteration(1)
    ids = [
        cl.assign(_analysis(m)).cluster_id
        for m in (
            "date filter missing",
            "date filter absent",
            "filter for dates omitted",
            "dates unfiltered entirely",
        )
    ]
    assert len(set(ids)) == 1, (
        f"one fault fragmented into {len(set(ids))} clusters: {ids}. Centroid "
        "drift moved the centroid below threshold for later phrasings."
    )


# ---------------------------------------------------------------------- #
# 3. at_cap must not force a below-threshold merge
# ---------------------------------------------------------------------- #
def test_at_cap_does_not_force_a_below_threshold_merge():
    """``or at_cap`` currently discards the similarity check entirely.

    Measured: with cap=2, an unrelated mechanism joined at cosine 0.822 with
    ``is_new_cluster=False``. Mixing two unrelated faults into one cell produces
    a high variance reading for a fix that does not exist -- worse than having no
    reading, because nothing looks broken.
    """
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        max_clusters_per_task=2,
        join_threshold=0.95,
    )
    cl.begin_iteration(1)
    cl.assign(_analysis("alpha alpha alpha"))
    cl.assign(_analysis("beta beta beta"))
    forced = cl.assign(_analysis("zeta completely unrelated wording here"))

    assert forced.similarity < 0.95, "test setup: expected a below-threshold pair"
    assert forced.cluster_id == "", (
        f"a below-threshold mechanism was absorbed into cluster "
        f"{forced.cluster_id!r} at similarity {forced.similarity:.3f} because the "
        "cluster cap was full. It must be reported as unassigned instead, so "
        "entropy treats the cell as unavailable rather than computing variance "
        "over two unrelated faults."
    )


def test_at_cap_refusal_is_reported_with_a_reason():
    """A refusal must be legible, not a silent empty id."""
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        max_clusters_per_task=1,
        join_threshold=0.95,
    )
    cl.begin_iteration(1)
    cl.assign(_analysis("alpha alpha alpha"))
    forced = cl.assign(_analysis("totally different words entirely here now"))
    assert forced.unassigned_reason, (
        "a cap refusal must name its cause so a coarse fallback is never "
        "mistaken for a real mechanism assignment"
    )
    assert "cap" in forced.unassigned_reason.lower()


def test_at_cap_still_joins_a_genuinely_similar_mechanism():
    """The cap must not block a *legitimate* join. Only forced ones."""
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        max_clusters_per_task=1,
        join_threshold=0.50,
    )
    cl.begin_iteration(1)
    first = cl.assign(_analysis("units were not verified before reporting"))
    same = cl.assign(_analysis("units were not verified before reporting"))
    assert same.cluster_id == first.cluster_id
    assert same.unassigned_reason is None


# ---------------------------------------------------------------------- #
# 4. Outages must never silently change a decision
# ---------------------------------------------------------------------- #
def test_adjudicator_outage_falls_back_to_cosine_and_records_it():
    """A raising adjudicator must not crash the run, and must not pretend.

    Conservative on missing evidence, as the retirement rule already is: fall
    back to the cosine decision and record that the adjudication was
    unavailable.
    """
    adj = _RaisingAdjudicator()
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        adjudicator=adj,
        band_low=0.0,
        band_high=1.0,
    )
    cl.begin_iteration(1)
    cl.assign(_analysis("date filter missing"))
    out = cl.assign(_analysis("date filter absent"))
    assert adj.calls >= 1, "expected the adjudicator to be attempted"
    assert out.cluster_id, "an outage must not leave the finding unassigned"
    assert out.adjudication_unavailable_reason, (
        "an unavailable adjudication must be recorded, not silently ignored"
    )


def test_abstaining_adjudicator_falls_back_to_cosine():
    """``None`` means "no opinion" and must be treated as an abstention."""
    adj = _RecordingAdjudicator(verdict=None)
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        adjudicator=adj,
        band_low=0.0,
        band_high=1.0,
    )
    cl.begin_iteration(1)
    cl.assign(_analysis("date filter missing"))
    out = cl.assign(_analysis("date filter absent"))
    assert adj.calls, "expected consultation"
    assert out.cluster_id, "an abstention must not leave the finding unassigned"


def test_no_adjudicator_keeps_todays_cosine_only_behaviour():
    """The default must be unchanged: no adjudicator, no model dependency."""
    cl = MechanismClusterer(task_id="t", embedder=LexicalEmbedder(dim=32))
    cl.begin_iteration(1)
    ids = [
        cl.assign(_analysis(m)).cluster_id
        for m in (
            "retrieval step omitted the date filter so stale rows returned",
            "the date filter was omitted in the retrieval step, returning stale rows",
            "the agent never called the pricing API and invented a number",
        )
    ]
    # Vocabulary-sharing paraphrases still join; the unrelated one still splits.
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


# ---------------------------------------------------------------------- #
# 5. Band validation
# ---------------------------------------------------------------------- #
def test_inverted_band_is_rejected():
    with pytest.raises(ValueError, match="band_low"):
        MechanismClusterer(
            task_id="t",
            embedder=LexicalEmbedder(dim=32),
            band_low=0.9,
            band_high=0.5,
        )


def test_band_bounds_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        MechanismClusterer(
            task_id="t", embedder=LexicalEmbedder(dim=32), band_low=-0.1
        )
    with pytest.raises(ValueError):
        MechanismClusterer(
            task_id="t", embedder=LexicalEmbedder(dim=32), band_high=1.5
        )
