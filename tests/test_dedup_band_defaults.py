"""SV-12 step 4: the adjudication band, and the seam that made it inert.

Step 2 built the adjudicator and step 3 wired a registry into the genetic path.
This module pins the two defects measured on 2026-08-21 that together meant no
production run has ever adjudicated a mechanism merge, plus the band widening
the live calibration justifies.

Measured against unmodified source before these tests were written:

1. **The adjudicator never attaches in production.**
   ``pipeline.cluster_registry_for_config`` passed ``base_url=dedup.base_url``,
   but :class:`MechanismDedupConfig`'s field is ``url``. Its fields are
   ``url/model/api_key/enabled/band_low/band_high`` and
   ``hasattr(MechanismDedupConfig, "base_url")`` is ``False``. The broad
   ``except Exception`` then caught the ``AttributeError`` and degraded to
   cosine-only clustering. Measured against a fully-configured dedup config:
   ``config.mechanism_dedup.enabled`` is ``True`` while
   ``registry.adjudicator`` is ``None``, and stderr carried
   ``AttributeError: 'MechanismDedupConfig' object has no attribute 'base_url'``.
   The band is only consulted when an adjudicator exists, so widening it while
   this held would have changed nothing observable.

2. **Four independent hardcoded band pairs.** ``0.60``/``0.85`` was written out
   at ``config.py`` module scope, on ``MechanismClusterer``, on
   ``ClusterRegistry``, and at ``pipeline.py`` module scope. Four copies of one
   policy number drift apart silently, and the drift is invisible because a
   wrong band produces a *plausible* clustering rather than an error.

3. **The dead window.** ``band_high`` below ``join_threshold`` leaves
   ``[band_high, join_threshold)`` neither ambiguous nor joining, so cosine
   decides it alone with no model call -- exactly the case the adjudicator
   exists to catch. Live embeddinggemma over 4 fault families, 66 pairs:
   band ``[0.45, 0.70)`` strands 3 true paraphrase pairs at cosine ``0.718``,
   ``0.749`` and ``0.726``. The band cannot be validated independently of the
   join threshold; only the pair is meaningful.

Band choice is measured, not assumed (``terminal_output/sv12/17-band-decision.log``,
live embeddinggemma, 12 same-fault and 54 different-fault pairs):

    band            adjudicated   silent-split   false-merge-risk
    [0.60,0.85)         9              2                0
    [0.45,0.75)        16              0                0
    [0.40,0.75)        35              0                0

``[0.45, 0.75)`` is the smallest measured band that silently splits **zero**
true paraphrase pairs. ``[0.40, 0.75)`` buys nothing and doubles the calls.

Caveat carried deliberately: the 12 calibration strings are synthetic
phrasings, not real CUGA analyzer output, so these defaults are evidence-based
but not a tuned optimum.
"""
from __future__ import annotations

import dataclasses
import io
import sys
from contextlib import redirect_stderr
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve import pipeline  # noqa: E402
from agent_evolve.core import config as config_mod  # noqa: E402
from agent_evolve.core.blame import (  # noqa: E402
    BlameGraph,
    BlameNode,
    CausalAnalysis,
)
from agent_evolve.core.clustering import (  # noqa: E402
    ClusterRegistry,
    LexicalEmbedder,
    MechanismClusterer,
    _cosine,
)
from agent_evolve.core.config import MechanismDedupConfig, resolve_profile  # noqa: E402


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
    """Counts calls so "was the model consulted" is a measurement, not a guess.

    Parameter names match :class:`MechanismAdjudicator` exactly; a protocol is
    satisfied by names as well as types.
    """

    def __init__(self, verdict: bool | None = True) -> None:
        self.verdict = verdict
        self.pairs: list[tuple[str, str]] = []

    def same_mechanism(self, left: str, right: str) -> bool | None:
        self.pairs.append((left, right))
        return self.verdict

    @property
    def calls(self) -> int:
        return len(self.pairs)


# --------------------------------------------------------------------------- #
# Defect 1: the config -> adapter mapping that made the band inert
# --------------------------------------------------------------------------- #
def test_dedup_config_has_no_base_url_attribute() -> None:
    """Pins the real field name, so a future rename cannot silently re-break it.

    This is the fact ``pipeline`` got wrong. Asserted on the dataclass rather
    than on an instance because ``slots=True`` makes the distinction observable.
    """
    names = {f.name for f in dataclasses.fields(MechanismDedupConfig)}
    assert "url" in names
    assert "base_url" not in names
    assert not hasattr(MechanismDedupConfig, "base_url")


def test_configured_dedup_actually_attaches_an_adjudicator() -> None:
    """The defect that made every earlier band number decoration.

    RED against unmodified source: ``registry.adjudicator`` was ``None`` even
    with dedup fully configured, because ``dedup.base_url`` raised
    ``AttributeError`` into the fallback path.
    """
    cfg = resolve_profile(
        "research_sequential",
        {
            "AE_MECHANISM_DEDUP_MODEL": "openai/some-model",
            "AE_MECHANISM_DEDUP_BASE_URL": "http://localhost:9999/v1",
            "AE_MECHANISM_DEDUP_API_KEY": "SECRET",
        },
        seed=0,
    )
    assert cfg.mechanism_dedup.enabled is True

    err = io.StringIO()
    with redirect_stderr(err):
        registry = pipeline.cluster_registry_for_config(
            cfg, embedder=LexicalEmbedder(dim=768)
        )

    assert registry.adjudicator is not None, (
        "dedup is enabled in config but no adjudicator reached the registry; "
        f"stderr said: {err.getvalue().strip()!r}"
    )
    assert "AttributeError" not in err.getvalue()


def test_adjudicator_construction_failure_still_degrades_to_cosine() -> None:
    """The fallback must survive the fix: an outage is not a run-ending error.

    Control test. Guards against "fixing" defect 1 by removing the safety net.
    """
    cfg = resolve_profile(
        "research_sequential",
        {
            "AE_MECHANISM_DEDUP_MODEL": "openai/some-model",
            "AE_MECHANISM_DEDUP_BASE_URL": "http://localhost:9999/v1",
        },
        seed=0,
    )

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("provider down")

    import agent_evolve.adapters.cuga_mechanism_adjudicator as adj_mod

    original = adj_mod.CugaMechanismAdjudicator
    adj_mod.CugaMechanismAdjudicator = _boom  # type: ignore[assignment]
    try:
        err = io.StringIO()
        with redirect_stderr(err):
            registry = pipeline.cluster_registry_for_config(
                cfg, embedder=LexicalEmbedder(dim=768)
            )
        assert registry.adjudicator is None
        assert "mechanism-dedup" in err.getvalue()
    finally:
        adj_mod.CugaMechanismAdjudicator = original  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Defect 2: one band policy, four copies
# --------------------------------------------------------------------------- #
def test_all_band_defaults_agree() -> None:
    """Four hardcoded pairs cannot be allowed to disagree.

    A drifted band does not raise; it produces a plausible-looking clustering
    with a different adjudication policy than the one documented.
    """
    clusterer = MechanismClusterer(task_id="t", embedder=LexicalEmbedder(dim=32))
    registry = ClusterRegistry(embedder_factory=lambda: LexicalEmbedder(dim=32))
    dedup = MechanismDedupConfig()

    lows = {
        "config module": config_mod._DEFAULT_DEDUP_BAND_LOW,
        "MechanismDedupConfig": dedup.band_low,
        "MechanismClusterer": clusterer.band_low,
        "ClusterRegistry": registry.band_low,
        "pipeline module": pipeline._DEFAULT_CLUSTER_BAND_LOW,
    }
    highs = {
        "config module": config_mod._DEFAULT_DEDUP_BAND_HIGH,
        "MechanismDedupConfig": dedup.band_high,
        "MechanismClusterer": clusterer.band_high,
        "ClusterRegistry": registry.band_high,
        "pipeline module": pipeline._DEFAULT_CLUSTER_BAND_HIGH,
    }
    assert len(set(lows.values())) == 1, f"band_low defaults disagree: {lows}"
    assert len(set(highs.values())) == 1, f"band_high defaults disagree: {highs}"


def test_band_default_is_the_measured_value() -> None:
    """Pins the calibrated band so a later edit must re-argue it.

    RED against unmodified source, which had ``0.60``/``0.85`` -- measured to
    silently split 2 of 12 true paraphrase pairs.
    """
    dedup = MechanismDedupConfig()
    assert dedup.band_low == pytest.approx(0.45)
    assert dedup.band_high == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Defect 3: the dead window between band_high and join_threshold
# --------------------------------------------------------------------------- #
def test_band_high_below_join_threshold_is_rejected() -> None:
    """The invariant that makes the band meaningful at all.

    With ``band_high < join_threshold`` the span ``[band_high, join_threshold)``
    is neither ambiguous nor joining, so cosine alone splits pairs there.
    Measured live: band ``[0.45,0.70)`` with threshold ``0.75`` stranded true
    pairs at ``0.718``, ``0.749`` and ``0.726``.
    """
    with pytest.raises(ValueError, match="join_threshold"):
        MechanismClusterer(
            task_id="t",
            embedder=LexicalEmbedder(dim=32),
            join_threshold=0.75,
            band_low=0.45,
            band_high=0.70,
            adjudicator=_RecordingAdjudicator(),
        )


def test_dead_window_is_allowed_without_an_adjudicator() -> None:
    """The invariant is scoped to the case where it can actually bite.

    The band is read only at the adjudicator gate, so with no adjudicator there
    is no dead window to create and raising would reject legitimate cosine-only
    configurations. Guards against over-applying the check.
    """
    c = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=32),
        join_threshold=0.95,
        band_low=0.45,
        band_high=0.75,
    )
    assert c.adjudicator is None
    assert c.band_high < c.join_threshold


def test_band_high_equal_to_join_threshold_is_allowed() -> None:
    """Boundary control: exactly equal leaves no gap, so it must be legal."""
    c = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=32),
        join_threshold=0.75,
        band_low=0.45,
        band_high=0.75,
    )
    assert c.band_high == pytest.approx(c.join_threshold)


def test_default_band_leaves_no_dead_window() -> None:
    """The shipped defaults must satisfy the invariant they introduce."""
    c = MechanismClusterer(task_id="t", embedder=LexicalEmbedder(dim=32))
    assert c.band_high >= c.join_threshold, (
        f"dead window [{c.band_high}, {c.join_threshold}) decided by cosine alone"
    )
    r = ClusterRegistry(embedder_factory=lambda: LexicalEmbedder(dim=32))
    assert r.band_high >= r.join_threshold


# --------------------------------------------------------------------------- #
# The behavioural consequence: who actually gets adjudicated
# --------------------------------------------------------------------------- #
def test_subthreshold_pair_in_widened_band_reaches_the_adjudicator() -> None:
    """The step-4 objective, stated behaviourally.

    A pair at cosine ``0.546`` -- below the old ``band_low`` of 0.60 and below
    the join threshold -- was decided by cosine alone and split. Under the
    widened band it must reach the adjudicator, and a ``same`` verdict must
    merge it.

    The similarity is *measured* and asserted to land in the intended range, so
    the test cannot pass for the wrong reason if the embedder changes.
    """
    adj = _RecordingAdjudicator(verdict=True)
    c = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=256),
        join_threshold=0.75,
        band_low=0.45,
        band_high=0.75,
        adjudicator=adj,
    )
    first = c.assign(_analysis("retriever returned a stale schema"))
    assert first.is_new_cluster

    second = c.assign(
        _analysis("schema from the retriever was out of date entirely", "skills/b.md")
    )
    assert 0.45 <= second.similarity < 0.75, (
        "fixture no longer lands in the band; similarity="
        f"{second.similarity:.3f} -- the test would prove nothing"
    )
    assert adj.calls == 1, "the ambiguous pair never reached the adjudicator"
    assert second.cluster_id == first.cluster_id
    assert second.is_new_cluster is False


def test_distinct_pair_in_band_is_still_refused() -> None:
    """Widening must not become merge-everything.

    Same in-band cosine as the merge case, opposite verdict: the pair must stay
    split. The cosine is measured from the embedder rather than read off the
    result, because a refusal opens a new cluster and the new-cluster path
    reports ``similarity=1.0`` by construction.
    """
    adj = _RecordingAdjudicator(verdict=False)
    embedder = LexicalEmbedder(dim=256)
    c = MechanismClusterer(
        task_id="t",
        embedder=embedder,
        join_threshold=0.75,
        band_low=0.45,
        band_high=0.75,
        adjudicator=adj,
    )
    a = _analysis("retriever returned a stale schema")
    b = _analysis("schema from the retriever was out of date entirely", "skills/b.md")
    measured = _cosine(
        embedder.embed(c._embed_text(a)), embedder.embed(c._embed_text(b))
    )
    assert 0.45 <= measured < 0.75, (
        f"fixture no longer lands in the band: {measured:.3f}"
    )

    first = c.assign(a)
    second = c.assign(b)
    assert adj.calls == 1, "the ambiguous pair never reached the adjudicator"
    assert second.is_new_cluster is True
    assert second.cluster_id != first.cluster_id


def test_clear_cases_still_cost_nothing() -> None:
    """The band widened, but the extremes must remain free.

    A confidently-distinct pair below ``band_low`` must not pay a model call;
    otherwise widening has quietly become "adjudicate everything".

    The cosine is computed from the embedder directly rather than read off the
    returned assignment: on the new-cluster path ``_add`` short-circuits
    ``_best_match`` and reports ``similarity=1.0`` by construction, a number
    that says nothing about proximity to existing clusters.
    """
    adj = _RecordingAdjudicator(verdict=True)
    embedder = LexicalEmbedder(dim=256)
    c = MechanismClusterer(
        task_id="t",
        embedder=embedder,
        join_threshold=0.75,
        band_low=0.45,
        band_high=0.75,
        adjudicator=adj,
    )
    a = _analysis("retriever returned a stale schema")
    b = _analysis("planner looped forever without progress", "x/y.md")
    measured = _cosine(
        embedder.embed(c._embed_text(a)), embedder.embed(c._embed_text(b))
    )
    assert measured < 0.45, f"fixture no longer below band_low: {measured:.3f}"

    c.assign(a)
    outcome = c.assign(b)
    assert outcome.is_new_cluster is True
    assert adj.calls == 0, "a clear case paid for a model call"
