"""SV-12 step 3: mechanism-keyed entropy evidence on the live genetic path.

What this file is about
-----------------------
Today the genetic path computes its DPP entropy term inline in
``SequentialGepaRunner._cell_entropy``, filtering the pool score tensor on
``m_id == self.mechanism_cluster_id`` -- a **constant** (``"mechanism-default"``
in production, ``_CLUSTER`` in the test harness). So the "per-mechanism
variance" that feeds issue selection is actually the spread of one score per
*candidate* inside a single synthetic bucket: candidates that failed for
completely unrelated reasons are pooled together, and run-to-run LLM answer
variation is indistinguishable from genuine mechanism diversity.

Meanwhile ``EntropyTracker`` -- which implements the spec's
``H(t, m) = Var * max(max_score, score_floor)`` *with* the evidence floors
(>=3 comparable candidates, >=2 rollouts each) -- is written only by the RHO
path and read by nobody. Two entropy implementations, one wired per path,
neither doing what the spec defines.

These tests pin the four properties step 3 must establish, per
``docs/COMPACTION-ANCHOR-SV12.md`` section 14:

1. a **producer**: the genetic path records its rollout scores into a tracker;
2. **mechanism-keyed cells**: the key comes from the clusterer, not the constant;
3. a **consumer**: the entropy the DPP sees comes from the tracker;
4. **coarse-vs-fine reporting**: a cell below the floors is reported
   unavailable, never silently substituted.

Two hard constraints are guarded here as well:

* the **pool**'s mechanism id stays constant. Keying pool cells by mechanism
  empties the champion comparison's shared-cell intersection and SV-2 fails
  *silently* (``dominates()`` correctly returns False on no overlap, and a
  frontier containing everything looks like healthy diversity). The dedicated
  guard lives in ``test_embedder_wiring.py``; the test here states the
  invariant from the tracker side, so a future change that unifies the two
  keyspaces trips at least one of them.
* a finding can now come back **unassigned** (``cluster_id=""``) when the
  cluster cap is full and the nearest cluster is below threshold. That refusal
  must never become a cell key.

``entropy.py`` is under a 0-diff constraint and needs no change: its write API
(``record_score`` / ``mark_comparable``) and read API (``entropy`` returning
``None`` below the floor, ``classify``) are already sufficient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_evolve.core.blame import (  # noqa: E402
    BlameGraph,
    BlameNode,
    CausalAnalysis,
    CausalFinding,
)
from agent_evolve.core.clustering import (  # noqa: E402
    ClusterRegistry,
    LexicalEmbedder,
    MechanismClusterer,
)
from agent_evolve.core.entropy import EntropyTracker  # noqa: E402


def _analysis(mechanism: str, artifact: str = "skills/a.md") -> CausalAnalysis:
    """A ``CausalAnalysis``, which is what ``MechanismClusterer.assign`` takes."""
    return CausalAnalysis(
        mechanism=mechanism,
        severity=0.6,
        score=0.2,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="agent", artifacts=(artifact,), blame=0.9),)
        ),
    )


def _finding(
    mechanism: str, *, task_id: str = "task-a", artifact: str = "skills/a.md"
) -> CausalFinding:
    """A ``CausalFinding``, which is what ``ClusterRegistry.assign`` takes.

    The two entry points have different parameter types: the per-task clusterer
    accepts either, but the registry is annotated ``CausalFinding`` and reads
    ``mechanism_description``. Passing an analysis there raises ``AttributeError``
    -- which is a harness fault, not evidence of the defect under test, so the
    two shapes are kept separate deliberately.
    """
    return CausalFinding(
        verdict_id=f"v-{mechanism[:12]}",
        candidate_id="c1",
        task_id=task_id,
        trace_id=f"tr-{task_id}",
        status="observed",
        mechanism_description=mechanism,
        # ``status="observed"`` requires this field, so it is set to satisfy the
        # model contract. It is NOT the value under test: the clusterer computes
        # a fresh assignment from the mechanism *text*, and it is that computed
        # id these tests inspect.
        mechanism_cluster_id="pre-existing",
        severity=0.6,
        confidence=0.8,
        rationale=mechanism,
        evidence_refs=(artifact,),
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="agent", artifacts=(artifact,), blame=0.9),)
        ),
    )


# ---------------------------------------------------------------------- #
# 1. The registry must not launder a refusal into a real-looking id
# ---------------------------------------------------------------------- #
def test_registry_does_not_namespace_a_refusal_into_a_truthy_id():
    """A refusal must stay falsy through the registry's task namespacing.

    ``ClusterRegistry.assign`` returns ``f"{task_id}:{inner_cluster_id}"``. When
    the per-task clusterer refuses, the inner id is ``""`` and that formatting
    produces ``"task-a:"`` -- which is **non-empty and therefore truthy**, so
    ``CellKey.__post_init__``'s ``if not mechanism_cluster_id: raise`` does not
    fire and the refusal is silently promoted to a legitimate-looking mechanism.
    A caller doing the obvious ``if assignment.cluster_id:`` check would be
    defeated by the namespacing alone.
    """
    registry = ClusterRegistry(
        embedder_factory=lambda: LexicalEmbedder(dim=32),
        join_threshold=0.95,
    )
    clusterer = registry.clusterer_for("task-a")
    clusterer.max_clusters_per_task = 1

    first = registry.assign("task-a", _finding("timeout waiting on the http api"))
    assert first.cluster_id, "the first observation should open a cluster"

    refused = registry.assign(
        "task-a", _finding("json schema mismatch in the parsed payload")
    )
    assert not refused.cluster_id, (
        "a cap-refused assignment must remain falsy after task namespacing; "
        f"got {refused.cluster_id!r}, which passes a truthiness check and would "
        "be accepted as a real mechanism id"
    )


def test_registry_propagates_the_refusal_reason():
    """The reason must survive the registry hop, or the refusal is unexplainable.

    ``MechanismClusterer`` sets ``unassigned_reason`` naming the cap and both
    numbers, but ``ClusterRegistry.assign`` reconstructs a fresh
    ``ClusterAssignment`` and copies only five of the eight fields, dropping
    ``unassigned_reason`` and ``adjudication_unavailable_reason``. A caller that
    must report *why* entropy is unavailable has nothing to report.
    """
    registry = ClusterRegistry(
        embedder_factory=lambda: LexicalEmbedder(dim=32),
        join_threshold=0.95,
    )
    registry.clusterer_for("task-a").max_clusters_per_task = 1
    registry.assign("task-a", _finding("timeout waiting on the http api"))
    refused = registry.assign(
        "task-a", _finding("json schema mismatch in the parsed payload")
    )

    assert refused.unassigned_reason, (
        "the cap refusal reason must reach the caller; it is the only evidence "
        "distinguishing 'no mechanism yet' from 'mechanism withheld at the cap'"
    )
    assert "cap" in refused.unassigned_reason.lower()


# ---------------------------------------------------------------------- #
# 2. A producer on the genetic path
# ---------------------------------------------------------------------- #
def test_runner_exposes_an_entropy_tracker():
    """The genetic runner must own a tracker at all.

    ``SequentialGepaRunner`` has neither an ``entropy`` nor a
    ``cluster_registry`` field: both live only on ``Orchestrator``, whose
    ``run_iteration`` has zero production callers. So the tracker cannot be fed
    from the live path as things stand.
    """
    from test_phase_6_orchestrator import _runner

    runner = _runner()
    assert hasattr(runner, "entropy"), (
        "the live genetic runner needs an EntropyTracker to record into"
    )
    assert isinstance(runner.entropy, EntropyTracker)


def test_genetic_rollouts_record_into_the_tracker():
    """Scores observed on the genetic path must reach the tracker.

    Currently ``_record_rollout_score`` writes only to the pool, so the tracker
    stays empty no matter how many rollouts run and the spec's floors never
    have anything to gate.
    """
    from test_phase_6_orchestrator import _runner, _task

    runner = _runner()
    runner.build_issues((_task("task-a"),))

    assert runner.entropy.all_cells(), (
        "after a genetic observation the tracker must hold at least one cell; "
        "an empty tracker means the evidence was recorded to the pool only"
    )


def test_tracker_cells_are_mechanism_keyed_not_the_pool_constant():
    """The tracker key must be the clusterer's id, not the constant.

    This is the whole point of step 3. If the tracker is keyed by
    ``self.mechanism_cluster_id`` it reproduces exactly the single synthetic
    bucket that ``_cell_entropy`` already computes over, and nothing is gained.
    """
    from test_phase_6_orchestrator import _runner, _task

    runner = _runner()
    runner.build_issues((_task("task-a"),))

    keys = runner.entropy.all_cells()
    assert keys, "no cells recorded"
    mechanisms = {k.mechanism_cluster_id for k in keys}
    assert mechanisms != {runner.mechanism_cluster_id}, (
        "tracker cells are keyed by the constant pool mechanism id "
        f"({runner.mechanism_cluster_id!r}); they must carry the clusterer's "
        "mechanism id or the cell measures cross-candidate spread, not "
        "per-mechanism variance"
    )


def test_unassigned_mechanism_never_becomes_a_cell_key():
    """A refused assignment must not be written as a cell.

    With the cap set to 1 and a high join threshold, the clusterer refuses.
    Whatever the runner does with that, it must not produce a cell keyed by an
    empty or namespace-only mechanism id.
    """
    from test_phase_6_orchestrator import _runner, _task

    runner = _runner()
    if hasattr(runner, "cluster_registry"):
        runner.cluster_registry.join_threshold = 0.99
        runner.cluster_registry.clusterer_for("task-a").max_clusters_per_task = 1

    runner.build_issues((_task("task-a"),))

    for key in runner.entropy.all_cells():
        mech = key.mechanism_cluster_id
        assert mech.strip(), f"cell keyed by a blank mechanism: {key!r}"
        assert not mech.endswith(":"), (
            f"cell keyed by a namespace-only mechanism {mech!r}: this is a "
            "laundered refusal, not a mechanism"
        )


# ---------------------------------------------------------------------- #
# 3. The pool key stays constant (SV-2 comparability)
# ---------------------------------------------------------------------- #
def test_pool_keys_stay_constant_while_tracker_keys_diverge():
    """Mechanism-keying the tracker must not leak into the pool.

    The two structures answer different questions and need opposite key
    policies: the pool asks "is c1 better than base?" and needs **shared**
    keys; the tracker asks "how much do candidates disagree on this
    mechanism?" and needs **separated** keys. Champion selection intersects on
    the exact full key, so a mechanism-keyed pool yields an empty intersection
    and SV-2 regresses without raising.
    """
    from test_phase_6_orchestrator import _runner, _task

    runner = _runner()
    runner.build_issues((_task("task-a"),))

    pool_mechanisms = {
        m_id
        for entry in runner.pool.all_entries()
        for (_t, m_id) in entry.score_tensor
    }
    assert pool_mechanisms <= {runner.mechanism_cluster_id}, (
        "pool cells must keep the constant mechanism id for champion "
        f"comparability; found {sorted(pool_mechanisms)}"
    )


# ---------------------------------------------------------------------- #
# 4. A consumer, and honest unavailability
# ---------------------------------------------------------------------- #
def test_sparse_cell_reports_entropy_unavailable_rather_than_zero():
    """Below the floors, the tier must say ``skip`` -- not a bare 0.0.

    ``EntropyTracker.entropy`` returns ``None`` below the floor precisely so a
    caller can distinguish "unavailable" from "measured zero". A single
    candidate's single rollout cannot support a variance claim, and reporting
    0.0 makes an unmeasured cell look identical to a measured-and-uniform one.
    """
    from test_phase_6_orchestrator import _runner, _task

    runner = _runner()
    issues = runner.build_issues((_task("task-a"),))
    if not issues:
        pytest.skip("no issue produced; nothing to assert a tier on")

    assert issues[0].entropy_tier == "skip", (
        "one candidate with one rollout is below both floors (>=3 comparable "
        f"candidates, >=2 rollouts each) yet the tier is "
        f"{issues[0].entropy_tier!r}, which lets an unsupported entropy term "
        "reach issue quality at full weight"
    )


def test_entropy_term_comes_from_the_tracker():
    """The DPP's entropy number must be the tracker's, not a second inline copy.

    Two implementations of one spec formula is the defect: whichever the DPP
    reads, the other is dead weight, and today the DPP reads the inline one
    that ignores the evidence floors entirely.
    """
    from test_phase_6_orchestrator import _runner, _task

    runner = _runner()
    seen: list[tuple[str, str]] = []
    real = runner.entropy

    class _SpyTracker:
        """Delegates everything, recording the entropy reads.

        A wrapper rather than a patched method because ``EntropyTracker`` is a
        ``slots=True`` dataclass, so assigning to ``tracker.entropy`` raises
        ``AttributeError: attribute 'entropy' is read-only``.
        """

        def __init__(self, inner: EntropyTracker) -> None:
            self._inner = inner

        def entropy(self, task_id: str, mechanism_cluster_id: str):
            seen.append((task_id, mechanism_cluster_id))
            return self._inner.entropy(task_id, mechanism_cluster_id)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    runner.entropy = _SpyTracker(real)  # type: ignore[assignment]
    runner.build_issues((_task("task-a"),))

    assert seen, (
        "build_issues never consulted EntropyTracker.entropy; the entropy "
        "feeding issue selection is still the inline pool-tensor computation"
    )


def test_unrelated_mechanisms_do_not_share_one_cell():
    """Two unrelated faults on one task must occupy two cells.

    This is the property the whole change exists to buy. Under the constant
    key they land in one bucket and their score difference reads as variance
    *within* a mechanism -- i.e. "a fix is reachable here" for a mechanism that
    does not exist.
    """
    tracker = EntropyTracker()
    clusterer = MechanismClusterer(
        task_id="task-a",
        embedder=LexicalEmbedder(dim=256),
        join_threshold=0.75,
    )
    a = clusterer.assign(_analysis("timeout waiting on the remote http endpoint"))
    b = clusterer.assign(_analysis("json schema mismatch parsing the reply body"))
    assert a.cluster_id != b.cluster_id, "harness: mechanisms should not cluster"

    tracker.record_score("task-a", a.cluster_id, "cand-1", 0.1)
    tracker.record_score("task-a", b.cluster_id, "cand-2", 0.9)

    assert len({k.mechanism_cluster_id for k in tracker.all_cells()}) == 2
    for key in tracker.all_cells():
        cell_scores = tracker.cell_entropy(key.task_id, key.mechanism_cluster_id)
        assert cell_scores == 0.0, (
            "a single candidate in a cell must not yield entropy; got "
            f"{cell_scores} for {key!r}"
        )


# ---------------------------------------------------------------------- #
# 5. The offline stack must stay offline
# ---------------------------------------------------------------------- #
def test_offline_stack_embedder_makes_no_network_call():
    """``build_offline_stack`` must not reach a network embedding endpoint.

    The resolved config's default embedding provider is ``ollama``, so wiring
    ``embedder_for_config`` into the *offline* builder makes the offline path
    depend on whether a local daemon happens to be running -- measured at
    ~0.18s per embed against a live Ollama, which is both a hidden network
    dependency and a large slowdown across a suite that embeds on every
    diagnosed rollout. Offline means deterministic and local.
    """
    from agent_evolve.core.embeddings import FallbackEmbedder

    import agent_evolve.pipeline as pipeline

    embedder = pipeline.LexicalEmbedder(dim=pipeline.DEFAULT_EMBEDDING_DIM)
    assert not isinstance(embedder, FallbackEmbedder)

    calls: list[str] = []

    def _forbidden(*args: object, **kwargs: object) -> object:
        calls.append("network")
        raise AssertionError("the offline stack attempted a network embed")

    monkeypatched = pipeline.embedder_for_config
    try:
        pipeline.embedder_for_config = _forbidden  # type: ignore[assignment]
        registry = pipeline.cluster_registry_for_config(
            pipeline.resolve_profile("research_sequential", seed=0),
            embedder=embedder,
        )
        # An explicit embedder must be honoured without consulting the network
        # builder at all.
        assert registry.embedder_factory() is embedder
    finally:
        pipeline.embedder_for_config = monkeypatched  # type: ignore[assignment]

    assert not calls, "cluster_registry_for_config built a network embedder"


def test_offline_embedder_dim_is_not_the_colliding_32():
    """Offline determinism must not cost the wider hash space.

    Staying lexical offline is correct, but reverting to ``dim=32`` would
    reintroduce the collision that fragments one mechanism across clusters:
    unrelated texts measured cosine 0.822 in a 32-slot space, above the 0.75
    default join threshold.
    """
    import agent_evolve.pipeline as pipeline

    assert pipeline.DEFAULT_EMBEDDING_DIM > 32
    embedder = pipeline.LexicalEmbedder(dim=pipeline.DEFAULT_EMBEDDING_DIM)
    assert embedder.dim == pipeline.DEFAULT_EMBEDDING_DIM


# ---------------------------------------------------------------------- #
# 6. The entropy value and its tier must describe the SAME cell
# ---------------------------------------------------------------------- #
def test_entropy_value_and_tier_come_from_the_same_mechanism_cell():
    """A task's entropy number and its tier must not be sourced independently.

    ``raw_issue_quality`` uses the tier to decide how to weight the entropy
    number: ``frontier_exploration`` damps it to ``frontier_weight`` (0.30),
    ``skip`` zeroes it, anything else applies it at full weight. So the tier is
    an instruction about *that specific number*.

    When a task holds several mechanism cells, picking the value from the
    strongest cell while picking the tier by "any non-skip tier present, prefer
    recombination_target" lets one cell's number inherit another cell's weight
    -- measured below as a ``frontier_exploration`` entropy being reported as
    ``recombination_target``, i.e. silently promoted from 30% to 100% weight.
    """
    from test_phase_6_orchestrator import _runner

    runner = _runner()
    tracker = runner.entropy

    # m1: floors met, LOW variance, HIGH max score -> recombination_target
    for cand, pair in {
        "c1": (0.80, 0.82),
        "c2": (0.84, 0.86),
        "c3": (0.88, 0.90),
    }.items():
        for value in pair:
            tracker.record_score("T", "T:m1", cand, value)
        tracker.mark_comparable("T", "T:m1", cand)

    # m2: floors met, HIGHER variance, LOW max score -> frontier_exploration
    for cand, pair in {
        "d1": (0.00, 0.02),
        "d2": (0.10, 0.12),
        "d3": (0.20, 0.29),
    }.items():
        for value in pair:
            tracker.record_score("T", "T:m2", cand, value)
        tracker.mark_comparable("T", "T:m2", cand)

    h1 = tracker.entropy("T", "T:m1")
    h2 = tracker.entropy("T", "T:m2")
    assert h1 is not None and h2 is not None
    assert h2 > h1, "harness: m2 must be the stronger cell for this to bite"
    assert tracker.classify("T", "T:m1") == "recombination_target"
    assert tracker.classify("T", "T:m2") == "frontier_exploration"

    value = runner._cell_entropy("T")
    tier = runner._entropy_tier("T")

    # Identify which cell actually supplied the number, then require the tier to
    # be that cell's own classification.
    source = "T:m1" if abs(value - h1) < 1e-12 else "T:m2"
    assert tier == tracker.classify("T", source), (
        f"entropy {value!r} came from {source} whose tier is "
        f"{tracker.classify('T', source)!r}, but the reported tier is "
        f"{tier!r}. The tier decides how this number is weighted, so a "
        "mismatched pair applies the wrong weight: a frontier_exploration "
        "value reported as recombination_target is promoted from 30% to 100%."
    )
