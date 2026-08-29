"""Task-local incremental mechanism clustering.

Free-form analyzer+judge mechanisms are embedded with task, phase/tool,
artifact, and counterfactual context. Base-harness mechanisms provide anchors.
Task-local incremental clusters assign ``mechanism_cluster_id``, which is the
cross-candidate alignment key.

Clusters are stable inside an outer iteration. New observations may join an
existing cluster; cluster creation is eager (each assignment either joins an
existing cluster or spawns a new one immediately). Automatic merge/split and
barrier-deferred creation are future work. Track cluster freshness and reduce
entropy weight when evidence is stale.

Implementation note
-------------------
This module deliberately does NOT depend on any embedding model. It exposes
a :class:`MechanismEmbedder` Protocol so an adapter or host application can
plug in a real embedder (e.g., sentence-transformers, an OpenAI embedding
endpoint, or the ``z-ai-web-dev-sdk`` VLM/LLM embeddings). The default
:class:`LexicalEmbedder` is a deterministic tf-idf-style bag-of-tokens embedder
suitable for tests and offline smoke runs. It is NOT a research-grade
embedder and must not be used to draw research conclusions.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Iterable, Protocol, Sequence, runtime_checkable

from agent_evolve.core.blame import CausalAnalysis, CausalFinding

#: Cosine at or above which two mechanism descriptions join without a model call.
DEFAULT_JOIN_THRESHOLD = 0.75

#: The ambiguous cosine band, inside which cosine is measurably unreliable and an
#: injected adjudicator decides instead.
#:
#: Calibrated against live ``embeddinggemma`` over 4 fault families with 3
#: analyzer rephrasings each -- 12 same-fault and 54 different-fault pairs
#: (``terminal_output/sv12/17-band-decision.log``). The two distributions
#: **overlap**: same-fault cosine ran 0.466 to 0.851 and different-fault ran
#: 0.244 to 0.502, a separation of ``-0.036``. No single threshold can separate
#: an analyzer paraphrase from a genuinely different fault, which is why the
#: adjudicator is load-bearing rather than a cost optimisation.
#:
#: Scored by *silent splits* -- true paraphrase pairs decided against merging by
#: cosine alone, with no model call:
#:
#: ===================  ===========  ============  ================
#: band                 adjudicated  silent-split  false-merge-risk
#: ===================  ===========  ============  ================
#: ``[0.60, 0.85)``     9 / 66       2 / 12        0
#: ``[0.45, 0.75)``     16 / 66      0 / 12        0
#: ``[0.40, 0.75)``     35 / 66      0 / 12        0
#: ===================  ===========  ============  ================
#:
#: ``[0.45, 0.75)`` is the smallest measured band that silently splits zero true
#: pairs; ``[0.40, 0.75)`` buys nothing and doubles the calls. The previous
#: ``[0.60, 0.85)`` split 2 of 12.
#:
#: These 12 strings are synthetic phrasings, not real CUGA analyzer output, so
#: these values are evidence-based but not a tuned optimum.
DEFAULT_BAND_LOW = 0.45

#: Upper edge of the ambiguous band. **Must not be below the join threshold**:
#: the span ``[band_high, join_threshold)`` would then be neither ambiguous nor
#: joining, so cosine would decide it alone -- precisely the region the
#: adjudicator exists to cover. Measured: band ``[0.45, 0.70)`` against
#: threshold ``0.75`` stranded true pairs at cosine 0.718, 0.749 and 0.726.
DEFAULT_BAND_HIGH = 0.75


def _tokenize(text: str) -> tuple[str, ...]:
    """Lowercase, split on non-alphanumeric, drop empties and 1-char tokens."""
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                if len(tok) > 1:
                    out.append(tok)
                cur = []
    if cur:
        tok = "".join(cur)
        if len(tok) > 1:
            out.append(tok)
    return tuple(out)


class MechanismEmbedder(Protocol):
    """Maps a mechanism string (with context) to a fixed-length vector."""

    dim: int

    def embed(self, text: str) -> tuple[float, ...]: ...


class EmbeddingProviderUnavailable(RuntimeError):
    """Raised by an embedder when its backing provider is unreachable.

    The clusterer catches only this sentinel to trigger the lexical fallback;
    any other exception (e.g. a ``TypeError`` or ``ValueError`` from an
    embedder bug) propagates to the caller.
    """


class LexicalEmbedder:
    """Deterministic lexical embedder for tests and offline demos.

    Builds a vocab from all tokens it has ever seen (in token-id order, sorted
    for determinism), then emits a sparse->dense L2-normalized vector.
    Vocab grows over the lifetime of the embedder; subsequent calls see the
    expanded vocab.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim
        self._vocab: dict[str, int] = {}

    def _token_id(self, token: str) -> int:
        # Hash into [0, dim) deterministically so we don't have to grow the
        # vector length. Collisions are acceptable for a test embedder.
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    def embed(self, text: str) -> tuple[float, ...]:
        tokens = _tokenize(text)
        if not tokens:
            return tuple(0.0 for _ in range(self.dim))
        counts: list[float] = [0.0] * self.dim
        for tok in tokens:
            counts[self._token_id(tok)] += 1.0
        # L2 normalize.
        norm = math.sqrt(sum(c * c for c in counts))
        if norm == 0.0:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(c / norm for c in counts)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@runtime_checkable
class MechanismAdjudicator(Protocol):
    """Decides whether two mechanism descriptions name the same fault.

    Injected, never imported: ``core/`` is agent-neutral, so the model-backed
    implementation lives adapter-side and arrives through this protocol exactly as
    :class:`MechanismEmbedder` does.

    Consulted **only** where embedding cosine is measurably unreliable -- inside
    the ambiguous similarity band, and on a forced merge at the cluster cap. Clear
    cases never reach it, which is what keeps a model in this path affordable.

    Returns ``True`` for the same mechanism, ``False`` for different ones, and
    ``None`` to abstain. An abstention or a raised exception must leave the cosine
    decision standing and be recorded: a dedup outage may never silently change a
    clustering decision.
    """

    def same_mechanism(self, left: str, right: str) -> bool | None: ...


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """Where one mechanism observation landed."""

    cluster_id: str
    similarity: float  # in [-1, 1]; cosine to the cluster centroid
    is_new_cluster: bool
    task_id: str = ""
    freshness_iteration: int = 0
    embedding_fallback_reason: str | None = None
    #: Set when no cluster could be assigned. ``cluster_id`` is then ``""`` and
    #: the observation must not contribute to a mechanism cell: variance over two
    #: unrelated faults reads as "a fix is reachable here" for a mechanism that
    #: does not exist.
    unassigned_reason: str | None = None
    #: Set when the adjudicator was consulted but could not answer (abstained or
    #: raised). The cosine decision stands; this records that it was not
    #: adjudicated, so a coarse decision is never mistaken for a fine one.
    adjudication_unavailable_reason: str | None = None


@dataclass(slots=True)
class _Cluster:
    cluster_id: str
    centroid: list[float]
    member_count: int = 0
    last_touched_iter: int = 0


@dataclass(slots=True)
class MechanismClusterer:
    """Task-local incremental clusterer.

    Each instance is scoped to one task. Anchors are the base-harness
    mechanisms; candidate mechanisms are matched against the existing
    centroids and either join (similarity >= ``join_threshold``) or spawn a
    new cluster.
    """

    task_id: str
    embedder: MechanismEmbedder
    join_threshold: float = DEFAULT_JOIN_THRESHOLD
    max_clusters_per_task: int = 12
    #: Optional model-backed tie-breaker for the ambiguous band. ``None`` keeps
    #: cosine-only behaviour and adds no model dependency.
    adjudicator: MechanismAdjudicator | None = None
    #: Below ``band_low`` a pair is confidently distinct; at or above
    #: ``band_high`` it is confidently the same mechanism. Only the span between
    #: is worth a model call. See :data:`DEFAULT_BAND_LOW` for the calibration.
    band_low: float = DEFAULT_BAND_LOW
    band_high: float = DEFAULT_BAND_HIGH
    _clusters: dict[str, _Cluster] = field(default_factory=dict)
    _next_id: int = 0
    _current_iter: int = 0
    _fallback_embedder: LexicalEmbedder | None = field(default=None, init=False)
    #: One representative mechanism text per cluster, kept so the adjudicator can
    #: be asked about *text* rather than about a centroid vector it cannot read.
    _exemplars: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.join_threshold <= 1.0:
            raise ValueError("join_threshold must be in [0, 1]")
        if not self.task_id:
            raise ValueError("task_id is required")
        if (
            isinstance(self.max_clusters_per_task, bool)
            or not isinstance(self.max_clusters_per_task, int)
            or self.max_clusters_per_task < 1
        ):
            raise ValueError("max_clusters_per_task must be a positive integer")
        if not 0.0 <= self.band_low <= 1.0:
            raise ValueError("band_low must be in [0, 1]")
        if not 0.0 <= self.band_high <= 1.0:
            raise ValueError("band_high must be in [0, 1]")
        if self.band_low > self.band_high:
            raise ValueError(
                f"band_low ({self.band_low}) must be <= band_high "
                f"({self.band_high}): an inverted band would make every pair "
                "ambiguous and every assignment a model call"
            )
        if self.adjudicator is not None and self.band_high < self.join_threshold:
            # Scoped to "an adjudicator exists" deliberately. The band is read
            # only at the adjudicator gate in ``_add``, so with no adjudicator
            # there is nothing to strand and a raise here would reject
            # legitimate cosine-only configurations -- measured rejecting 7
            # existing tests that raise the join threshold with no adjudicator.
            raise ValueError(
                f"band_high ({self.band_high}) must be >= join_threshold "
                f"({self.join_threshold}) when an adjudicator is attached: the "
                f"span [{self.band_high}, {self.join_threshold}) would be "
                "neither ambiguous nor joining, so cosine alone would split "
                "pairs there with no adjudicator call -- measured stranding "
                "true paraphrase pairs at cosine 0.718, 0.749 and 0.726"
            )

    # ------------------------------------------------------------------ #
    # Iteration barriers
    # ------------------------------------------------------------------ #
    def begin_iteration(self, iteration: int) -> None:
        if iteration < 0:
            raise ValueError("iteration must be >= 0")
        if iteration <= self._current_iter:
            raise ValueError(
                f"iteration must increase: current={self._current_iter}, got={iteration}"
            )
        self._current_iter = iteration

    @property
    def current_iteration(self) -> int:
        return self._current_iter

    def refresh_barriers(self) -> None:
        """Called at a refresh barrier.

        Per the architecture doc, cluster create/merge/split occurs at refresh
        barriers. We do not implement automatic merge/split here (it would
        require a full clustering pass over all members); we expose this hook
        so the orchestrator can mark a barrier, and subclasses can override.
        """
        # Hook for future work: re-cluster from scratch if drift detected.
        return None

    # ------------------------------------------------------------------ #
    # Anchors
    # ------------------------------------------------------------------ #
    def add_anchor(self, mechanism: str) -> ClusterAssignment:
        """Add a base-harness mechanism as a new cluster anchor."""
        return self._add(mechanism, force_new=True)

    # ------------------------------------------------------------------ #
    # Incremental assignment
    # ------------------------------------------------------------------ #
    def assign(self, analysis: CausalAnalysis | CausalFinding) -> ClusterAssignment:
        """Assign one analysis's mechanism to a cluster.

        The embedder input is the mechanism string joined with the actor and
        artifact context, mirroring the architecture doc's requirement that
        embedding carry "task, phase/tool, artifact, and counterfactual
        context".
        """
        if isinstance(analysis, CausalFinding):
            return self.assign_finding(analysis)
        text = self._embed_text(analysis)
        return self._add(text, force_new=False)

    def assign_finding(self, finding: CausalFinding) -> ClusterAssignment:
        """Assign a trace-backed :class:`CausalFinding` to a cluster."""
        text = self._embed_text_finding(finding)
        return self._add(text, force_new=False)

    def _embed_text(self, analysis: CausalAnalysis) -> str:
        parts = [analysis.mechanism]
        for n in analysis.blame_graph.nodes:
            parts.append(n.actor_id)
            parts.extend(n.artifacts)
        parts.extend(analysis.counterfactual_evidence)
        return " ".join(parts)

    def _embed_text_finding(self, finding: CausalFinding) -> str:
        parts: list[str] = []
        if finding.mechanism_description:
            parts.append(finding.mechanism_description)
        for n in finding.blame_graph.nodes:
            parts.append(n.actor_id)
            parts.extend(n.artifacts)
        parts.extend(finding.counterfactual_notes)
        return " ".join(parts)

    def _embed(self, text: str) -> tuple[list[float], str | None]:
        """Embed text, falling back to a lexical embedder on provider failure."""
        try:
            return list(self.embedder.embed(text)), None
        except EmbeddingProviderUnavailable:
            if self._fallback_embedder is None:
                self._fallback_embedder = LexicalEmbedder()
            return list(self._fallback_embedder.embed(text)), "provider_unavailable"

    def _consult(self, text: str, cluster_id: str) -> tuple[bool | None, str | None]:
        """Ask the adjudicator whether ``text`` names the cluster's mechanism.

        Returns ``(verdict, unavailable_reason)``. A raised exception or a missing
        exemplar is an unavailability, never a verdict: a dedup outage must leave
        the cosine decision standing rather than silently splitting or merging.
        """
        if self.adjudicator is None:
            return None, None
        exemplar = self._exemplars.get(cluster_id)
        if not exemplar:
            return None, "no_exemplar"
        try:
            return self.adjudicator.same_mechanism(exemplar, text), None
        except Exception as exc:  # noqa: BLE001 - any provider failure degrades
            return None, f"adjudicator_error: {type(exc).__name__}"

    def _add(self, text: str, force_new: bool) -> ClusterAssignment:
        vec, fallback_reason = self._embed(text)
        unavailable: str | None = None
        if not force_new and self._clusters:
            best_id, best_sim = self._best_match(vec)
            at_cap = len(self._clusters) >= self.max_clusters_per_task

            # Cosine decides the clear cases for free. The adjudicator is
            # consulted only where cosine is measurably unreliable: inside the
            # ambiguous band, and on a forced merge at the cap.
            join = best_sim >= self.join_threshold
            ambiguous = self.band_low <= best_sim < self.band_high
            if self.adjudicator is not None and (ambiguous or (at_cap and not join)):
                verdict, unavailable = self._consult(text, best_id)
                if verdict is True:
                    join = True
                elif verdict is False:
                    join = False

            if join:
                self._update_cluster(best_id, vec)
                return ClusterAssignment(
                    cluster_id=best_id,
                    similarity=best_sim,
                    is_new_cluster=False,
                    task_id=self.task_id,
                    freshness_iteration=self._current_iter,
                    embedding_fallback_reason=fallback_reason,
                    adjudication_unavailable_reason=unavailable,
                )

            if at_cap:
                # Previously ``best_sim >= join_threshold or at_cap`` absorbed
                # this observation into the nearest cluster whatever its
                # similarity -- measured joining an unrelated mechanism at cosine
                # 0.822. Two unrelated faults in one cell yield a *high* variance
                # reading, i.e. "a fix is reachable here" for a mechanism that
                # does not exist, which is worse than no reading because nothing
                # looks broken. Refuse instead and say why.
                return ClusterAssignment(
                    cluster_id="",
                    similarity=best_sim,
                    is_new_cluster=False,
                    task_id=self.task_id,
                    freshness_iteration=self._current_iter,
                    embedding_fallback_reason=fallback_reason,
                    unassigned_reason=(
                        f"cluster cap reached ({self.max_clusters_per_task}) and "
                        f"nearest cluster similarity {best_sim:.3f} is below the "
                        f"join threshold {self.join_threshold:.3f}"
                    ),
                    adjudication_unavailable_reason=unavailable,
                )
        # New cluster.
        cluster_id = f"c{self._next_id}"
        self._next_id += 1
        self._clusters[cluster_id] = _Cluster(
            cluster_id=cluster_id,
            centroid=vec,
            member_count=1,
            last_touched_iter=self._current_iter,
        )
        self._exemplars[cluster_id] = text
        return ClusterAssignment(
            cluster_id=cluster_id,
            similarity=1.0,
            is_new_cluster=True,
            task_id=self.task_id,
            freshness_iteration=self._current_iter,
            embedding_fallback_reason=fallback_reason,
            adjudication_unavailable_reason=unavailable,
        )

    def _best_match(self, vec: list[float]) -> tuple[str, float]:
        best_id = ""
        best_sim = -2.0
        for cid, cluster in self._clusters.items():
            sim = _cosine(vec, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id, best_sim

    def _update_cluster(self, cluster_id: str, vec: list[float]) -> None:
        c = self._clusters[cluster_id]
        n = c.member_count
        # Running mean update: new_centroid = (old*n + vec) / (n+1)
        c.centroid = [(n * c.centroid[i] + vec[i]) / (n + 1) for i in range(len(vec))]
        c.member_count = n + 1
        c.last_touched_iter = self._current_iter

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def cluster_count(self) -> int:
        return len(self._clusters)

    def cluster_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._clusters.keys()))

    def cluster_exemplars(self) -> tuple[str, ...]:
        """Representative mechanism text per cluster, in cluster-id order.

        The exemplar is the first mechanism text that formed the cluster; it is
        what a consumer (e.g. the positivity judge) needs to decide *which
        cluster* a new observation names. Sorted by cluster id for determinism.
        """
        return tuple(self._exemplars[cid] for cid in sorted(self._clusters))

    def cluster_size(self, cluster_id: str) -> int:
        if cluster_id not in self._clusters:
            raise KeyError(cluster_id)
        return self._clusters[cluster_id].member_count

    def cluster_freshness(self, cluster_id: str) -> int:
        """How stale a cluster is, in iterations.

        Returns ``current_iteration - last_touched_iter``. Higher = staler.
        Stale clusters should have their entropy weight reduced per the
        architecture doc.
        """
        if cluster_id not in self._clusters:
            raise KeyError(cluster_id)
        return self._current_iter - self._clusters[cluster_id].last_touched_iter


@dataclass(slots=True)
class ClusterRegistry:
    """Holds one clusterer per task; the orchestrator-level container."""

    embedder_factory: Callable[[], MechanismEmbedder]
    join_threshold: float = DEFAULT_JOIN_THRESHOLD
    #: Optional shared adjudicator, handed to every per-task clusterer. ``None``
    #: keeps cosine-only behaviour with no model dependency.
    adjudicator: MechanismAdjudicator | None = None
    band_low: float = DEFAULT_BAND_LOW
    band_high: float = DEFAULT_BAND_HIGH
    _clusterers: dict[str, MechanismClusterer] = field(default_factory=dict)

    def clusterer_for(self, task_id: str) -> MechanismClusterer:
        if task_id not in self._clusterers:
            self._clusterers[task_id] = MechanismClusterer(
                task_id=task_id,
                embedder=self.embedder_factory(),
                join_threshold=self.join_threshold,
                adjudicator=self.adjudicator,
                band_low=self.band_low,
                band_high=self.band_high,
            )
        return self._clusterers[task_id]

    def assign(self, task_id: str, finding: CausalFinding) -> ClusterAssignment:
        """Assign a finding via the per-task clusterer, namespaced by task.

        A **refusal is preserved as a refusal**. The per-task clusterer returns
        ``cluster_id=""`` when the cluster cap is full and the nearest cluster
        is below the join threshold; namespacing that unconditionally would
        yield ``f"{task_id}:"`` -- a *non-empty* string. That is worse than
        useless: ``CellKey`` rejects only a falsy mechanism id, so a laundered
        refusal passes the guard and is filed as a legitimate mechanism, and
        even a caller checking ``if assignment.cluster_id:`` is defeated by the
        namespacing alone. Both reason fields are forwarded for the same reason:
        a caller that must report *why* entropy is unavailable cannot invent it.
        """
        assignment = self.clusterer_for(task_id).assign_finding(finding)
        namespaced = (
            f"{task_id}:{assignment.cluster_id}" if assignment.cluster_id else ""
        )
        return ClusterAssignment(
            cluster_id=namespaced,
            similarity=assignment.similarity,
            is_new_cluster=assignment.is_new_cluster,
            task_id=task_id,
            freshness_iteration=assignment.freshness_iteration,
            embedding_fallback_reason=assignment.embedding_fallback_reason,
            unassigned_reason=assignment.unassigned_reason,
            adjudication_unavailable_reason=assignment.adjudication_unavailable_reason,
        )

    def begin_iteration(self, iteration: int) -> None:
        for c in self._clusterers.values():
            c.begin_iteration(iteration)

    def refresh_barriers(self) -> None:
        for c in self._clusterers.values():
            c.refresh_barriers()

    def all_cluster_ids(self) -> tuple[str, ...]:
        out: list[str] = []
        for task_id in sorted(self._clusterers.keys()):
            for cid in self._clusterers[task_id].cluster_ids():
                out.append(f"{task_id}:{cid}")
        return tuple(out)
