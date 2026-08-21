"""SV-12 step 1: the production stack must honour ``config.embedding``.

Three defects are pinned here, all measured against unmodified source before
these tests were written.

**1. The embedder is unwired.** ``config.embedding`` defaults to
``provider="ollama"`` (``config.py:316``) and a real semantic embedder exists
(``core/embeddings.py``), but ``build_embedder`` has zero callers in ``src/`` or
``scripts/`` and ``config.embedding`` is never read to construct anything.
``pipeline.py`` hardcodes ``LexicalEmbedder(dim=32)`` instead. So the config
advertises semantic embeddings while production runs hashed-token cosine.

**2. The pool cell key must stay mechanism-constant.** Champion comparison
intersects on the *full* ``(task_id, mechanism_cluster_id)`` key
(``pool.py:449-451``), and an offspring's diagnosed mechanism is *supposed* to
differ from its parent's. Mechanism-keying the pool therefore empties the
comparable-cell overlap and an unambiguously better candidate stops dominating
-- silently, with a frontier that looks like healthy diversity. Only the entropy
tracker gets mechanism-keyed; the pool key stays constant. This is a guard on
closed SV-2, requested by the user.

**3. ``task_id`` must not be embedded.** The task name is not evidence about a
failure *mechanism*, and with a real semantic embedder it biases clustering
toward same-task grouping -- the opposite of the cross-task evidence pooling the
floors need.

Nothing here asserts that clustering *quality* improved; that is step 2. These
tests only fix the wiring and the keys.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import EmbeddingConfig, resolve_profile  # noqa: E402
from agent_evolve.core.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    FallbackEmbedder,
    OllamaEmbedder,
)
from agent_evolve.core.pool import PersistentPool  # noqa: E402

from test_pool import _candidate, _prov  # noqa: E402


# ---------------------------------------------------------------------- #
# Fake transport
# ---------------------------------------------------------------------- #
class _StubTransport:
    """Returns a canned unit vector so no network call is made."""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls: list[tuple[str, dict[str, object], float]] = []

    def __call__(
        self, url: str, payload: Mapping[str, object], timeout: float
    ) -> Mapping[str, object]:
        self.calls.append((url, dict(payload), timeout))
        return {"embeddings": [[1.0] + [0.0] * (self.dim - 1)]}


# ---------------------------------------------------------------------- #
# 1. The pipeline must build its embedder from the resolved config
# ---------------------------------------------------------------------- #
def test_pipeline_exposes_an_embedder_builder_that_reads_config():
    """A seam must exist that turns ``config.embedding`` into an embedder.

    Without this, ``config.embedding`` is decoration: the profile declares
    ``provider="ollama"`` and production silently runs lexical instead.
    """
    import agent_evolve.pipeline as pipeline

    assert hasattr(pipeline, "embedder_for_config"), (
        "pipeline must expose embedder_for_config(config, ...) so the resolved "
        "config.embedding actually determines the embedder. Today "
        "build_embedder has zero callers in src/ and pipeline.py hardcodes "
        "LexicalEmbedder(dim=32), so the ollama default is never honoured."
    )


def test_ollama_provider_config_yields_a_semantic_embedder():
    """``provider="ollama"`` must not silently degrade to lexical."""
    import agent_evolve.pipeline as pipeline

    cfg = resolve_profile("research_sequential", {})
    assert cfg.embedding.provider == "ollama", "profile default changed"

    embedder = pipeline.embedder_for_config(
        cfg, transport=_StubTransport(), dim=DEFAULT_EMBEDDING_DIM
    )
    # Either the provider itself, or the provider wrapped in its declared
    # fallback -- but never a bare LexicalEmbedder, which would mean the
    # ollama config was discarded.
    assert isinstance(embedder, (OllamaEmbedder, FallbackEmbedder)), (
        f"config declared provider=ollama but got {type(embedder).__name__}; "
        "a bare LexicalEmbedder means config.embedding was ignored"
    )


def test_lexical_provider_config_is_still_honoured():
    """``provider="lexical"`` must stay offline and deterministic."""
    import agent_evolve.pipeline as pipeline

    cfg = resolve_profile(
        "research_sequential",
        {},
        embedding=EmbeddingConfig(
            url="http://unused", model="unused", provider="lexical"
        ),
    )
    embedder = pipeline.embedder_for_config(cfg)
    assert isinstance(embedder, LexicalEmbedder), (
        "an explicit lexical provider must produce LexicalEmbedder"
    )


def test_embedder_dim_is_not_silently_downgraded_to_32():
    """The 768-dim default must not be overridden by a hardcoded 32.

    ``pipeline.py`` passes ``dim=32`` at both stack-build sites while
    ``DEFAULT_EMBEDDING_DIM`` is 768. A 32-slot hashed bucket collides far more
    aggressively, which is one reason one fault fragments into many clusters.
    """
    import agent_evolve.pipeline as pipeline

    cfg = resolve_profile(
        "research_sequential",
        {},
        embedding=EmbeddingConfig(
            url="http://unused", model="unused", provider="lexical"
        ),
    )
    embedder = pipeline.embedder_for_config(cfg)
    dim = len(embedder.embed("any mechanism text"))
    assert dim == DEFAULT_EMBEDDING_DIM, (
        f"embedding dim is {dim}; expected DEFAULT_EMBEDDING_DIM="
        f"{DEFAULT_EMBEDDING_DIM}. A hardcoded 32 was the production value."
    )


def test_fallback_reason_is_reported_not_silent():
    """Degrading to lexical must be *recorded*, never silent.

    Per ``selection-algorithms.md`` and ``orchestration-lifecycle.md``, a
    provider outage may degrade only when the profile permits it and the reason
    must be recorded. A silently substituted embedder makes a coarse clustering
    indistinguishable from a fine one.
    """
    import agent_evolve.pipeline as pipeline

    def _dead_transport(url: str, payload: Mapping[str, object], timeout: float):
        raise OSError("connection refused")

    cfg = resolve_profile("research_sequential", {})
    embedder = pipeline.embedder_for_config(
        cfg, transport=_dead_transport, dim=DEFAULT_EMBEDDING_DIM
    )
    if not isinstance(embedder, FallbackEmbedder):
        pytest.skip("profile does not declare a lexical fallback")

    vec = embedder.embed("some mechanism")
    assert len(vec) == DEFAULT_EMBEDDING_DIM
    assert embedder.used_fallback is True, (
        "a provider outage degraded to lexical without recording it"
    )
    assert embedder.fallback_reason, "fallback_reason must name the cause"


# ---------------------------------------------------------------------- #
# 2. Pool cell key guard -- protects closed SV-2
# ---------------------------------------------------------------------- #
def _pool_with(mech_of) -> PersistentPool:
    """base scores 0.5 everywhere, c1 scores 0.9 everywhere: c1 is better."""
    p = PersistentPool()
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1", parents=("base",)))
    for cid, score in (("base", 0.5), ("c1", 0.9)):
        for task in ("task-a", "task-b"):
            for r in range(2):
                s, prov = _prov(task, mech_of(cid, task), rollout=r, score=score)
                p.record_score(cid, s, prov)
    return p


def test_shared_mechanism_key_lets_the_better_candidate_dominate():
    """The control: with a shared key, 0.9 beats 0.5."""
    p = _pool_with(lambda c, t: "mechanism-default")
    assert len(p.comparable_cells("base", "c1")) == 2
    assert p.dominates("c1", "base") is True
    assert p.pareto_frontier() == ("c1",)


def test_diverging_mechanism_keys_destroy_comparability():
    """Documents *why* the pool key must stay constant.

    This is the failure mode a future "make the pool mechanism-aware" change
    would introduce. It is asserted rather than described so the consequence
    stays measured: same tasks, same scores, same obvious winner -- and the
    comparison can no longer see it.
    """
    p = _pool_with(lambda c, t: f"{t}:mech-of-{c}")
    assert p.comparable_cells("base", "c1") == ()
    assert p.dominates("c1", "base") is False
    # Both "on the frontier" reads like diversity; it means nothing compared.
    assert set(p.pareto_frontier()) == {"base", "c1"}


def test_production_pool_writes_use_a_constant_mechanism_id():
    """Guard on closed SV-2, requested by the user.

    Every pool write on the live genetic path must use one constant mechanism
    id. If a future change keys pool cells by diagnosed mechanism, champion
    comparison degrades silently -- see the test above.
    """
    from test_phase_6_orchestrator import _runner, _task

    r = _runner(seed=0)
    r.run_attempt((_task("task-a"), _task("task-b")))

    mech_ids = {
        m_id
        for entry in r.pool.all_entries()
        for (_t_id, m_id) in entry.score_tensor.keys()
    }
    assert len(mech_ids) == 1, (
        f"production pool writes used {len(mech_ids)} distinct mechanism ids "
        f"({sorted(mech_ids)}). The pool's mechanism key must stay constant: "
        "champion comparison intersects on the full (task, mechanism) key, so "
        "diverging ids empty the overlap and a strictly better candidate stops "
        "dominating. Mechanism-key the EntropyTracker instead."
    )


# ---------------------------------------------------------------------- #
# 4. The dedup adjudicator config: opt-in, and never leaks its credential
# ---------------------------------------------------------------------- #
def test_dedup_is_disabled_without_configuration():
    """An unconfigured deployment keeps today's cosine-only behaviour.

    The adjudicator must never acquire a model dependency implicitly: absent env
    vars mean disabled, not "enabled against an empty endpoint".
    """
    cfg = resolve_profile("research_sequential", {})
    assert cfg.mechanism_dedup.enabled is False
    assert cfg.mechanism_dedup.model == ""


def test_dedup_enables_from_its_own_env_vars():
    """Its endpoint, model and key are addressed independently of every other
    model role, so a small cheap model can serve it whatever the others use."""
    cfg = resolve_profile(
        "research_sequential",
        {
            "AE_MECHANISM_DEDUP_MODEL": "small-dedup-model",
            "AE_MECHANISM_DEDUP_BASE_URL": "http://localhost:11434",
            "AE_MECHANISM_DEDUP_API_KEY": "unit-test-key",
        },
    )
    assert cfg.mechanism_dedup.enabled is True
    assert cfg.mechanism_dedup.model == "small-dedup-model"
    assert cfg.mechanism_dedup.url == "http://localhost:11434"


def test_dedup_api_key_never_reaches_the_manifest():
    """Credentials must never be persisted. The manifest is written to disk."""
    import json

    secret = "SECRET-must-not-appear-anywhere"
    cfg = resolve_profile(
        "research_sequential",
        {
            "AE_MECHANISM_DEDUP_MODEL": "small-dedup-model",
            "AE_MECHANISM_DEDUP_BASE_URL": "http://localhost:11434",
            "AE_MECHANISM_DEDUP_API_KEY": secret,
        },
    )
    blob = json.dumps(cfg.manifest_payload(), default=str)
    assert secret not in blob, "the dedup API key leaked into the manifest"
    assert "api_key" not in blob, "no api_key field may be serialized at all"
    # The non-secret settings must still be recorded, or a run is unreproducible.
    assert "mechanism_dedup" in blob
    assert "small-dedup-model" in blob


def test_inverted_dedup_band_is_rejected():
    """An inverted band would make every pair ambiguous and every assignment a
    model call -- the exact cost blow-up the band exists to prevent."""
    from agent_evolve.core.config import MechanismDedupConfig

    with pytest.raises(ValueError, match="band_low"):
        MechanismDedupConfig(band_low=0.9, band_high=0.5)


def test_malformed_band_env_var_fails_loudly():
    """A typo must not silently run the session on an unintended threshold."""
    with pytest.raises(ValueError, match="AE_MECHANISM_DEDUP_BAND_LOW"):
        resolve_profile(
            "research_sequential", {"AE_MECHANISM_DEDUP_BAND_LOW": "hgh"}
        )


def test_enabled_dedup_requires_a_model_and_endpoint():
    """Refuse to enable an adjudicator that cannot be called."""
    from agent_evolve.core.config import MechanismDedupConfig

    with pytest.raises(ValueError, match="no model was configured"):
        MechanismDedupConfig(enabled=True, url="http://x", model="")
    with pytest.raises(ValueError, match="no endpoint was configured"):
        MechanismDedupConfig(enabled=True, url="", model="m")


# ---------------------------------------------------------------------- #
# 5. task_id must not be embedded
# ---------------------------------------------------------------------- #
def test_embed_finding_does_not_encode_task_id():
    """The task name is not evidence about a failure mechanism.

    Asserted on the *text handed to the embedder*, not on the resulting vectors.
    Comparing vectors is vacuous here: ``LexicalEmbedder`` hashes ``task-a`` and
    ``task-b`` into the same bucket, so the two embeddings come back identical
    whether or not ``task_id`` was included, and the test would pass with the
    defect fully present. Measured before this test was trusted.
    """
    from agent_evolve.core.blame import BlameGraph, BlameNode, CausalFinding
    from test_phase_6_orchestrator import _runner, _task

    r = _runner(seed=0)

    def _finding(task_id: str) -> CausalFinding:
        """Same mechanism and same evidence; only the task differs."""
        return CausalFinding(
            verdict_id=f"v-{task_id}",
            candidate_id="c1",
            task_id=task_id,
            trace_id=f"tr-{task_id}",
            status="observed",
            mechanism_description="units were not verified before reporting",
            mechanism_cluster_id="m0",
            severity=0.6,
            confidence=0.8,
            rationale="units unchecked",
            evidence_refs=("skills/a.md",),
            blame_graph=BlameGraph(
                nodes=(
                    BlameNode(
                        actor_id="agent", artifacts=("skills/a.md",), blame=0.9
                    ),
                )
            ),
        )

    seen: list[str] = []

    class _SpyEmbedder:
        dim = 8

        def embed(self, text: str) -> tuple[float, ...]:
            seen.append(text)
            return (0.0,) * 8

    r.embedder = _SpyEmbedder()
    r._embed_finding(_finding("task-alpha"), _task("task-alpha"))
    r._embed_finding(_finding("task-beta"), _task("task-beta"))

    assert len(seen) == 2, "expected one embed call per finding"
    assert "task-alpha" not in seen[0], (
        f"task_id leaked into the embedded text: {seen[0]!r}. The task name is "
        "not evidence about a failure mechanism, and with a real semantic "
        "embedder it biases clustering toward same-task grouping -- the "
        "opposite of the cross-task evidence pooling the floors need."
    )
    assert "task-beta" not in seen[1], f"task_id leaked: {seen[1]!r}"
    # The same mechanism on two tasks must produce identical embedder input.
    assert seen[0] == seen[1], (
        f"same mechanism embedded differently across tasks: {seen[0]!r} vs "
        f"{seen[1]!r}"
    )
    # The mechanism and its attributed artifact must still be present.
    assert "units were not verified" in seen[0]
    assert "skills/a.md" in seen[0]
