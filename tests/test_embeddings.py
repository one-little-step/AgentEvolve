"""Behavioral tests for the Ollama embedding provider and its fallback.

Governing contracts:
* ``docs/architecture/selection-algorithms.md:276-280`` — embedding fallback is
  permitted only when the profile allows it, and the reason must be recorded.
  Silent substitution is forbidden.
* ``docs/architecture/orchestration-lifecycle.md:123`` — "Embedding unavailable:
  use configured deterministic lexical fallback only if profile permits; record
  fallback in manifest."

Every test here is offline: the HTTP transport is injected. A single opt-in live
test exercises the real ``embeddinggemma`` service and skips when it is absent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_evolve.core.clustering import (  # noqa: E402
    EmbeddingProviderUnavailable,
    LexicalEmbedder,
)
from agent_evolve.core.config import EmbeddingConfig  # noqa: E402
from agent_evolve.core.embeddings import (  # noqa: E402
    FallbackEmbedder,
    OllamaEmbedder,
    build_embedder,
)


# ---------------------------------------------------------------------- #
# Fake transports
# ---------------------------------------------------------------------- #
class _RecordingTransport:
    """Captures calls and returns a canned payload."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, Mapping[str, object], float]] = []

    def __call__(
        self, url: str, payload: Mapping[str, object], timeout: float
    ) -> Mapping[str, object]:
        self.calls.append((url, dict(payload), timeout))
        return self.payload


class _FailingTransport:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def __call__(
        self, url: str, payload: Mapping[str, object], timeout: float
    ) -> Mapping[str, object]:
        self.calls += 1
        raise self.exc


def _unit_vector(dim: int) -> list[float]:
    vec = [0.0] * dim
    vec[0] = 1.0
    return vec


# ---------------------------------------------------------------------- #
# Happy path
# ---------------------------------------------------------------------- #
def test_embed_returns_vector_from_api_embed_response() -> None:
    """The ``/api/embed`` batch response shape yields a float tuple."""
    transport = _RecordingTransport({"embeddings": [_unit_vector(4)]})
    embedder = OllamaEmbedder(
        url="http://localhost:11434", model="embeddinggemma", dim=4, transport=transport
    )

    vector = embedder.embed("retrieve top_k docs")

    assert vector == (1.0, 0.0, 0.0, 0.0)
    assert all(isinstance(value, float) for value in vector)


def test_embed_posts_model_and_input_to_api_embed() -> None:
    """The request targets ``/api/embed`` and carries the configured model."""
    transport = _RecordingTransport({"embeddings": [_unit_vector(4)]})
    embedder = OllamaEmbedder(
        url="http://localhost:11434/",  # trailing slash must not double up
        model="embeddinggemma",
        dim=4,
        timeout=7.5,
        transport=transport,
    )

    embedder.embed("policy: execute tool")

    assert len(transport.calls) == 1
    url, payload, timeout = transport.calls[0]
    assert url == "http://localhost:11434/api/embed"
    assert payload == {"model": "embeddinggemma", "input": "policy: execute tool"}
    assert timeout == 7.5


def test_embed_accepts_single_embedding_response_shape() -> None:
    """The older ``{"embedding": [...]}`` shape is also accepted."""
    transport = _RecordingTransport({"embedding": _unit_vector(4)})
    embedder = OllamaEmbedder(
        url="http://localhost:11434", model="embeddinggemma", dim=4, transport=transport
    )

    assert embedder.embed("x") == (1.0, 0.0, 0.0, 0.0)


def test_dim_is_exposed_for_the_mechanism_embedder_protocol() -> None:
    embedder = OllamaEmbedder(
        url="http://localhost:11434",
        model="embeddinggemma",
        dim=768,
        transport=_RecordingTransport({"embeddings": [_unit_vector(768)]}),
    )

    assert embedder.dim == 768


# ---------------------------------------------------------------------- #
# Caching (keeps the DPP path from re-billing identical mechanism text)
# ---------------------------------------------------------------------- #
def test_repeated_text_is_served_from_cache() -> None:
    transport = _RecordingTransport({"embeddings": [_unit_vector(4)]})
    embedder = OllamaEmbedder(
        url="http://localhost:11434", model="embeddinggemma", dim=4, transport=transport
    )

    first = embedder.embed("same mechanism text")
    second = embedder.embed("same mechanism text")

    assert first == second
    assert len(transport.calls) == 1


def test_blank_text_returns_zero_vector_without_a_network_call() -> None:
    """Blank mechanism text must not cost a request; zeros are a valid vector."""
    transport = _RecordingTransport({"embeddings": [_unit_vector(4)]})
    embedder = OllamaEmbedder(
        url="http://localhost:11434", model="embeddinggemma", dim=4, transport=transport
    )

    assert embedder.embed("   ") == (0.0, 0.0, 0.0, 0.0)
    assert transport.calls == []


# ---------------------------------------------------------------------- #
# Unavailability: must raise the sentinel, never return a wrong-shape vector
# ---------------------------------------------------------------------- #
def test_transport_failure_raises_provider_unavailable() -> None:
    transport = _FailingTransport(OSError("connection refused"))
    embedder = OllamaEmbedder(
        url="http://localhost:11434", model="embeddinggemma", dim=4, transport=transport
    )

    with pytest.raises(EmbeddingProviderUnavailable):
        embedder.embed("anything")


def test_dimension_mismatch_raises_provider_unavailable() -> None:
    """An unexpected dimension is an unavailability, not a silent reshape."""
    transport = _RecordingTransport({"embeddings": [_unit_vector(3)]})
    embedder = OllamaEmbedder(
        url="http://localhost:11434", model="embeddinggemma", dim=4, transport=transport
    )

    with pytest.raises(EmbeddingProviderUnavailable):
        embedder.embed("anything")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"embeddings": []},
        {"embeddings": [[]]},
        {"embedding": None},
        {"embeddings": [["not-a-number", 0.0, 0.0, 0.0]]},
    ],
)
def test_malformed_response_raises_provider_unavailable(
    payload: Mapping[str, object],
) -> None:
    embedder = OllamaEmbedder(
        url="http://localhost:11434",
        model="embeddinggemma",
        dim=4,
        transport=_RecordingTransport(payload),
    )

    with pytest.raises(EmbeddingProviderUnavailable):
        embedder.embed("anything")


def test_failed_lookup_is_not_cached() -> None:
    """A transient failure must not poison the cache with a bad entry."""
    transport = _FailingTransport(OSError("boom"))
    embedder = OllamaEmbedder(
        url="http://localhost:11434", model="embeddinggemma", dim=4, transport=transport
    )

    for _ in range(2):
        with pytest.raises(EmbeddingProviderUnavailable):
            embedder.embed("anything")

    assert transport.calls == 2


# ---------------------------------------------------------------------- #
# Constructor validation
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [0, -1, True])
def test_invalid_dim_is_rejected(dim: object) -> None:
    with pytest.raises(ValueError):
        OllamaEmbedder(url="http://x", model="m", dim=dim)  # type: ignore[arg-type]


def test_blank_url_or_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        OllamaEmbedder(url="", model="m")
    with pytest.raises(ValueError):
        OllamaEmbedder(url="http://x", model="")


# ---------------------------------------------------------------------- #
# FallbackEmbedder: records the reason, never substitutes silently
# ---------------------------------------------------------------------- #
def test_fallback_embedder_uses_primary_and_records_no_reason() -> None:
    primary = OllamaEmbedder(
        url="http://localhost:11434",
        model="embeddinggemma",
        dim=4,
        transport=_RecordingTransport({"embeddings": [_unit_vector(4)]}),
    )
    embedder = FallbackEmbedder(primary=primary, fallback=LexicalEmbedder(dim=4))

    vector = embedder.embed("mechanism text")

    assert vector == (1.0, 0.0, 0.0, 0.0)
    assert embedder.fallback_reason is None
    assert embedder.used_fallback is False


def test_fallback_embedder_falls_back_and_records_reason() -> None:
    """Provider unavailability degrades to lexical AND records the reason."""
    primary = OllamaEmbedder(
        url="http://localhost:11434",
        model="embeddinggemma",
        dim=4,
        transport=_FailingTransport(OSError("down")),
    )
    embedder = FallbackEmbedder(primary=primary, fallback=LexicalEmbedder(dim=4))

    vector = embedder.embed("mechanism text")

    assert len(vector) == 4
    assert embedder.used_fallback is True
    assert embedder.fallback_reason == "provider_unavailable"


def test_fallback_embedder_dim_follows_the_active_embedder() -> None:
    primary = OllamaEmbedder(
        url="http://x", model="m", dim=4, transport=_FailingTransport(OSError("down"))
    )
    embedder = FallbackEmbedder(primary=primary, fallback=LexicalEmbedder(dim=4))

    assert embedder.dim == 4


def test_fallback_embedder_rejects_dim_mismatch_between_primary_and_fallback() -> None:
    """A fallback of a different width would silently break the DPP kernel."""
    primary = OllamaEmbedder(url="http://x", model="m", dim=768)
    with pytest.raises(ValueError):
        FallbackEmbedder(primary=primary, fallback=LexicalEmbedder(dim=64))


def test_fallback_embedder_keeps_dimensions_stable_across_a_mixed_run() -> None:
    """Once degraded, every later vector keeps the same width (kernel safety)."""
    transport = _RecordingTransport({"embeddings": [_unit_vector(4)]})
    primary = OllamaEmbedder(
        url="http://x", model="m", dim=4, transport=transport
    )
    embedder = FallbackEmbedder(primary=primary, fallback=LexicalEmbedder(dim=4))

    good = embedder.embed("first")
    transport_failure = _FailingTransport(OSError("down"))
    primary._transport = transport_failure  # type: ignore[attr-defined]
    degraded = embedder.embed("second")

    assert len(good) == len(degraded) == 4


# ---------------------------------------------------------------------- #
# build_embedder: wires EmbeddingConfig to a concrete provider
# ---------------------------------------------------------------------- #
def test_build_embedder_returns_ollama_backed_fallback_for_ollama_provider() -> None:
    config = EmbeddingConfig(
        url="http://localhost:11434",
        model="embeddinggemma",
        provider="ollama",
        fallback="lexical",
    )

    embedder = build_embedder(config, dim=4, transport=_RecordingTransport({}))

    assert isinstance(embedder, FallbackEmbedder)
    assert isinstance(embedder.primary, OllamaEmbedder)
    assert isinstance(embedder.fallback, LexicalEmbedder)


def test_build_embedder_returns_lexical_only_when_fallback_is_the_provider() -> None:
    config = EmbeddingConfig(
        url="", model="", provider="lexical", fallback="lexical"
    )

    embedder = build_embedder(config, dim=4)

    assert isinstance(embedder, LexicalEmbedder)


def test_build_embedder_rejects_an_unknown_provider() -> None:
    config = EmbeddingConfig(
        url="http://x", model="m", provider="mystery-net", fallback="lexical"
    )

    with pytest.raises(ValueError):
        build_embedder(config, dim=4)


def test_build_embedder_rejects_an_unknown_fallback() -> None:
    config = EmbeddingConfig(
        url="http://x", model="m", provider="ollama", fallback="guess"
    )

    with pytest.raises(ValueError):
        build_embedder(config, dim=4)


# ---------------------------------------------------------------------- #
# Opt-in live check against the real service
# ---------------------------------------------------------------------- #
def _live_embeddings_enabled() -> bool:
    return os.environ.get("AGENT_EVOLVE_LIVE_EMBEDDINGS") == "1"


@pytest.mark.skipif(
    not _live_embeddings_enabled(),
    reason="set AGENT_EVOLVE_LIVE_EMBEDDINGS=1 to exercise the real Ollama service",
)
def test_live_embeddinggemma_returns_a_normalized_768_vector() -> None:
    embedder = OllamaEmbedder(
        url=os.environ.get("OLLAMA_EMBEDDING_URL", "http://localhost:11434"),
        model=os.environ.get("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
        dim=768,
    )

    first = embedder.embed("retrieve(query): return top_k docs by bm25")
    second = OllamaEmbedder(
        url=os.environ.get("OLLAMA_EMBEDDING_URL", "http://localhost:11434"),
        model=os.environ.get("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
        dim=768,
    ).embed("retrieve(query): return top_k docs by bm25")

    assert len(first) == 768
    assert sum(value * value for value in first) == pytest.approx(1.0, abs=1e-3)
    assert first == second  # deterministic for identical input
