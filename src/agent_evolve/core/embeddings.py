"""Embedding providers for mechanism and issue embeddings.

The generic core needs real embeddings for two things:

* mechanism clustering (:mod:`agent_evolve.core.clustering`), and
* the DPP similarity kernel (:mod:`agent_evolve.core.issues`), where an empty or
  dimension-incompatible embedding forces a recorded fallback and the joint
  quality/diversity objective degrades to quality-only ordering.

Per ``docs/architecture/selection-algorithms.md:276-280`` and
``docs/architecture/orchestration-lifecycle.md:123``, a provider outage may
degrade to the configured deterministic lexical fallback **only** when the
profile permits it, and the fallback reason must be recorded. Silent
substitution is forbidden, so :class:`OllamaEmbedder` raises
:class:`EmbeddingProviderUnavailable` rather than returning a wrong-shape or
zero-filled vector, and :class:`FallbackEmbedder` records
``fallback_reason`` whenever it degrades.

This module performs no imports of any agent runtime; the HTTP transport is
injectable so every unit test runs offline.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Mapping, Protocol

from agent_evolve.core.clustering import (
    EmbeddingProviderUnavailable,
    LexicalEmbedder,
    MechanismEmbedder,
)
from agent_evolve.core.config import EmbeddingConfig

# ``embeddinggemma`` emits 768-dimensional, already L2-normalized vectors.
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_TIMEOUT_SECONDS = 30.0

_KNOWN_PROVIDERS = frozenset({"ollama", "lexical"})
_KNOWN_FALLBACKS = frozenset({"lexical", "none"})


class Transport(Protocol):
    """A POST-JSON-return-JSON callable, injected so tests stay offline."""

    def __call__(
        self, url: str, payload: Mapping[str, object], timeout: float
    ) -> Mapping[str, object]: ...


def _urllib_transport(
    url: str, payload: Mapping[str, object], timeout: float
) -> Mapping[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class OllamaEmbedder:
    """Ollama ``/api/embed`` embedder implementing :class:`MechanismEmbedder`.

    Vectors are cached per text so repeated mechanism strings inside one
    selection pass cost a single request. Any transport error, malformed
    response, or unexpected dimension raises
    :class:`EmbeddingProviderUnavailable` so the caller can apply the
    profile-permitted fallback and record the reason.
    """

    def __init__(
        self,
        url: str,
        model: str,
        dim: int = DEFAULT_EMBEDDING_DIM,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Transport | None = None,
    ) -> None:
        if not url:
            raise ValueError("url is required")
        if not model:
            raise ValueError("model is required")
        self.dim = _require_positive_int("dim", dim)
        if timeout <= 0.0:
            raise ValueError("timeout must be > 0")
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self._transport: Transport = transport or _urllib_transport
        self._cache: dict[str, tuple[float, ...]] = {}

    @property
    def endpoint(self) -> str:
        return f"{self.url}/api/embed"

    def embed(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            # Blank mechanism text carries no signal; a zero vector is honest
            # and costs no request. The DPP path treats it as low similarity.
            return tuple(0.0 for _ in range(self.dim))
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = self._fetch(text)
        self._cache[text] = vector
        return vector

    def _fetch(self, text: str) -> tuple[float, ...]:
        payload = {"model": self.model, "input": text}
        try:
            response = self._transport(self.endpoint, payload, self.timeout)
        except EmbeddingProviderUnavailable:
            raise
        except Exception as exc:  # transport, socket, decode, HTTP status
            raise EmbeddingProviderUnavailable(
                f"ollama embedding request failed: {type(exc).__name__}"
            ) from exc
        return self._parse(response)

    def _parse(self, response: Mapping[str, object]) -> tuple[float, ...]:
        raw = self._extract_vector(response)
        try:
            vector = tuple(
                float(value)  # type: ignore[arg-type]
                for value in raw
            )
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderUnavailable(
                "ollama embedding response contained a non-numeric component"
            ) from exc
        if len(vector) != self.dim:
            raise EmbeddingProviderUnavailable(
                f"ollama embedding dimension mismatch: expected {self.dim}, "
                f"got {len(vector)}"
            )
        return vector

    def _extract_vector(self, response: Mapping[str, object]) -> list[object]:
        if not isinstance(response, Mapping):
            raise EmbeddingProviderUnavailable(
                "ollama embedding response was not a JSON object"
            )
        batch = response.get("embeddings")
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise EmbeddingProviderUnavailable(
                    "ollama embedding response carried an empty batch"
                )
            first = batch[0]
            if not isinstance(first, (list, tuple)) or not first:
                raise EmbeddingProviderUnavailable(
                    "ollama embedding response carried an empty vector"
                )
            return list(first)
        single = response.get("embedding")
        if isinstance(single, (list, tuple)) and single:
            return list(single)
        raise EmbeddingProviderUnavailable(
            "ollama embedding response carried no usable embedding"
        )


class FallbackEmbedder:
    """Primary embedder with a recorded, profile-permitted lexical fallback.

    ``fallback_reason`` is ``None`` until the primary provider reports
    unavailability; from then on it records why the run degraded so the caller
    can persist it in the manifest. The fallback must share the primary's
    dimension, otherwise the DPP kernel would silently switch widths mid-run.
    """

    def __init__(
        self, primary: MechanismEmbedder, fallback: MechanismEmbedder
    ) -> None:
        if primary.dim != fallback.dim:
            raise ValueError(
                "fallback embedder dimension must match the primary "
                f"({primary.dim} != {fallback.dim})"
            )
        self.primary = primary
        self.fallback = fallback
        self.dim = primary.dim
        self.fallback_reason: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.fallback_reason is not None

    def embed(self, text: str) -> tuple[float, ...]:
        try:
            return self.primary.embed(text)
        except EmbeddingProviderUnavailable:
            self.fallback_reason = "provider_unavailable"
            return self.fallback.embed(text)


def build_embedder(
    config: EmbeddingConfig,
    dim: int = DEFAULT_EMBEDDING_DIM,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    transport: Transport | None = None,
) -> MechanismEmbedder:
    """Build the embedder described by ``config``.

    ``provider="lexical"`` returns the deterministic offline embedder directly.
    ``provider="ollama"`` returns a :class:`FallbackEmbedder` when the config
    declares the ``lexical`` fallback, and a bare :class:`OllamaEmbedder` when
    the config declares ``none`` (fail loud instead of degrading).
    """
    if config.provider not in _KNOWN_PROVIDERS:
        raise ValueError(
            f"unknown embedding provider: {config.provider!r} "
            f"(known: {sorted(_KNOWN_PROVIDERS)})"
        )
    if config.fallback not in _KNOWN_FALLBACKS:
        raise ValueError(
            f"unknown embedding fallback: {config.fallback!r} "
            f"(known: {sorted(_KNOWN_FALLBACKS)})"
        )
    dim = _require_positive_int("dim", dim)
    if config.provider == "lexical":
        return LexicalEmbedder(dim=dim)
    primary = OllamaEmbedder(
        url=config.url,
        model=config.model,
        dim=dim,
        timeout=timeout,
        transport=transport,
    )
    if config.fallback == "none":
        return primary
    return FallbackEmbedder(primary=primary, fallback=LexicalEmbedder(dim=dim))
