"""Environment variables must actually reach the resolved config.

``resolve_profile(name, environ=None)`` defaults ``environ`` to an **empty
dict**, not ``os.environ``. Every ``env.get(...)`` inside it therefore reads
nothing unless a caller passes a mapping explicitly -- and neither production
call site (``pipeline.py:1066`` in ``build_offline_stack``,
``pipeline.py:1272`` in ``build_live_stack``) does.

The consequence is silent rather than loud, which is why it survived: every var
has a default, so the config resolves successfully with default values and
nothing raises. Specifically:

* ``AE_MECHANISM_DEDUP_MODEL`` / ``_BASE_URL`` / ``_API_KEY`` default to ``""``,
  so ``enabled=bool(model and url)`` is always ``False`` and the dedup
  adjudicator can never switch on in production no matter what the operator
  exports. This is documented in ``docs/USER-MANUAL.md`` as a working feature.
* ``OLLAMA_EMBEDDING_URL`` / ``_MODEL`` default to ``http://localhost:11434``
  and ``embeddinggemma``. These *happen* to be right on a default local Ollama,
  which is why live embedding appeared to work -- it was the default, not the
  environment. An operator pointing at any other endpoint is silently ignored.

These tests pin the contract at the boundary that matters: the config a
production stack builder produces must reflect the process environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_evolve.core.config import resolve_profile  # noqa: E402


_DEDUP_ENV = {
    "AE_MECHANISM_DEDUP_MODEL": "openai/aws/gpt-oss-120b",
    "AE_MECHANISM_DEDUP_BASE_URL": "https://example.invalid",
    "AE_MECHANISM_DEDUP_API_KEY": "test-key-not-a-real-secret",
}


def test_resolve_profile_reads_the_process_environment_by_default():
    """Omitting ``environ`` must consult ``os.environ``, not an empty dict.

    Every caller in ``src/`` omits the argument, so if the default is an empty
    mapping then no environment variable in this function is ever honoured in
    production.
    """
    import os

    for key, value in _DEDUP_ENV.items():
        os.environ[key] = value
    try:
        config = resolve_profile("research_sequential", seed=0)
    finally:
        for key in _DEDUP_ENV:
            os.environ.pop(key, None)

    assert config.mechanism_dedup.enabled, (
        "AE_MECHANISM_DEDUP_* were exported yet dedup resolved disabled: "
        "resolve_profile defaulted environ to an empty dict, so no env var "
        "reaches the config in production"
    )
    assert config.mechanism_dedup.model == _DEDUP_ENV["AE_MECHANISM_DEDUP_MODEL"]
    assert config.mechanism_dedup.url == _DEDUP_ENV["AE_MECHANISM_DEDUP_BASE_URL"]


def test_explicit_environ_still_wins_over_the_process_environment():
    """An explicitly passed mapping must remain authoritative.

    The whole test suite passes ``environ={...}`` to get deterministic configs.
    Reading ``os.environ`` as a *default* must not let ambient state leak into a
    test that named its environment explicitly, or offline runs become dependent
    on the developer's shell.
    """
    import os

    os.environ["AE_MECHANISM_DEDUP_MODEL"] = "ambient-should-not-win"
    os.environ["AE_MECHANISM_DEDUP_BASE_URL"] = "https://ambient.invalid"
    try:
        config = resolve_profile("research_sequential", environ={}, seed=0)
    finally:
        os.environ.pop("AE_MECHANISM_DEDUP_MODEL", None)
        os.environ.pop("AE_MECHANISM_DEDUP_BASE_URL", None)

    assert not config.mechanism_dedup.enabled, (
        "an explicit empty environ must mean 'no env', otherwise ambient shell "
        "state leaks into offline runs and tests stop being deterministic"
    )
    assert config.mechanism_dedup.model == ""


def test_embedding_endpoint_is_not_silently_pinned_to_the_default():
    """An operator pointing at a non-default Ollama must be honoured.

    The defaults happen to match a stock local Ollama, so live embedding
    *appeared* to work while the environment was in fact being ignored. That
    coincidence is what hid the defect.
    """
    import os

    os.environ["OLLAMA_EMBEDDING_URL"] = "http://elsewhere.invalid:9999"
    os.environ["OLLAMA_EMBEDDING_MODEL"] = "some-other-embed-model"
    try:
        config = resolve_profile("research_sequential", seed=0)
    finally:
        os.environ.pop("OLLAMA_EMBEDDING_URL", None)
        os.environ.pop("OLLAMA_EMBEDDING_MODEL", None)

    assert config.embedding.url == "http://elsewhere.invalid:9999"
    assert config.embedding.model == "some-other-embed-model"


def test_dedup_api_key_still_absent_from_the_manifest():
    """Reading the environment must not start leaking the key into artifacts.

    Making env vars live means a real credential now reaches this object on
    every production run, so the manifest exclusion matters more than it did
    when the field was always empty.
    """
    import os

    for key, value in _DEDUP_ENV.items():
        os.environ[key] = value
    try:
        config = resolve_profile("research_sequential", seed=0)
        payload = config.manifest_payload()
    finally:
        for key in _DEDUP_ENV:
            os.environ.pop(key, None)

    rendered = repr(payload)
    assert _DEDUP_ENV["AE_MECHANISM_DEDUP_API_KEY"] not in rendered, (
        "the dedup api key reached the manifest payload"
    )
    assert "api_key" not in rendered
