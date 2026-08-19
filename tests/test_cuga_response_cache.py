"""Upstream response-cache control for rollout sampling.

An identical prompt repeated N times can be served from the gateway's response
cache, returning ONE observation N times while every counter reports N rollouts.
Verified live on 2026-08-18 against both ``azure/gpt-5.6-luna`` and
``gcp/gemini-3.6-flash``: four sequential identical requests shared a single
response ``id`` and identical text; with the cache disabled the same four
requests produced four distinct ids and four distinct completions.

This matters for evolution specifically because rollout diversity IS the
evidence: RHO's ``G`` group rollouts and the genetic path's ``R`` repeats both
exist to sample variance for a fixed harness. Cached repeats make that variance
structurally zero, inflate confidence in whatever single sample was drawn, and
understate the measured noise floor.

Only one injection form works, and the two plausible alternatives fail
*silently* -- see ``reference/cuga_example_wrapper/docs/cuga-integration-learnings.md``
("Sampling, Temperature, And The Upstream Response Cache"):

* ``extra_params={"caching": False}`` is dropped by the langchain wrapper, which
  is a pydantic model with no such field and no extras.
* A bare ``model_kwargs={"caching": False}`` reaches litellm but is consumed as
  litellm's own client-side setting and never forwarded upstream.
* ``model_kwargs={"extra_body": {...}}`` travels in the HTTP body, which the
  gateway reads. This is the only form that defeats the cache.

These tests pin the payload shape and the no-clobber contract offline. They do
not assert live diversity; that is verified by a live run.
"""
from __future__ import annotations

from agent_evolve.cuga_wrapper import (
    CACHE_BYPASS_EXTRA_BODY,
    apply_response_cache_policy,
    install_response_cache_policy,
    response_cache_disabled,
)


class _FakeModel:
    """Stands in for a constructed langchain chat client."""

    def __init__(self, model_kwargs: dict | None = None) -> None:
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}


def test_cache_bypass_lands_in_extra_body_not_as_a_bare_kwarg() -> None:
    """The flag must be nested under ``extra_body``.

    A bare ``caching`` kwarg is accepted by litellm and then consumed locally,
    so it reaches the wire and still returns cached text. Asserting the nesting
    is what separates the working form from the silently-useless one.
    """
    model = _FakeModel()

    apply_response_cache_policy(model, disable_cache=True)

    assert model.model_kwargs["extra_body"] == CACHE_BYPASS_EXTRA_BODY
    assert "caching" not in model.model_kwargs


def test_disabled_policy_leaves_the_model_untouched() -> None:
    """Cache control must be opt-in so a default run is unchanged."""
    model = _FakeModel()

    apply_response_cache_policy(model, disable_cache=False)

    assert model.model_kwargs == {}


def test_existing_extra_body_keys_are_preserved() -> None:
    """Merging, not replacing.

    ``extra_body`` is a shared passthrough channel; clobbering it would silently
    drop an unrelated provider parameter set by another layer.
    """
    model = _FakeModel({"extra_body": {"custom_flag": 7}})

    apply_response_cache_policy(model, disable_cache=True)

    assert model.model_kwargs["extra_body"]["custom_flag"] == 7
    for key, value in CACHE_BYPASS_EXTRA_BODY.items():
        assert model.model_kwargs["extra_body"][key] == value


def test_unrelated_model_kwargs_are_preserved() -> None:
    """Sampling kwargs set elsewhere must survive."""
    model = _FakeModel({"top_p": 0.9})

    apply_response_cache_policy(model, disable_cache=True)

    assert model.model_kwargs["top_p"] == 0.9
    assert model.model_kwargs["extra_body"] == CACHE_BYPASS_EXTRA_BODY


def test_missing_model_kwargs_attribute_is_tolerated() -> None:
    """Never abort a rollout over cache control.

    ``model_kwargs`` is a langchain implementation detail. If a future client
    lacks it, degrading to a cached (less diverse) rollout is strictly better
    than crashing the run.
    """

    class _NoKwargs:
        pass

    model = _NoKwargs()

    apply_response_cache_policy(model, disable_cache=True)  # must not raise


def test_none_model_kwargs_is_initialized() -> None:
    """CUGA's own updater treats ``model_kwargs=None`` as a valid state."""

    class _NoneKwargs:
        model_kwargs: dict | None = None

    model = _NoneKwargs()

    apply_response_cache_policy(model, disable_cache=True)

    assert model.model_kwargs is not None
    assert model.model_kwargs["extra_body"] == CACHE_BYPASS_EXTRA_BODY


def test_repeated_application_is_idempotent() -> None:
    """``get_model`` is called per agent role, so this runs many times per run."""
    model = _FakeModel()

    apply_response_cache_policy(model, disable_cache=True)
    apply_response_cache_policy(model, disable_cache=True)

    assert model.model_kwargs["extra_body"] == CACHE_BYPASS_EXTRA_BODY


def test_install_patches_every_get_model_path() -> None:
    """The policy must reach LLMs created for EVERY agent role.

    ``LLMManager`` is a process-wide singleton with its own model cache, and
    ``get_model`` returns clients through three distinct paths (pre-instantiated,
    cached, freshly created). All three funnel through
    ``_update_model_parameters``, which is therefore the only choke point that
    covers a whole run. Patching ``_create_llm_instance`` instead would miss
    every cache hit -- i.e. most calls in a multi-role run.
    """

    class _FakeManager:
        """Mimics the singleton's relevant surface."""

        def _update_model_parameters(self, model, **kwargs):
            return model

    manager = _FakeManager()

    install_response_cache_policy(manager, disable_cache=True)

    assert "_update_model_parameters" in vars(manager)
    model = _FakeModel()
    returned = manager._update_model_parameters(model, temperature=0.1)
    assert returned is model
    assert model.model_kwargs["extra_body"] == CACHE_BYPASS_EXTRA_BODY


def test_install_is_a_noop_when_cache_control_is_disabled() -> None:
    """Default runs must keep CUGA's own behavior byte for byte."""

    class _FakeManager:
        def _update_model_parameters(self, model, **kwargs):
            return model

    manager = _FakeManager()

    install_response_cache_policy(manager, disable_cache=False)

    # A bound method is rebuilt on every attribute access, so identity cannot be
    # compared directly. Absence of an instance override is the real assertion.
    assert "_update_model_parameters" not in vars(manager)
    model = _FakeModel()
    manager._update_model_parameters(model, temperature=0.1)
    assert model.model_kwargs == {}


def test_install_does_not_double_wrap() -> None:
    """``from_settings`` may run more than once per process."""

    class _FakeManager:
        def _update_model_parameters(self, model, **kwargs):
            return model

    manager = _FakeManager()
    install_response_cache_policy(manager, disable_cache=True)
    once = vars(manager)["_update_model_parameters"]
    install_response_cache_policy(manager, disable_cache=True)

    assert vars(manager)["_update_model_parameters"] is once


def test_env_var_opts_out_of_cache_control(monkeypatch) -> None:
    """Rollout workers are separate processes, so the switch must cross a fork.

    ``--isolation process`` runs every rollout in a child that builds its own
    wrapper; a constructor argument in the parent cannot reach it. The env var is
    the only channel both paths share, which is why it -- not the parameter --
    is what the CLI sets.
    """
    monkeypatch.setenv("AGENT_EVOLVE_ALLOW_RESPONSE_CACHE", "1")

    assert response_cache_disabled(default=True) is False


def test_env_var_absent_keeps_cache_disabled(monkeypatch) -> None:
    """Cache-free sampling is the safe default for evidence collection."""
    monkeypatch.delenv("AGENT_EVOLVE_ALLOW_RESPONSE_CACHE", raising=False)

    assert response_cache_disabled(default=True) is True


def test_env_var_falsey_values_are_ignored(monkeypatch) -> None:
    """``0``/``false``/empty must not silently re-enable caching."""
    for value in ("0", "false", "False", "", "no"):
        monkeypatch.setenv("AGENT_EVOLVE_ALLOW_RESPONSE_CACHE", value)
        assert response_cache_disabled(default=True) is True
