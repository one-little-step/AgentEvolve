"""Resolved research configuration and budget accounting tests."""
from __future__ import annotations

import json

import pytest

from agent_evolve.core.config import (
    PROFILE_GATES,
    BudgetLimits,
    BudgetUsage,
    EmbeddingConfig,
    FeatureGates,
    ResolvedConfig,
    resolve_profile,
)
from agent_evolve.core.errors import BudgetExceededError
from agent_evolve.core.run_logging import ALL_LOG_CHANNELS, LogCaptureConfig


def test_research_parallel_is_resolved_but_inactive_for_json_storage() -> None:
    config = resolve_profile("research_parallel", environ={})
    assert config.features.parallel_execution is False
    assert config.deferred_features == ("parallel_execution",)


def test_budget_refuses_operation_above_limit() -> None:
    limits = BudgetLimits(max_rollouts=1)
    usage = BudgetUsage(rollouts=1)
    with pytest.raises(BudgetExceededError):
        usage.reserve(limits, rollouts=1)


def test_embedding_config_reads_ollama_values_without_network_call() -> None:
    config = resolve_profile("minimal", environ={
        "OLLAMA_EMBEDDING_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "embeddinggemma",
    })
    assert config.embedding.url == "http://localhost:11434"
    assert config.embedding.model == "embeddinggemma"


def test_minimal_profile_gates_all_false() -> None:
    config = resolve_profile("minimal", environ={})
    assert config.features == FeatureGates()
    assert config.deferred_features == ()
    assert config.profile_name == "minimal"


def test_research_sequential_profile_gates() -> None:
    config = resolve_profile("research_sequential", environ={})
    assert config.features.use_causal_blame is True
    assert config.features.use_edit_memory is True
    assert config.features.use_focused_validation is True
    assert config.features.use_entropy_selection is False
    assert config.features.parallel_execution is False
    assert config.deferred_features == ()


def test_research_parallel_profile_gates() -> None:
    config = resolve_profile("research_parallel", environ={})
    assert config.features.use_causal_blame is True
    assert config.features.use_edit_memory is True
    assert config.features.use_focused_validation is True
    assert config.features.use_entropy_selection is True
    assert config.features.parallel_execution is False


def test_full_ablation_profile_gates() -> None:
    config = resolve_profile("full_ablation", environ={})
    assert config.features.use_causal_blame is True
    assert config.features.use_edit_memory is True
    assert config.features.use_focused_validation is True
    assert config.features.use_entropy_selection is True
    assert config.features.parallel_execution is False
    assert config.deferred_features == ("parallel_execution",)


def test_positivity_judge_gate_defaults_off_in_every_profile() -> None:
    """S1.2: Judge 2 is opt-in spend. Off by default means a live run is
    byte-identical to today until the operator turns it on."""
    for name in PROFILE_GATES:
        config = resolve_profile(name, environ={})
        assert config.features.use_positivity_judge is False, name


def test_positivity_judge_gate_is_overridable() -> None:
    config = resolve_profile(
        "research_sequential",
        environ={},
        features=FeatureGates(**{**PROFILE_GATES["research_sequential"],
                                 "use_positivity_judge": True}),
    )
    assert config.features.use_positivity_judge is True


def test_positivity_judge_gate_in_manifest() -> None:
    config = resolve_profile("research_sequential", environ={})
    payload = config.manifest_payload()
    assert payload["features"]["use_positivity_judge"] is False


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_profile("not_a_profile", environ={})


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_profile("minimal", environ={}, dpp_theta_typo=0.5)


def test_overrides_apply_to_resolved_config() -> None:
    config = resolve_profile("minimal", environ={}, seed=42, dpp_theta=0.5)
    assert config.seed == 42
    assert config.dpp_theta == 0.5


def test_overrides_reject_identity_fields() -> None:
    """``profile_name`` and ``deferred_features`` stay non-overridable.

    They are the run's identity, not tuning: letting a caller pass
    ``profile_name="full_ablation"`` while the gates came from ``minimal`` would
    stamp a manifest that misdescribes the run it recorded.
    """
    with pytest.raises(ValueError):
        resolve_profile("minimal", profile_name="full_ablation")
    with pytest.raises(ValueError):
        resolve_profile("minimal", deferred_features=("parallel_execution",))


def test_features_are_overridable_for_per_gate_ablation() -> None:
    """``features`` IS overridable, deliberately.

    An ablation study has to move exactly one gate while holding the rest at the
    profile's values. Previously the only lever was ``--profile``, which swaps
    all five gates at once, so "same profile, entropy selection off" was
    unreachable from the CLI. ``PROFILE_GATES`` exposes the profile's own bundle
    so a caller starts from it and changes one member.
    """
    base = PROFILE_GATES["research_sequential"]
    config = resolve_profile(
        "research_sequential",
        environ={},
        features=FeatureGates(**{**base, "use_entropy_selection": True}),
    )
    assert config.features.use_entropy_selection is True
    # The untouched gates still match the profile.
    assert config.features.use_causal_blame is base["use_causal_blame"]
    assert config.features.use_edit_memory is base["use_edit_memory"]
    # And the profile name still describes which profile was asked for.
    assert config.profile_name == "research_sequential"


def test_profile_gates_matches_every_profile() -> None:
    """``PROFILE_GATES`` is derived, so it cannot drift from the profile table."""
    for name in PROFILE_GATES:
        gates = resolve_profile(name, environ={}).features
        assert PROFILE_GATES[name] == {
            "use_causal_blame": gates.use_causal_blame,
            "use_edit_memory": gates.use_edit_memory,
            "use_focused_validation": gates.use_focused_validation,
            "use_entropy_selection": gates.use_entropy_selection,
            "parallel_execution": gates.parallel_execution,
            "use_positivity_judge": gates.use_positivity_judge,
        }


def test_budgets_are_overridable_so_a_run_can_be_capped() -> None:
    """Without this, no caller can cap spend.

    Every :class:`BudgetLimits` field defaults to ``None`` (unlimited) and
    ``resolve_profile`` hardcoded ``BudgetLimits()``, so a run had no reachable
    ceiling on rollouts, attempts or editor calls.
    """
    config = resolve_profile(
        "minimal",
        environ={},
        budgets=BudgetLimits(max_rollouts=50, max_attempts=4),
    )
    assert config.budgets.max_rollouts == 50
    assert config.budgets.max_attempts == 4
    assert resolve_profile("minimal", environ={}).budgets.max_rollouts is None


def test_embedding_defaults_when_environ_absent() -> None:
    config = resolve_profile("minimal", environ={})
    assert config.embedding.url == "http://localhost:11434"
    assert config.embedding.model == "embeddinggemma"
    assert config.embedding.provider == "ollama"
    assert config.embedding.fallback == "lexical"


def test_manifest_payload_is_json_safe() -> None:
    config = resolve_profile("research_parallel", environ={
        "OLLAMA_EMBEDDING_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "embeddinggemma",
    })
    payload = config.manifest_payload()

    json.dumps(payload)

    assert payload["profile_name"] == "research_parallel"
    assert payload["deferred_features"] == ["parallel_execution"]
    assert payload["embedding"]["url"] == "http://localhost:11434"
    assert payload["embedding"]["model"] == "embeddinggemma"
    assert payload["embedding"]["provider"] == "ollama"
    assert payload["embedding"]["fallback"] == "lexical"
    assert payload["seed"] == 0
    assert payload["dpp_theta"] == 0.7
    assert payload["champion_alpha"] == 0.55


def test_budget_rejects_negative_increment() -> None:
    limits = BudgetLimits(max_rollouts=10)
    usage = BudgetUsage()
    with pytest.raises(ValueError):
        usage.reserve(limits, rollouts=-1)


def test_budget_reserve_is_atomic_on_failure() -> None:
    limits = BudgetLimits(max_rollouts=2, max_model_tokens=5)
    usage = BudgetUsage(rollouts=1, model_tokens=3)
    with pytest.raises(BudgetExceededError):
        usage.reserve(limits, rollouts=1, model_tokens=3)
    assert usage.rollouts == 1
    assert usage.model_tokens == 3


def test_budget_reserve_applies_increments() -> None:
    limits = BudgetLimits(max_rollouts=10)
    usage = BudgetUsage()
    usage.reserve(limits, rollouts=3, analyzer_judge_calls=2)
    assert usage.rollouts == 3
    assert usage.analyzer_judge_calls == 2


def test_unlimited_field_has_no_ceiling() -> None:
    limits = BudgetLimits()
    usage = BudgetUsage()
    usage.reserve(limits, embedding_calls=1000)
    assert usage.embedding_calls == 1000


def test_resolved_config_rejects_out_of_range_theta() -> None:
    with pytest.raises(ValueError):
        ResolvedConfig(
            profile_name="minimal",
            features=FeatureGates(),
            budgets=BudgetLimits(),
            embedding=EmbeddingConfig(
                url="http://localhost:11434", model="embeddinggemma", provider="ollama",
            ),
            dpp_theta=1.5,
        )


def test_resolved_config_rejects_negative_integer() -> None:
    with pytest.raises(ValueError):
        ResolvedConfig(
            profile_name="minimal",
            features=FeatureGates(),
            budgets=BudgetLimits(),
            embedding=EmbeddingConfig(
                url="http://localhost:11434", model="embeddinggemma", provider="ollama",
            ),
            dpp_max_items=-1,
        )


def test_resolved_config_rejects_nan_champion_weight() -> None:
    with pytest.raises(ValueError):
        ResolvedConfig(
            profile_name="minimal",
            features=FeatureGates(),
            budgets=BudgetLimits(),
            embedding=EmbeddingConfig(
                url="http://localhost:11434", model="embeddinggemma", provider="ollama",
            ),
            champion_alpha=float("nan"),
        )


def test_log_capture_defaults_to_disabled() -> None:
    """Capture is opt-in: a measurement run must not silently start writing logs."""
    config = resolve_profile("minimal", environ={})
    assert config.log_capture.enabled is False


def test_log_capture_override_reaches_resolved_config(tmp_path) -> None:
    """An operator enabling capture at profile resolution is how a run turns it on."""
    capture = LogCaptureConfig(enabled=True, root=tmp_path)
    config = resolve_profile("minimal", environ={}, log_capture=capture)
    assert config.log_capture is capture
    assert config.log_capture.enabled is True
    assert config.log_capture.root == tmp_path


def test_manifest_payload_records_log_capture(tmp_path) -> None:
    """A run whose manifest omits its logging config cannot be reproduced."""
    config = resolve_profile(
        "minimal",
        environ={},
        log_capture=LogCaptureConfig(enabled=True, root=tmp_path),
    )
    payload = config.manifest_payload()

    json.dumps(payload)

    assert payload["log_capture"]["enabled"] is True
    assert payload["log_capture"]["root"] == str(tmp_path)
    assert payload["log_capture"]["channels"] == list(ALL_LOG_CHANNELS)


def test_resolved_config_rejects_non_log_capture_config() -> None:
    """A bare path or dict would silently disable capture instead of failing loudly."""
    with pytest.raises(ValueError):
        ResolvedConfig(
            profile_name="minimal",
            features=FeatureGates(),
            budgets=BudgetLimits(),
            embedding=EmbeddingConfig(
                url="http://localhost:11434", model="embeddinggemma", provider="ollama",
            ),
            log_capture={"enabled": True},
        )
