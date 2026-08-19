"""Resolved research configuration, feature gates, and budget accounting."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_evolve.core.errors import BudgetExceededError
from agent_evolve.core.run_logging import LogCaptureConfig

ProfileName = Literal["minimal", "research_sequential", "research_parallel", "full_ablation"]


@dataclass(frozen=True, slots=True)
class FeatureGates:
    use_causal_blame: bool = False
    use_edit_memory: bool = False
    use_focused_validation: bool = False
    use_entropy_selection: bool = False
    parallel_execution: bool = False


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    url: str
    model: str
    provider: str
    fallback: str = "lexical"


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_attempts: int | None = None
    max_accepted_edits: int | None = None
    max_model_tokens: int | None = None
    max_rollouts: int | None = None
    max_judge_verdicts: int | None = None
    max_editor_calls: int | None = None
    edit_max_retries: int = 3
    max_wall_seconds: float | None = None
    max_pool_candidates: int | None = None
    max_history_records: int | None = None
    max_rag_context_tokens: int | None = None


@dataclass(slots=True)
class BudgetUsage:
    rollouts: int = 0
    analyzer_judge_calls: int = 0
    editor_calls: int = 0
    validation_calls: int = 0
    embedding_calls: int = 0
    model_tokens: int = 0
    attempts: int = 0
    accepted_edits: int = 0

    def reserve(self, limits: BudgetLimits, **increments: int) -> None:
        limit_fields = {
            "rollouts": "max_rollouts",
            "analyzer_judge_calls": "max_judge_verdicts",
            "editor_calls": "max_editor_calls",
            "model_tokens": "max_model_tokens",
            "attempts": "max_attempts",
            "accepted_edits": "max_accepted_edits",
        }
        for field, increment in increments.items():
            if increment < 0:
                raise ValueError("budget increments must be non-negative")
            if field not in limit_fields:
                continue
            limit = getattr(limits, limit_fields[field])
            if limit is not None and getattr(self, field) + increment > limit:
                raise BudgetExceededError(f"{field} budget exceeded")
        for field, increment in increments.items():
            setattr(self, field, getattr(self, field) + increment)


_FLOAT_UNIT_FIELDS = (
    "dpp_theta",
    "dpp_score_floor",
    "dpp_min_gain",
    "entropy_score_floor",
    "entropy_recombination_score_threshold",
    "entropy_frontier_weight",
    "cluster_similarity_threshold",
    "probe_budget_fraction",
    "champion_alpha",
    "champion_beta",
    "champion_gamma",
    "champion_delta",
    "champion_min_coverage_fraction",
)

_POSITIVE_INT_FIELDS = (
    "dpp_max_items",
    "entropy_min_comparable_candidates",
    "entropy_min_rollouts_per_candidate",
    "max_clusters_per_task",
    "max_analyzer_workers",
)


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    profile_name: ProfileName
    features: FeatureGates
    budgets: BudgetLimits
    embedding: EmbeddingConfig
    dpp_max_items: int = 100
    dpp_theta: float = 0.7
    dpp_score_floor: float = 0.1
    dpp_min_gain: float = 1e-12
    entropy_refresh_mode: Literal["outer_iteration", "accepted_edits", "pool_growth"] = "outer_iteration"
    entropy_score_floor: float = 0.15
    entropy_recombination_score_threshold: float = 0.30
    entropy_frontier_weight: float = 0.30
    entropy_min_comparable_candidates: int = 3
    entropy_min_rollouts_per_candidate: int = 2
    cluster_similarity_threshold: float = 0.80
    max_clusters_per_task: int = 12
    # Bounded fan-out for trajectory analysis. Analyzing distinct
    # (candidate, task) rollout groups is independent, read-only work, so it
    # parallelizes; analysis is LLM-latency-bound. 1 means inline/sequential.
    # This bounds analyzer concurrency only -- never artifact writes.
    max_analyzer_workers: int = 1
    generalization_probe_mode: Literal["deferred", "enabled"] = "deferred"
    probe_budget_fraction: float = 0.15
    champion_alpha: float = 0.55
    champion_beta: float = 0.20
    champion_gamma: float = 0.15
    champion_delta: float = 0.10
    champion_min_coverage_fraction: float = 0.0
    seed: int = 0
    deferred_features: tuple[str, ...] = ()
    # Off by default: capture is opt-in so a measurement run writes nothing
    # unless asked. Agent-neutral -- ``run_logging`` imports no adapter.
    log_capture: LogCaptureConfig = field(default_factory=LogCaptureConfig)

    def __post_init__(self) -> None:
        for name in _FLOAT_UNIT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
            if not math.isfinite(float(value)) or not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        for name in _POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.log_capture, LogCaptureConfig):
            raise ValueError("log_capture must be a LogCaptureConfig")

    def manifest_payload(self) -> dict:
        return {
            "profile_name": self.profile_name,
            "features": {
                "use_causal_blame": self.features.use_causal_blame,
                "use_edit_memory": self.features.use_edit_memory,
                "use_focused_validation": self.features.use_focused_validation,
                "use_entropy_selection": self.features.use_entropy_selection,
                "parallel_execution": self.features.parallel_execution,
            },
            "budgets": {
                "max_attempts": self.budgets.max_attempts,
                "max_accepted_edits": self.budgets.max_accepted_edits,
                "max_model_tokens": self.budgets.max_model_tokens,
                "max_rollouts": self.budgets.max_rollouts,
                "max_judge_verdicts": self.budgets.max_judge_verdicts,
                "max_editor_calls": self.budgets.max_editor_calls,
                "edit_max_retries": self.budgets.edit_max_retries,
                "max_wall_seconds": self.budgets.max_wall_seconds,
                "max_pool_candidates": self.budgets.max_pool_candidates,
                "max_history_records": self.budgets.max_history_records,
                "max_rag_context_tokens": self.budgets.max_rag_context_tokens,
            },
            "embedding": {
                "url": self.embedding.url,
                "model": self.embedding.model,
                "provider": self.embedding.provider,
                "fallback": self.embedding.fallback,
            },
            "dpp_max_items": self.dpp_max_items,
            "dpp_theta": self.dpp_theta,
            "dpp_score_floor": self.dpp_score_floor,
            "dpp_min_gain": self.dpp_min_gain,
            "entropy_refresh_mode": self.entropy_refresh_mode,
            "entropy_score_floor": self.entropy_score_floor,
            "entropy_recombination_score_threshold": self.entropy_recombination_score_threshold,
            "entropy_frontier_weight": self.entropy_frontier_weight,
            "entropy_min_comparable_candidates": self.entropy_min_comparable_candidates,
            "entropy_min_rollouts_per_candidate": self.entropy_min_rollouts_per_candidate,
            "cluster_similarity_threshold": self.cluster_similarity_threshold,
            "max_clusters_per_task": self.max_clusters_per_task,
            "max_analyzer_workers": self.max_analyzer_workers,
            "generalization_probe_mode": self.generalization_probe_mode,
            "probe_budget_fraction": self.probe_budget_fraction,
            "champion_alpha": self.champion_alpha,
            "champion_beta": self.champion_beta,
            "champion_gamma": self.champion_gamma,
            "champion_delta": self.champion_delta,
            "champion_min_coverage_fraction": self.champion_min_coverage_fraction,
            "seed": self.seed,
            "deferred_features": list(self.deferred_features),
            "log_capture": self.log_capture.manifest_payload(),
        }


_DEFAULT_EMBEDDING_URL = "http://localhost:11434"
_DEFAULT_EMBEDDING_MODEL = "embeddinggemma"

_PROFILES: dict[str, dict] = {
    "minimal": {
        "gates": (False, False, False, False, False),
        "deferred": (),
    },
    "research_sequential": {
        "gates": (True, True, True, False, False),
        "deferred": (),
    },
    "research_parallel": {
        "gates": (True, True, True, True, False),
        "deferred": ("parallel_execution",),
    },
    "full_ablation": {
        "gates": (True, True, True, True, False),
        "deferred": ("parallel_execution",),
    },
}

_GATE_ORDER = (
    "use_causal_blame",
    "use_edit_memory",
    "use_focused_validation",
    "use_entropy_selection",
    "parallel_execution",
)

#: ``{profile_name: {gate_name: bool}}``, derived from :data:`_PROFILES` so the
#: two cannot drift. Public because a per-gate ablation has to start from the
#: profile's own bundle and move exactly one gate.
PROFILE_GATES: dict[str, dict[str, bool]] = {
    name: dict(zip(_GATE_ORDER, spec["gates"], strict=True))
    for name, spec in _PROFILES.items()
}

_VALID_OVERRIDES = {
    "dpp_max_items",
    "dpp_theta",
    "dpp_score_floor",
    "dpp_min_gain",
    "entropy_refresh_mode",
    "entropy_score_floor",
    "entropy_recombination_score_threshold",
    "entropy_frontier_weight",
    "entropy_min_comparable_candidates",
    "entropy_min_rollouts_per_candidate",
    "cluster_similarity_threshold",
    "max_clusters_per_task",
    "max_analyzer_workers",
    "generalization_probe_mode",
    "probe_budget_fraction",
    "champion_alpha",
    "champion_beta",
    "champion_gamma",
    "champion_delta",
    "champion_min_coverage_fraction",
    # Composite members. ``budgets`` in particular must be overridable or a
    # caller has no way to cap spend: every BudgetLimits field defaults to None
    # (unlimited), so a run without an override is an uncapped run.
    "budgets",
    "embedding",
    "features",
    "log_capture",
}


def resolve_profile(
    name: str,
    environ: Mapping[str, str] | None = None,
    *,
    seed: int = 0,
    **overrides: Any,
) -> ResolvedConfig:
    profile = _PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown profile: {name!r}")

    env = environ if environ is not None else {}

    features = FeatureGates(*profile["gates"])
    embedding = EmbeddingConfig(
        url=env.get("OLLAMA_EMBEDDING_URL", _DEFAULT_EMBEDDING_URL),
        model=env.get("OLLAMA_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
        provider="ollama",
    )

    kwargs: dict[str, Any] = {
        "profile_name": name,
        "features": features,
        "budgets": BudgetLimits(),
        "embedding": embedding,
        "seed": seed,
        "deferred_features": profile["deferred"],
    }
    for key, value in overrides.items():
        if key not in _VALID_OVERRIDES:
            raise ValueError(f"unknown config override: {key!r}")
        kwargs[key] = value

    return ResolvedConfig(**kwargs)
