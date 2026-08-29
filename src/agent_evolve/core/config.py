"""Resolved research configuration, feature gates, and budget accounting."""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_evolve.core.clustering import DEFAULT_BAND_HIGH, DEFAULT_BAND_LOW
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
    #: D5/J2B: attach the positivity judge (Judge 2) on the live path. Off by
    #: default: it costs one model call per passing rollout, so a run is
    #: byte-identical to today until the operator opts in.
    use_positivity_judge: bool = False


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    url: str
    model: str
    provider: str
    fallback: str = "lexical"


#: Ambiguity band defaults for the mechanism-dedup adjudicator. Module-level
#: constants rather than class-attribute reads: ``slots=True`` turns a dataclass
#: field into a ``member_descriptor``, so ``MechanismDedupConfig.band_low`` is a
#: descriptor object, not the number.
#:
#: **Re-exported, not redefined.** The band is one policy decision, and it was
#: previously written out at four independent sites -- here, on
#: ``MechanismClusterer``, on ``ClusterRegistry`` and in ``pipeline`` -- which
#: can drift apart silently, because a wrong band produces a plausible-looking
#: clustering rather than an error. ``core.clustering`` owns the numbers because
#: it owns the decision that consumes them; see :data:`DEFAULT_BAND_LOW` there
#: for the live calibration behind these values. Core-to-core import only.
_DEFAULT_DEDUP_BAND_LOW = DEFAULT_BAND_LOW
_DEFAULT_DEDUP_BAND_HIGH = DEFAULT_BAND_HIGH


@dataclass(frozen=True, slots=True)
class MechanismDedupConfig:
    """The small model that adjudicates ambiguous mechanism-cluster merges.

    Deliberately separate from every other model role. Embedding cosine decides
    the clear cases for free; this model is consulted only where cosine is
    measurably unreliable -- inside a similarity band around the join threshold,
    and on a forced merge at the cluster cap. Keeping its endpoint, model id and
    key independent of the rollout/analyzer/judge/editor roles is what makes it
    affordable to run a *small* model here regardless of how large those are.

    ``enabled=False`` is the default, so the adjudicator never fires unless it
    was configured on purpose: an unconfigured deployment keeps exactly today's
    cosine-only behaviour rather than silently acquiring a model dependency.

    ``band_low``/``band_high`` bracket the ambiguous region in cosine similarity.
    A pair below ``band_low`` is confidently distinct and a pair at or above
    ``band_high`` is confidently the same mechanism; only the span between them
    is worth a model call.
    """

    url: str = ""
    model: str = ""
    api_key: str = ""
    enabled: bool = False
    band_low: float = _DEFAULT_DEDUP_BAND_LOW
    band_high: float = _DEFAULT_DEDUP_BAND_HIGH

    def __post_init__(self) -> None:
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
        if self.enabled and not self.model:
            raise ValueError(
                "mechanism dedup is enabled but no model was configured; "
                "refusing to enable an adjudicator that cannot be called"
            )
        if self.enabled and not self.url:
            raise ValueError(
                "mechanism dedup is enabled but no endpoint was configured"
            )


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
    mechanism_dedup: MechanismDedupConfig = field(
        default_factory=MechanismDedupConfig
    )
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
    #: Ablation switch for the RHO pairwise acceptance gate (SV-4).
    #:
    #: ``False`` (default) is *paper* behaviour: a candidate is eligible to be
    #: champion only when its symmetric pairwise preference ``S_j > 0``, per RHO
    #: Algorithm 1. ``True`` disables the gate and restores ranking by the
    #: grader aggregate alone.
    #:
    #: The default is deliberately ``False``. The gate reflects the published
    #: algorithm and the judge is already paid for on every round; defaulting to
    #: ``True`` would mean an ordinary run silently discards the preference
    #: evidence it just bought, which is the SV-4 defect. The flag exists so an
    #: ablation can *measure* the gate's contribution, not so the gate is opt-in.
    experimental_candidate_promotion: bool = False
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
                "use_positivity_judge": self.features.use_positivity_judge,
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
            # ``api_key`` is deliberately absent: this mapping is written to run
            # manifests and logs, and a credential must never be persisted.
            "mechanism_dedup": {
                "url": self.mechanism_dedup.url,
                "model": self.mechanism_dedup.model,
                "enabled": self.mechanism_dedup.enabled,
                "band_low": self.mechanism_dedup.band_low,
                "band_high": self.mechanism_dedup.band_high,
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
            # Recorded so a run manifest states whether the paper gate was
            # active. Without it a champion.json from an ablation arm is
            # indistinguishable from one produced under the paper algorithm.
            "experimental_candidate_promotion": self.experimental_candidate_promotion,
            "seed": self.seed,
            "deferred_features": list(self.deferred_features),
            "log_capture": self.log_capture.manifest_payload(),
        }


_DEFAULT_EMBEDDING_URL = "http://localhost:11434"
_DEFAULT_EMBEDDING_MODEL = "embeddinggemma"

_PROFILES: dict[str, dict] = {
    "minimal": {
        "gates": (False, False, False, False, False, False),
        "deferred": (),
    },
    "research_sequential": {
        "gates": (True, True, True, False, False, False),
        "deferred": (),
    },
    "research_parallel": {
        "gates": (True, True, True, True, False, False),
        "deferred": ("parallel_execution",),
    },
    "full_ablation": {
        "gates": (True, True, True, True, False, False),
        "deferred": ("parallel_execution",),
    },
}

_GATE_ORDER = (
    "use_causal_blame",
    "use_edit_memory",
    "use_focused_validation",
    "use_entropy_selection",
    "parallel_execution",
    "use_positivity_judge",
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
    "experimental_candidate_promotion",
    # Composite members. ``budgets`` in particular must be overridable or a
    # caller has no way to cap spend: every BudgetLimits field defaults to None
    # (unlimited), so a run without an override is an uncapped run.
    "budgets",
    "embedding",
    "features",
    "log_capture",
}


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    """Read a float from the environment, or fail loudly.

    A malformed value raises rather than falling back to the default: silently
    ignoring ``BAND_LOW=hgh`` would run the whole session on a threshold the
    operator did not choose and believes they set.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{key} must be a number, got {raw!r}"
        ) from exc


def resolve_profile(
    name: str,
    environ: Mapping[str, str] | None = None,
    *,
    seed: int = 0,
    **overrides: Any,
) -> ResolvedConfig:
    """Resolve a named profile into a concrete configuration.

    ``environ`` defaults to :data:`os.environ`. It previously defaulted to an
    empty dict, which made **every** ``env.get(...)`` below dead in production:
    neither stack builder passes the argument, so nothing an operator exported
    was ever read. The failure was silent because each var has a default, so the
    config resolved successfully with default values and nothing raised --
    ``AE_MECHANISM_DEDUP_*`` could never enable the adjudicator, and the Ollama
    endpoint appeared to work only because its default happens to match a stock
    local install.

    Passing an explicit mapping (including ``{}``) still wins, so tests stay
    deterministic and ambient shell state cannot leak into an offline run.
    """
    profile = _PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown profile: {name!r}")

    env = environ if environ is not None else os.environ

    features = FeatureGates(*profile["gates"])
    embedding = EmbeddingConfig(
        url=env.get("OLLAMA_EMBEDDING_URL", _DEFAULT_EMBEDDING_URL),
        model=env.get("OLLAMA_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
        provider="ollama",
    )
    # The mechanism-dedup adjudicator is opt-in and independently addressed, so a
    # small cheap model can serve it whatever the other roles use. Absent env
    # vars leave it disabled, which preserves today's cosine-only behaviour.
    dedup_model = env.get("AE_MECHANISM_DEDUP_MODEL", "")
    dedup_url = env.get("AE_MECHANISM_DEDUP_BASE_URL", "")
    mechanism_dedup = MechanismDedupConfig(
        url=dedup_url,
        model=dedup_model,
        api_key=env.get("AE_MECHANISM_DEDUP_API_KEY", ""),
        enabled=bool(dedup_model and dedup_url),
        band_low=_env_float(
            env, "AE_MECHANISM_DEDUP_BAND_LOW", _DEFAULT_DEDUP_BAND_LOW
        ),
        band_high=_env_float(
            env, "AE_MECHANISM_DEDUP_BAND_HIGH", _DEFAULT_DEDUP_BAND_HIGH
        ),
    )

    kwargs: dict[str, Any] = {
        "profile_name": name,
        "features": features,
        "budgets": BudgetLimits(),
        "embedding": embedding,
        "mechanism_dedup": mechanism_dedup,
        "seed": seed,
        "deferred_features": profile["deferred"],
    }
    for key, value in overrides.items():
        if key not in _VALID_OVERRIDES:
            raise ValueError(f"unknown config override: {key!r}")
        kwargs[key] = value

    return ResolvedConfig(**kwargs)
