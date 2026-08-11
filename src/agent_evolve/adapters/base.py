"""Runtime-checkable adapter validation helpers."""
from __future__ import annotations

from agent_evolve.core.contracts import EvolutionAdapter


REQUIRED_ADAPTER_METHODS = (
    "artifact_inventory",
    "read_artifacts",
    "materialize_candidate",
    "apply_structured_edits",
    "run_full_rollout",
    "capture_trace",
    "supports_counterfactual_replay",
    "discover_checkpoints",
    "replay_from_checkpoint",
)


def validate_adapter(adapter: EvolutionAdapter) -> None:
    """Fail early when a configured adapter omits a required capability."""
    missing = [name for name in REQUIRED_ADAPTER_METHODS if not callable(getattr(adapter, name, None))]
    if missing:
        raise TypeError(f"adapter is missing capabilities: {', '.join(missing)}")
    if not getattr(adapter, "adapter_name", ""):
        raise TypeError("adapter_name is required")
