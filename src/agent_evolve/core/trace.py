"""Agent-neutral persisted causal trace models and deterministic canonicalization.

This module intentionally contains no CUGA, LangChain, or adapter imports. It
defines the rich persisted :class:`CausalTrace` used by the wrapper for later
diagnosis, plus the conversion back into the minimal adapter-facing
:class:`~agent_evolve.core.contracts.ExecutionTrace`.
"""
from __future__ import annotations

import json
import math
from enum import Enum
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evolve.core.contracts import ExecutionTrace, TraceEvent


class PayloadLevel(str, Enum):
    """Persisted payload preservation policy for a causal trace."""

    STRUCTURAL = "structural"
    CAUSAL_SUFFICIENT = "causal_sufficient"
    RAW_OPT_IN = "raw_opt_in"


CaptureStatus = Literal[
    "captured",
    "disabled_by_config",
    "unavailable_no_sdk_surface",
    "unavailable_no_checkpointer",
    "runtime_failure",
]


class FacilityCapability(BaseModel):
    """Capability-honest report for one optional collection facility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: CaptureStatus
    reason: str | None = None


TraceCapabilities = Mapping[str, FacilityCapability]


def _normalize_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot canonicalize a non-finite float")
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return _normalize_json_value(value.model_dump())
    if hasattr(value, "__dict__"):
        return _normalize_json_value(vars(value))
    raise ValueError(f"cannot canonicalize value of type {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return a deterministic, sorted, compact JSON representation of ``value``.

    Rejects non-finite floats and values that cannot be reduced to JSON. Mapping
    keys are normalized to strings and sorted; sequence order is preserved.
    """
    normalized = _normalize_json_value(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CausalEvent(BaseModel):
    """One normalized persisted event in a causal trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    kind: str = Field(min_length=1)
    actor_id: str | None = None
    parent_event_id: str | None = None
    timestamp: str | None = None
    payload: Mapping[str, object] = Field(default_factory=dict)


class StateSnapshot(BaseModel):
    """A captured graph-state snapshot with an explicit replay-safety flag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=0)
    checkpoint_id: str | None = None
    state_hash: str | None = None
    payload: object = None
    replay_safe: bool = False


class ToolObservation(BaseModel):
    """One recorded tool invocation with deterministic replay eligibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=0)
    tool_name: str = Field(min_length=1)
    canonical_arguments: str = ""
    result: object = None
    truncated: bool = False
    original_bytes: int = Field(default=0, ge=0)
    retained_bytes: int = Field(default=0, ge=0)
    content_digest: str | None = None
    replay_eligible: bool = False
    withheld_reason: str | None = None
    error: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_replay_eligibility(self) -> "ToolObservation":
        if self.replay_eligible and (
            self.truncated or self.withheld_reason is not None or self.error is not None
        ):
            raise ValueError(
                "a replay_eligible observation must not be truncated, withheld, or errored"
            )
        return self


class CausalTrace(BaseModel):
    """Rich, agent-neutral persisted trace kept separately from the minimal adapter trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    thread_id: str | None = None
    thread_id_source: str | None = None
    harness_version: str = "unversioned"
    status: str = Field(min_length=1)
    final_output: str = ""
    model: str | None = None
    events: tuple[CausalEvent, ...] = ()
    checkpoints: tuple[StateSnapshot, ...] = ()
    tool_observations: tuple[ToolObservation, ...] = ()
    capabilities: dict[str, FacilityCapability] = Field(default_factory=dict)
    events_truncated: bool = False
    captured_event_count: int = Field(default=0, ge=0)
    dropped_event_count: int = Field(default=0, ge=0)
    started_at: str | None = None
    completed_at: str | None = None

    def to_execution_trace(self, *, candidate_id: str, trace_id: str) -> ExecutionTrace:
        """Convert persisted evidence into the minimal adapter-facing trace.

        Only replay-safe snapshots contribute checkpoint IDs; the conversion
        never fabricates checkpoint evidence from unreplayable state.
        """
        events = tuple(
            TraceEvent(
                event_id=event.event_id,
                kind=event.kind,
                actor_id=event.actor_id,
                parent_event_id=event.parent_event_id,
                payload=dict(event.payload),
            )
            for event in self.events
        )
        checkpoint_ids = tuple(
            snapshot.checkpoint_id
            for snapshot in self.checkpoints
            if snapshot.replay_safe and snapshot.checkpoint_id is not None
        )
        return ExecutionTrace(
            trace_id=trace_id,
            candidate_id=candidate_id,
            task_id=self.task_id,
            events=events,
            final_output=self.final_output,
            status=self.status,
            checkpoint_ids=checkpoint_ids,
        )
