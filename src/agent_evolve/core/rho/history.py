"""Historical (task, trajectory) corpus loading for the RHO coreset stage.

RHO is retrospective: it needs past trajectories to judge difficulty and
fingerprint failure structure. Two trace formats exist in this repository and
only one is usable.

A *stale-format* trace carries generic ``stream_event`` entries with
``actor_id=None`` and no ``tool_call`` events. Group diagnosis cannot attribute
anything in such a trace, so accepting it would produce confident diagnoses
about an agent whose behaviour was never recorded. It is rejected with a reason
instead.

When no valid record is found the loader reports a *cold start* rather than
raising: RHO can still run by selecting a coreset without difficulty weighting
and generating its own evidence through group rollouts.

This module is agent-neutral: stdlib only, no ``cuga``, no ``litellm``, no
``agent_evolve.adapters``.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

TRACE_FILENAME = "causal-trace.json"


class StaleTraceFormat(ValueError):
    """A trace exists but cannot support diagnosis."""


@dataclass(frozen=True, slots=True)
class HistoricalRecord:
    """One past (task, trajectory) pair usable as RHO evidence."""

    task_id: str
    input_text: str
    trace_path: str
    raw_trace: Mapping[str, object]
    final_output: str
    tool_observation_count: int
    harness_version: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class HistoryLoadReport:
    """Loaded records plus every rejection, so nothing is silently dropped."""

    records: tuple[HistoricalRecord, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()

    @property
    def is_cold_start(self) -> bool:
        """True when there is no usable historical evidence at all."""
        return not self.records


def _validate_current_format(payload: Mapping[str, object]) -> None:
    """Raise :class:`StaleTraceFormat` for a trace that cannot be diagnosed."""
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise StaleTraceFormat("trace has no events")
    mappings = [event for event in events if isinstance(event, Mapping)]
    if not mappings:
        raise StaleTraceFormat("trace has no structured events")
    actors = {event.get("actor_id") for event in mappings}
    if actors == {None}:
        raise StaleTraceFormat(
            "every event has actor_id=None: stale trace format, nothing can be "
            "causally attributed"
        )
    kinds = {event.get("kind") for event in mappings}
    if kinds == {"stream_event"}:
        raise StaleTraceFormat(
            "every event kind is 'stream_event': stale trace format"
        )


def load_history(root: Path) -> HistoryLoadReport:
    """Load every ``<root>/<run_id>/causal-trace.json`` under ``root``.

    A missing ``root`` is a cold start, not an error. Unreadable or
    stale-format traces are recorded in :attr:`HistoryLoadReport.rejected`
    with a human-readable reason and the run continues on valid records.
    """
    root = Path(root)
    if not root.exists():
        return HistoryLoadReport()

    records: list[HistoricalRecord] = []
    rejected: list[tuple[str, str]] = []
    for path in sorted(root.rglob(TRACE_FILENAME)):
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            rejected.append((str(path), f"unreadable: {exc}"))
            continue
        if not isinstance(payload, dict):
            rejected.append((str(path), "trace root is not an object"))
            continue
        try:
            _validate_current_format(payload)
        except StaleTraceFormat as exc:
            rejected.append((str(path), str(exc)))
            continue
        observations = payload.get("tool_observations")
        records.append(
            HistoricalRecord(
                task_id=str(payload.get("task_id") or path.parent.name),
                input_text=str(payload.get("input_text") or ""),
                trace_path=str(path),
                raw_trace=payload,
                final_output=str(payload.get("final_output") or ""),
                tool_observation_count=(
                    len(observations) if isinstance(observations, list) else 0
                ),
                harness_version=str(payload.get("harness_version") or "unknown"),
                content_hash=f"sha256:{sha256(raw.encode('utf-8')).hexdigest()}",
            )
        )
    return HistoryLoadReport(records=tuple(records), rejected=tuple(rejected))
