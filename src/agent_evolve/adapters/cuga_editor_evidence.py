"""Read-only evidence view handed to the CUGA editor agent.

Boundary (spec §8). The editor may see:
    mechanism, blame graph, artifact content, edit history, task input_text,
    trace event metadata, and tool_call payloads.

The editor may NOT see:
    task.expected_contract, trace.final_output, payload blob contents.

Because tool_call payloads are exposed and a tool result can contain
answer-shaped free text, a fail-closed contamination guard drops any payload
containing an expected-contract value. The guard consumes expected_contract to
build its term list; it never emits it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_evolve.core.blame import CausalAnalysis
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace

# Content-addressed blob references. Forwarded nowhere: blob bodies carry raw
# prompts and AgentState.
_REF_SUFFIX = "_ref"

# Terms shorter than this are unsafe to scan for: they match incidental text and
# would redact legitimate evidence.
_MIN_TERM_LENGTH = 3


def contamination_terms_from(task: EvolutionTask) -> tuple[str, ...]:
    """Extract every scannable string in a task's expected contract, at any depth.

    Recursion is load-bearing, not defensive generality. A shallow
    ``.values()`` scan sees ``{"expected_substring": "tok"}`` but misses
    ``{"expected_any": ["tok"]}`` and ``{"grader": {"expected": "tok"}}``,
    yielding zero terms -- and a guard with zero terms passes every payload
    through. A contract shape the extractor cannot see is a contract the guard
    cannot enforce, so the extractor must not assume a flat mapping.
    """
    terms: list[str] = []
    _collect_terms(task.expected_contract, terms)
    # Deduplicate while preserving first-seen order for deterministic output.
    return tuple(dict.fromkeys(terms))


def _collect_terms(value: object, out: list[str]) -> None:
    """Walk an arbitrary contract value, appending scannable strings."""
    if isinstance(value, str):
        if len(value) >= _MIN_TERM_LENGTH:
            out.append(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _collect_terms(item, out)
        return
    # Sequences of values, excluding the str case handled above.
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_terms(item, out)


@dataclass(slots=True)
class EvidenceView:
    """Bounded, guarded projection of one rollout's causal evidence."""

    analysis: CausalAnalysis
    trace: ExecutionTrace
    task: EvolutionTask
    contamination_terms: tuple[str, ...] = ()
    _redactions: int = field(default=0, repr=False)

    @property
    def redaction_count(self) -> int:
        return self._redactions

    def mechanism(self) -> dict[str, object]:
        return {
            "mechanism": self.analysis.mechanism,
            "severity": self.analysis.severity,
        }

    def blamed_actors(self) -> tuple[dict[str, object], ...]:
        nodes = sorted(
            self.analysis.blame_graph.nodes,
            key=lambda n: (-n.blame, n.actor_id),
        )
        return tuple(
            {
                "actor_id": n.actor_id,
                "blame": n.blame,
                "artifacts": n.artifacts,
            }
            for n in nodes
        )

    def task_input(self) -> str:
        """The task's input_text only.

        Safe by construction: this is exactly what the agent under test already
        received, so it reveals nothing the rollout did not already see.
        """
        return self.task.input_text

    def actors(self) -> tuple[str, ...]:
        seen: list[str] = []
        for event in self.trace.events:
            if event.actor_id and event.actor_id not in seen:
                seen.append(event.actor_id)
        return tuple(seen)

    def events(
        self,
        kind: str | None = None,
        actor_id: str | None = None,
        limit: int = 50,
    ) -> tuple[dict[str, object], ...]:
        out: list[dict[str, object]] = []
        for event in self.trace.events:
            if kind is not None and event.kind != kind:
                continue
            if actor_id is not None and event.actor_id != actor_id:
                continue
            payload, redacted = self._safe_payload(event.kind, event.payload)
            out.append(
                {
                    "event_id": event.event_id,
                    "kind": event.kind,
                    "actor_id": event.actor_id,
                    "parent_event_id": event.parent_event_id,
                    "payload": payload,
                    "payload_redacted": redacted,
                }
            )
            if len(out) >= limit:
                break
        return tuple(out)

    def _safe_payload(
        self, kind: str, payload: object
    ) -> tuple[dict[str, object], bool]:
        """Strip blob refs, keep tool_call evidence, drop contaminated payloads."""
        if not isinstance(payload, dict):
            return {}, False
        # Only tool_call payloads carry environment evidence worth exposing.
        if kind != "tool_call":
            return {}, False
        cleaned = {
            key: value
            for key, value in payload.items()
            if not key.endswith(_REF_SUFFIX)
        }
        if self._is_contaminated(cleaned):
            self._redactions += 1
            return {}, True
        return cleaned, False

    def _is_contaminated(self, payload: dict[str, object]) -> bool:
        if not self.contamination_terms:
            return False
        blob = repr(payload)
        return any(term in blob for term in self.contamination_terms)
