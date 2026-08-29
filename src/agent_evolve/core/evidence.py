"""Answer-key contamination guard and the (task, traces) -> report bridge.

Two responsibilities, both agent-neutral:

1. **Contamination primitives.** ``contamination_terms_from`` extracts every
   scannable string from a task's expected contract so callers can refuse to
   forward evidence containing an answer-key value. These primitives live here,
   in the core, because both the analyzer boundary and the editor boundary need
   them and a security guard with two implementations is a guard that will
   drift. :mod:`agent_evolve.adapters.cuga_editor_evidence` imports them from
   here; the dependency never points the other way.

2. **The bridge.** ``rollout_group_report`` converts the orchestrator's
   ``(EvolutionTask, ExecutionTrace...)`` into the ``RolloutGroupReport`` the
   report-based :class:`~agent_evolve.core.analysis.AnalyzerJudge` consumes.

Why sanitize for the analyzer at all
------------------------------------
``ExecutionTrace.final_output`` is the rollout's answer and
``EvolutionTask.expected_contract`` is the answer key. An LLM analyzer given
either can "diagnose" by comparing them, producing a fluent verdict that
contains no causal reasoning and does not generalize. The report therefore
carries event metadata, actors, tool_call payloads, and ``input_text`` -- never
the final output, never contract values.

What the analyzer may see (mirrors the editor boundary, spec §8):
    task input_text, trace event metadata, actor identities, tool_call payloads.
What it may not see:
    task.expected_contract, trace.final_output, payload blob bodies.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_evolve.core.analysis import RolloutGroupReport
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace

# Content-addressed blob references. Forwarded nowhere: blob bodies carry raw
# prompts and AgentState.
_REF_SUFFIX = "_ref"

# Terms shorter than this are unsafe to scan for: they match incidental text and
# would redact legitimate evidence.
_MIN_TERM_LENGTH = 3

# Only tool_call payloads carry environment evidence worth exposing. Other
# payload kinds (llm_call_start, checkpoints) carry prompt bodies and state.
_PAYLOAD_BEARING_KINDS = frozenset({"tool_call"})

#: S4-9: tool_call names that LOAD an editable surface. A load's payload names
#: the loaded artifact id in its arguments, which is how the surface summary
#: knows the surface was exercised. Only names, never contents.
_SURFACE_LOAD_TOOLS: dict[str, str] = {
    "load_skill": "skills",
    "load_policy": "policies",
    "load_memory": "memory",
}

#: S4-9: per-surface artifact ids actually exercised by a trace. Keyed by the
#: same surface vocabulary as ``CausalFinding.absent_surfaces``.
_SURFACE_KEYS = ("skills", "policies", "memory")

_DEFAULT_MAX_EVENTS_PER_TRACE = 50


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


def is_contaminated(payload: object, terms: Sequence[str]) -> bool:
    """True if any contract term appears anywhere in ``payload``.

    Scans ``repr`` so nested containers are covered without a second recursive
    walk. With no terms there is nothing to detect: an unlabeled task is a
    legitimate case, not a guard failure.
    """
    if not terms:
        return False
    blob = repr(payload)
    return any(term in blob for term in terms)


def strip_blob_refs(payload: Mapping[str, object]) -> dict[str, object]:
    """Drop content-addressed blob references from a payload."""
    return {
        key: value
        for key, value in payload.items()
        if not key.endswith(_REF_SUFFIX)
    }


# ---------------------------------------------------------------------- #
# Bridge
# ---------------------------------------------------------------------- #
def rollout_group_report(
    task: EvolutionTask,
    traces: ExecutionTrace | Sequence[ExecutionTrace],
    *,
    rollout_ids: Sequence[str] | None = None,
    max_events_per_trace: int = _DEFAULT_MAX_EVENTS_PER_TRACE,
) -> RolloutGroupReport:
    """Build a sanitized :class:`RolloutGroupReport` for one candidate/task group.

    A single trace is accepted directly, so the orchestrator's current
    one-rollout path does not have to wrap it in a list.

    All traces must share ``candidate_id`` and ``task_id``: a rollout group is
    one candidate on one task, and mixing candidates would make any
    cross-rollout variance computed from the group meaningless.
    """
    group = (
        (traces,) if isinstance(traces, ExecutionTrace) else tuple(traces)
    )
    if not group:
        raise ValueError("a rollout group needs at least one trace")

    candidate_ids = {t.candidate_id for t in group}
    if len(candidate_ids) > 1:
        raise ValueError(
            f"a rollout group must be one candidate, got: {sorted(candidate_ids)}"
        )
    task_ids = {t.task_id for t in group}
    if len(task_ids) > 1:
        raise ValueError(
            f"a rollout group must be one task, got: {sorted(task_ids)}"
        )

    if rollout_ids is None:
        resolved_rollout_ids = tuple(t.trace_id for t in group)
    else:
        resolved_rollout_ids = tuple(rollout_ids)
        if len(resolved_rollout_ids) != len(group):
            raise ValueError(
                "rollout_ids must have one entry per trace: "
                f"{len(resolved_rollout_ids)} ids for {len(group)} traces"
            )

    if max_events_per_trace < 0:
        raise ValueError("max_events_per_trace must be >= 0")

    terms = contamination_terms_from(task)
    evidence = tuple(
        _trace_evidence(task, trace, terms, max_events_per_trace)
        for trace in group
    )

    return RolloutGroupReport(
        candidate_id=group[0].candidate_id,
        task_id=group[0].task_id,
        trace_refs=tuple(t.trace_id for t in group),
        rollout_ids=resolved_rollout_ids,
        sanitized_evidence=evidence,
    )


def surface_activity_from(
    trace: ExecutionTrace, terms: Sequence[str]
) -> tuple[dict[str, list[str]], int]:
    """S4-9: which artifact ids did this trace actually exercise, per surface?

    Returns ``(summary, withheld_load_count)``. Derived ONLY from ``tool_call``
    payloads whose tool name loads a surface (``load_skill`` etc.): the loaded
    id appears in the call's arguments, so a load is direct evidence that the
    surface was used. Nothing else is read -- no prompt bodies, no artifact
    contents -- and a load id that carries answer-key material is withheld
    (counted in ``withheld_load_count``), mirroring payload handling.

    The summary maps each surface to sorted deduplicated ids, with EMPTY
    members when the surface was never exercised: explicit absence is the
    point (S4-9), so the summary is meaningful exactly when it is empty.
    """
    loads: dict[str, set[str]] = {key: set() for key in _SURFACE_KEYS}
    redactions = 0
    for event in trace.events:
        if event.kind != "tool_call" or not isinstance(event.payload, Mapping):
            continue
        call = event.payload.get("tool_call")
        if not isinstance(call, Mapping):
            continue
        surface = _SURFACE_LOAD_TOOLS.get(str(call.get("name", "")))
        if surface is None:
            continue
        arguments = call.get("arguments")
        artifact_id = ""
        if isinstance(arguments, Mapping):
            candidate = arguments.get("name") or arguments.get("id")
            if isinstance(candidate, str):
                artifact_id = candidate.strip()
        if not artifact_id:
            continue
        if is_contaminated(artifact_id, terms):
            redactions += 1
            continue
        loads[surface].add(artifact_id)
    return {key: list(sorted(loads[key])) for key in _SURFACE_KEYS}, redactions


def _trace_evidence(
    task: EvolutionTask,
    trace: ExecutionTrace,
    terms: Sequence[str],
    max_events: int,
) -> dict[str, object]:
    """Sanitized evidence for one trace.

    ``final_output`` is deliberately absent. ``status`` is included: knowing a
    rollout errored versus completed is causal information that does not reveal
    what the correct answer was.
    """
    actors: list[str] = []
    for event in trace.events:
        if event.actor_id and event.actor_id not in actors:
            actors.append(event.actor_id)

    events: list[dict[str, object]] = []
    redactions = 0
    for event in trace.events[:max_events]:
        payload, redacted = _safe_payload(event.kind, event.payload, terms)
        if redacted:
            redactions += 1
        events.append(
            {
                "event_id": event.event_id,
                "kind": event.kind,
                "actor_id": event.actor_id,
                "parent_event_id": event.parent_event_id,
                "payload": payload,
                "payload_redacted": redacted,
            }
        )

    surface, withheld_loads = surface_activity_from(trace, terms)
    return {
        "trace_id": trace.trace_id,
        "status": trace.status,
        "task_input": task.input_text,
        "actors": tuple(actors),
        "events": tuple(events),
        "events_truncated": len(trace.events) > max_events,
        "redaction_count": redactions + withheld_loads,
        # S4-9: which surfaces this trace exercised. Computed over the FULL
        # event list (a load past the trim window still proves use), never
        # leaking contents. Empty members mean the surface was never loaded;
        # a withheld load id is folded into redaction_count so it reads as
        # withheld rather than indistinguishable from never-happened.
        "surface_activity": surface,
    }


def _safe_payload(
    kind: str,
    payload: object,
    terms: Sequence[str],
) -> tuple[dict[str, object], bool]:
    """Strip blob refs, keep tool_call evidence, drop contaminated payloads.

    Returns ``(payload, was_redacted)``. The flag distinguishes "this event had
    no payload to show" from "this event's payload was withheld", so an analyzer
    can reason about a gap instead of silently treating it as absence.
    """
    if not isinstance(payload, Mapping):
        return {}, False
    if kind not in _PAYLOAD_BEARING_KINDS:
        return {}, False
    cleaned = strip_blob_refs(payload)
    if is_contaminated(cleaned, terms):
        return {}, True
    return cleaned, False
