"""TapeIndex — offline index over a recorded trace directory (design R2/R3).

Loads ``events.jsonl`` plus the verbatim ``payloads\\`` store of a trace and
exposes:

* ordered LLM call boundaries (``messages_ref`` -> ``response_ref``),
* tool observations paired by ``run_id``,
* node starts (``node``, ``step``, ``state_before_ref``) for resume
  addressing (RQ5),
* content-addressed lazy resolution: blobs are read from disk only on demand
  and their sha256 is re-verified against the ref at every read, so memory
  stays flat regardless of trace size. This is why there is no truncation
  threshold anywhere: nothing large is ever held in the index itself
  (closes RQ1's deferred sub-question — verbatim wins, laziness pays for it).

Classification policy lives in :class:`ToolTapeClassifier`, a caller-supplied
pattern registry. The core ships NO tool-name table: an unregistered tool is
conservatively ``UNRECORDABLE`` with an explicit reason, never guessed.

This module is agent-neutral: it reads our own trace file format and imports
nothing from any adapter or model library.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "ToolClass",
    "TapeIntegrityError",
    "LLMBoundary",
    "ToolTapeEntry",
    "NodeStart",
    "DryRunReport",
    "ToolTapeClassifier",
    "TapeIndex",
    "boundary_for_fault",
]


class ToolClass(str, Enum):
    """Replay behaviour classes (design R3)."""

    PURE = "pure"
    EXTERNAL = "external"
    STATEFUL_LOCAL = "stateful_local"
    UNRECORDABLE = "unrecordable"


class TapeIntegrityError(RuntimeError):
    """A payload blob is missing or fails its sha256 verification."""


@dataclass(frozen=True)
class LLMBoundary:
    sequence: int
    messages_ref: str
    response_ref: str


@dataclass(frozen=True)
class ToolTapeEntry:
    sequence: int
    run_id: str
    tool_name: str | None
    args_ref: str | None
    output_ref: str | None


@dataclass(frozen=True)
class NodeStart:
    sequence: int
    node: str
    step: int
    state_before_ref: str
    #: W2: event id + nesting parent, carried so resume addressing can
    #: disambiguate the same actor at different subgraph depths.
    event_id: str = ""
    parent_event_id: str | None = None


@dataclass
class DryRunReport:
    total_calls: int
    counts: Counter
    unclassified_names: list[str]


@dataclass
class ToolTapeClassifier:
    """Pattern -> class registry; first matching registration wins."""

    _rules: list[tuple[str, ToolClass]] = field(default_factory=list)

    def register(self, pattern: str, tool_class: ToolClass) -> None:
        self._rules.append((pattern, tool_class))

    def classify(self, tool_name: str | None) -> tuple[ToolClass, str | None]:
        """Return ``(class, withheld_reason)``; reason is None unless refused."""
        if tool_name:
            for pattern, tool_class in self._rules:
                if fnmatch.fnmatchcase(tool_name, pattern):
                    return tool_class, None
        return ToolClass.UNRECORDABLE, f"unclassified_tool: {tool_name!r}"


def _iter_refs(payload: dict) -> list[str]:
    return [value for key, value in sorted(payload.items())
            if isinstance(key, str) and key.endswith("_ref") and isinstance(value, str)]


class TapeIndex:
    def __init__(self, events: list[dict], payloads_dir: Path) -> None:
        self._events = sorted(events, key=lambda e: e.get("sequence", 0))
        self._payloads_dir = payloads_dir
        self._llm_boundaries = self._pair_llm()
        self._tool_entries = self._pair_tools()
        self._node_starts = [
            NodeStart(
                sequence=int(e["sequence"]),
                node=str(e["payload"]["node"]),
                step=int(e["payload"].get("step", 0)),
                state_before_ref=e["payload"]["state_before_ref"],
                event_id=str(e.get("event_id", "")),
                parent_event_id=(
                    str(e["parent_event_id"])
                    if e.get("parent_event_id") is not None
                    else None
                ),
            )
            for e in self._events if e.get("kind") == "graph_node_start"
        ]

    @classmethod
    def load(cls, trace_dir: Path) -> "TapeIndex":
        events_path = Path(trace_dir) / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(events=events, payloads_dir=Path(trace_dir) / "payloads")

    # -- indexes ----------------------------------------------------------

    @property
    def llm_boundaries(self) -> list[LLMBoundary]:
        return list(self._llm_boundaries)

    @property
    def tool_entries(self) -> list[ToolTapeEntry]:
        return list(self._tool_entries)

    @property
    def node_starts(self) -> list[NodeStart]:
        return list(self._node_starts)

    def _pair_llm(self) -> list[LLMBoundary]:
        messages_by_run: dict[str, tuple[int, str]] = {}
        boundaries: list[LLMBoundary] = []
        for event in self._events:
            kind = event.get("kind")
            payload = event.get("payload", {})
            if kind == "llm_call_start":
                messages_by_run[payload["run_id"]] = (
                    int(event["sequence"]), payload["messages_ref"])
            elif kind == "llm_call_end":
                start_seq, messages_ref = messages_by_run.pop(
                    payload["run_id"], (int(event["sequence"]), payload.get("messages_ref", "")))
                boundaries.append(LLMBoundary(
                    sequence=start_seq,
                    messages_ref=messages_ref,
                    response_ref=payload["response_ref"],
                ))
        return boundaries

    def _pair_tools(self) -> list[ToolTapeEntry]:
        starts: dict[str, dict] = {}
        entries: dict[str, ToolTapeEntry] = {}
        order: list[str] = []
        for event in self._events:
            kind = event.get("kind")
            payload = event.get("payload", {})
            if kind == "graph_tool_start":
                run_id = payload["run_id"]
                starts[run_id] = payload
                order.append(run_id)
                entries[run_id] = ToolTapeEntry(
                    sequence=int(event["sequence"]),
                    run_id=run_id,
                    tool_name=payload.get("tool_name"),
                    args_ref=payload.get("args_ref"),
                    output_ref=None,
                )
            elif kind == "graph_tool_end":
                run_id = payload["run_id"]
                base = entries.get(run_id)
                if base is None:  # end without start: keep honest record
                    order.append(run_id)
                    base = ToolTapeEntry(
                        sequence=int(event["sequence"]), run_id=run_id,
                        tool_name=payload.get("tool_name"),
                        args_ref=None, output_ref=None,
                    )
                    entries[run_id] = base
                entries[run_id] = ToolTapeEntry(
                    sequence=base.sequence, run_id=base.run_id,
                    tool_name=base.tool_name or payload.get("tool_name"),
                    args_ref=base.args_ref,
                    output_ref=payload.get("output_ref"),
                )
        return [entries[run_id] for run_id in order]

    # -- content addressing -----------------------------------------------

    def resolve(self, ref: str) -> bytes:
        """Read a payload blob and re-verify its sha256 against the ref."""
        path = self._payloads_dir / f"{ref}.json"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise TapeIntegrityError(f"missing payload blob: {ref}") from exc
        if hashlib.sha256(raw).hexdigest() != ref:
            raise TapeIntegrityError(f"sha256 mismatch for payload blob: {ref}")
        return raw

    def verify_all_refs(self) -> None:
        refs = [
            ref
            for event in self._events
            for ref in _iter_refs(event.get("payload", {}))
        ]
        problems: list[str] = []
        for ref in refs:
            try:
                self.resolve(ref)
            except TapeIntegrityError as exc:
                problems.append(str(exc))
        if problems:
            raise TapeIntegrityError(
                f"{len(problems)} integrity failure(s): " + "; ".join(problems))

    # -- strictness -------------------------------------------------------

    def dry_classify(self, classifier: ToolTapeClassifier) -> DryRunReport:
        counts: Counter = Counter()
        unclassified: list[str] = []
        for entry in self._tool_entries:
            tool_class, reason = classifier.classify(entry.tool_name)
            counts[tool_class] += 1
            if reason and reason.startswith("unclassified_tool"):
                name = entry.tool_name if entry.tool_name else "<unnamed>"
                if name not in unclassified:
                    unclassified.append(name)
        return DryRunReport(
            total_calls=len(self._tool_entries),
            counts=counts,
            unclassified_names=unclassified,
        )


def boundary_for_fault(tape_index: "TapeIndex", analysis: object) -> int | None:
    """Map a blame graph to a resume boundary (W2).

    Returns the number of LLM boundaries to tape before going live -- the
    ``--resume N`` semantics RQ5 settled (boundaries ``< N`` taped, ``>= N``
    live) -- or ``None`` to fall through to full validation.

    Matching: the max-blame actor (ties broken by ``actor_id``) names the
    failing node; its LAST ``NodeStart`` occurrence is the failing cycle, and
    the resume boundary is the count of LLM boundaries with a strictly lower
    ``sequence``. ``None`` when there is no blame, the actor does not appear,
    the fault precedes the first boundary (nothing to tape), or it sits at or
    after the last boundary (tail ~= whole run, replay pointless).

    ``analysis`` is typed ``object`` (duck-typed for ``.blame_graph.nodes``) so
    this module needs no import of the blame model; it stays agent-neutral.
    """
    nodes = getattr(getattr(analysis, "blame_graph", None), "nodes", ())
    if not nodes:
        return None
    # Max-blame actor; ties broken by actor_id for determinism.
    blamed = max(nodes, key=lambda n: (getattr(n, "blame", 0.0), getattr(n, "actor_id", "")))
    actor_id = getattr(blamed, "actor_id", "")
    if not actor_id:
        return None

    starts = [s for s in tape_index.node_starts if s.node == actor_id]
    if not starts:
        return None
    failing = max(starts, key=lambda s: s.sequence)

    boundaries = tape_index.llm_boundaries
    resume = sum(1 for b in boundaries if b.sequence < failing.sequence)
    if resume == 0 or resume == len(boundaries):
        return None
    return resume
