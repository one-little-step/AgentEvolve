"""Edit-memory RAG with worked/failed/regression state and retry budget.

Per docs/architecture/target-rho-parallel-gepa.md:

    Edit-memory RAG
    worked, failed, regression state

    Retry state is scoped to issue, artifact group, and lineage, with a
    default maximum of three attempts.

This module is pure data + small query helpers. It does NOT call an LLM; the
editor caller (:mod:`agent_evolve.core.editor`) is responsible for producing
:class:`EditAttempt` records and the orchestrator for committing them here.

Sanitization rule
-----------------
The architecture doc and AGENTS.md require that no credentials, expected
answers, evaluator internals, labels, or regexes be persisted to edit memory.
This module trusts its callers to pre-sanitize; we additionally refuse to
store any payload whose keys match a denylist (``expected_*``, ``label``,
``regex``, ``secret``, ``token``, ``password``).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from agent_evolve.core.contracts import MemoryRecord, RedactionReport
from agent_evolve.core.storage import JSONFileStorage


# ---------------------------------------------------------------------- #
# Status enum
# ---------------------------------------------------------------------- #
class AttemptStatus(str, Enum):
    """Outcome of a single edit attempt."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REGRESSION = "regression"
    EXHAUSTED = "exhausted"


# ---------------------------------------------------------------------- #
# Sanitization
# ---------------------------------------------------------------------- #
_DENYLIST_KEYS = frozenset(
    {
        "expected_answer",
        "expected_label",
        "expected_regex",
        "expected",
        "label",
        "labels",
        "regex",
        "secret",
        "token",
        "password",
        "api_key",
        "apikey",
    }
)


def sanitize_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Strip denylisted keys before persisting an edit attempt.

    Returns a new mapping; the input is not mutated. Raises ``ValueError`` if
    a denied key is encountered, because silently dropping a key the editor
    thought was harmless would mask a contract violation.
    """
    bad = _DENYLIST_KEYS & set(payload.keys())
    if bad:
        raise ValueError(
            f"refusing to persist denied payload keys (likely credential/label/regex): {sorted(bad)}"
        )
    return dict(payload)


# ---------------------------------------------------------------------- #
# Records
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EditAttempt:
    """One structured edit attempt recorded in edit memory.

    Attributes
    ----------
    attempt_id:
        Unique within the iteration that produced it.
    candidate_id:
        The candidate version the edit was applied to.
    issue_id:
        The mechanism cluster + task + iteration key that drove this attempt.
    artifact_ids:
        The artifacts this attempt touched.
    operation:
        Adapter-defined operation name (e.g. "replace", "append").
    sanitized_reasoning:
        Editor reasoning, post-sanitization. May be empty.
    sanitized_diff:
        Adapter-returned change summary; never raw file contents if those
        might contain credentials.
    evidence_refs:
        IDs of traces/analyses the editor cited as justification.
    history_refs:
        IDs of prior attempts the editor consulted.
    validation_summary:
        Outcome counts: {"origin": pass, "worked": pass, "regression": fail}.
    status:
        Final status of the attempt.
    """

    attempt_id: str
    candidate_id: str
    issue_id: str
    artifact_ids: tuple[str, ...]
    operation: str
    sanitized_reasoning: str
    sanitized_diff: Mapping[str, object]
    evidence_refs: tuple[str, ...] = ()
    history_refs: tuple[str, ...] = ()
    validation_summary: Mapping[str, str] = field(default_factory=dict)
    status: AttemptStatus = AttemptStatus.PENDING
    created_at: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("attempt_id is required")
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not self.issue_id:
            raise ValueError("issue_id is required")
        if not self.artifact_ids:
            raise ValueError("artifact_ids is required (cannot be empty)")
        if not self.operation:
            raise ValueError("operation is required")
        # Sanitize diff: refuse denied keys.
        sanitize_payload(self.sanitized_diff)


# ---------------------------------------------------------------------- #
# Retry budget
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class RetryBudget:
    """Per-(issue, artifact_group, lineage) retry budget.

    Default maximum is three attempts per the architecture doc. Once a
    budget is exhausted, further attempts for the same scope are rejected
    and the orchestrator must move on or wait for the next outer iteration.
    """

    max_attempts: int = 3
    _counts: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _exhausted: set[tuple[str, str, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")

    def _key(self, issue_id: str, artifact_group: str, lineage: str) -> tuple[str, str, str]:
        if not issue_id or not artifact_group or not lineage:
            raise ValueError("issue_id, artifact_group, and lineage are all required")
        return (issue_id, artifact_group, lineage)

    def record_attempt(self, issue_id: str, artifact_group: str, lineage: str) -> int:
        """Increment the count for this scope and return the new count."""
        k = self._key(issue_id, artifact_group, lineage)
        n = self._counts.get(k, 0) + 1
        self._counts[k] = n
        if n >= self.max_attempts:
            self._exhausted.add(k)
        return n

    def remaining(self, issue_id: str, artifact_group: str, lineage: str) -> int:
        k = self._key(issue_id, artifact_group, lineage)
        used = self._counts.get(k, 0)
        return max(0, self.max_attempts - used)

    def is_exhausted(self, issue_id: str, artifact_group: str, lineage: str) -> bool:
        return self._key(issue_id, artifact_group, lineage) in self._exhausted

    def reset(self, issue_id: str, artifact_group: str, lineage: str) -> None:
        """Clear the budget for one scope (used on outer iteration refresh)."""
        k = self._key(issue_id, artifact_group, lineage)
        self._counts.pop(k, None)
        self._exhausted.discard(k)


# ---------------------------------------------------------------------- #
# Retry state (issue/artifact/lineage scoped)
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class RetryState:
    """Scoped retry accounting keyed by (issue, artifacts, lineage).

    Distinct from :class:`RetryBudget`, which is keyed by
    (issue_id, artifact_group, lineage) and enforces a default maximum of
    three. ``RetryState`` is the append-only-record counterpart: it stores an
    explicit per-scope count and lets callers pass the limit per check.
    """

    attempts_by_scope: dict[tuple[str, tuple[str, ...], str], int] = field(
        default_factory=dict
    )

    def record(self, issue: str, artifacts: tuple[str, ...], lineage: str) -> int:
        key = (issue, artifacts, lineage)
        n = self.attempts_by_scope.get(key, 0) + 1
        self.attempts_by_scope[key] = n
        return n

    def exhausted(
        self, issue: str, artifacts: tuple[str, ...], lineage: str, limit: int
    ) -> bool:
        return self.attempts_by_scope.get((issue, artifacts, lineage), 0) >= limit


# ---------------------------------------------------------------------- #
# Edit memory
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class EditMemory:
    """Append-only edit-memory RAG.

    Three logical stores per the architecture doc:

    * ``worked``: attempts that were accepted and survived regression probes.
    * ``failed``: attempts that were rejected or caused regressions.
    * ``regression``: attempts that introduced a regression on a previously
      worked artifact.

    Plus a retry budget scoped to (issue, artifact_group, lineage).

    When ``storage`` is set, every persisted object is routed through the
    storage backend's recursive sanitizer: raw editor/task/evaluator content is
    never written, and unsafe content fails closed with
    :class:`PersistenceSafetyError`.
    """

    storage: JSONFileStorage | None = None
    retry_budget: RetryBudget = field(default_factory=RetryBudget)
    max_records: int | None = None
    _attempts: list[EditAttempt] = field(default_factory=list)
    _by_id: dict[str, EditAttempt] = field(default_factory=dict)
    _by_artifact: dict[str, list[str]] = field(default_factory=dict)
    _by_issue: dict[str, list[str]] = field(default_factory=dict)
    _records: list[MemoryRecord] = field(default_factory=list)
    _by_record_id: dict[str, MemoryRecord] = field(default_factory=dict)
    _records_by_issue: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be > 0 or None")

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def record(self, attempt: EditAttempt, artifact_group: str, lineage: str) -> None:
        if attempt.attempt_id in self._by_id:
            raise ValueError(f"duplicate attempt_id: {attempt.attempt_id!r}")
        self._attempts.append(attempt)
        self._by_id[attempt.attempt_id] = attempt
        for aid in attempt.artifact_ids:
            self._by_artifact.setdefault(aid, []).append(attempt.attempt_id)
        self._by_issue.setdefault(attempt.issue_id, []).append(attempt.attempt_id)
        # Account for the retry budget.
        self.retry_budget.record_attempt(attempt.issue_id, artifact_group, lineage)
        # Persist a sanitized, reference-based memory record when backed.
        if self.storage is not None:
            self.append(_attempt_to_memory_record(attempt))

    def _require_storage(self) -> JSONFileStorage:
        if self.storage is None:
            raise ValueError(
                "EditMemory has no storage backend; cannot persist memory records"
            )
        return self.storage

    def append(self, memory_record: MemoryRecord) -> None:
        """Sanitize, persist, and index an append-only memory record."""
        storage = self._require_storage()
        storage.write_record(
            "memory",
            memory_record.memory_record_id,
            memory_record.model_dump(mode="json"),
        )
        self._index_memory_record(memory_record)

    def append_payload(self, payload: Mapping[str, object]) -> None:
        """Recursively sanitize ``payload`` and persist it under ``memory``.

        Raises :class:`PersistenceSafetyError` when the storage sanitizer
        rejects prohibited content. Any supplied ``memory_record_id`` is used
        as the record ID; otherwise a fresh ID is generated.
        """
        storage = self._require_storage()
        record_id = str(payload.get("memory_record_id") or uuid.uuid4().hex)
        storage.write_record("memory", record_id, payload)

    def _index_memory_record(self, memory_record: MemoryRecord) -> None:
        self._records.append(memory_record)
        self._by_record_id[memory_record.memory_record_id] = memory_record
        self._records_by_issue.setdefault(
            memory_record.issue_fingerprint, []
        ).append(memory_record.memory_record_id)
        self._enforce_bound()

    def _enforce_bound(self) -> None:
        if self.max_records is None:
            return
        while len(self._records) > self.max_records:
            oldest = self._records.pop(0)
            self._by_record_id.pop(oldest.memory_record_id, None)
            ids = self._records_by_issue.get(oldest.issue_fingerprint, [])
            if oldest.memory_record_id in ids:
                ids.remove(oldest.memory_record_id)
            if not ids:
                self._records_by_issue.pop(oldest.issue_fingerprint, None)

    # ------------------------------------------------------------------ #
    # Read (RAG-style retrieval)
    # ------------------------------------------------------------------ #
    def retrieve(
        self, issue_fingerprint: str, max_records: int = 1
    ) -> tuple[MemoryRecord, ...]:
        """Bounded retrieval of memory records for an issue fingerprint.

        Returns up to ``max_records`` matching records, most recent first.
        """
        ids = self._records_by_issue.get(issue_fingerprint, ())
        if max_records <= 0:
            return ()
        recent = ids[-max_records:]
        return tuple(self._by_record_id[i] for i in reversed(recent))

    def get(self, attempt_id: str) -> EditAttempt:
        if attempt_id not in self._by_id:
            raise KeyError(attempt_id)
        return self._by_id[attempt_id]

    def for_artifact(self, artifact_id: str) -> tuple[EditAttempt, ...]:
        ids = self._by_artifact.get(artifact_id, [])
        return tuple(self._by_id[i] for i in ids)

    def for_issue(self, issue_id: str) -> tuple[EditAttempt, ...]:
        ids = self._by_issue.get(issue_id, [])
        return tuple(self._by_id[i] for i in ids)

    def worked(self) -> tuple[EditAttempt, ...]:
        return tuple(a for a in self._attempts if a.status == AttemptStatus.ACCEPTED)

    def failed(self) -> tuple[EditAttempt, ...]:
        return tuple(
            a for a in self._attempts if a.status in (AttemptStatus.REJECTED, AttemptStatus.EXHAUSTED)
        )

    def regressions(self) -> tuple[EditAttempt, ...]:
        return tuple(a for a in self._attempts if a.status == AttemptStatus.REGRESSION)

    def __len__(self) -> int:
        return len(self._attempts)

    def all_attempts(self) -> tuple[EditAttempt, ...]:
        return tuple(self._attempts)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
_ATTEMPT_STATUS_TO_OUTCOME = {
    AttemptStatus.ACCEPTED: "accepted",
    AttemptStatus.REJECTED: "rejected",
    AttemptStatus.REGRESSION: "rejected",
    AttemptStatus.EXHAUSTED: "exhausted",
    AttemptStatus.PENDING: "unavailable",
}


def _attempt_to_memory_record(attempt: EditAttempt) -> MemoryRecord:
    """Map a sanitized :class:`EditAttempt` to a reference-based MemoryRecord.

    Only sanitized fields are carried forward; raw editor/task/evaluator
    content never leaves the attempt's sanitized diff/reasoning boundary.
    """
    summary = attempt.sanitized_reasoning or attempt.operation
    return MemoryRecord(
        memory_record_id=attempt.attempt_id,
        attempt_id=attempt.attempt_id,
        artifact_ids=attempt.artifact_ids,
        issue_fingerprint=attempt.issue_id,
        outcome=_ATTEMPT_STATUS_TO_OUTCOME[attempt.status],
        summary=summary,
        evidence_refs=attempt.evidence_refs,
        redaction_report=RedactionReport(rule_hits=(), truncations=0),
    )


def make_attempt_id(iteration: int, seq: int) -> str:
    """Deterministic attempt ID for tests and demos."""
    return f"att-i{iteration:03d}-s{seq:04d}"


def artifact_group_of(artifact_ids: Iterable[str]) -> str:
    """Stable group key for an artifact set, used by RetryBudget.

    Two attempts that touch the same set of artifacts share a group; the
    order of artifact_ids in the input does not matter.
    """
    return "|".join(sorted(set(artifact_ids)))
