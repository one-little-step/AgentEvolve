"""Tests for edit-memory RAG, retry budget, and sanitization."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_evolve.core.contracts import MemoryRecord, RedactionReport
from agent_evolve.core.errors import PersistenceSafetyError
from agent_evolve.core.memory import (
    AttemptStatus,
    EditAttempt,
    EditMemory,
    RetryBudget,
    RetryState,
    artifact_group_of,
    make_attempt_id,
    sanitize_payload,
)
from agent_evolve.core.storage import JSONFileStorage


def valid_memory_record() -> MemoryRecord:
    return MemoryRecord(
        memory_record_id="mem-1",
        attempt_id="att-1",
        artifact_ids=("skills/r1",),
        issue_fingerprint="issue-1",
        outcome="accepted",
        summary="fix retrieval",
        evidence_refs=("trace-1",),
        redaction_report=RedactionReport(rule_hits=(), truncations=0),
    )


def _memory_record(
    memory_record_id: str,
    issue_fingerprint: str = "issue-1",
    summary: str = "fix retrieval",
) -> MemoryRecord:
    return MemoryRecord(
        memory_record_id=memory_record_id,
        attempt_id=f"att-{memory_record_id}",
        artifact_ids=("skills/r1",),
        issue_fingerprint=issue_fingerprint,
        outcome="accepted",
        summary=summary,
        evidence_refs=("trace-1",),
        redaction_report=RedactionReport(rule_hits=(), truncations=0),
    )


class _FailingStorage:
    """Storage backend whose write always fails closed."""

    def write_record(self, record_type: str, record_id: str, payload) -> None:
        raise PersistenceSafetyError("refusing to persist denied field")


def _attempt(
    attempt_id: str = "a1",
    candidate_id: str = "c1",
    issue_id: str = "issue-1",
    artifact_ids: tuple[str, ...] = ("skills/r1",),
    operation: str = "replace",
    status: AttemptStatus = AttemptStatus.ACCEPTED,
    diff: dict | None = None,
) -> EditAttempt:
    return EditAttempt(
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        issue_id=issue_id,
        artifact_ids=artifact_ids,
        operation=operation,
        sanitized_reasoning="fix retrieval",
        sanitized_diff=diff if diff is not None else {"line_count": 5},
        evidence_refs=("trace-1",),
        history_refs=(),
        validation_summary={"origin": "pass"},
        status=status,
    )


# ---------------------------------------------------------------------- #
# Sanitization
# ---------------------------------------------------------------------- #
def test_sanitize_payload_passes_clean_payload():
    p = {"line_count": 5, "operation": "replace"}
    assert sanitize_payload(p) == p


def test_sanitize_payload_rejects_expected_answer():
    with pytest.raises(ValueError):
        sanitize_payload({"expected_answer": "42"})


def test_sanitize_payload_rejects_password():
    with pytest.raises(ValueError):
        sanitize_payload({"password": "hunter2"})


def test_sanitize_payload_rejects_regex():
    with pytest.raises(ValueError):
        sanitize_payload({"regex": ".*"})


def test_edit_attempt_rejects_diff_with_denied_key():
    with pytest.raises(ValueError):
        _attempt(diff={"expected_label": "correct"})


# ---------------------------------------------------------------------- #
# EditAttempt validation
# ---------------------------------------------------------------------- #
def test_edit_attempt_requires_artifact_ids():
    with pytest.raises(ValueError):
        EditAttempt(
            attempt_id="a1",
            candidate_id="c1",
            issue_id="i1",
            artifact_ids=(),
            operation="replace",
            sanitized_reasoning="",
            sanitized_diff={},
        )


def test_edit_attempt_requires_operation():
    with pytest.raises(ValueError):
        EditAttempt(
            attempt_id="a1",
            candidate_id="c1",
            issue_id="i1",
            artifact_ids=("x",),
            operation="",
            sanitized_reasoning="",
            sanitized_diff={},
        )


def test_attempt_status_is_enum_and_string():
    assert AttemptStatus.ACCEPTED == "accepted"
    assert AttemptStatus("rejected") is AttemptStatus.REJECTED


# ---------------------------------------------------------------------- #
# RetryBudget
# ---------------------------------------------------------------------- #
def test_retry_budget_default_max_is_three():
    b = RetryBudget()
    assert b.max_attempts == 3


def test_retry_budget_counts_attempts():
    b = RetryBudget(max_attempts=3)
    assert b.remaining("i", "g", "l") == 3
    n = b.record_attempt("i", "g", "l")
    assert n == 1
    assert b.remaining("i", "g", "l") == 2


def test_retry_budget_exhausts_at_max():
    b = RetryBudget(max_attempts=2)
    b.record_attempt("i", "g", "l")
    assert not b.is_exhausted("i", "g", "l")
    b.record_attempt("i", "g", "l")
    assert b.is_exhausted("i", "g", "l")
    assert b.remaining("i", "g", "l") == 0


def test_retry_budget_scopes_independently():
    b = RetryBudget(max_attempts=2)
    b.record_attempt("i1", "g", "l")
    b.record_attempt("i1", "g", "l")
    assert b.is_exhausted("i1", "g", "l")
    # Different scope is independent.
    assert not b.is_exhausted("i2", "g", "l")
    assert b.remaining("i2", "g", "l") == 2


def test_retry_budget_rejects_empty_scope_parts():
    b = RetryBudget()
    with pytest.raises(ValueError):
        b.record_attempt("", "g", "l")
    with pytest.raises(ValueError):
        b.remaining("i", "", "l")


def test_retry_budget_reset_clears_one_scope():
    b = RetryBudget(max_attempts=2)
    b.record_attempt("i", "g", "l")
    b.record_attempt("i", "g", "l")
    assert b.is_exhausted("i", "g", "l")
    b.reset("i", "g", "l")
    assert not b.is_exhausted("i", "g", "l")
    assert b.remaining("i", "g", "l") == 2


def test_retry_budget_rejects_zero_max():
    with pytest.raises(ValueError):
        RetryBudget(max_attempts=0)


# ---------------------------------------------------------------------- #
# EditMemory
# ---------------------------------------------------------------------- #
def test_edit_memory_records_and_retrieves():
    m = EditMemory()
    a = _attempt()
    m.record(a, artifact_group="skills/r1", lineage="base")
    assert len(m) == 1
    assert m.get("a1") is a


def test_edit_memory_rejects_duplicate_attempt_id():
    m = EditMemory()
    m.record(_attempt(), artifact_group="g", lineage="l")
    with pytest.raises(ValueError):
        m.record(_attempt(), artifact_group="g", lineage="l")


def test_edit_memory_for_artifact_returns_in_order():
    m = EditMemory()
    a1 = _attempt(attempt_id="a1")
    a2 = _attempt(attempt_id="a2", status=AttemptStatus.REJECTED)
    m.record(a1, artifact_group="g", lineage="l")
    m.record(a2, artifact_group="g", lineage="l")
    by_art = m.for_artifact("skills/r1")
    assert by_art == (a1, a2)


def test_edit_memory_for_issue_groups_attempts():
    m = EditMemory()
    a1 = _attempt(attempt_id="a1", issue_id="issue-A")
    a2 = _attempt(attempt_id="a2", issue_id="issue-A")
    a3 = _attempt(attempt_id="a3", issue_id="issue-B")
    m.record(a1, artifact_group="g", lineage="l")
    m.record(a2, artifact_group="g", lineage="l")
    m.record(a3, artifact_group="g", lineage="l")
    assert {a.attempt_id for a in m.for_issue("issue-A")} == {"a1", "a2"}
    assert {a.attempt_id for a in m.for_issue("issue-B")} == {"a3"}


def test_edit_memory_worked_failed_regression_filters():
    m = EditMemory()
    m.record(_attempt(attempt_id="w1", status=AttemptStatus.ACCEPTED), "g", "l")
    m.record(_attempt(attempt_id="f1", status=AttemptStatus.REJECTED), "g", "l")
    m.record(_attempt(attempt_id="r1", status=AttemptStatus.REGRESSION), "g", "l")
    m.record(_attempt(attempt_id="e1", status=AttemptStatus.EXHAUSTED), "g", "l")
    assert {a.attempt_id for a in m.worked()} == {"w1"}
    assert {a.attempt_id for a in m.failed()} == {"f1", "e1"}
    assert {a.attempt_id for a in m.regressions()} == {"r1"}


def test_edit_memory_record_consumes_retry_budget():
    m = EditMemory(retry_budget=RetryBudget(max_attempts=2))
    m.record(_attempt(attempt_id="a1"), artifact_group="g", lineage="l")
    m.record(_attempt(attempt_id="a2"), artifact_group="g", lineage="l")
    assert m.retry_budget.is_exhausted("issue-1", "g", "l")
    # Third record would still succeed (EditMemory doesn't enforce the budget;
    # the orchestrator does). But the budget count should now be 2, remaining 0.
    assert m.retry_budget.remaining("issue-1", "g", "l") == 0


# ---------------------------------------------------------------------- #
# Storage-backed append-only records
# ---------------------------------------------------------------------- #
def test_memory_persists_only_redacted_reference_record(tmp_path: Path) -> None:
    memory = EditMemory(JSONFileStorage(tmp_path), max_records=2)
    memory.append(valid_memory_record())
    assert memory.retrieve(issue_fingerprint="issue-1", max_records=1) == (valid_memory_record(),)


def test_memory_rejects_raw_nested_editor_response(tmp_path: Path) -> None:
    with pytest.raises(PersistenceSafetyError):
        EditMemory(JSONFileStorage(tmp_path)).append_payload({"editor": {"raw_response": "secret"}})


def test_redaction_report_reflects_actual_hits(tmp_path: Path) -> None:
    memory = EditMemory(JSONFileStorage(tmp_path))
    memory.append(_memory_record("mem-long", summary="x" * 2500))
    stored = memory.retrieve(issue_fingerprint="issue-1", max_records=1)
    assert len(stored) == 1
    report = stored[0].redaction_report
    assert "string_bounded" in report.rule_hits
    assert report.truncations >= 1


def test_record_is_atomic_on_persistence_failure() -> None:
    memory = EditMemory(_FailingStorage(), retry_budget=RetryBudget(max_attempts=2))
    with pytest.raises(PersistenceSafetyError):
        memory.record(_attempt(attempt_id="a1"), artifact_group="g", lineage="l")
    assert len(memory) == 0
    assert memory.for_issue("issue-1") == ()
    assert memory.for_artifact("skills/r1") == ()
    assert memory.retrieve("issue-1", max_records=10) == ()
    assert memory.retry_budget.remaining("issue-1", "g", "l") == 2
    assert not memory.retry_budget.is_exhausted("issue-1", "g", "l")


def test_append_rejects_duplicate_memory_record_id(tmp_path: Path) -> None:
    memory = EditMemory(JSONFileStorage(tmp_path))
    memory.append(_memory_record("mem-1"))
    with pytest.raises(ValueError):
        memory.append(_memory_record("mem-1"))


def test_max_history_records_bound_honored_by_append(tmp_path: Path) -> None:
    memory = EditMemory(JSONFileStorage(tmp_path), max_history_records=2)
    memory.append(_memory_record("mem-1"))
    memory.append(_memory_record("mem-2"))
    memory.append(_memory_record("mem-3"))
    stored = memory.retrieve("issue-1", max_records=10)
    assert {r.memory_record_id for r in stored} == {"mem-2", "mem-3"}


def test_max_history_records_bound_honored_by_append_payload(tmp_path: Path) -> None:
    memory = EditMemory(JSONFileStorage(tmp_path), max_history_records=2)
    memory.append_payload(_memory_record("mem-1").model_dump(mode="json"))
    memory.append_payload(_memory_record("mem-2").model_dump(mode="json"))
    memory.append_payload(_memory_record("mem-3").model_dump(mode="json"))
    stored = memory.retrieve("issue-1", max_records=10)
    assert {r.memory_record_id for r in stored} == {"mem-2", "mem-3"}


# ---------------------------------------------------------------------- #
# RetryState (issue/artifact/lineage scoped)
# ---------------------------------------------------------------------- #
def test_retry_state_exhausted_at_limit():
    s = RetryState()
    assert not s.exhausted("i", ("a",), "l", 2)
    s.attempts_by_scope[("i", ("a",), "l")] = 2
    assert s.exhausted("i", ("a",), "l", 2)
    assert not s.exhausted("i2", ("a",), "l", 2)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def test_make_attempt_id_is_deterministic_and_unique_per_seq():
    assert make_attempt_id(1, 1) == "att-i001-s0001"
    assert make_attempt_id(1, 2) != make_attempt_id(1, 1)
    assert make_attempt_id(2, 1) != make_attempt_id(1, 1)


def test_artifact_group_of_is_order_independent():
    assert artifact_group_of(["a", "b", "c"]) == artifact_group_of(["c", "b", "a"])
    assert artifact_group_of(["a", "a", "b"]) == artifact_group_of(["a", "b"])
