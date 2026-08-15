# Unified CUGA Editor Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `FakeEditor` (which injects expected answers) with a CUGA-agent-backed editor that proposes artifact edits from causal evidence, handling refinement and multi-parent combination in one invocation.

**Architecture:** A new `CugaEditorAgent` in `adapters/` satisfies the existing `core.editor.Editor` protocol. Its multi-turn agent loop lives entirely inside `propose_edit`, so `core/` never learns the editor is a CUGA agent. The agent reads evidence and writes staged edits through adapter-interfaced tool clusters; a terminal `submit_edit_plan` tool captures the plan, and the agent's prose answer is ignored.

**Tech Stack:** Python 3.14, pydantic v2, `cuga==0.3.1` SDK, `langchain_core.tools`, pytest.

## Global Constraints

- `src/agent_evolve/core/` must never import `cuga` or any agent implementation.
- Design doc: `docs/superpowers/specs/2026-08-15-unified-cuga-editor-agent-design.md`. Where this plan and the spec disagree, the spec governs.
- The editor must never receive `task.expected_contract`, `trace.final_output`, or trace payload blob contents.
- Tests precede implementation. Every task writes a failing test first.
- Capture test runs with `2>&1 | tee terminal_output/cuga-editor/<name>.log`.
- Never commit unless the user explicitly asks. Steps that say "Commit" mean: stage the listed files and report readiness; do not run `git commit` without approval.
- Created artifact ids must match `skills/generated-<name>`; per-attempt cap 2; pool-wide cap 10.
- `supports_counterfactual_replay()` stays `False`.
- Run tests with `uv run pytest`.

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `src/agent_evolve/core/config.py` (modify) | 1 | add `max_editor_calls` to `BudgetLimits`, wire into `reserve()`, expose in `manifest_payload()` |
| `src/agent_evolve/adapters/cuga_editor_state.py` (create) | 2 | `EditStagingArea`: staged edits, caps, authorization, parent-read ledger. No CUGA import. |
| `src/agent_evolve/adapters/cuga_editor_evidence.py` (create) | 3 | `EvidenceView` + contamination guard. Enforces the evidence boundary. No CUGA import. |
| `src/agent_evolve/adapters/cuga_adapter.py` (modify) | 4 | `create` edit operation; `creatable_prefix`; `created_artifact_count` |
| `src/agent_evolve/core/editor.py` (modify) | 5 | `ParentContext`, `EditorOutcome`; `parents` / `creatable_prefix` / `pool_created_count` fields |
| `src/agent_evolve/adapters/cuga_editor_skills.py` (create) | 6 | editor `special_instructions` and the four skill bodies |
| `src/agent_evolve/adapters/cuga_editor_tools.py` (create) | 7 | tool cluster builders. CUGA import deferred into `build_editor_tools`. |
| `src/agent_evolve/adapters/cuga_editor.py` (create) | 8 | `CugaEditorAgent.propose_edit`, isolation kwargs, outcome classification |
| `src/agent_evolve/core/orchestrator.py` (modify) | 9, 11 | `select_parents`, observed lineage in `commit_to_pool`, editor wiring in `propose_edits` |
| `tests/test_editor_budget.py` (create) | 1 | budget cap + manifest exposure |
| `tests/test_cuga_editor_state.py` (create) | 2 | staging: caps, namespace, authorization, ledger |
| `tests/test_cuga_editor_evidence.py` (create) | 3 | redaction, contamination guard, leak audit |
| `tests/test_cuga_adapter_create.py` (create) | 4 | create operation, harness delivery |
| `tests/test_editor_request_parents.py` (create) | 5 | parent fields, outcome enum |
| `tests/test_cuga_editor_skills.py` (create) | 6 | skill/instruction content invariants |
| `tests/test_cuga_editor_tools.py` (create) | 7 | tool bodies end to end |
| `tests/test_cuga_editor_agent.py` (create) | 8 | `propose_edit` with a stubbed agent; taxonomy |
| `tests/test_orchestrator_multiparent.py` (create) | 9 | `select_parents`, observed lineage |
| `tests/test_orchestrator_editor_wiring.py` (create) | 11 | `propose_edits` passes parents/prefix; lineage from observed reads |
| `scripts/verify_editor_against_live_trace.py` (create) | 10 | one live editor invocation over the reference trace |

**Dependency order:**

```
Task 1 (budget)      -- independent
Task 2 (staging)     -- independent
Task 3 (evidence)    -- independent
Task 4 (adapter)     -- independent
Task 5 (contracts)   -- independent
Task 6 (skills)      -- independent
Task 7 (tools)       -- needs 2, 3, 5
Task 8 (agent)       -- needs 6, 7
Task 9 (pool/lineage)-- needs 5
Task 10 (live verify)-- needs 8
Task 11 (wiring)     -- needs 8, 9
```

Tasks 1-6 can be executed in any order or in parallel.

---

### Task 1: Editor call budget cap

Spec §12. `BudgetUsage.editor_calls` exists (`config.py:49`) but `reserve()`'s
`limit_fields` map has no entry for it, so editor calls are counted and never
capped. With 10-40 internal LLM calls per editor call this must be capped before
any experiment runs.

**Files:**
- Modify: `src/agent_evolve/core/config.py:33-42` (BudgetLimits), `:57-63` (reserve limit_fields), `:156-166` (manifest_payload budgets)
- Test: `tests/test_editor_budget.py` (create)

**Interfaces:**
- Consumes: `BudgetLimits`, `BudgetUsage`, `BudgetExceededError` from `agent_evolve.core.config` / `.errors`
- Produces: `BudgetLimits.max_editor_calls: int | None = None`; `reserve(limits, editor_calls=N)` raises `BudgetExceededError` when exceeded

- [ ] **Step 1: Write the failing tests**

Create `tests/test_editor_budget.py`:

```python
"""Editor-call budget cap (spec §12).

Each editor call becomes 10-40 internal LLM calls, so the invocation count
must be capable of being capped. Default stays None (uncapped) so existing
profiles and tests are unaffected.
"""
from __future__ import annotations

import pytest

from agent_evolve.core.config import BudgetLimits, BudgetUsage, resolve_profile
from agent_evolve.core.errors import BudgetExceededError


def test_editor_calls_budget_refuses_operation_above_limit() -> None:
    limits = BudgetLimits(max_editor_calls=2)
    usage = BudgetUsage(editor_calls=2)
    with pytest.raises(BudgetExceededError):
        usage.reserve(limits, editor_calls=1)


def test_editor_calls_budget_allows_operation_within_limit() -> None:
    limits = BudgetLimits(max_editor_calls=2)
    usage = BudgetUsage(editor_calls=1)
    usage.reserve(limits, editor_calls=1)
    assert usage.editor_calls == 2


def test_editor_calls_uncapped_by_default() -> None:
    limits = BudgetLimits()
    usage = BudgetUsage()
    assert limits.max_editor_calls is None
    usage.reserve(limits, editor_calls=1000)
    assert usage.editor_calls == 1000


def test_manifest_payload_exposes_editor_call_cap() -> None:
    config = resolve_profile("research_sequential", environ={})
    assert "max_editor_calls" in config.manifest_payload()["budgets"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_editor_budget.py -v 2>&1 | tee terminal_output/cuga-editor/task1-red.log`

Expected: FAIL. `TypeError: BudgetLimits.__init__() got an unexpected keyword argument 'max_editor_calls'` for the first three; `KeyError: 'max_editor_calls'` for the fourth.

- [ ] **Step 3: Add the field to BudgetLimits**

In `src/agent_evolve/core/config.py`, add to `BudgetLimits` after `max_judge_verdicts`:

```python
    max_editor_calls: int | None = None
```

- [ ] **Step 4: Wire it into reserve()**

In `BudgetUsage.reserve`, add to the `limit_fields` dict:

```python
            "editor_calls": "max_editor_calls",
```

- [ ] **Step 5: Expose it in manifest_payload()**

In `ResolvedConfig.manifest_payload`, add inside the `"budgets"` dict:

```python
                "max_editor_calls": self.budgets.max_editor_calls,
```

This is required, not cosmetic: `manifest_payload` enumerates budget fields
explicitly and no existing test asserts key completeness, so an omission would
pass silently and the cap would be absent from every run manifest.

- [ ] **Step 6: Run the new tests**

Run: `uv run pytest tests/test_editor_budget.py -v 2>&1 | tee terminal_output/cuga-editor/task1-green.log`
Expected: 4 passed.

- [ ] **Step 7: Run the full suite for regressions**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task1-suite.log`
Expected: 668 passed, 1 skipped (664 + 4 new). No failures.

- [ ] **Step 8: Stage (do not commit without approval)**

```bash
git add src/agent_evolve/core/config.py tests/test_editor_budget.py
```

Report: files staged, suite result. Await explicit commit approval.

---

### Task 2: EditStagingArea — staged writes, caps, authorization, parent ledger

Spec §5 (harness cluster, creation namespace), §9 (provenance). This is the
pure-Python core of the editor's write boundary. It has **no CUGA import**, so
the entire authorization and cap surface is testable offline.

**Files:**
- Create: `src/agent_evolve/adapters/cuga_editor_state.py`
- Test: `tests/test_cuga_editor_state.py` (create)

**Interfaces:**
- Consumes: `ArtifactEdit` from `agent_evolve.core.contracts`
- Produces:
  - `StageOutcome` frozen dataclass: `accepted: bool`, `reason: str = ""`
  - `EditStagingArea(write_set: tuple[str,...], creatable_prefix: str = "skills/generated-", per_attempt_create_cap: int = 2, pool_created_count: int = 0, pool_create_cap: int = 10)`
  - `.stage_replace(artifact_id: str, content: str) -> StageOutcome`
  - `.stage_create(artifact_id: str, content: str) -> StageOutcome`
  - `.unstage(artifact_id: str) -> StageOutcome`
  - `.staged_ids() -> tuple[str, ...]`
  - `.edits() -> tuple[ArtifactEdit, ...]` (operation `"replace"` or `"create"`)
  - `.record_parent_read(parent_id: str) -> None`
  - `.parents_read() -> tuple[str, ...]` (sorted)
  - `.created_count` property

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cuga_editor_state.py`:

```python
"""EditStagingArea: the editor's write boundary (spec §5, §9).

Every rule here is enforced in a tool body at staging time so the agent gets
per-artifact feedback while it works. Rejections are returned, never raised:
raising inside a CUGA tool body can abort the agent run.
"""
from __future__ import annotations

from agent_evolve.adapters.cuga_editor_state import EditStagingArea


def _area(**kwargs) -> EditStagingArea:
    defaults = dict(write_set=("skills/retrieval", "instructions"))
    defaults.update(kwargs)
    return EditStagingArea(**defaults)


# ------------------------------------------------------------------ #
# stage_replace authorization
# ------------------------------------------------------------------ #
def test_stage_replace_accepts_write_set_member() -> None:
    area = _area()
    outcome = area.stage_replace("skills/retrieval", "new body")
    assert outcome.accepted
    assert area.staged_ids() == ("skills/retrieval",)


def test_stage_replace_rejects_id_outside_write_set() -> None:
    area = _area()
    outcome = area.stage_replace("policies/execution", "x")
    assert not outcome.accepted
    assert "not in the authorized write set" in outcome.reason
    assert area.staged_ids() == ()


def test_stage_replace_returns_rejection_rather_than_raising() -> None:
    area = _area()
    # Must not raise: an exception inside a CUGA tool body can abort the run.
    outcome = area.stage_replace("nope/absent", "x")
    assert outcome.accepted is False


# ------------------------------------------------------------------ #
# stage_create namespace
# ------------------------------------------------------------------ #
def test_stage_create_accepts_namespaced_skill_id() -> None:
    area = _area()
    outcome = area.stage_create("skills/generated-recovery", "body")
    assert outcome.accepted
    assert area.created_count == 1


def test_stage_create_rejects_flat_generated_prefix() -> None:
    """A flat 'generated/' id would raise ValueError in _harness_slot.

    cuga_adapter._harness_slot accepts only 'instructions' or a
    skills|policies|memory/<name> prefix, so the CUGA group must come first.
    """
    area = _area()
    outcome = area.stage_create("generated/recovery", "body")
    assert not outcome.accepted
    assert "skills/generated-" in outcome.reason


def test_stage_create_rejects_policies_and_memory_namespaces() -> None:
    area = _area()
    for artifact_id in ("policies/generated-x", "memory/generated-x"):
        outcome = area.stage_create(artifact_id, "body")
        assert not outcome.accepted, artifact_id


def test_stage_create_rejects_existing_write_set_id() -> None:
    area = _area()
    outcome = area.stage_create("skills/retrieval", "body")
    assert not outcome.accepted
    assert "already exists" in outcome.reason


# ------------------------------------------------------------------ #
# caps
# ------------------------------------------------------------------ #
def test_stage_create_enforces_per_attempt_cap_of_two() -> None:
    area = _area()
    assert area.stage_create("skills/generated-a", "a").accepted
    assert area.stage_create("skills/generated-b", "b").accepted
    third = area.stage_create("skills/generated-c", "c")
    assert not third.accepted
    assert "per-attempt" in third.reason
    assert area.created_count == 2


def test_stage_create_enforces_pool_wide_cap() -> None:
    area = _area(pool_created_count=10, pool_create_cap=10)
    outcome = area.stage_create("skills/generated-a", "a")
    assert not outcome.accepted
    assert "pool" in outcome.reason


def test_pool_cap_counts_existing_plus_staged() -> None:
    area = _area(pool_created_count=9, pool_create_cap=10)
    assert area.stage_create("skills/generated-a", "a").accepted
    second = area.stage_create("skills/generated-b", "b")
    assert not second.accepted


# ------------------------------------------------------------------ #
# unstage / edits
# ------------------------------------------------------------------ #
def test_unstage_removes_a_staged_edit() -> None:
    area = _area()
    area.stage_replace("skills/retrieval", "x")
    assert area.unstage("skills/retrieval").accepted
    assert area.staged_ids() == ()


def test_unstage_rejects_unknown_id() -> None:
    area = _area()
    assert not area.unstage("skills/retrieval").accepted


def test_unstage_frees_a_create_slot() -> None:
    area = _area()
    area.stage_create("skills/generated-a", "a")
    area.stage_create("skills/generated-b", "b")
    area.unstage("skills/generated-b")
    assert area.stage_create("skills/generated-c", "c").accepted


def test_edits_carry_the_correct_operation() -> None:
    area = _area()
    area.stage_replace("skills/retrieval", "r")
    area.stage_create("skills/generated-a", "a")
    ops = {e.artifact_id: e.operation for e in area.edits()}
    assert ops == {
        "skills/retrieval": "replace",
        "skills/generated-a": "create",
    }


def test_edits_are_sorted_for_determinism() -> None:
    area = _area()
    area.stage_replace("instructions", "i")
    area.stage_replace("skills/retrieval", "r")
    assert [e.artifact_id for e in area.edits()] == [
        "instructions",
        "skills/retrieval",
    ]


def test_restaging_the_same_id_replaces_the_content() -> None:
    area = _area()
    area.stage_replace("skills/retrieval", "first")
    area.stage_replace("skills/retrieval", "second")
    assert area.staged_ids() == ("skills/retrieval",)
    assert area.edits()[0].payload["content"] == "second"


# ------------------------------------------------------------------ #
# parent read ledger (provenance, spec §9)
# ------------------------------------------------------------------ #
def test_parents_read_is_empty_before_any_read() -> None:
    assert _area().parents_read() == ()


def test_parents_read_records_reads_deduplicated_and_sorted() -> None:
    area = _area()
    area.record_parent_read("cand-b")
    area.record_parent_read("cand-a")
    area.record_parent_read("cand-b")
    assert area.parents_read() == ("cand-a", "cand-b")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cuga_editor_state.py -v 2>&1 | tee terminal_output/cuga-editor/task2-red.log`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_evolve.adapters.cuga_editor_state'`

- [ ] **Step 3: Write the implementation**

Create `src/agent_evolve/adapters/cuga_editor_state.py`:

```python
"""Staged-write boundary for the CUGA editor agent.

Pure Python: no CUGA, no filesystem, no network. The editor agent stages edits
incrementally through tool bodies and finalizes once, so every authorization
and cap rule is enforced at staging time with per-artifact feedback.

Rejections are RETURNED, never raised. An exception inside a CUGA tool body can
abort the whole agent run, which would turn a recoverable authorization mistake
into a lost attempt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_evolve.core.contracts import ArtifactEdit

# Created ids must carry the CUGA group first. ``_harness_slot`` in
# cuga_adapter.py accepts only ``instructions`` or a
# ``skills|policies|memory/<name>`` prefix, so a flat ``generated/<name>`` would
# raise ValueError at registration and the creation path would be dead code.
DEFAULT_CREATABLE_PREFIX = "skills/generated-"
DEFAULT_PER_ATTEMPT_CREATE_CAP = 2
DEFAULT_POOL_CREATE_CAP = 10


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Result of one staging operation."""

    accepted: bool
    reason: str = ""


@dataclass(slots=True)
class EditStagingArea:
    """Accumulates authorized edits for one editor attempt."""

    write_set: tuple[str, ...]
    creatable_prefix: str = DEFAULT_CREATABLE_PREFIX
    per_attempt_create_cap: int = DEFAULT_PER_ATTEMPT_CREATE_CAP
    pool_created_count: int = 0
    pool_create_cap: int = DEFAULT_POOL_CREATE_CAP
    _replaced: dict[str, str] = field(default_factory=dict)
    _created: dict[str, str] = field(default_factory=dict)
    _parents_read: set[str] = field(default_factory=set)

    # -------------------------------------------------------------- #
    # Writes
    # -------------------------------------------------------------- #
    def stage_replace(self, artifact_id: str, content: str) -> StageOutcome:
        if artifact_id not in self.write_set:
            return StageOutcome(
                False,
                f"{artifact_id!r} is not in the authorized write set "
                f"{sorted(self.write_set)}",
            )
        if not isinstance(content, str):
            return StageOutcome(False, "content must be a string")
        self._replaced[artifact_id] = content
        return StageOutcome(True, f"staged replacement for {artifact_id!r}")

    def stage_create(self, artifact_id: str, content: str) -> StageOutcome:
        if artifact_id in self.write_set:
            return StageOutcome(
                False,
                f"{artifact_id!r} already exists; use stage_replace instead",
            )
        if not artifact_id.startswith(self.creatable_prefix):
            return StageOutcome(
                False,
                f"created artifacts must start with {self.creatable_prefix!r}; "
                f"got {artifact_id!r}",
            )
        if len(artifact_id) <= len(self.creatable_prefix):
            return StageOutcome(False, "created artifact needs a name after the prefix")
        if not isinstance(content, str):
            return StageOutcome(False, "content must be a string")
        if artifact_id not in self._created and (
            len(self._created) >= self.per_attempt_create_cap
        ):
            return StageOutcome(
                False,
                f"per-attempt creation cap of {self.per_attempt_create_cap} reached",
            )
        if artifact_id not in self._created and (
            self.pool_created_count + len(self._created) >= self.pool_create_cap
        ):
            return StageOutcome(
                False,
                f"pool-wide creation cap of {self.pool_create_cap} reached",
            )
        self._created[artifact_id] = content
        return StageOutcome(True, f"staged creation of {artifact_id!r}")

    def unstage(self, artifact_id: str) -> StageOutcome:
        if self._replaced.pop(artifact_id, None) is not None:
            return StageOutcome(True, f"unstaged {artifact_id!r}")
        if self._created.pop(artifact_id, None) is not None:
            return StageOutcome(True, f"unstaged {artifact_id!r}")
        return StageOutcome(False, f"{artifact_id!r} is not staged")

    # -------------------------------------------------------------- #
    # Reads
    # -------------------------------------------------------------- #
    @property
    def created_count(self) -> int:
        return len(self._created)

    def staged_ids(self) -> tuple[str, ...]:
        return tuple(sorted({*self._replaced, *self._created}))

    def edits(self) -> tuple[ArtifactEdit, ...]:
        staged: list[ArtifactEdit] = [
            ArtifactEdit(
                artifact_id=artifact_id,
                operation="replace",
                payload={"content": content},
            )
            for artifact_id, content in self._replaced.items()
        ]
        staged.extend(
            ArtifactEdit(
                artifact_id=artifact_id,
                operation="create",
                payload={"content": content},
            )
            for artifact_id, content in self._created.items()
        )
        return tuple(sorted(staged, key=lambda e: e.artifact_id))

    # -------------------------------------------------------------- #
    # Provenance ledger
    # -------------------------------------------------------------- #
    def record_parent_read(self, parent_id: str) -> None:
        """Record that donor ``parent_id`` was actually read.

        This ledger -- not the agent's prose -- is the provenance source for
        ``parent_ids`` (spec §9), following the same tool-execution-over-
        narration principle that makes ``ingest_sdk_tool_calls`` correct.
        """
        if parent_id:
            self._parents_read.add(parent_id)

    def parents_read(self) -> tuple[str, ...]:
        return tuple(sorted(self._parents_read))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cuga_editor_state.py -v 2>&1 | tee terminal_output/cuga-editor/task2-green.log`
Expected: 20 passed.

- [ ] **Step 5: Verify no CUGA import leaked in**

Run: `grep -n "cuga\|langchain" src/agent_evolve/adapters/cuga_editor_state.py`
Expected: matches only in the module docstring/comments, never in an `import` statement.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task2-suite.log`
Expected: 688 passed, 1 skipped. No failures.

- [ ] **Step 7: Stage (do not commit without approval)**

```bash
git add src/agent_evolve/adapters/cuga_editor_state.py tests/test_cuga_editor_state.py
```

---

### Task 3: Evidence view and contamination guard

Spec §8. The editor sees blame, artifacts, history and the task's `input_text` —
never `expected_contract`, `final_output`, or blob contents. Because `tool_call`
payloads ARE exposed, a fail-closed guard scans assembled payloads for
`expected_contract` string values and drops matches.

Pure Python, no CUGA import.

**Files:**
- Create: `src/agent_evolve/adapters/cuga_editor_evidence.py`
- Test: `tests/test_cuga_editor_evidence.py` (create)

**Interfaces:**
- Consumes: `ExecutionTrace`, `TraceEvent`, `EvolutionTask` from `agent_evolve.core.contracts`; `CausalAnalysis` from `agent_evolve.core.blame`
- Produces:
  - `EvidenceView(analysis, trace, task, contamination_terms: tuple[str,...])`
  - `.mechanism() -> dict[str, object]`
  - `.blamed_actors() -> tuple[dict[str, object], ...]`
  - `.task_input() -> str`
  - `.actors() -> tuple[str, ...]`
  - `.events(kind: str | None = None, actor_id: str | None = None, limit: int = 50) -> tuple[dict[str, object], ...]`
  - `.redaction_count` property
  - `contamination_terms_from(task) -> tuple[str, ...]` module function

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cuga_editor_evidence.py`:

```python
"""Editor evidence boundary and contamination guard (spec §8).

The guard CONSUMES expected_contract to build its term list but never SHOWS it.
Key-name denylisting alone is insufficient: sanitize_payload matches keys such
as 'expected_answer', not an expected answer appearing as free text inside a
tool result string.
"""
from __future__ import annotations

from agent_evolve.adapters.cuga_editor_evidence import (
    EvidenceView,
    contamination_terms_from,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent

_SECRET = "token-a"


def _task() -> EvolutionTask:
    return EvolutionTask(
        task_id="task-a",
        input_text="produce the A capability",
        expected_contract={"expected_substring": _SECRET},
    )


def _analysis() -> CausalAnalysis:
    return CausalAnalysis(
        mechanism="skill never loaded",
        severity=0.9,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(
                BlameNode(actor_id="call_model", blame=0.7, artifacts=("skills/retrieval",)),
                BlameNode(actor_id="prepare", blame=0.3, artifacts=()),
            )
        ),
    )


def _trace(events: tuple[TraceEvent, ...] | None = None) -> ExecutionTrace:
    if events is None:
        events = (
            TraceEvent(
                event_id="graph:1",
                kind="llm_call",
                actor_id="call_model",
                parent_event_id="graph:0",
                payload={"messages_ref": "a" * 64, "sequence": 1},
            ),
            TraceEvent(
                event_id="graph:2",
                kind="tool_call",
                actor_id="sandbox",
                parent_event_id="graph:1",
                payload={"name": "run_command", "result": "exit 0"},
            ),
        )
    return ExecutionTrace(
        trace_id="trace-1",
        candidate_id="cand-1",
        task_id="task-a",
        events=events,
        final_output=f"the answer is {_SECRET}",
        status="completed",
    )


def _view(trace: ExecutionTrace | None = None) -> EvidenceView:
    task = _task()
    return EvidenceView(
        analysis=_analysis(),
        trace=trace if trace is not None else _trace(),
        task=task,
        contamination_terms=contamination_terms_from(task),
    )


# ------------------------------------------------------------------ #
# term extraction
# ------------------------------------------------------------------ #
def test_contamination_terms_extracts_string_values() -> None:
    assert contamination_terms_from(_task()) == (_SECRET,)


def test_contamination_terms_ignores_short_and_nonstring_values() -> None:
    task = EvolutionTask(
        task_id="t",
        input_text="i",
        expected_contract={"expected_substring": "ab", "threshold": 0.5},
    )
    # 2-char terms are too short to scan safely (false positives everywhere).
    assert contamination_terms_from(task) == ()


# ------------------------------------------------------------------ #
# what the editor may see
# ------------------------------------------------------------------ #
def test_mechanism_exposes_description_and_severity() -> None:
    assert _view().mechanism() == {
        "mechanism": "skill never loaded",
        "severity": 0.9,
    }


def test_blamed_actors_are_sorted_by_blame_descending() -> None:
    actors = _view().blamed_actors()
    assert [a["actor_id"] for a in actors] == ["call_model", "prepare"]
    assert actors[0]["artifacts"] == ("skills/retrieval",)


def test_task_input_exposes_input_text() -> None:
    assert _view().task_input() == "produce the A capability"


def test_actors_lists_distinct_trace_actors() -> None:
    assert _view().actors() == ("call_model", "sandbox")


# ------------------------------------------------------------------ #
# what the editor may NOT see
# ------------------------------------------------------------------ #
def test_events_strip_ref_payload_keys() -> None:
    events = _view().events()
    llm = next(e for e in events if e["kind"] == "llm_call")
    assert "messages_ref" not in llm["payload"]


def test_events_keep_tool_call_payloads() -> None:
    events = _view().events(kind="tool_call")
    assert events[0]["payload"]["name"] == "run_command"


def test_contamination_guard_drops_payload_containing_expected_value() -> None:
    dirty = (
        TraceEvent(
            event_id="graph:3",
            kind="tool_call",
            actor_id="sandbox",
            parent_event_id=None,
            payload={"name": "run_command", "result": f"found {_SECRET} here"},
        ),
    )
    view = _view(_trace(dirty))
    events = view.events()
    assert events[0]["payload"] == {}
    assert events[0]["payload_redacted"] is True
    assert view.redaction_count == 1


def test_no_view_output_contains_the_expected_value() -> None:
    """Full leak audit across every exposed surface."""
    view = _view()
    blob = repr(
        (
            view.mechanism(),
            view.blamed_actors(),
            view.task_input(),
            view.actors(),
            view.events(limit=100),
        )
    )
    assert _SECRET not in blob


def test_no_view_output_contains_the_final_output() -> None:
    view = _view()
    blob = repr((view.mechanism(), view.events(limit=100), view.task_input()))
    assert "the answer is" not in blob


# ------------------------------------------------------------------ #
# filtering and bounding
# ------------------------------------------------------------------ #
def test_events_filter_by_kind() -> None:
    events = _view().events(kind="tool_call")
    assert [e["kind"] for e in events] == ["tool_call"]


def test_events_filter_by_actor() -> None:
    events = _view().events(actor_id="call_model")
    assert [e["actor_id"] for e in events] == ["call_model"]


def test_events_respect_the_limit() -> None:
    assert len(_view().events(limit=1)) == 1


def test_events_preserve_dag_fields() -> None:
    llm = _view().events(kind="llm_call")[0]
    assert llm["event_id"] == "graph:1"
    assert llm["parent_event_id"] == "graph:0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cuga_editor_evidence.py -v 2>&1 | tee terminal_output/cuga-editor/task3-red.log`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_evolve.adapters.cuga_editor_evidence'`

- [ ] **Step 3: Write the implementation**

Create `src/agent_evolve/adapters/cuga_editor_evidence.py`:

```python
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
    """Extract scannable string values from a task's expected contract."""
    terms: list[str] = []
    for value in task.expected_contract.values():
        if isinstance(value, str) and len(value) >= _MIN_TERM_LENGTH:
            terms.append(value)
    return tuple(terms)


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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cuga_editor_evidence.py -v 2>&1 | tee terminal_output/cuga-editor/task3-green.log`
Expected: 16 passed.

- [ ] **Step 5: Verify no CUGA import**

Run: `grep -n "^from cuga\|^import cuga\|langchain" src/agent_evolve/adapters/cuga_editor_evidence.py`
Expected: no output.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task3-suite.log`
Expected: 704 passed, 1 skipped.

- [ ] **Step 7: Stage**

```bash
git add src/agent_evolve/adapters/cuga_editor_evidence.py tests/test_cuga_editor_evidence.py
```

---

### Task 4: Adapter create operation and creatable-prefix declaration

Spec §5. `apply_structured_edits` currently raises `KeyError` for any id not
already present (`cuga_adapter.py:109-110`) and accepts only `replace`
(`:107`), so creation is impossible. This task adds the `create` operation.

**Files:**
- Modify: `src/agent_evolve/adapters/cuga_adapter.py:102-115` (`apply_structured_edits`)
- Test: `tests/test_cuga_adapter_create.py` (create)

**Interfaces:**
- Consumes: `EditStagingArea.DEFAULT_CREATABLE_PREFIX` from Task 2
- Produces:
  - `CugaAdapter.apply_structured_edits` accepts `operation="create"`
  - `CugaAdapter.creatable_prefix` class attribute (`"skills/generated-"`)
  - `CugaAdapter.created_artifact_count(version) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cuga_adapter_create.py`:

```python
"""Adapter support for artifact creation (spec §5).

Creation must map onto a real CUGA harness slot. A flat 'generated/<name>' id
would raise ValueError in _harness_slot, so the CUGA group comes first:
'skills/generated-<name>'.
"""
from __future__ import annotations

import pytest

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.core.contracts import ArtifactEdit
from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime


def _adapter() -> CugaAdapter:
    adapter = CugaAdapter(wrapper=CugaWrapper(runtime=InMemoryRuntime()))
    adapter.register_candidate("base-v0", {"skills/retrieval": "body"})
    return adapter


def test_create_adds_a_new_artifact() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    result = adapter.apply_structured_edits(
        ws,
        (
            ArtifactEdit(
                artifact_id="skills/generated-recovery",
                operation="create",
                payload={"content": "new skill body"},
            ),
        ),
    )
    assert result["skills/generated-recovery"] == "new skill body"


def test_created_artifact_appears_in_inventory() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    adapter.apply_structured_edits(
        ws,
        (
            ArtifactEdit(
                artifact_id="skills/generated-recovery",
                operation="create",
                payload={"content": "b"},
            ),
        ),
    )
    ids = [d.artifact_id for d in adapter.artifact_inventory(ws.version)]
    assert "skills/generated-recovery" in ids


def test_created_artifact_reaches_the_harness_config() -> None:
    """A created skill must actually be delivered to CUGA, not merely stored."""
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    adapter.apply_structured_edits(
        ws,
        (
            ArtifactEdit(
                artifact_id="skills/generated-recovery",
                operation="create",
                payload={"content": "b"},
            ),
        ),
    )
    from agent_evolve.core.contracts import EvolutionTask

    config = adapter._harness_config(
        ws.version, EvolutionTask(task_id="t", input_text="i")
    )
    assert config["skills"]["generated-recovery"] == "b"


def test_create_rejects_an_unmappable_id() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    with pytest.raises(ValueError, match="does not map to a CUGA harness slot"):
        adapter.apply_structured_edits(
            ws,
            (
                ArtifactEdit(
                    artifact_id="generated/recovery",
                    operation="create",
                    payload={"content": "b"},
                ),
            ),
        )


def test_create_rejects_an_existing_id() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    with pytest.raises(ValueError, match="already exists"):
        adapter.apply_structured_edits(
            ws,
            (
                ArtifactEdit(
                    artifact_id="skills/retrieval",
                    operation="create",
                    payload={"content": "b"},
                ),
            ),
        )


def test_replace_still_rejects_an_absent_id() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    with pytest.raises(KeyError):
        adapter.apply_structured_edits(
            ws,
            (
                ArtifactEdit(
                    artifact_id="skills/absent",
                    operation="replace",
                    payload={"content": "b"},
                ),
            ),
        )


def test_created_artifact_count_tracks_generated_prefix() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    assert adapter.created_artifact_count(ws.version) == 0
    adapter.apply_structured_edits(
        ws,
        (
            ArtifactEdit(
                artifact_id="skills/generated-a",
                operation="create",
                payload={"content": "a"},
            ),
        ),
    )
    assert adapter.created_artifact_count(ws.version) == 1


def test_creatable_prefix_is_declared() -> None:
    assert CugaAdapter.creatable_prefix == "skills/generated-"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cuga_adapter_create.py -v 2>&1 | tee terminal_output/cuga-editor/task4-red.log`
Expected: FAIL — `ValueError: unsupported CUGA wrapper edit operation: create` and `AttributeError: creatable_prefix`.

- [ ] **Step 3: Add the prefix declaration**

In `src/agent_evolve/adapters/cuga_adapter.py`, inside the `CugaAdapter`
dataclass beside `adapter_name`:

```python
    # Created artifacts must carry the CUGA group first so ``_harness_slot``
    # accepts them; a flat ``generated/<name>`` would raise ValueError.
    creatable_prefix: str = "skills/generated-"
```

- [ ] **Step 4: Replace apply_structured_edits**

Replace the body of `apply_structured_edits` (currently `cuga_adapter.py:102-115`):

```python
    def apply_structured_edits(
        self, workspace: CandidateWorkspace, edits: Sequence[ArtifactEdit]
    ) -> Mapping[str, str]:
        artifacts = self._workspaces[workspace.version]
        for edit in edits:
            content = edit.payload.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"{edit.operation} edits require a string payload.content"
                )
            if edit.operation == "replace":
                if edit.artifact_id not in artifacts:
                    raise KeyError(edit.artifact_id)
            elif edit.operation == "create":
                if edit.artifact_id in artifacts:
                    raise ValueError(
                        f"artifact {edit.artifact_id!r} already exists; "
                        "use operation='replace'"
                    )
                # Fail loudly on an id CUGA cannot receive. A silently dropped
                # creation would report a successful edit that never reached
                # the agent.
                self._harness_slot(edit.artifact_id)
            else:
                raise ValueError(
                    f"unsupported CUGA wrapper edit operation: {edit.operation}"
                )
            artifacts[edit.artifact_id] = content
        return dict(artifacts)

    def created_artifact_count(self, version: str) -> int:
        """How many artifacts in this version came from editor creation."""
        return sum(
            1
            for artifact_id in self._artifacts_for(version)
            if artifact_id.startswith(self.creatable_prefix)
        )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_cuga_adapter_create.py -v 2>&1 | tee terminal_output/cuga-editor/task4-green.log`
Expected: 8 passed.

- [ ] **Step 6: Run the existing adapter tests for regressions**

Run: `uv run pytest tests/test_cuga_adapter.py tests/test_cuga_adapter_wiring.py -v 2>&1 | tee terminal_output/cuga-editor/task4-adapter.log`
Expected: all pass. The `replace`-rejects-absent-id behavior must be unchanged.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task4-suite.log`
Expected: 712 passed, 1 skipped.

- [ ] **Step 8: Stage**

```bash
git add src/agent_evolve/adapters/cuga_adapter.py tests/test_cuga_adapter_create.py
```

---

### Task 5: EditorRequest extension and EditorOutcome taxonomy

Spec §7, §10. `EditorRequest` gains optional parent context and a creatable
prefix. `EditorOutcome` makes `no_tool_call` distinct from `no_op`.

Note `EditorRequest.__post_init__` (`editor.py:77-82`) requires
`current_artifacts ⊆ write_set`, pinned by
`tests/test_editor.py:68`. That guard stays intact: donor content is fetched
through a tool at run time, never placed in `current_artifacts`.

**Files:**
- Modify: `src/agent_evolve/core/editor.py:54-82` (`EditorRequest`), and add `EditorOutcome`
- Test: `tests/test_editor_request_parents.py` (create)

**Interfaces:**
- Produces:
  - `EditorRequest.parents: tuple[ParentContext, ...] = ()`
  - `EditorRequest.creatable_prefix: str = ""` (empty disables creation)
  - `EditorRequest.pool_created_count: int = 0`
  - `ParentContext` frozen dataclass: `candidate_id: str`, `version: str`, `is_primary: bool`, `score_summary: Mapping[str, float]`
  - `EditorOutcome` str-enum: `VALID`, `NO_TOOL_CALL`, `NO_OP`, `UNAVAILABLE`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_editor_request_parents.py`:

```python
"""Multi-parent editor request fields and outcome taxonomy (spec §7, §10)."""
from __future__ import annotations

import pytest

from agent_evolve.core.blame import BlameGraph, CausalAnalysis
from agent_evolve.core.contracts import CandidateWorkspace, EvolutionTask
from agent_evolve.core.editor import (
    EditorOutcome,
    EditorRequest,
    ParentContext,
)
from pathlib import Path


def _request(**kwargs) -> EditorRequest:
    defaults = dict(
        base_workspace=CandidateWorkspace("att-1", "v1", Path("."), "v0"),
        task=EvolutionTask(task_id="t", input_text="i"),
        analysis=CausalAnalysis(
            mechanism="m", severity=0.5, score=0.0,
            blame_graph=BlameGraph(nodes=()),
        ),
        issue_id="issue-1",
        write_set=("skills/a",),
    )
    defaults.update(kwargs)
    return EditorRequest(**defaults)


def test_parents_defaults_to_empty() -> None:
    assert _request().parents == ()


def test_creatable_prefix_defaults_to_disabled() -> None:
    assert _request().creatable_prefix == ""


def test_pool_created_count_defaults_to_zero() -> None:
    assert _request().pool_created_count == 0


def test_request_accepts_parent_context() -> None:
    primary = ParentContext(
        candidate_id="cand-1", version="v1", is_primary=True,
        score_summary={"task-a": 0.5},
    )
    donor = ParentContext(
        candidate_id="cand-2", version="v2", is_primary=False,
        score_summary={"task-a": 0.9},
    )
    request = _request(parents=(primary, donor))
    assert [p.candidate_id for p in request.parents] == ["cand-1", "cand-2"]


def test_request_rejects_more_than_one_primary_parent() -> None:
    a = ParentContext(candidate_id="c1", version="v1", is_primary=True, score_summary={})
    b = ParentContext(candidate_id="c2", version="v2", is_primary=True, score_summary={})
    with pytest.raises(ValueError, match="exactly one primary parent"):
        _request(parents=(a, b))


def test_request_rejects_parents_without_a_primary() -> None:
    a = ParentContext(candidate_id="c1", version="v1", is_primary=False, score_summary={})
    with pytest.raises(ValueError, match="exactly one primary parent"):
        _request(parents=(a,))


def test_current_artifacts_subset_guard_is_unchanged() -> None:
    """The existing write_set guard must survive the extension."""
    with pytest.raises(ValueError, match="outside write_set"):
        _request(current_artifacts={"skills/b": "x"})


def test_outcome_distinguishes_no_tool_call_from_no_op() -> None:
    assert EditorOutcome.NO_TOOL_CALL != EditorOutcome.NO_OP
    assert EditorOutcome.NO_TOOL_CALL.value == "no_tool_call"
    assert EditorOutcome.NO_OP.value == "no_op"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_editor_request_parents.py -v 2>&1 | tee terminal_output/cuga-editor/task5-red.log`
Expected: FAIL — `ImportError: cannot import name 'EditorOutcome'`.

- [ ] **Step 3: Add ParentContext and EditorOutcome**

In `src/agent_evolve/core/editor.py`, after the imports and before `EditorRequest`:

```python
class EditorOutcome(str, Enum):
    """How one editor invocation terminated.

    ``NO_TOOL_CALL`` must stay distinct from ``NO_OP``. Collapsing them would
    let "the agent did not engage" masquerade as "the agent judged no edit
    warranted" -- the same category of error that produced the retracted
    Phase 8 E2E PASS.
    """

    VALID = "valid"
    NO_TOOL_CALL = "no_tool_call"
    NO_OP = "no_op"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ParentContext:
    """One candidate exposed to the editor as an edit source.

    The primary parent owns the workspace being written. Donors are read-only:
    the editor may draw content from a donor but always writes into the
    primary's workspace.
    """

    candidate_id: str
    version: str
    is_primary: bool
    score_summary: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not self.version:
            raise ValueError("version is required")
```

`Enum` is already imported at `editor.py:28`; `field` at `:27`; `Mapping` at `:29`.

- [ ] **Step 4: Extend EditorRequest**

Add these fields to `EditorRequest` after `correction_request`:

```python
    # Candidates the editor may draw from. Empty means single-parent editing.
    # Exactly one entry must be primary when non-empty.
    parents: tuple[ParentContext, ...] = ()
    # Prefix new artifact ids must carry. Empty disables creation.
    creatable_prefix: str = ""
    # Generated artifacts already present pool-wide, for the creation cap.
    pool_created_count: int = 0
```

And append to `__post_init__`:

```python
        if self.parents:
            primaries = [p for p in self.parents if p.is_primary]
            if len(primaries) != 1:
                raise ValueError(
                    "parents must contain exactly one primary parent, "
                    f"got {len(primaries)}"
                )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_editor_request_parents.py -v 2>&1 | tee terminal_output/cuga-editor/task5-green.log`
Expected: 9 passed.

- [ ] **Step 6: Verify FakeEditor and existing editor tests still pass**

Run: `uv run pytest tests/test_editor.py tests/test_editor_floors.py tests/test_orchestrator.py tests/test_phase_6_orchestrator.py -v 2>&1 | tee terminal_output/cuga-editor/task5-compat.log`
Expected: all pass. New fields are optional with defaults, so existing callers are unaffected.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task5-suite.log`
Expected: 721 passed, 1 skipped.

- [ ] **Step 8: Stage**

```bash
git add src/agent_evolve/core/editor.py tests/test_editor_request_parents.py
```

---

### Task 6: Editor skills and instructions text

Spec §6. The agent needs to know HOW to evolve a harness, not merely which
tools exist. Split by CUGA's injection semantics: invariants go in
`special_instructions` (always present), procedures go in on-demand skills.

Pure text module, no imports beyond typing. Testable by assertion on content.

**Files:**
- Create: `src/agent_evolve/adapters/cuga_editor_skills.py`
- Test: `tests/test_cuga_editor_skills.py` (create)

**Interfaces:**
- Produces:
  - `EDITOR_INSTRUCTIONS: str`
  - `EDITOR_SKILLS: dict[str, str]` with keys `refine-artifact`, `combine-parents`, `create-artifact`, `learn-from-history`
  - `build_editor_prompt(evidence_summary: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cuga_editor_skills.py`:

```python
"""Editor agent instructions and skills (spec §6)."""
from __future__ import annotations

from agent_evolve.adapters.cuga_editor_skills import (
    EDITOR_INSTRUCTIONS,
    EDITOR_SKILLS,
    build_editor_prompt,
)


def test_four_skills_are_defined() -> None:
    assert set(EDITOR_SKILLS) == {
        "refine-artifact",
        "combine-parents",
        "create-artifact",
        "learn-from-history",
    }


def test_instructions_state_the_authorization_invariant() -> None:
    assert "authorized" in EDITOR_INSTRUCTIONS.lower()


def test_instructions_require_finalizing_even_when_declining() -> None:
    lowered = EDITOR_INSTRUCTIONS.lower()
    assert "submit_edit_plan" in lowered
    assert "declin" in lowered


def test_instructions_nudge_both_refine_and_combine() -> None:
    lowered = EDITOR_INSTRUCTIONS.lower()
    assert "refine" in lowered
    assert "combine" in lowered


def test_instructions_never_mention_the_expected_contract() -> None:
    """The editor must not be told an expected answer exists to look for."""
    lowered = EDITOR_INSTRUCTIONS.lower()
    for banned in ("expected_contract", "expected answer", "expected_substring"):
        assert banned not in lowered


def test_no_skill_mentions_the_expected_contract() -> None:
    for name, body in EDITOR_SKILLS.items():
        lowered = body.lower()
        for banned in ("expected_contract", "expected answer", "expected_substring"):
            assert banned not in lowered, f"{name} leaks {banned}"


def test_every_skill_is_non_trivial() -> None:
    for name, body in EDITOR_SKILLS.items():
        assert len(body.strip()) > 200, f"{name} is too thin to guide behavior"


def test_prompt_embeds_the_evidence_summary() -> None:
    prompt = build_editor_prompt("MECHANISM: skill never loaded")
    assert "MECHANISM: skill never loaded" in prompt


def test_prompt_directs_the_agent_to_finalize() -> None:
    assert "submit_edit_plan" in build_editor_prompt("x")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cuga_editor_skills.py -v 2>&1 | tee terminal_output/cuga-editor/task6-red.log`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `src/agent_evolve/adapters/cuga_editor_skills.py`:

```python
"""Instructions and skills that teach the editor agent how to evolve a harness.

Split follows CUGA's injection semantics (see
feedback/gpt_context/cuga_skills_polices_etc.md):

* ``special_instructions`` are always-present behavioral configuration, so they
  carry the invariants that must hold whatever the agent decides to do.
* Skills are on-demand procedures loaded via ``load_skill``, so they carry the
  per-strategy playbooks.

These texts are hand-authored. Edit quality is therefore bounded by them; that
limitation is recorded in the design doc §13.
"""
from __future__ import annotations

EDITOR_INSTRUCTIONS = """\
You improve an AI agent's harness by editing its artifacts. You are given
evidence about one failed run and a set of artifacts you may change.

Invariants, which hold no matter what you decide to do:

1. Write only where you are authorized. If a staging call is rejected, the
   answer is to rethink the target, not to retry the same write.
2. Prefer the smallest change that addresses the evidence. Do not rewrite an
   artifact wholesale when a targeted change suffices.
3. Ground every change in the blame evidence you were given. Do not make
   general "improvements" that the evidence does not support.
4. Declining to change anything is a legitimate, useful outcome. If the
   evidence does not justify an edit, say so.
5. Always finish by calling submit_edit_plan, including when you are declining.
   Work that is never finalized is discarded and counts as no engagement.

You have two ways to change the harness, and you choose between them from the
evidence:

* Refine: change an artifact the primary parent already owns. Choose this when
  blame points at a specific artifact whose content is wrong or incomplete.
* Combine: take content from a donor parent that performs better on the failing
  task. Choose this when a donor already solves what the primary cannot.

You may also create a new artifact when no existing artifact covers the failure
at all. Use each mechanism when the evidence calls for it; do not default to one
because it is easier.

Read before you write. Consult past attempts before repeating a strategy.
"""

_REFINE = """\
# Refining an existing artifact

Use when the blame graph points at an artifact the primary parent already owns.

Procedure:

1. Call get_mechanism and list_blamed_actors. Note which artifacts the
   highest-blame actors are attributed.
2. Call list_artifacts to see what you may write, then read_artifact on the
   attributed artifact. Never edit content you have not read.
3. Locate the specific gap the mechanism describes. A mechanism such as
   "skill never loaded" points at discoverability; "wrong argument order"
   points at a procedure step. These need different changes.
4. Call read_trace_events, filtered by the blamed actor, to see how far
   execution got. An actor that never appears did not run, which is different
   from an actor that ran and produced the wrong result.
5. Make the smallest change that closes the gap. Preserve working content:
   the artifact may already be succeeding on tasks you cannot see.
6. Call stage_replace, then submit_edit_plan with a rationale naming the
   mechanism and what you changed.
"""

_COMBINE = """\
# Combining content from a donor parent

Use when a donor parent performs better than the primary on the failing task.

Donors are read-only. You always write into the primary parent's artifacts.

Procedure:

1. Call list_parents. Compare each donor's score summary against the primary's
   on the failing task. A donor with no advantage is not worth reading.
2. Call read_parent_artifact on the donor artifact matching the blamed
   artifact, then read_artifact on the primary's version of it.
3. Compare them. Identify precisely what the donor does that the primary does
   not. That difference, not the donor's whole text, is what you want.
4. Transplant the difference into the primary's content. Do not paste the
   donor artifact over the primary wholesale: the primary may contain
   improvements the donor lacks, and you would silently discard them.
5. Call stage_replace on the primary's artifact id, then submit_edit_plan with
   a rationale naming the donor and the transplanted capability.

Reading a donor is recorded as provenance, so read the donors you actually use.
"""

_CREATE = """\
# Creating a new artifact

Use only when no existing artifact addresses the failure mechanism. This is a
strong claim: check list_artifacts and read the plausible candidates first.

The clearest case is a mechanism describing a capability that is entirely
absent, rather than one that is present but wrong.

Procedure:

1. Call list_artifacts and read_artifact on every artifact that could
   plausibly cover the mechanism. Confirm none does.
2. Choose an id beginning with the required creation prefix. The group comes
   first, then the generated marker, then your name.
3. Write a focused artifact covering the missing capability. A new artifact
   competing with an existing one splits behavior unpredictably; a new
   artifact covering a genuine gap does not.
4. Call stage_create, then submit_edit_plan explaining what was absent and why
   an existing artifact could not carry it.

Creation is capped per attempt and pool-wide. If a cap rejects your call, the
answer is to refine instead.
"""

_HISTORY = """\
# Learning from previous attempts

Consult history before proposing a strategy that may already have been tried.

Procedure:

1. Call search_edit_history for this issue. Results are bounded.
2. Call get_attempt_outcome on relevant attempts. Each is worked, failed, or
   regression.
3. Interpret them:
   * worked: the approach was accepted. Build on it rather than replacing it.
   * failed: the approach did not improve the outcome. A variation may still
     work, but repeating it verbatim will not.
   * regression: the approach broke something that previously worked. Treat
     this as a boundary, not a starting point.
4. If several attempts on this issue failed the same way, the artifact you are
   editing may not be the cause. Consider a different target, or decline with
   a rationale naming what you ruled out.
"""

EDITOR_SKILLS: dict[str, str] = {
    "refine-artifact": _REFINE,
    "combine-parents": _COMBINE,
    "create-artifact": _CREATE,
    "learn-from-history": _HISTORY,
}


def build_editor_prompt(evidence_summary: str) -> str:
    """Build the single user message that starts the editor agent's run."""
    return (
        "A harness run failed. Evidence:\n\n"
        f"{evidence_summary}\n\n"
        "Investigate with the tools available to you, then change the harness "
        "so this failure is less likely. Finish by calling submit_edit_plan, "
        "including if you decide no change is warranted."
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cuga_editor_skills.py -v 2>&1 | tee terminal_output/cuga-editor/task6-green.log`
Expected: 9 passed.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task6-suite.log`
Expected: 730 passed, 1 skipped.

- [ ] **Step 6: Stage**

```bash
git add src/agent_evolve/adapters/cuga_editor_skills.py tests/test_cuga_editor_skills.py
```

---

### Task 7: Tool cluster builder

Spec §5. Builds the LangChain tools the editor agent calls, clustered by
`tracked_tool(app_name=...)`. Every tool closes over the request's staging area,
evidence view and adapter — never global state. The CUGA/LangChain import is
deferred into `build_editor_tools` so the module imports offline.

The tool *bodies* are built by `build_tool_callables`, which has no CUGA
dependency at all, so every rule is unit-testable without the SDK.

**Files:**
- Create: `src/agent_evolve/adapters/cuga_editor_tools.py`
- Test: `tests/test_cuga_editor_tools.py` (create)

**Interfaces:**
- Consumes: `EditStagingArea`, `StageOutcome` (Task 2); `EvidenceView` (Task 3); `CugaAdapter` (Task 4); `EditorRequest`, `ParentContext` (Task 5)
- Produces:
  - `EditorToolContext` mutable dataclass (`slots=True`): `staging`, `evidence`, `request`, `adapter`, `memory`, private `_plan`
  - `build_tool_callables(ctx: EditorToolContext) -> dict[str, Callable[..., str]]` — plain functions, JSON-string returns, no CUGA
  - `build_editor_tools(ctx: EditorToolContext) -> list` — wraps the callables via `tracked_tool` + `tool`
  - `TOOL_APP_NAMES: dict[str, str]` — tool name to cluster name
  - `submitted_plan(ctx) -> dict | None` — the captured plan, or None if never finalized

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cuga_editor_tools.py`:

```python
"""Editor tool bodies (spec §5).

build_tool_callables returns plain functions with no CUGA dependency, so the
entire authorization, evidence and capture surface is testable offline.

Every tool returns a JSON string: CUGA tools must return strings, and a
structured error string keeps one failing tool from aborting the agent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.adapters.cuga_editor_evidence import (
    EvidenceView,
    contamination_terms_from,
)
from agent_evolve.adapters.cuga_editor_state import EditStagingArea
from agent_evolve.adapters.cuga_editor_tools import (
    TOOL_APP_NAMES,
    EditorToolContext,
    build_tool_callables,
    submitted_plan,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import (
    CandidateWorkspace,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.editor import EditorRequest, ParentContext
from agent_evolve.core.memory import EditMemory
from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime

_SECRET = "token-a"


def _ctx(**overrides) -> EditorToolContext:
    adapter = CugaAdapter(wrapper=CugaWrapper(runtime=InMemoryRuntime()))
    adapter.register_candidate("v-primary", {"skills/retrieval": "primary body"})
    adapter.register_candidate("v-donor", {"skills/retrieval": "donor body"})

    task = EvolutionTask(
        task_id="task-a",
        input_text="produce the A capability",
        expected_contract={"expected_substring": _SECRET},
    )
    analysis = CausalAnalysis(
        mechanism="skill never loaded",
        severity=0.9,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="call_model", blame=1.0, artifacts=("skills/retrieval",)),)
        ),
    )
    trace = ExecutionTrace(
        trace_id="t-1",
        candidate_id="cand-1",
        task_id="task-a",
        events=(
            TraceEvent(
                event_id="graph:1", kind="llm_call", actor_id="call_model",
                parent_event_id=None, payload={"messages_ref": "a" * 64},
            ),
            TraceEvent(
                event_id="graph:2", kind="tool_call", actor_id="sandbox",
                parent_event_id="graph:1",
                payload={"name": "run_command", "result": "exit 0"},
            ),
        ),
        final_output=f"answer {_SECRET}",
        status="completed",
    )
    request = EditorRequest(
        base_workspace=CandidateWorkspace("att-1", "v-primary", Path("."), "v0"),
        task=task,
        analysis=analysis,
        issue_id="issue-1",
        write_set=("skills/retrieval",),
        current_artifacts={"skills/retrieval": "primary body"},
        creatable_prefix="skills/generated-",
        parents=(
            ParentContext("cand-1", "v-primary", True, {"task-a": 0.0}),
            ParentContext("cand-2", "v-donor", False, {"task-a": 0.9}),
        ),
    )
    ctx = EditorToolContext(
        staging=EditStagingArea(
            write_set=request.write_set,
            creatable_prefix=request.creatable_prefix,
        ),
        evidence=EvidenceView(
            analysis=analysis, trace=trace, task=task,
            contamination_terms=contamination_terms_from(task),
        ),
        request=request,
        adapter=adapter,
        memory=EditMemory(),
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def _tools(ctx: EditorToolContext | None = None):
    ctx = ctx if ctx is not None else _ctx()
    return ctx, build_tool_callables(ctx)


# ------------------------------------------------------------------ #
# shape
# ------------------------------------------------------------------ #
def test_every_tool_returns_a_json_string() -> None:
    _, tools = _tools()
    for name, fn in tools.items():
        if name in {"get_mechanism", "list_blamed_actors", "get_task_input",
                    "list_trace_actors", "list_artifacts", "list_staged",
                    "list_parents"}:
            out = fn()
            assert isinstance(out, str), name
            json.loads(out)


def test_every_tool_has_a_cluster_assignment() -> None:
    _, tools = _tools()
    assert set(tools) <= set(TOOL_APP_NAMES)
    for name in tools:
        assert TOOL_APP_NAMES[name]


def test_expected_cluster_names() -> None:
    assert set(TOOL_APP_NAMES.values()) == {
        "evidence", "harness", "history", "parents", "submit",
    }


# ------------------------------------------------------------------ #
# evidence cluster
# ------------------------------------------------------------------ #
def test_get_mechanism_returns_mechanism_and_severity() -> None:
    _, tools = _tools()
    payload = json.loads(tools["get_mechanism"]())
    assert payload["mechanism"] == "skill never loaded"


def test_get_task_input_returns_input_text() -> None:
    _, tools = _tools()
    assert json.loads(tools["get_task_input"]())["input_text"] == (
        "produce the A capability"
    )


def test_read_trace_events_strips_blob_refs() -> None:
    _, tools = _tools()
    events = json.loads(tools["read_trace_events"]())
    llm = next(e for e in events if e["kind"] == "llm_call")
    assert "messages_ref" not in llm["payload"]


def test_no_tool_output_contains_the_expected_value() -> None:
    """Leak audit across every readable tool."""
    ctx, tools = _tools()
    blob = ""
    for name in ("get_mechanism", "list_blamed_actors", "get_task_input",
                 "list_trace_actors", "list_artifacts", "list_parents"):
        blob += tools[name]()
    blob += tools["read_trace_events"]()
    blob += tools["read_artifact"]("skills/retrieval")
    blob += tools["read_parent_artifact"]("cand-2", "skills/retrieval")
    assert _SECRET not in blob


def test_no_tool_output_contains_the_final_output() -> None:
    _, tools = _tools()
    blob = tools["read_trace_events"]() + tools["get_mechanism"]()
    assert "answer " not in blob


# ------------------------------------------------------------------ #
# harness cluster
# ------------------------------------------------------------------ #
def test_read_artifact_returns_current_content() -> None:
    _, tools = _tools()
    assert json.loads(tools["read_artifact"]("skills/retrieval"))["content"] == (
        "primary body"
    )


def test_read_artifact_rejects_unknown_id() -> None:
    _, tools = _tools()
    payload = json.loads(tools["read_artifact"]("skills/absent"))
    assert payload["status"] == "error"


def test_stage_replace_accepts_authorized_write() -> None:
    ctx, tools = _tools()
    payload = json.loads(tools["stage_replace"]("skills/retrieval", "new"))
    assert payload["accepted"] is True
    assert ctx.staging.staged_ids() == ("skills/retrieval",)


def test_stage_replace_returns_rejection_without_raising() -> None:
    _, tools = _tools()
    payload = json.loads(tools["stage_replace"]("policies/x", "new"))
    assert payload["accepted"] is False
    assert "authorized write set" in payload["reason"]


def test_stage_create_enforces_namespace() -> None:
    _, tools = _tools()
    payload = json.loads(tools["stage_create"]("generated/x", "body"))
    assert payload["accepted"] is False


def test_stage_create_accepts_namespaced_id() -> None:
    _, tools = _tools()
    payload = json.loads(tools["stage_create"]("skills/generated-x", "body"))
    assert payload["accepted"] is True


def test_list_staged_reflects_staging() -> None:
    _, tools = _tools()
    tools["stage_replace"]("skills/retrieval", "new")
    assert json.loads(tools["list_staged"]())["staged"] == ["skills/retrieval"]


def test_unstage_removes_an_edit() -> None:
    _, tools = _tools()
    tools["stage_replace"]("skills/retrieval", "new")
    tools["unstage"]("skills/retrieval")
    assert json.loads(tools["list_staged"]())["staged"] == []


# ------------------------------------------------------------------ #
# parents cluster
# ------------------------------------------------------------------ #
def test_list_parents_marks_the_primary() -> None:
    _, tools = _tools()
    parents = json.loads(tools["list_parents"]())
    primary = [p for p in parents if p["is_primary"]]
    assert [p["candidate_id"] for p in primary] == ["cand-1"]


def test_read_parent_artifact_returns_donor_content() -> None:
    _, tools = _tools()
    payload = json.loads(tools["read_parent_artifact"]("cand-2", "skills/retrieval"))
    assert payload["content"] == "donor body"


def test_read_parent_artifact_records_provenance() -> None:
    ctx, tools = _tools()
    tools["read_parent_artifact"]("cand-2", "skills/retrieval")
    assert ctx.staging.parents_read() == ("cand-2",)


def test_read_parent_artifact_rejects_unknown_parent() -> None:
    ctx, tools = _tools()
    payload = json.loads(tools["read_parent_artifact"]("cand-99", "skills/retrieval"))
    assert payload["status"] == "error"
    assert ctx.staging.parents_read() == ()


# ------------------------------------------------------------------ #
# submit cluster
# ------------------------------------------------------------------ #
def test_submitted_plan_is_none_before_finalizing() -> None:
    ctx, _ = _tools()
    assert submitted_plan(ctx) is None


def test_submit_captures_the_staged_plan() -> None:
    ctx, tools = _tools()
    tools["stage_replace"]("skills/retrieval", "new")
    payload = json.loads(tools["submit_edit_plan"]("because evidence", "some risk", "fix"))
    assert payload["accepted"] is True
    plan = submitted_plan(ctx)
    assert plan is not None
    assert [e.artifact_id for e in plan["edits"]] == ["skills/retrieval"]
    assert plan["rationale"] == "because evidence"


def test_submit_with_nothing_staged_is_an_explicit_decline() -> None:
    ctx, tools = _tools()
    payload = json.loads(tools["submit_edit_plan"]("no change warranted", "", ""))
    assert payload["accepted"] is True
    plan = submitted_plan(ctx)
    assert plan is not None
    assert plan["edits"] == ()
    assert plan["declined"] is True


def test_submit_requires_a_rationale_when_declining() -> None:
    ctx, tools = _tools()
    payload = json.loads(tools["submit_edit_plan"]("", "", ""))
    assert payload["accepted"] is False
    assert submitted_plan(ctx) is None


def test_submit_is_idempotent_last_call_wins() -> None:
    ctx, tools = _tools()
    tools["stage_replace"]("skills/retrieval", "first")
    tools["submit_edit_plan"]("first rationale", "", "")
    tools["stage_replace"]("skills/retrieval", "second")
    tools["submit_edit_plan"]("second rationale", "", "")
    plan = submitted_plan(ctx)
    assert plan["rationale"] == "second rationale"
    assert plan["edits"][0].payload["content"] == "second"


def test_module_imports_without_cuga_installed() -> None:
    """build_tool_callables must not require the SDK."""
    import agent_evolve.adapters.cuga_editor_tools as mod

    assert not hasattr(mod, "tracked_tool")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cuga_editor_tools.py -v 2>&1 | tee terminal_output/cuga-editor/task7-red.log`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_evolve.adapters.cuga_editor_tools'`

- [ ] **Step 3: Write the implementation**

Create `src/agent_evolve/adapters/cuga_editor_tools.py`:

```python
"""Tool clusters handed to the CUGA editor agent.

Two layers on purpose:

* ``build_tool_callables`` returns plain functions with NO CUGA dependency, so
  every authorization, evidence and capture rule is unit-testable offline.
* ``build_editor_tools`` wraps those callables with ``tracked_tool`` + ``tool``,
  deferring the SDK import into the function body.

Every tool returns a JSON string. CUGA tools must return strings, and returning
a structured error keeps one failing tool from aborting the agent run. Nothing
here raises into the agent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from agent_evolve.adapters.cuga_editor_evidence import EvidenceView
from agent_evolve.adapters.cuga_editor_state import EditStagingArea
from agent_evolve.core.editor import EditorRequest
from agent_evolve.core.memory import EditMemory

TOOL_APP_NAMES: dict[str, str] = {
    # evidence
    "get_mechanism": "evidence",
    "list_blamed_actors": "evidence",
    "get_task_input": "evidence",
    "list_trace_actors": "evidence",
    "read_trace_events": "evidence",
    # harness
    "list_artifacts": "harness",
    "read_artifact": "harness",
    "stage_replace": "harness",
    "stage_create": "harness",
    "list_staged": "harness",
    "unstage": "harness",
    # history
    "search_edit_history": "history",
    "get_attempt_outcome": "history",
    # parents
    "list_parents": "parents",
    "read_parent_artifact": "parents",
    # submit
    "submit_edit_plan": "submit",
}

_MAX_HISTORY_RECORDS = 5


@dataclass(slots=True)
class EditorToolContext:
    """Per-request state the tools close over.

    Bound to one ``propose_edit`` call. Nothing here is global, so concurrent
    editors cannot interfere.
    """

    staging: EditStagingArea
    evidence: EvidenceView
    request: EditorRequest
    adapter: object
    memory: EditMemory
    _plan: dict | None = field(default=None, repr=False)


def submitted_plan(ctx: EditorToolContext) -> dict | None:
    """The finalized plan, or ``None`` if the agent never finalized."""
    return ctx._plan


def _ok(**payload: object) -> str:
    return json.dumps(payload, default=str)


def _err(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def build_tool_callables(ctx: EditorToolContext) -> dict[str, Callable[..., str]]:
    """Build the tool bodies for one editor request."""

    # ---------------------------------------------------------- evidence
    def get_mechanism() -> str:
        return _ok(**ctx.evidence.mechanism())

    def list_blamed_actors() -> str:
        return json.dumps(list(ctx.evidence.blamed_actors()), default=str)

    def get_task_input() -> str:
        return _ok(input_text=ctx.evidence.task_input())

    def list_trace_actors() -> str:
        return json.dumps(list(ctx.evidence.actors()))

    def read_trace_events(
        kind: str = "", actor_id: str = "", limit: int = 50
    ) -> str:
        try:
            events = ctx.evidence.events(
                kind=kind or None,
                actor_id=actor_id or None,
                limit=max(1, min(int(limit), 200)),
            )
        except Exception as exc:  # noqa: BLE001 - never raise into the agent
            return _err(f"read_trace_events failed: {exc}")
        return json.dumps(list(events), default=str)

    # ---------------------------------------------------------- harness
    def list_artifacts() -> str:
        return _ok(
            writable=list(ctx.request.write_set),
            creatable_prefix=ctx.request.creatable_prefix,
        )

    def read_artifact(artifact_id: str) -> str:
        content = ctx.request.current_artifacts.get(artifact_id)
        if content is None:
            return _err(
                f"{artifact_id!r} is not readable; call list_artifacts first"
            )
        return _ok(artifact_id=artifact_id, content=content)

    def stage_replace(artifact_id: str, content: str) -> str:
        outcome = ctx.staging.stage_replace(artifact_id, content)
        return _ok(accepted=outcome.accepted, reason=outcome.reason)

    def stage_create(artifact_id: str, content: str) -> str:
        outcome = ctx.staging.stage_create(artifact_id, content)
        return _ok(accepted=outcome.accepted, reason=outcome.reason)

    def list_staged() -> str:
        return _ok(staged=list(ctx.staging.staged_ids()))

    def unstage(artifact_id: str) -> str:
        outcome = ctx.staging.unstage(artifact_id)
        return _ok(accepted=outcome.accepted, reason=outcome.reason)

    # ---------------------------------------------------------- history
    def search_edit_history() -> str:
        try:
            records = ctx.memory.retrieve(
                ctx.request.issue_id, max_records=_MAX_HISTORY_RECORDS
            )
        except Exception as exc:  # noqa: BLE001
            return _err(f"search_edit_history failed: {exc}")
        return json.dumps(
            [
                {
                    "attempt_id": r.attempt_id,
                    "artifact_ids": list(r.artifact_ids),
                    "outcome": r.outcome,
                    "summary": r.summary,
                }
                for r in records
            ],
            default=str,
        )

    def get_attempt_outcome(attempt_id: str) -> str:
        try:
            attempt = ctx.memory.get(attempt_id)
        except KeyError:
            return _err(f"unknown attempt_id: {attempt_id!r}")
        return _ok(
            attempt_id=attempt.attempt_id,
            status=attempt.status.value,
            artifact_ids=list(attempt.artifact_ids),
            summary=attempt.sanitized_reasoning,
        )

    # ---------------------------------------------------------- parents
    def list_parents() -> str:
        return json.dumps(
            [
                {
                    "candidate_id": p.candidate_id,
                    "is_primary": p.is_primary,
                    "score_summary": dict(p.score_summary),
                }
                for p in ctx.request.parents
            ],
            default=str,
        )

    def read_parent_artifact(parent_id: str, artifact_id: str) -> str:
        parent = next(
            (p for p in ctx.request.parents if p.candidate_id == parent_id), None
        )
        if parent is None:
            return _err(f"unknown parent: {parent_id!r}; call list_parents first")
        try:
            contents = ctx.adapter.read_artifacts(parent.version, (artifact_id,))
        except Exception as exc:  # noqa: BLE001
            return _err(f"read_parent_artifact failed: {exc}")
        # Record only a read that actually returned content: provenance must
        # reflect what the editor used, not what it attempted.
        ctx.staging.record_parent_read(parent_id)
        return _ok(
            parent_id=parent_id,
            artifact_id=artifact_id,
            content=contents[artifact_id],
        )

    # ---------------------------------------------------------- submit
    def submit_edit_plan(
        rationale: str, risks: str = "", expected_effect: str = ""
    ) -> str:
        if not rationale.strip():
            return _ok(
                accepted=False,
                reason="rationale is required, including when declining",
            )
        edits = ctx.staging.edits()
        ctx._plan = {
            "edits": edits,
            "rationale": rationale,
            "risks": risks,
            "expected_effect": expected_effect,
            "declined": not edits,
            "parents_read": ctx.staging.parents_read(),
        }
        return _ok(
            accepted=True,
            staged=list(ctx.staging.staged_ids()),
            declined=not edits,
        )

    return {
        "get_mechanism": get_mechanism,
        "list_blamed_actors": list_blamed_actors,
        "get_task_input": get_task_input,
        "list_trace_actors": list_trace_actors,
        "read_trace_events": read_trace_events,
        "list_artifacts": list_artifacts,
        "read_artifact": read_artifact,
        "stage_replace": stage_replace,
        "stage_create": stage_create,
        "list_staged": list_staged,
        "unstage": unstage,
        "search_edit_history": search_edit_history,
        "get_attempt_outcome": get_attempt_outcome,
        "list_parents": list_parents,
        "read_parent_artifact": read_parent_artifact,
        "submit_edit_plan": submit_edit_plan,
    }


def build_editor_tools(ctx: EditorToolContext) -> list:
    """Wrap the tool bodies as tracked LangChain tools.

    This is the only place the CUGA SDK is imported from this module, mirroring
    ``cuga_wrapper.tools.build_tools``.
    """
    from langchain_core.tools import tool

    from cuga import tracked_tool

    built = []
    for name, fn in build_tool_callables(ctx).items():
        fn.__name__ = name
        wrapped = tracked_tool(app_name=TOOL_APP_NAMES[name])(fn)
        built.append(tool(wrapped))
    return built
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cuga_editor_tools.py -v 2>&1 | tee terminal_output/cuga-editor/task7-green.log`
Expected: 27 passed.

- [ ] **Step 5: Verify the module imports without touching CUGA**

Run: `uv run python -c "import agent_evolve.adapters.cuga_editor_tools as m; print(sorted(m.TOOL_APP_NAMES.values())[:3])"`
Expected: prints cluster names, no CUGA import error.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task7-suite.log`
Expected: 757 passed, 1 skipped.

- [ ] **Step 7: Stage**

```bash
git add src/agent_evolve/adapters/cuga_editor_tools.py tests/test_cuga_editor_tools.py
```

---

### Task 8: CugaEditorAgent — propose_edit and outcome classification

Spec §4, §7, §10. The `Editor` protocol implementation. The multi-turn agent loop
lives entirely inside `propose_edit`, so `core/` never learns the editor is a
CUGA agent. Tracing is detached and no workspace is bound (spec isolation
decision).

**Files:**
- Create: `src/agent_evolve/adapters/cuga_editor.py`
- Test: `tests/test_cuga_editor_agent.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 2-7
- Produces:
  - `CugaEditorAgent(adapter, memory, agent_factory=None, editor_model_id="cuga-editor-agent")`
  - `.propose_edit(request: EditorRequest) -> EditorResponse` (raises `EditorDeclined` when no plan)
  - `.last_outcome: EditorOutcome`
  - `.last_parents_read: tuple[str, ...]`
  - `.last_tools_called: tuple[str, ...]`
  - `EditorDeclined(RuntimeError)` — carries `.outcome`

Design note: `EditorResponse.__post_init__` requires non-empty `edits`
(`editor.py:103-104`), so a decline cannot be expressed as an `EditorResponse`.
`propose_edit` therefore raises `EditorDeclined`, which
`repair_once_then_classify` already converts to a recorded non-promotion via
`_propose_safely` (`editor.py:374-378`). The distinct outcome survives on
`.last_outcome` for the caller to record.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cuga_editor_agent.py`:

```python
"""CugaEditorAgent.propose_edit with a stubbed agent (spec §4, §7, §10).

The stub invokes a scripted tool sequence, so the whole classification and
capture path is testable without the SDK or a network call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.adapters.cuga_editor import CugaEditorAgent, EditorDeclined
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import (
    CandidateWorkspace,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.editor import (
    EditorOutcome,
    EditorRequest,
    ParentContext,
    repair_once_then_classify,
)
from agent_evolve.core.memory import EditMemory
from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime


class ScriptedAgent:
    """Calls a fixed sequence of tools, then returns prose.

    Mirrors the real contract: the prose answer is irrelevant and must be
    ignored by propose_edit.
    """

    def __init__(self, script, answer="I have finished my analysis."):
        self.script = script
        self.answer = answer
        self.called: list[str] = []

    def run(self, tools: dict, prompt: str) -> str:
        for name, args in self.script:
            self.called.append(name)
            tools[name](*args)
        return self.answer


def _adapter() -> CugaAdapter:
    adapter = CugaAdapter(wrapper=CugaWrapper(runtime=InMemoryRuntime()))
    adapter.register_candidate("v-primary", {"skills/retrieval": "primary body"})
    adapter.register_candidate("v-donor", {"skills/retrieval": "donor body"})
    return adapter


def _request() -> EditorRequest:
    task = EvolutionTask(task_id="task-a", input_text="do A",
                         expected_contract={"expected_substring": "token-a"})
    analysis = CausalAnalysis(
        mechanism="skill never loaded", severity=0.9, score=0.0,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="call_model", blame=1.0,
                             artifacts=("skills/retrieval",)),)
        ),
    )
    return EditorRequest(
        base_workspace=CandidateWorkspace("att-1", "v-primary", Path("."), "v0"),
        task=task,
        analysis=analysis,
        issue_id="issue-1",
        write_set=("skills/retrieval",),
        current_artifacts={"skills/retrieval": "primary body"},
        creatable_prefix="skills/generated-",
        parents=(
            ParentContext("cand-1", "v-primary", True, {"task-a": 0.0}),
            ParentContext("cand-2", "v-donor", False, {"task-a": 0.9}),
        ),
    )


def _editor(script, answer="done") -> tuple[CugaEditorAgent, ScriptedAgent]:
    stub = ScriptedAgent(script, answer)
    editor = CugaEditorAgent(
        adapter=_adapter(),
        memory=EditMemory(),
        agent_factory=lambda tools, prompt: stub.run(tools, prompt),
    )
    return editor, stub


_TRACE = ExecutionTrace(
    trace_id="t-1", candidate_id="cand-1", task_id="task-a",
    events=(TraceEvent(event_id="graph:1", kind="llm_call",
                       actor_id="call_model", parent_event_id=None, payload={}),),
    final_output="", status="completed",
)


# ------------------------------------------------------------------ #
# happy path
# ------------------------------------------------------------------ #
def test_propose_edit_returns_the_staged_plan() -> None:
    editor, _ = _editor([
        ("get_mechanism", ()),
        ("read_artifact", ("skills/retrieval",)),
        ("stage_replace", ("skills/retrieval", "improved body")),
        ("submit_edit_plan", ("addresses the mechanism",)),
    ])
    response = editor.propose_edit(_request())
    assert [e.artifact_id for e in response.edits] == ["skills/retrieval"]
    assert response.edits[0].payload["content"] == "improved body"
    assert response.rationale == "addresses the mechanism"
    assert editor.last_outcome is EditorOutcome.VALID


def test_prose_answer_is_ignored() -> None:
    editor, _ = _editor(
        [("stage_replace", ("skills/retrieval", "x")),
         ("submit_edit_plan", ("r",))],
        answer='{"edits": [{"artifact_id": "skills/HACKED"}]}',
    )
    response = editor.propose_edit(_request())
    assert [e.artifact_id for e in response.edits] == ["skills/retrieval"]


def test_editor_model_id_is_reported() -> None:
    editor, _ = _editor([("stage_replace", ("skills/retrieval", "x")),
                         ("submit_edit_plan", ("r",))])
    response = editor.propose_edit(_request())
    assert response.editor_model_id == "cuga-editor-agent"


def test_tools_called_are_recorded() -> None:
    editor, _ = _editor([("get_mechanism", ()),
                         ("stage_replace", ("skills/retrieval", "x")),
                         ("submit_edit_plan", ("r",))])
    editor.propose_edit(_request())
    assert "get_mechanism" in editor.last_tools_called


# ------------------------------------------------------------------ #
# provenance
# ------------------------------------------------------------------ #
def test_parents_read_are_recorded_for_provenance() -> None:
    editor, _ = _editor([
        ("read_parent_artifact", ("cand-2", "skills/retrieval")),
        ("stage_replace", ("skills/retrieval", "donor body")),
        ("submit_edit_plan", ("transplanted from donor",)),
    ])
    editor.propose_edit(_request())
    assert editor.last_parents_read == ("cand-2",)


def test_unread_donors_are_not_recorded() -> None:
    editor, _ = _editor([("stage_replace", ("skills/retrieval", "x")),
                         ("submit_edit_plan", ("r",))])
    editor.propose_edit(_request())
    assert editor.last_parents_read == ()


# ------------------------------------------------------------------ #
# outcome taxonomy (spec §10)
# ------------------------------------------------------------------ #
def test_never_calling_submit_is_no_tool_call() -> None:
    editor, _ = _editor([("get_mechanism", ())])
    with pytest.raises(EditorDeclined) as excinfo:
        editor.propose_edit(_request())
    assert excinfo.value.outcome is EditorOutcome.NO_TOOL_CALL
    assert editor.last_outcome is EditorOutcome.NO_TOOL_CALL


def test_staging_without_finalizing_is_no_tool_call() -> None:
    """Staged-but-unfinalized work is discarded, not silently applied."""
    editor, _ = _editor([("stage_replace", ("skills/retrieval", "x"))])
    with pytest.raises(EditorDeclined) as excinfo:
        editor.propose_edit(_request())
    assert excinfo.value.outcome is EditorOutcome.NO_TOOL_CALL


def test_explicit_decline_is_no_op_not_no_tool_call() -> None:
    editor, _ = _editor([("submit_edit_plan", ("evidence does not justify a change",))])
    with pytest.raises(EditorDeclined) as excinfo:
        editor.propose_edit(_request())
    assert excinfo.value.outcome is EditorOutcome.NO_OP
    assert editor.last_outcome is EditorOutcome.NO_OP


def test_agent_error_is_unavailable() -> None:
    def exploding_factory(tools, prompt):
        raise RuntimeError("CUGA execution failed")

    editor = CugaEditorAgent(
        adapter=_adapter(), memory=EditMemory(),
        agent_factory=exploding_factory,
    )
    with pytest.raises(EditorDeclined) as excinfo:
        editor.propose_edit(_request())
    assert excinfo.value.outcome is EditorOutcome.UNAVAILABLE


# ------------------------------------------------------------------ #
# integration with the core repair protocol
# ------------------------------------------------------------------ #
def test_repair_protocol_treats_a_decline_as_a_non_promotion() -> None:
    editor, _ = _editor([("submit_edit_plan", ("declining",))])
    result = repair_once_then_classify(editor, _request())
    assert result.status == "malformed"
    assert result.response is None
    # The distinct outcome survives for the caller to record.
    assert editor.last_outcome is EditorOutcome.NO_OP


def test_repair_protocol_passes_a_valid_plan_through_unchanged() -> None:
    editor, _ = _editor([("stage_replace", ("skills/retrieval", "x")),
                         ("submit_edit_plan", ("r",))])
    result = repair_once_then_classify(editor, _request())
    assert result.status == "valid"
    assert result.correction_requests == 0


# ------------------------------------------------------------------ #
# isolation (spec §13)
# ------------------------------------------------------------------ #
def test_editor_agent_construction_detaches_tracing() -> None:
    """The editor's own LLM calls must never enter a rollout trace."""
    from agent_evolve.adapters.cuga_editor import editor_agent_kwargs

    kwargs = editor_agent_kwargs()
    assert kwargs["callbacks"] == []
    assert kwargs["cuga_folder"] is None


def test_editor_agent_construction_binds_no_workspace(monkeypatch) -> None:
    import os

    from agent_evolve.adapters.cuga_editor import prepare_editor_environment

    monkeypatch.setenv("CUGA_FOLDER", "/some/rollout/workspace")
    prepare_editor_environment()
    assert "CUGA_FOLDER" not in os.environ
```

- [ ] **Step 2: Verify the test file compiles**

Run: `uv run python -m py_compile tests/test_cuga_editor_agent.py`
Expected: no output (exit 0). A `SyntaxError` here means the file was
transcribed incorrectly; fix it before running pytest, since a syntax error
would otherwise look like a collection failure.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cuga_editor_agent.py -v 2>&1 | tee terminal_output/cuga-editor/task8-red.log`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_evolve.adapters.cuga_editor'`

- [ ] **Step 4: Write the implementation**

Create `src/agent_evolve/adapters/cuga_editor.py`:

```python
"""CUGA-agent-backed editor implementing the core ``Editor`` protocol.

The whole multi-turn agent loop lives inside ``propose_edit``, so
``agent_evolve.core`` never learns the editor is a CUGA agent.

Isolation (design doc §13): the editor agent is constructed with tracing
detached and no workspace bound, so its own LLM calls cannot enter a rollout
trace and it cannot read a candidate's skills directory. CUGA's singleton
ActivityTracker and global policy DB remain shared in-process; that residual
risk is accepted and guarded by test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from agent_evolve.adapters.cuga_editor_evidence import (
    EvidenceView,
    contamination_terms_from,
)
from agent_evolve.adapters.cuga_editor_skills import (
    EDITOR_INSTRUCTIONS,
    build_editor_prompt,
)
from agent_evolve.adapters.cuga_editor_state import EditStagingArea
from agent_evolve.adapters.cuga_editor_tools import (
    EditorToolContext,
    build_tool_callables,
    submitted_plan,
)
from agent_evolve.core.contracts import ExecutionTrace
from agent_evolve.core.editor import EditorOutcome, EditorRequest, EditorResponse
from agent_evolve.core.memory import EditMemory


class EditorDeclined(RuntimeError):
    """The editor produced no usable plan.

    Carries the distinct :class:`EditorOutcome` so a caller can tell
    ``no_tool_call`` (the agent did not engage) from ``no_op`` (the agent
    judged no edit warranted). ``repair_once_then_classify`` converts this into
    a recorded non-promotion.
    """

    def __init__(self, outcome: EditorOutcome, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome


def prepare_editor_environment() -> None:
    """Unbind any rollout workspace before constructing the editor agent.

    CUGA reads ``CUGA_FOLDER`` in the sandbox and in ``prepare_node``, so a
    leftover value from a rollout would give the editor a candidate's skills.
    """
    os.environ.pop("CUGA_FOLDER", None)


def editor_agent_kwargs() -> dict[str, object]:
    """Construction arguments that keep the editor out of rollout traces."""
    return {
        # No callbacks: the GraphEventCollector must never see editor LLM calls,
        # or the editor would pollute the evidence the analyzer reads.
        "callbacks": [],
        "cuga_folder": None,
        "special_instructions": EDITOR_INSTRUCTIONS,
        "enable_skills": True,
        "auto_load_policies": False,
    }


def _evidence_summary(view: EvidenceView) -> str:
    mechanism = view.mechanism()
    actors = ", ".join(
        f"{a['actor_id']} (blame {a['blame']})" for a in view.blamed_actors()
    ) or "none attributed"
    return (
        f"MECHANISM: {mechanism['mechanism']}\n"
        f"SEVERITY: {mechanism['severity']}\n"
        f"BLAMED ACTORS: {actors}\n"
        f"TASK: {view.task_input()}"
    )


@dataclass(slots=True)
class CugaEditorAgent:
    """Editor backed by a multi-turn CUGA agent."""

    adapter: object
    memory: EditMemory
    # Injected for tests: (tool_callables, prompt) -> agent answer. When None,
    # a real CugaAgent is constructed.
    agent_factory: Callable[[dict, str], str] | None = None
    editor_model_id: str = "cuga-editor-agent"
    trace: ExecutionTrace | None = None
    last_outcome: EditorOutcome = EditorOutcome.UNAVAILABLE
    last_parents_read: tuple[str, ...] = ()
    last_tools_called: tuple[str, ...] = field(default_factory=tuple)

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        ctx = self._build_context(request)
        callables = build_tool_callables(ctx)
        recorded, names = self._recording_wrapper(callables)
        prompt = build_editor_prompt(_evidence_summary(ctx.evidence))

        try:
            self._run_agent(recorded, prompt)
        except Exception as exc:  # noqa: BLE001 - classify, never propagate raw
            self.last_tools_called = tuple(names)
            self.last_outcome = EditorOutcome.UNAVAILABLE
            raise EditorDeclined(
                EditorOutcome.UNAVAILABLE, f"editor agent failed: {exc}"
            ) from exc

        self.last_tools_called = tuple(names)
        plan = submitted_plan(ctx)

        if plan is None:
            # Includes the case where edits were staged but never finalized:
            # unfinalized work is discarded, not silently applied.
            self.last_outcome = EditorOutcome.NO_TOOL_CALL
            self.last_parents_read = ()
            raise EditorDeclined(
                EditorOutcome.NO_TOOL_CALL,
                "editor agent never called submit_edit_plan",
            )

        self.last_parents_read = tuple(plan["parents_read"])

        if plan["declined"]:
            self.last_outcome = EditorOutcome.NO_OP
            raise EditorDeclined(
                EditorOutcome.NO_OP,
                f"editor declined to edit: {plan['rationale']}",
            )

        self.last_outcome = EditorOutcome.VALID
        writes = {
            edit.artifact_id: str(edit.payload.get("content", ""))
            for edit in plan["edits"]
        }
        return EditorResponse(
            rationale=plan["rationale"],
            edits=plan["edits"],
            reads=dict(request.current_artifacts),
            writes=writes,
            risks={"summary": plan["risks"]} if plan["risks"] else {},
            expected_effects=(
                {"summary": plan["expected_effect"]}
                if plan["expected_effect"]
                else {}
            ),
            editor_model_id=self.editor_model_id,
        )

    # -------------------------------------------------------------- #
    # Internals
    # -------------------------------------------------------------- #
    def _build_context(self, request: EditorRequest) -> EditorToolContext:
        pool_created = request.pool_created_count
        staging = EditStagingArea(
            write_set=request.write_set,
            creatable_prefix=request.creatable_prefix,
            pool_created_count=pool_created,
        )
        trace = self.trace or ExecutionTrace(
            trace_id="unavailable",
            candidate_id=request.base_workspace.version,
            task_id=request.task.task_id,
            events=(),
            final_output="",
            status="unavailable",
        )
        evidence = EvidenceView(
            analysis=request.analysis,
            trace=trace,
            task=request.task,
            contamination_terms=contamination_terms_from(request.task),
        )
        return EditorToolContext(
            staging=staging,
            evidence=evidence,
            request=request,
            adapter=self.adapter,
            memory=self.memory,
        )

    @staticmethod
    def _recording_wrapper(
        callables: dict[str, Callable[..., str]],
    ) -> tuple[dict[str, Callable[..., str]], list[str]]:
        names: list[str] = []

        def wrap(name: str, fn: Callable[..., str]) -> Callable[..., str]:
            def recorded(*args, **kwargs) -> str:
                names.append(name)
                return fn(*args, **kwargs)

            recorded.__name__ = name
            return recorded

        return {name: wrap(name, fn) for name, fn in callables.items()}, names

    def _run_agent(self, callables: dict, prompt: str) -> str:
        if self.agent_factory is not None:
            return self.agent_factory(callables, prompt)
        return self._run_cuga_agent(callables, prompt)

    def _run_cuga_agent(self, callables: dict, prompt: str) -> str:
        """Construct and run a real CUGA agent. SDK import stays local."""
        import asyncio

        from agent_evolve.adapters.cuga_editor_tools import build_editor_tools
        from cuga import CugaAgent

        prepare_editor_environment()
        kwargs = editor_agent_kwargs()
        agent = CugaAgent(tools=build_editor_tools(self._active_ctx), **kwargs)

        async def run() -> str:
            await agent.initialize()
            result = await agent.invoke(prompt)
            return str(result)

        return asyncio.run(run())
```

Note: `_run_cuga_agent` references `self._active_ctx`. Add that field and set it
in `propose_edit` before the run:

```python
    _active_ctx: EditorToolContext | None = None
```

and in `propose_edit`, immediately after `ctx = self._build_context(request)`:

```python
        self._active_ctx = ctx
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_cuga_editor_agent.py -v 2>&1 | tee terminal_output/cuga-editor/task8-green.log`
Expected: 15 passed.

- [ ] **Step 6: Verify core has no adapter dependency**

Run: `grep -rn "cuga" src/agent_evolve/core/*.py | grep -v "^.*#" | grep import`
Expected: no output. `core/` must not import any adapter or CUGA module.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task8-suite.log`
Expected: 772 passed, 1 skipped.

- [ ] **Step 8: Stage**

```bash
git add src/agent_evolve/adapters/cuga_editor.py tests/test_cuga_editor_agent.py
```

---

### Task 9: Orchestrator multi-parent selection and observed lineage

Spec §7, §9. `select_parent()` (`orchestrator.py:1105`) returns one entry;
unified editing needs primary + donors. `commit_to_pool` (`orchestrator.py:1221`)
hardcodes `parent_ids=(parent_entry.candidate_id,)`, which would misreport
lineage when the editor drew from a donor.

**Files:**
- Modify: `src/agent_evolve/core/orchestrator.py:1105-1129` (add `select_parents`), `:1205-1244` (`commit_to_pool`)

`propose_edits` wiring is Task 11, so this task stays reviewable on its own.
- Test: `tests/test_orchestrator_multiparent.py` (create)

**Interfaces:**
- Consumes: `PersistentPool.pareto_frontier()`, `.parent_frequencies()`, `.get()`; `ParentContext` (Task 5); `CugaEditorAgent.last_parents_read` (Task 8)
- Produces:
  - `SequentialGepaRunner.select_parents(k: int = 3) -> tuple[PoolEntry, ...]` — primary first, then donors
  - `SequentialGepaRunner.donor_count: int = 2` attribute (K−1)
  - `commit_to_pool(..., extra_parent_ids: Sequence[str] = ())`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator_multiparent.py`:

```python
"""Multi-parent selection and observed lineage (spec §7, §9).

Parent-set size is bounded so prompt size does not grow with the pool: the
primary keeps the architecture's frequency-proportional semantics, donors come
from the Pareto frontier.
"""
from __future__ import annotations

import pytest

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import PersistentPool, ScoreProvenance

# Reuse the established fake harness from the phase 6 tests.
from tests.test_phase_6_orchestrator import _runner  # type: ignore


def _candidate(candidate_id: str) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=candidate_id,
        version=f"{candidate_id}-v",
        artifact_hashes={"skills/retrieval": "sha256:" + "0" * 64},
    )


def test_select_parents_returns_primary_first() -> None:
    runner = _runner()
    parents = runner.select_parents(k=3)
    assert parents
    assert parents[0].candidate_id == runner.pool.base.candidate_id


def test_select_parents_is_bounded_by_k() -> None:
    runner = _runner()
    for i in range(6):
        runner.pool.add_candidate(_candidate(f"extra-{i}"))
    assert len(runner.select_parents(k=3)) <= 3


def test_select_parents_never_repeats_the_primary_as_a_donor() -> None:
    runner = _runner()
    for i in range(4):
        runner.pool.add_candidate(_candidate(f"extra-{i}"))
    parents = runner.select_parents(k=4)
    ids = [p.candidate_id for p in parents]
    assert len(ids) == len(set(ids))


def test_select_parents_with_k_one_returns_only_the_primary() -> None:
    runner = _runner()
    runner.pool.add_candidate(_candidate("extra-0"))
    assert len(runner.select_parents(k=1)) == 1


def test_select_parents_rejects_k_below_one() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="k must be >= 1"):
        runner.select_parents(k=0)


def test_select_parents_draws_donors_from_the_pareto_frontier() -> None:
    runner = _runner()
    runner.pool.add_candidate(_candidate("frontier-cand"))
    frontier = set(runner.pool.pareto_frontier())
    parents = runner.select_parents(k=3)
    donors = [p.candidate_id for p in parents[1:]]
    assert all(d in frontier for d in donors)


def test_commit_to_pool_records_only_the_primary_by_default() -> None:
    runner = _runner()
    entry = runner._commit_single_parent_for_test()
    assert entry.candidate.parent_ids == (runner.pool.base.candidate_id,)


def test_commit_to_pool_records_observed_extra_parents() -> None:
    """Lineage must reflect donors actually read, not donors merely offered."""
    runner = _runner()
    runner.pool.add_candidate(_candidate("donor-1"))
    entry = runner._commit_with_extra_parents_for_test(("donor-1",))
    assert set(entry.candidate.parent_ids) == {
        runner.pool.base.candidate_id, "donor-1",
    }


def test_observed_extra_parents_appear_in_ancestors() -> None:
    runner = _runner()
    runner.pool.add_candidate(_candidate("donor-1"))
    entry = runner._commit_with_extra_parents_for_test(("donor-1",))
    assert "donor-1" in entry.candidate.ancestor_ids


def test_parent_ids_are_deduplicated_and_sorted() -> None:
    runner = _runner()
    runner.pool.add_candidate(_candidate("donor-1"))
    base_id = runner.pool.base.candidate_id
    entry = runner._commit_with_extra_parents_for_test((base_id, "donor-1", "donor-1"))
    assert entry.candidate.parent_ids == tuple(sorted({base_id, "donor-1"}))


def test_lineage_of_is_stable_for_multiple_parents() -> None:
    """Confirms the qf30 §15 verification with a regression test."""
    from pathlib import Path

    from agent_evolve.core.contracts import CandidateWorkspace
    from agent_evolve.core.editor import lineage_of

    ws = CandidateWorkspace("att-1", "v1", Path("."), "v0")
    assert lineage_of(ws, ("v-b", "v-a")) == "v-a|v-b"
    assert lineage_of(ws, ("v-a", "v-b")) == "v-a|v-b"
    assert lineage_of(ws) == "v0"
```

- [ ] **Step 2: Add the two test-support helpers**

`_commit_single_parent_for_test` and `_commit_with_extra_parents_for_test` do not
exist. Add them to `SequentialGepaRunner` as thin wrappers so the lineage logic
is testable without running a full attempt:

```python
    def _commit_single_parent_for_test(self) -> PoolEntry:
        """Test seam: commit the base's workspace with no extra parents."""
        return self._commit_for_test(())

    def _commit_with_extra_parents_for_test(
        self, extra_parent_ids: Sequence[str]
    ) -> PoolEntry:
        """Test seam: commit with observed donor parents."""
        return self._commit_for_test(extra_parent_ids)

    def _commit_for_test(self, extra_parent_ids: Sequence[str]) -> PoolEntry:
        from agent_evolve.core.editor import (
            FocusedValidationReport as _Report,
        )

        parent = self.pool.base
        attempt_id = self._next_attempt_id()
        workspace = self.adapter.materialize_candidate(parent.version, attempt_id)
        return self.commit_to_pool(
            parent,
            workspace,
            attempt_id,
            _Report(origin=(), worked=(), regression=()),
            empty_analysis(),
            extra_parent_ids=extra_parent_ids,
        )
```

`empty_analysis` must be imported from `agent_evolve.core.blame` at the top of
`orchestrator.py` if it is not already.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator_multiparent.py -v 2>&1 | tee terminal_output/cuga-editor/task9-red.log`
Expected: FAIL — `AttributeError: 'SequentialGepaRunner' object has no attribute 'select_parents'`

- [ ] **Step 4: Add select_parents**

In `src/agent_evolve/core/orchestrator.py`, after `select_parent`:

```python
    def select_parents(self, k: int = 3) -> tuple[PoolEntry, ...]:
        """Select the primary parent plus up to ``k - 1`` donor parents.

        The primary keeps the architecture's frequency-proportional sampling and
        owns the workspace being written. Donors come from the Pareto frontier
        and are exposed read-only, so an editor can transplant a capability
        without the prompt growing with the pool.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be >= 1")
        primary = self.select_parent()
        if k == 1:
            return (primary,)
        donors: list[PoolEntry] = []
        for candidate_id in self.pool.pareto_frontier():
            if candidate_id == primary.candidate_id:
                continue
            donors.append(self.pool.get(candidate_id))
            if len(donors) >= k - 1:
                break
        return (primary, *donors)
```

- [ ] **Step 5: Add the donor_count attribute**

In `SequentialGepaRunner`'s field list, beside the other tuning attributes:

```python
    donor_count: int = 2
```

- [ ] **Step 6: Thread observed parents through commit_to_pool**

Change the `commit_to_pool` signature and the `EvolutionCandidate` construction
(currently `orchestrator.py:1205-1225`):

```python
    def commit_to_pool(
        self,
        parent_entry: PoolEntry,
        workspace: CandidateWorkspace,
        attempt_id: str,
        report: FocusedValidationReport,
        analysis: CausalAnalysis,
        extra_parent_ids: Sequence[str] = (),
    ) -> PoolEntry:
        """Publish an accepted candidate with its post-edit score evidence.

        ``extra_parent_ids`` carries donor parents the editor actually read.
        They come from tool-execution evidence, never from editor narration, so
        lineage cannot claim a donor the editor merely had access to.
        """
        parent_ids = tuple(
            sorted({parent_entry.candidate_id, *extra_parent_ids})
        )
        candidate = EvolutionCandidate(
            candidate_id=workspace.version,
            version=workspace.version,
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in self.adapter.artifact_inventory(workspace.version)
            },
            parent_ids=parent_ids,
            ancestor_ids=tuple(
                sorted(set(parent_entry.candidate.ancestor_ids) | set(parent_ids))
            ),
            attempt_ids=(attempt_id,),
        )
```

The rest of the method body is unchanged.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_orchestrator_multiparent.py -v 2>&1 | tee terminal_output/cuga-editor/task9-green.log`
Expected: 11 passed.

- [ ] **Step 8: Run the orchestrator regression tests**

Run: `uv run pytest tests/test_orchestrator.py tests/test_phase_6_orchestrator.py tests/test_phase_6_b1.py -v 2>&1 | tee terminal_output/cuga-editor/task9-regress.log`
Expected: all pass. `extra_parent_ids` defaults to empty, so single-parent lineage is byte-identical to before.

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task9-suite.log`
Expected: 783 passed, 1 skipped.

- [ ] **Step 10: Stage**

```bash
git add src/agent_evolve/core/orchestrator.py tests/test_orchestrator_multiparent.py
```

---

### Task 10: Live verification script

Spec §11. One real editor invocation against the existing 56-event reference
trace. This is the step that reveals whether the model actually calls the tools —
the highest tracked risk in the design (§13).

**Files:**
- Create: `scripts/verify_editor_against_live_trace.py`

**Interfaces:**
- Consumes: `CugaEditorAgent` (Task 8), `CugaAdapter` (Task 4), the trace at `data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781/`
- Produces: a JSON report at `terminal_output/cuga-editor/live/editor-report.json`

- [ ] **Step 1: Write the script**

Create `scripts/verify_editor_against_live_trace.py`:

```python
"""One real editor-agent invocation over the reference live trace.

This answers the design's highest tracked risk (§13): does the model actually
call the editor tools? A model that never calls submit_edit_plan makes the
editor inert, and that must be visible immediately rather than after a full
experiment.

Reports: tools called, outcome classification, staged edits, parents read, and
whether the contamination guard fired. Makes ONE live inference.

Usage:
    uv run python scripts/verify_editor_against_live_trace.py \
        2>&1 | tee terminal_output/cuga-editor/live/editor-run.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_evolve.adapters.cuga_adapter import CugaAdapter  # noqa: E402
from agent_evolve.adapters.cuga_editor import (  # noqa: E402
    CugaEditorAgent,
    EditorDeclined,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    CandidateWorkspace,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.editor import EditorRequest, ParentContext  # noqa: E402
from agent_evolve.core.memory import EditMemory  # noqa: E402
from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime  # noqa: E402

TRACE_DIR = ROOT / "data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781"
REPORT_DIR = ROOT / "terminal_output/cuga-editor/live"


def load_reference_trace() -> ExecutionTrace:
    """Map the persisted causal trace into an ExecutionTrace."""
    causal = json.loads((TRACE_DIR / "causal-trace.json").read_text())
    events = tuple(
        TraceEvent(
            event_id=str(e["event_id"]),
            kind=str(e["kind"]),
            actor_id=(str(e["actor_id"]) if e.get("actor_id") else None),
            parent_event_id=(
                str(e["parent_event_id"]) if e.get("parent_event_id") else None
            ),
            payload=dict(e.get("payload") or {}),
        )
        for e in causal["events"]
    )
    return ExecutionTrace(
        trace_id="live-reference",
        candidate_id="cand-primary",
        task_id="reference-task",
        events=events,
        final_output="",
        status="completed",
    )


def build_request(adapter: CugaAdapter) -> EditorRequest:
    task = EvolutionTask(
        task_id="reference-task",
        input_text=(
            "Fetch the alpha token, exchange it for a beta token, then report "
            "the beta checksum."
        ),
    )
    analysis = CausalAnalysis(
        mechanism="the agent reported a final answer without verifying the checksum",
        severity=0.9,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(
                BlameNode(actor_id="call_model", blame=0.7,
                          artifacts=("skills/token-workflow",)),
                BlameNode(actor_id="FinalAnswerAgent", blame=0.3, artifacts=()),
            )
        ),
    )
    return EditorRequest(
        base_workspace=CandidateWorkspace(
            "live-att-1", "v-primary", Path("."), "v0"
        ),
        task=task,
        analysis=analysis,
        issue_id="live-issue-1",
        write_set=("skills/token-workflow",),
        current_artifacts={
            "skills/token-workflow": (
                "# Token workflow\n\n"
                "1. Call fetch_alpha_token.\n"
                "2. Call exchange_alpha_for_beta.\n"
                "3. Report the result.\n"
            )
        },
        creatable_prefix=CugaAdapter.creatable_prefix,
        parents=(
            ParentContext("cand-primary", "v-primary", True, {"reference-task": 0.0}),
            ParentContext("cand-donor", "v-donor", False, {"reference-task": 0.8}),
        ),
    )


def main() -> int:
    if not TRACE_DIR.is_dir():
        print(f"FAIL: reference trace missing at {TRACE_DIR}")
        return 1
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    adapter = CugaAdapter(wrapper=CugaWrapper(runtime=InMemoryRuntime()))
    adapter.register_candidate(
        "v-primary",
        {"skills/token-workflow": "# Token workflow\n\n1. Fetch.\n2. Report.\n"},
    )
    adapter.register_candidate(
        "v-donor",
        {
            "skills/token-workflow": (
                "# Token workflow\n\n"
                "1. Fetch alpha.\n2. Exchange for beta.\n"
                "3. Verify the checksum before reporting.\n"
            )
        },
    )

    editor = CugaEditorAgent(
        adapter=adapter, memory=EditMemory(), trace=load_reference_trace()
    )
    request = build_request(adapter)

    outcome = "unknown"
    edits: list[dict[str, object]] = []
    rationale = ""
    error = ""
    try:
        response = editor.propose_edit(request)
        outcome = editor.last_outcome.value
        rationale = response.rationale
        edits = [
            {
                "artifact_id": e.artifact_id,
                "operation": e.operation,
                "content_length": len(str(e.payload.get("content", ""))),
            }
            for e in response.edits
        ]
    except EditorDeclined as exc:
        outcome = exc.outcome.value
        error = str(exc)

    report = {
        "outcome": outcome,
        "tools_called": list(editor.last_tools_called),
        "distinct_tools_called": sorted(set(editor.last_tools_called)),
        "tool_call_count": len(editor.last_tools_called),
        "parents_read": list(editor.last_parents_read),
        "edits": edits,
        "rationale": rationale,
        "error": error,
    }
    (REPORT_DIR / "editor-report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print()
    if outcome == "valid":
        print("PASS: the editor produced a plan from real evidence.")
        return 0
    if outcome == "no_op":
        print("INCONCLUSIVE: the editor declined explicitly. Read the rationale.")
        return 0
    if outcome == "no_tool_call":
        print(
            "FAIL: the agent never finalized. This is the tracked "
            "tool-invocation risk (design §13), not a code defect."
        )
        return 1
    print(f"FAIL: editor unavailable: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the reference trace exists**

Run: `ls data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781/causal-trace.json`
Expected: the file is listed. If absent, stop and ask — this script needs the real trace.

- [ ] **Step 3: Run the script (ONE live inference)**

Run: `mkdir -p terminal_output/cuga-editor/live && uv run python scripts/verify_editor_against_live_trace.py 2>&1 | tee terminal_output/cuga-editor/live/editor-run.log`

Expected: a JSON report plus one of the four verdicts. Record the outcome
verbatim. `no_tool_call` is a legitimate research finding about the model, not a
bug to hide — report it plainly.

- [ ] **Step 4: Stage**

```bash
git add scripts/verify_editor_against_live_trace.py
```

Report the outcome, the distinct tools called, and whether any contamination was
detected. Await direction before any further live runs.

---

### Task 11: Wire the editor into propose_edits and run_attempt

Without this task the new editor is unreachable: `propose_edits`
(`orchestrator.py:1134-1163`) still builds a single-parent `EditorRequest` with
no `parents`, no `creatable_prefix`, and no way to carry observed donor reads
into `commit_to_pool`. This is the task that makes Tasks 5, 8 and 9 load-bearing.

**Files:**
- Modify: `src/agent_evolve/core/orchestrator.py:1134-1163` (`propose_edits`), `:1249-1321` (`run_attempt`)
- Test: `tests/test_orchestrator_editor_wiring.py` (create)

**Interfaces:**
- Consumes: `select_parents` (Task 9), `ParentContext` (Task 5), `commit_to_pool(..., extra_parent_ids=)` (Task 9)
- Produces:
  - `propose_edits(...) -> tuple[CandidateWorkspace, EditorResponse | None, int, tuple[str, ...]]` — the fourth element is the observed parent ids
  - `run_attempt` threads those ids into `commit_to_pool`

Note this changes `propose_edits`'s return arity from 3 to 4. Callers in
`run_attempt` and in `tests/test_phase_6_orchestrator.py` must be updated; the
test suite is the guard.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator_editor_wiring.py`:

```python
"""propose_edits must deliver parents and creation authority to the editor.

Without this wiring the multi-parent editor is unreachable and the loop silently
falls back to single-parent editing.
"""
from __future__ import annotations

from agent_evolve.core.contracts import ArtifactEdit
from agent_evolve.core.editor import EditorRequest, EditorResponse

from tests.test_phase_6_orchestrator import _runner  # type: ignore


class RecordingEditor:
    """Captures the request it was given and returns a minimal valid edit."""

    editor_model_id = "recording-editor"

    def __init__(self) -> None:
        self.seen: EditorRequest | None = None
        self.last_parents_read: tuple[str, ...] = ()

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        self.seen = request
        target = request.write_set[0]
        content = request.current_artifacts.get(target, "") + " edited"
        return EditorResponse(
            rationale="recorded",
            edits=(
                ArtifactEdit(
                    artifact_id=target,
                    operation="replace",
                    payload={"content": content},
                ),
            ),
            reads=dict(request.current_artifacts),
            writes={target: content},
            risks={},
            expected_effects={},
            editor_model_id=self.editor_model_id,
        )


def _wired() -> tuple[object, RecordingEditor]:
    runner = _runner()
    editor = RecordingEditor()
    runner.editor = editor
    return runner, editor


def _first_issue_and_task(runner):
    tasks = runner._tasks_for_test()
    issues = runner.build_issues(tasks)
    assert issues, "expected at least one failing task to produce an issue"
    issue = issues[0]
    task = runner._task_for(issue, tasks)
    return issue, task


def test_propose_edits_passes_parents_to_the_editor() -> None:
    runner, editor = _wired()
    issue, task = _first_issue_and_task(runner)
    _, analysis = runner.observe(runner.pool.base, task)
    runner.propose_edits(
        runner.pool.base, issue, task, analysis, "att-wiring-1"
    )
    assert editor.seen is not None
    assert editor.seen.parents, "editor received no parent context"
    primaries = [p for p in editor.seen.parents if p.is_primary]
    assert len(primaries) == 1


def test_propose_edits_passes_the_creatable_prefix() -> None:
    runner, editor = _wired()
    issue, task = _first_issue_and_task(runner)
    _, analysis = runner.observe(runner.pool.base, task)
    runner.propose_edits(
        runner.pool.base, issue, task, analysis, "att-wiring-2"
    )
    assert editor.seen.creatable_prefix != ""


def test_propose_edits_returns_observed_parent_ids() -> None:
    runner, editor = _wired()
    issue, task = _first_issue_and_task(runner)
    _, analysis = runner.observe(runner.pool.base, task)
    result = runner.propose_edits(
        runner.pool.base, issue, task, analysis, "att-wiring-3"
    )
    assert len(result) == 4
    assert result[3] == ()


def test_observed_parent_ids_come_from_the_editor_not_the_offer() -> None:
    """An editor that reads a donor reports it; merely offering does not."""
    runner, editor = _wired()
    editor.last_parents_read = ("donor-x",)
    issue, task = _first_issue_and_task(runner)
    _, analysis = runner.observe(runner.pool.base, task)
    result = runner.propose_edits(
        runner.pool.base, issue, task, analysis, "att-wiring-4"
    )
    assert result[3] == ("donor-x",)


def test_run_attempt_still_completes_with_the_new_arity() -> None:
    runner, _ = _wired()
    tasks = runner._tasks_for_test()
    outcome = runner.run_attempt(tasks)
    assert outcome.attempt_id
```

- [ ] **Step 2: Add the task-list test seam**

`_tasks_for_test` does not exist. Add it to `SequentialGepaRunner` so the wiring
tests can reuse the same coreset the phase 6 tests use:

```python
    def _tasks_for_test(self) -> tuple[EvolutionTask, ...]:
        """Test seam: the task coreset this runner was constructed against."""
        return tuple(self._test_tasks)
```

and add the backing field, set by the phase 6 test helper:

```python
    _test_tasks: tuple[EvolutionTask, ...] = ()
```

If `tests/test_phase_6_orchestrator._runner` does not already set a task tuple on
the runner, set `runner._test_tasks = TASKS` inside that helper, where `TASKS` is
the coreset it already builds.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator_editor_wiring.py -v 2>&1 | tee terminal_output/cuga-editor/task11-red.log`
Expected: FAIL — `assert editor.seen.parents` is empty, and `len(result) == 4` fails with 3.

- [ ] **Step 4: Rewrite propose_edits**

Replace `propose_edits` (`orchestrator.py:1134-1163`):

```python
    def propose_edits(
        self,
        parent_entry: PoolEntry,
        issue: TargetIssue,
        task: EvolutionTask,
        analysis: CausalAnalysis,
        attempt_id: str,
    ) -> tuple[CandidateWorkspace, EditorResponse | None, int, tuple[str, ...]]:
        """Materialize a workspace and obtain a validated editor response.

        The editor receives the primary parent plus up to ``donor_count`` donor
        parents, so one call can refine the primary or transplant a capability
        from a donor. Donors are read-only: writes always land in the primary's
        workspace.

        The fourth return element is the donor parents the editor actually read.
        It comes from the editor's own tool-execution ledger, never from its
        prose, so lineage cannot claim a donor that was merely offered.
        """
        workspace = self.adapter.materialize_candidate(
            parent_entry.version, attempt_id
        )
        write_set = tuple(issue.writable_artifact_ids)
        current = self.adapter.read_artifacts(parent_entry.version, write_set)

        entries = self.select_parents(k=self.donor_count + 1)
        parents = tuple(
            ParentContext(
                candidate_id=entry.candidate_id,
                version=entry.version,
                is_primary=entry.candidate_id == parent_entry.candidate_id,
                score_summary={
                    t_id: cell.mean
                    for (t_id, _m), cell in entry.score_tensor.items()
                },
            )
            for entry in entries
        )
        # select_parents samples independently, so the chosen parent may not be
        # in the returned set. The workspace owner must always be the primary.
        if not any(p.is_primary for p in parents):
            parents = (
                ParentContext(
                    candidate_id=parent_entry.candidate_id,
                    version=parent_entry.version,
                    is_primary=True,
                    score_summary={
                        t_id: cell.mean
                        for (t_id, _m), cell in parent_entry.score_tensor.items()
                    },
                ),
                *(p for p in parents if not p.is_primary),
            )

        request = EditorRequest(
            base_workspace=workspace,
            task=task,
            analysis=analysis,
            issue_id=issue.issue_id,
            write_set=write_set,
            current_artifacts=dict(current),
            parents=parents,
            creatable_prefix=getattr(self.adapter, "creatable_prefix", ""),
            pool_created_count=self._pool_created_count(),
        )
        repair = repair_once_then_classify(self.editor, request)
        observed = tuple(getattr(self.editor, "last_parents_read", ()))
        return workspace, repair.response, repair.correction_requests, observed

    def _pool_created_count(self) -> int:
        """Generated artifacts already present, for the creation cap."""
        counter = getattr(self.adapter, "created_artifact_count", None)
        if counter is None:
            return 0
        return max(
            (counter(entry.version) for entry in self.pool.all_entries()),
            default=0,
        )
```

Import `ParentContext` from `agent_evolve.core.editor` at the top of
`orchestrator.py` alongside the existing editor imports.

- [ ] **Step 5: Update run_attempt for the new arity**

In `run_attempt` (`orchestrator.py:1274-1276`), change the unpacking:

```python
        workspace, response, corrections, observed_parents = self.propose_edits(
            parent, issue, task, analysis, attempt_id
        )
```

and in the acceptance branch (`orchestrator.py:1302-1306`):

```python
        if decision.accepted:
            committed = self.commit_to_pool(
                parent,
                workspace,
                attempt_id,
                validation,
                analysis,
                extra_parent_ids=observed_parents,
            )
            result_candidate_id = committed.candidate_id
```

- [ ] **Step 6: Run the wiring tests**

Run: `uv run pytest tests/test_orchestrator_editor_wiring.py -v 2>&1 | tee terminal_output/cuga-editor/task11-green.log`
Expected: 5 passed.

- [ ] **Step 7: Fix any existing caller broken by the arity change**

Run: `grep -rn "propose_edits" tests/ examples/ src/ | grep -v __pycache__`

Any call site unpacking three values must be updated to four. Update them, then re-run.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/task11-suite.log`
Expected: 788 passed, 1 skipped. Any failure here is a real integration break, not noise.

- [ ] **Step 9: Stage**

```bash
git add src/agent_evolve/core/orchestrator.py tests/test_orchestrator_editor_wiring.py
```

---

## Completion checklist

- [ ] Full suite green: `uv run pytest 2>&1 | tee terminal_output/cuga-editor/final-suite.log`
- [ ] `grep -rn "^from cuga\|^import cuga" src/agent_evolve/core/` returns nothing (core stays agent-neutral)
- [ ] `grep -rn "expected_contract" src/agent_evolve/adapters/cuga_editor*.py` appears only in the contamination guard, which consumes it and never emits it
- [ ] `FakeEditor` is untouched and its tests still pass (it remains a valid fixture, never a reported result)
- [ ] Task 10 live verification outcome recorded verbatim, including a `no_tool_call` result
- [ ] Nothing committed without explicit user approval

## Spec coverage

| Spec section | Task |
|---|---|
| §4 architecture / three invariants | 8 (protocol + isolation), 7 (request-scoped tools) |
| §5 evidence cluster | 3, 7 |
| §5 harness cluster + staging | 2, 7 |
| §5 creation namespace and caps | 2, 4 |
| §5 history cluster | 7 |
| §5 parents cluster | 7 |
| §5 submit terminal tool | 7 |
| §6 instructions and skills | 6 |
| §7 request/response flow | 5, 11 |
| §7 primary + K-1 donors | 9, 11 |
| §8 evidence boundary | 3 |
| §8 contamination guard | 3 |
| §9 provenance from tool execution | 2 (ledger), 8 (surfacing), 9 + 11 (lineage) |
| §10 outcome taxonomy | 5 (enum), 8 (classification) |
| §11 offline unit tests | every task |
| §11 stubbed-agent tests | 8 |
| §11 isolation regression test | 8 |
| §11 live verification script | 10 |
| §12 budget cap | 1 |

Deliberately not implemented, per spec: simulation tools (§5 omissions),
`merge.py` defect fixes (§14), per-component instruction artifacts (§13).
