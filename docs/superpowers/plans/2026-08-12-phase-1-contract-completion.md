# Phase-1 Standard Pydantic Contract Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete binding neutral persisted-record schemas with standard frozen Pydantic models before Phase 2.

**Architecture:** Each persisted record directly subclasses `BaseModel` with `ConfigDict(frozen=True)`. Pydantic’s standard direct construction, `model_validate*` APIs, coercion, mutable nested values, exception chains, and `ValidationError` behavior are accepted. Ordinary `ValueError` validators enforce documented relationships.

**Tech Stack:** Python 3.12, Pydantic 2, pytest 8, uv.

## Global Constraints

- The explicit user decision accepts standard Pydantic coercion and nested mutability.
- Do not add `_TypedBoundaryModel`, custom construction factories, typed error wrappers, recursive freezing, or Pydantic API blocking.
- Every persisted record directly subclasses `BaseModel` with `ConfigDict(frozen=True)`.
- Tests expect Pydantic `ValidationError`, not project-specific domain errors.
- Core remains agent-neutral and may not import CUGA, Gaia, concrete adapters, storage, pool, selection, parallel, or orchestrator modules.
- Do not edit `config.py`, `storage.py`, `memory.py`, `editor.py`, `merge.py`, `parallel.py`, `orchestrator.py`, or adapters.
- Write tests first and capture every test command with `2>&1 | tee terminal_output/phase-1-contract-completion/<name>.log`.
- Do not commit unless explicitly requested.

---

### Task 1: Restore Standard Pydantic Boundaries

**Files:**
- Modify: `src/agent_evolve/core/contracts.py`
- Modify: `tests/test_contracts_immutability.py`

**Governing sources:** `docs/superpowers/specs/2026-08-12-phase-1-contract-completion-design.md` and `docs/architecture/data-contracts.md:5-29`.

**Produces:** Direct frozen `BaseModel` persisted boundaries and normal Pydantic API behavior.

- [ ] **Step 1: Write failing standard-behavior tests**

```python
import pytest
from pydantic import ValidationError


def test_standard_constructor_and_model_validate_are_supported() -> None:
    values = score_cell_values()
    assert ScoreCell(**values).candidate_id == "candidate-1"
    assert ScoreCell.model_validate(values).candidate_id == "candidate-1"


def test_frozen_model_blocks_attribute_assignment_but_nested_mapping_is_mutable() -> None:
    cell = ScoreCell(**score_cell_values())
    with pytest.raises(ValidationError):
        cell.score = 0.0
    cell.artifact_versions["artifact-2"] = "sha256:abcdef"
    assert cell.artifact_versions["artifact-2"] == "sha256:abcdef"
```

- [ ] **Step 2: Verify RED**

```bash
mkdir -p terminal_output/phase-1-contract-completion
uv run --extra dev pytest tests/test_contracts_immutability.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/standard-pydantic-red.log
```

Expected: FAIL because the previous implementation blocks normal Pydantic APIs or freezes nested values.

- [ ] **Step 3: Implement direct standard models**

Remove `_TypedBoundaryModel`, factory methods, error mapping, recursive-freezing
helpers, and overridden Pydantic APIs. Make `ScoreCell`, `AttemptRecord`,
`EditPlan`, `ArtifactMergeDecision`, and `MergeProvenance` directly subclass
`BaseModel` with `model_config = ConfigDict(frozen=True)`. Keep ordinary field
constraints, content-hash lexical validation, and ordinary `ValueError` model
validators.

- [ ] **Step 4: Verify GREEN**

```bash
uv run --extra dev pytest tests/test_contracts_immutability.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/standard-pydantic-green.log
git diff --check 2>&1 | tee terminal_output/phase-1-contract-completion/task-1-diff-check.log
```

Expected: all tests PASS and diff check exits 0.

### Task 2: Complete ScoreCell And AttemptRecord

**Files:**
- Modify: `src/agent_evolve/core/contracts.py`
- Modify: `tests/test_contracts_validation.py`

**Governing sources:** `docs/architecture/data-contracts.md:31-79,130-159`.

**Produces:** Complete `ScoreCell` and `AttemptRecord` fields and ordinary Pydantic cross-field validators.

- [ ] **Step 1: Write failing relation tests**

```python
def test_score_cell_requires_reason_for_unavailable_coverage() -> None:
    with pytest.raises(ValidationError, match="coverage_reason"):
        ScoreCell(**score_cell_values(coverage="unavailable", coverage_reason=None))


def test_score_cell_requires_stability_after_one_rollout() -> None:
    with pytest.raises(ValidationError, match="stability"):
        ScoreCell(**score_cell_values(rollout_count=2, stability=None))


def test_rejected_attempt_requires_evidence_references() -> None:
    with pytest.raises(ValidationError, match="memory_refs"):
        AttemptRecord(**attempt_record_values(status="rejected", memory_refs=()))


def test_unsealed_attempt_rejects_hashes_after() -> None:
    with pytest.raises(ValidationError, match="hashes_after"):
        AttemptRecord(**attempt_record_values(workspace_sealed=False, hashes_after={"artifact-1": "sha256:abcdef"}))
```

- [ ] **Step 2: Verify RED**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/score-attempt-red.log
```

Expected: FAIL for missing relation behavior.

- [ ] **Step 3: Implement fields and validators**

Add `coverage_reason` and require it exactly when coverage is `unavailable` or
`excluded`. Preserve score ranges, rollout cardinality/uniqueness, verdict and
artifact version requirements, and the stability relation. Add
`workspace_sealed`; require non-empty `hashes_after` iff sealed, evidence
references unless unavailable, and the accepted/rejected candidate and validation
relations. Validators raise ordinary `ValueError`.

- [ ] **Step 4: Verify GREEN**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/score-attempt-green.log
git diff --check 2>&1 | tee terminal_output/phase-1-contract-completion/task-2-diff-check.log
```

Expected: all tests PASS.

### Task 3: Complete Edit And Validation Contracts

**Files:**
- Modify: `src/agent_evolve/core/contracts.py`
- Modify: `tests/test_contracts_validation.py`

**Governing sources:** `docs/architecture/data-contracts.md:105-176`.

**Produces:** `ArtifactEdit`, `ExpectedEffect`, `EditPlan`, `ValidationCase`, and `ValidationResult` standard Pydantic models.

- [ ] **Step 1: Write failing tests**

```python
def test_edit_plan_rejects_edit_outside_authorized_writes() -> None:
    with pytest.raises(ValidationError, match="authorized_writes"):
        EditPlan(
            attempt_id="attempt-1", issue_fingerprint="issue-1",
            read_requests=("artifact-read",), authorized_writes=("artifact-write",),
            edits=(ArtifactEdit(artifact_id="artifact-read", operation="replace", payload={}),),
            rationale="safe rationale", risks=(),
            expected_effect=ExpectedEffect(mechanism_cluster_refs=("cluster-1",)),
        )


def test_validation_result_rejects_accept_with_violated_protected_floor() -> None:
    with pytest.raises(ValidationError, match="protected_floor_outcome"):
        ValidationResult(**validation_result_values(protected_floor_outcome="violated", decision="accept"))
```

- [ ] **Step 2: Verify RED**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/edit-validation-red.log
```

Expected: FAIL because the schemas are incomplete.

- [ ] **Step 3: Implement standard models**

Define `ArtifactEdit` and `ExpectedEffect` with required non-empty identifiers.
Replace `EditPlan.edit_targets` with non-empty `edits`, validating every edit
artifact against `authorized_writes`. Add `ValidationCase` and `ValidationResult`
with all fields mandated in `data-contracts.md:161-176`; reject protected-floor
violation paired with acceptance using `ValueError`.

- [ ] **Step 4: Verify GREEN**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/edit-validation-green.log
git diff --check 2>&1 | tee terminal_output/phase-1-contract-completion/task-3-diff-check.log
```

Expected: all tests PASS.

### Task 4: Complete Merge And Memory Contracts

**Files:**
- Modify: `src/agent_evolve/core/contracts.py`
- Modify: `tests/test_contracts_validation.py`

**Governing sources:** `docs/architecture/data-contracts.md:178-231`.

**Produces:** `ArtifactMergeDecision`, `MergeProvenance`, `RedactionReport`, and `MemoryRecord` standard Pydantic models.

- [ ] **Step 1: Write failing tests**

```python
def test_refined_merge_requires_refinement_request() -> None:
    with pytest.raises(ValidationError, match="refinement_request_ref"):
        ArtifactMergeDecision(**merge_decision_values(inheritance="refined", refinement_request_ref=None))


def test_ancestor_result_cannot_emit_operation() -> None:
    with pytest.raises(ValidationError, match="operation_emitted"):
        ArtifactMergeDecision(**merge_decision_values(resulting_hash="sha256:ancestor", operation_emitted=True))


def test_admitted_merge_requires_child_candidate() -> None:
    with pytest.raises(ValidationError, match="child_candidate_id"):
        MergeProvenance(**merge_provenance_values(child_admitted=True, child_candidate_id=None))


def test_memory_record_forbids_raw_prompt() -> None:
    with pytest.raises(ValidationError, match="raw_prompt"):
        MemoryRecord(**memory_record_values(), raw_prompt="secret")
```

- [ ] **Step 2: Verify RED**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/merge-memory-red.log
```

Expected: FAIL for missing schema relations.

- [ ] **Step 3: Implement standard models**

Add refinement-request and operation-emitted fields to merge decisions; enforce
refined/shared/ancestor/no-op relations with `ValueError`. Add `child_admitted`
and child-ID relation to merge provenance. Add frozen `RedactionReport` and
`MemoryRecord` with `ConfigDict(frozen=True, extra="forbid")`; no recursive
redaction scanner belongs in Phase 1.

- [ ] **Step 4: Verify GREEN**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py tests/test_contracts_immutability.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/merge-memory-green.log
git diff --check 2>&1 | tee terminal_output/phase-1-contract-completion/task-4-diff-check.log
```

Expected: all tests PASS.

### Task 3.5: Align Legacy Tests With ArtifactEdit Contract

**Files:**
- Modify: `tests/test_editor.py`
- Modify: `tests/test_parallel_rollback.py`

**Governing sources:** `docs/architecture/data-contracts.md:105-128` and the
explicit standard-Pydantic decision in
`docs/superpowers/specs/2026-08-12-phase-1-contract-completion-design.md`.

**Produces:** Existing editor and parallel rollback tests that exercise the
approved construction-time `ArtifactEdit` validation and keyword-only Pydantic
model API without weakening the persisted contract.

- [ ] **Step 1: Write the failing compatibility expectations**

In `tests/test_editor.py`, replace any expectation that a blank `ArtifactEdit`
can be constructed with this assertion:

```python
from pydantic import ValidationError


def test_artifact_edit_rejects_empty_artifact_id_at_construction() -> None:
    with pytest.raises(ValidationError, match="artifact_id"):
        ArtifactEdit(artifact_id="", operation="replace", payload={})
```

In `tests/test_parallel_rollback.py`, preserve the test's staged-result
behavior while changing the edit construction to keywords:

```python
edits=(
    ArtifactEdit(
        artifact_id=f"artifact-{attempt_id}",
        operation="replace",
        payload={},
    ),
),
```

- [ ] **Step 2: Verify the pre-fix regression**

```bash
uv run --extra dev pytest tests/test_editor.py tests/test_parallel_rollback.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/artifact-edit-integration-red.log
```

Expected before implementation: the legacy empty-ID or positional-constructor
expectations fail against the approved Pydantic `ArtifactEdit` contract.

- [ ] **Step 3: Keep tests aligned with the contract**

Do not modify `ArtifactEdit`, add positional compatibility, defer validation, or
change the construction-time non-empty identifier requirement. Apply only the
test updates from Step 1.

- [ ] **Step 4: Verify integration behavior**

```bash
uv run --extra dev pytest tests/test_editor.py tests/test_parallel_rollback.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/artifact-edit-integration-green.log
git diff --check 2>&1 | tee terminal_output/phase-1-contract-completion/task-3-5-diff-check.log
```

Expected: both tests pass and the diff check exits 0.

### Task 4.5: Complete Global Identity And Merge-Hash Validation

**Files:**
- Modify: `src/agent_evolve/core/contracts.py`
- Modify: `tests/test_contracts_validation.py`

**Governing sources:** `docs/architecture/data-contracts.md:23-29,46-49,146-150,184-212,220-227`.

**Produces:** Standard Pydantic validation of non-empty ID elements and lexical
content hashes wherever a Phase-1 persisted schema declares IDs or `ContentHash`
fields.

- [ ] **Step 1: Write failing identity and merge-hash tests**

```python
@pytest.mark.parametrize("changes", (
    {"rollout_ids": ("", "rollout-2")},
    {"verdict_refs": ("",)},
    {"artifact_versions": {"": "sha256:abcdef"}},
))
def test_score_cell_rejects_blank_references_and_artifact_ids(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ScoreCell(**score_cell_values(**changes))


def test_attempt_record_rejects_blank_task_and_evidence_references() -> None:
    with pytest.raises(ValidationError):
        AttemptRecord(**attempt_record_values(task_refs=("",)))
    with pytest.raises(ValidationError):
        AttemptRecord(**attempt_record_values(memory_refs=("",)))


@pytest.mark.parametrize("field", ("ancestor_hash", "left_hash", "right_hash", "resulting_hash"))
def test_merge_decision_rejects_non_content_hash(field: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactMergeDecision(**merge_decision_values(**{field: "not-a-hash"}))


def test_memory_record_rejects_blank_artifact_or_evidence_reference() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(**memory_record_values(artifact_ids=("",)))
    with pytest.raises(ValidationError):
        MemoryRecord(**memory_record_values(evidence_refs=("",)))
```

- [ ] **Step 2: Verify RED**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/identity-hash-red.log
```

Expected: FAIL because blank nested IDs and non-lexical merge hashes are accepted.

- [ ] **Step 3: Add ordinary Pydantic validators**

Add private helpers that raise `ValueError` for blank tuple IDs, blank mapping
keys, invalid mapping hash values, and invalid scalar content hashes. Call them
from ordinary `model_validator(mode="after")` methods on `ScoreCell`,
`AttemptRecord`, `EditPlan`, `ExpectedEffect`, `ArtifactMergeDecision`,
`MergeProvenance`, and `MemoryRecord`. Do not validate `mechanism_ids`, which
are explicitly free-form in `data-contracts.md:40`. Replace only merge fixture
placeholder hashes needed for this validation with valid hexadecimal hashes.

- [ ] **Step 4: Verify GREEN**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py tests/test_contracts_immutability.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/identity-hash-green.log
git diff --check 2>&1 | tee terminal_output/phase-1-contract-completion/task-4-5-diff-check.log
```

Expected: all tests PASS and diff check exits 0.

### Task 5: Verify Phase 1 And Reopen Phase 2

**Files:**
- Modify: `docs/superpowers/specs/2026-08-12-architecture-enforcement-design.md`
- Test: `tests/test_contracts_validation.py`
- Test: `tests/test_contracts_immutability.py`
- Test: `tests/test_contracts.py`

- [ ] **Step 1: Run focused suite**

```bash
uv run --extra dev pytest tests/test_contracts_validation.py tests/test_contracts_immutability.py tests/test_contracts.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/focused-suite.log
```

Expected: all tests PASS.

- [ ] **Step 2: Run full suite**

```bash
uv run --extra dev pytest 2>&1 | tee terminal_output/phase-1-contract-completion/full-suite.log
```

Expected: all tests PASS.

- [ ] **Step 3: Record bounded verification**

Append after passing suites:

```markdown
## Phase-1 Verification Record

Phase 1 is verified by logs under `terminal_output/phase-1-contract-completion/`.
The verified scope is standard frozen Pydantic fields and cross-field validation
for neutral persisted contracts. Standard Pydantic coercion, `ValidationError`,
exception behavior, and nested mutability are accepted by explicit user decision.
This does not claim Phase-2 or later implementation.
```

- [ ] **Step 4: Check final scope**

```bash
uv run python -c "from pathlib import Path; assert 'import cuga' not in Path('src/agent_evolve/core/contracts.py').read_text().lower()" 2>&1 | tee terminal_output/phase-1-contract-completion/core-neutrality.log
git diff --check 2>&1 | tee terminal_output/phase-1-contract-completion/final-diff-check.log
git status --short 2>&1 | tee terminal_output/phase-1-contract-completion/final-status.log
```

Expected: neutrality and diff checks exit 0.

## Plan Self-Review

- The plan follows the explicit standard-Pydantic override and contains no factory-only or typed-error work.
- Tasks 2-4 cover all Phase-1 record families without entering Phase-2 storage/config behavior.
- All rejection tests use Pydantic `ValidationError`.
