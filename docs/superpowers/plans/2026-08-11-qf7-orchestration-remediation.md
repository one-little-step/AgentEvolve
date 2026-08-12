# QF7 Orchestration Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove qf7's verified parallel-evidence, fabricated-blame, write-authorization, minimal-scoring, and recursive-redaction defects.

**Architecture:** Preserve measured parallel validation data through the barrier and record it only after candidate admission. Reject evidence-free reconstructed issues before editor dispatch, enforce write authorization at both orchestrator entry points, fail closed on missing minimal evaluation contracts, and recursively inspect all persisted memory payloads.

**Tech Stack:** Python 3.14, dataclasses, Pydantic, pytest, existing fake adapter and fake editor.

## Global Constraints

- `src/agent_evolve/core/` remains agent-neutral and must not import CUGA or a concrete adapter.
- Do not fabricate blame nodes, trace events, or artifact attribution.
- Edits outside `EditorRequest.write_set` fail before workspace mutation or lease acquisition.
- Missing minimal evaluation contracts fail closed; they cannot count as success.
- Sanitization recursively rejects expected answers, evaluator internals, labels, regexes, and credentials.
- Capture verification commands with `2>&1 | tee terminal_output/qf7-orchestration/<name>.log`.
- Do not implement `core/storage.py` or `core/config.py` in this increment.

---

### Task 1: Enforce Write Authorization

**Files:**
- Modify: `src/agent_evolve/core/orchestrator.py`
- Create: `tests/test_orchestrator_logic.py`

**Interfaces:**
- Consumes: `EditorRequest.write_set: tuple[str, ...]`, `EditorResponse.edits: tuple[ArtifactEdit, ...]`.
- Produces: `WriteAuthorizationError` before adapter mutation or lease acquisition.

- [ ] **Step 1: Write failing tests for sequential and parallel unauthorized edits**

```python
def test_editor_write_authorization_guard_blocks_sequential_mutation():
    editor = UnauthorizedEditor("outside-write-set")
    orch = _orchestrator(RESEARCH_SEQUENTIAL, editor=editor)
    with pytest.raises(WriteAuthorizationError):
        orch.run_iteration([_task()])
    assert orch.adapter.applied_edits == []


def test_editor_write_authorization_guard_blocks_parallel_lease_acquisition():
    editor = UnauthorizedEditor("outside-write-set")
    orch = _orchestrator(RESEARCH_PARALLEL, editor=editor)
    with pytest.raises(WriteAuthorizationError):
        orch.run_iteration([_task()])
    assert orch.adapter.applied_edits == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_orchestrator_logic.py -k authorization -v`

Expected: FAIL because an unauthorized `ArtifactEdit` reaches the adapter.

- [ ] **Step 3: Add one shared authorization guard and call it from both paths**

```python
from agent_evolve.core.errors import WriteAuthorizationError


def _validate_editor_writes(request: EditorRequest, response: EditorResponse) -> None:
    unauthorized = {
        edit.artifact_id for edit in response.edits
        if edit.artifact_id not in request.write_set
    }
    if unauthorized:
        raise WriteAuthorizationError(
            f"editor proposed artifacts outside write_set: {sorted(unauthorized)}"
        )
```

Call this immediately after each `self.editor.propose_edit(request)`, before
`apply_structured_edits`, clash checks, or lease acquisition.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_orchestrator_logic.py -k authorization -v`

Expected: PASS.

### Task 2: Make Minimal Scoring Fail Closed

**Files:**
- Modify: `src/agent_evolve/core/orchestrator.py`
- Modify: `tests/test_orchestrator_logic.py`

**Interfaces:**
- Consumes: `EvolutionTask.expected_contract`.
- Produces: score `0.0` and an empty blame graph for absent `expected_substring`.

- [ ] **Step 1: Write the failing rollout test**

```python
def test_minimal_rollout_without_declared_contract_is_not_a_success():
    orch = _orchestrator(MINIMAL)
    task = EvolutionTask(task_id="task-no-contract", input_text="work")
    workspace = orch.adapter.materialize_candidate("base-v0", "attempt-no-contract")

    _trace, analysis = orch._rollout(workspace, task, "rollout-no-contract")

    assert analysis.score == 0.0
    assert analysis.blame_graph.nodes == ()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_orchestrator_logic.py -k no_contract -v`

Expected: FAIL because `"" in trace.final_output` currently awards score `1.0`.

- [ ] **Step 3: Require an explicitly declared expected substring**

```python
expected = task.expected_contract.get("expected_substring")
if expected is None:
    analysis = CausalAnalysis(
        mechanism=f"insufficient_evidence:{task.task_id}",
        severity=0.0,
        score=0.0,
        blame_graph=BlameGraph(nodes=()),
    )
elif str(expected) in trace.final_output:
    analysis = empty_analysis()
else:
    # Preserve current minimal failure handling only when a contract exists.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev pytest tests/test_orchestrator_logic.py -k no_contract -v`

Expected: PASS.

### Task 3: Exclude Evidence-Free Reconstructed Issues

**Files:**
- Modify: `src/agent_evolve/core/orchestrator.py`
- Modify: `tests/test_orchestrator_logic.py`

**Interfaces:**
- Consumes: base score tensor cells lacking retained causal trace evidence.
- Produces: no `BlameNode` and no editor request for that reconstructed issue.

- [ ] **Step 1: Write the failing reconstruction test**

```python
def test_score_tensor_reconstruction_never_fabricates_blame_or_dispatches_editor():
    orch = _orchestrator(RESEARCH_SEQUENTIAL)
    orch.pool.record_score("base", 0.0, _provenance("task-1", "cluster-1"))

    result = orch.run_iteration([_task()])

    assert result.attempts == ()
    assert orch.editor.requests == []
```

Use a recording editor fixture and invoke the reconstruction branch without a
fresh causal analysis. The assertion proves the score tensor alone cannot
invent an editable causal issue.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_orchestrator_logic.py -k reconstruction -v`

Expected: FAIL because the code creates a `BlameNode(actor_id="agent", ...)`
and dispatches it to the editor.

- [ ] **Step 3: Remove the fabricated graph and skip the issue**

```python
# A score cell records outcome evidence, not causal attribution. Do not turn it
# into an editable issue after its trace-backed analysis has been discarded.
continue
```

Remove the `fake_analysis` import and construction entirely. Do not replace it
with an empty analysis passed to the editor: an empty graph has no trace-backed
artifact attribution and is ineligible for editing.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev pytest tests/test_orchestrator_logic.py -k reconstruction -v`

Expected: PASS.

### Task 4: Record Real Parallel Admission Evidence

**Files:**
- Modify: `src/agent_evolve/core/parallel.py`
- Modify: `src/agent_evolve/core/orchestrator.py`
- Modify: `tests/test_orchestrator_logic.py`

**Interfaces:**
- Extend `WorkerResult` with `task: EvolutionTask`, `issue_id: str`, and
  `validation_results: tuple[ValidationResult, ...]`.
- `on_committed` records only validation-derived score cells for admitted
  candidates.

- [ ] **Step 1: Write the failing parallel evidence test**

```python
def test_parallel_admissions_record_validation_scores_in_the_pool():
    orch = _orchestrator(RESEARCH_PARALLEL)

    result = orch.run_iteration([_task()])

    admitted = next(candidate_id for candidate_id in result.accepted)
    entry = orch.pool.get(admitted)
    assert entry.score_tensor
    assert all(cell.rollout_count > 0 for cell in entry.score_tensor.values())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_orchestrator_logic.py -k parallel_admissions -v`

Expected: FAIL because accepted parallel candidates currently have empty score
tensors.

- [ ] **Step 3: Preserve and record only actual validation evidence**

```python
@dataclass(frozen=True, slots=True)
class WorkerResult:
    attempt_id: str
    workspace: CandidateWorkspace
    edits: tuple[ArtifactEdit, ...]
    trace: ExecutionTrace
    attempt: EditAttempt
    task: EvolutionTask
    issue_id: str
    validation_results: tuple[ValidationResult, ...]
```

At worker staging, set the three new fields from the existing local task,
issue ID, and `report.all_results`. At barrier admission, add the candidate,
then for every validation result call `_record_score` with the staged task, a
`CausalAnalysis` containing the real validation score and an empty `BlameGraph`,
the issue mechanism cluster, and the validation trace ID. Do not synthesize an
actor, causal mechanism, or trace.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev pytest tests/test_orchestrator_logic.py -k parallel_admissions -v`

Expected: PASS.

### Task 5: Recursively Sanitize Memory Payloads

**Files:**
- Modify: `src/agent_evolve/core/memory.py`
- Modify: `tests/test_memory.py`

**Interfaces:**
- Changes `sanitize_payload(payload: object) -> object`.
- Existing callers may pass mappings; clean values retain their shape.

- [ ] **Step 1: Write failing nested key and string tests**

```python
def test_sanitize_payload_rejects_nested_denylisted_key():
    with pytest.raises(ValueError, match="expected_answer"):
        sanitize_payload({"change": {"expected_answer": "secret"}})


def test_sanitize_payload_rejects_sensitive_string_in_sequence():
    with pytest.raises(ValueError, match="sensitive string"):
        sanitize_payload({"notes": ["contains api_key=secret"]})


def test_sanitize_payload_preserves_clean_nested_shape():
    payload = {"change": [{"artifact": "a"}, ("safe", 1)]}
    assert sanitize_payload(payload) == payload
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_memory.py -k sanitize -v`

Expected: FAIL because only top-level mapping keys are inspected.

- [ ] **Step 3: Implement recursive fail-closed traversal**

```python
def sanitize_payload(payload: object) -> object:
    if isinstance(payload, Mapping):
        bad = _DENYLIST_KEYS & {str(key).lower() for key in payload}
        if bad:
            raise ValueError(f"refusing to persist denied payload keys: {sorted(bad)}")
        return {key: sanitize_payload(value) for key, value in payload.items()}
    if isinstance(payload, str):
        lower = payload.lower()
        if any(marker in lower for marker in ("expected_answer", "api_key", "password")):
            raise ValueError("refusing to persist sensitive string pattern in payload")
        return payload
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return type(payload)(sanitize_payload(value) for value in payload)
    return payload
```

Import `Sequence` from `collections.abc`. Keep mappings and sequences recursive
and fail closed rather than stripping sensitive values.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_memory.py -k sanitize -v`

Expected: PASS.

### Task 6: Verify the Remediation

**Files:**
- Create: `terminal_output/qf7-orchestration/focused.log`
- Create: `terminal_output/qf7-orchestration/full-suite.log`

- [ ] **Step 1: Run the focused qf7 regression suite**

Run:

```bash
mkdir -p terminal_output/qf7-orchestration
uv run --extra dev pytest tests/test_orchestrator_logic.py tests/test_memory.py -v 2>&1 | tee terminal_output/qf7-orchestration/focused.log
```

Expected: PASS with all qf7 regression tests collected.

- [ ] **Step 2: Run the full suite**

Run:

```bash
uv run --extra dev pytest 2>&1 | tee terminal_output/qf7-orchestration/full-suite.log
```

Expected: PASS with no regressions.
