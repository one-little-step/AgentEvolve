# Superseded: Step-2 Config And Transactional Storage Foundation Implementation Plan

> Superseded by the approved Phase 1-4 research storage exception in
> `docs/superpowers/specs/2026-08-12-architecture-enforcement-design.md` and
> the design in `docs/superpowers/specs/2026-08-12-phase-1-4-research-core-design.md`.
> Do not execute this SQLite/WAL plan for the current research path. SQLite WAL
> remains required when Phase 5 parallel execution begins.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the validated profile resolver and SQLite WAL transactional metadata store required before any further evidence, selection, parallel, or orchestration implementation.

**Architecture:** `core.config` resolves only the four named, agent-neutral experiment profiles into immutable feature gates and bounded budgets, and serializes that resolved state for manifests. `core.storage` owns a single SQLite coordinator connection, recursive fail-closed redaction, content-addressed blob staging, atomic metadata publication, idempotency, and recovery; workers receive no write API. It stores opaque, sanitized records rather than interpreting adapters, selection, or causal semantics.

**Tech Stack:** Python 3.12, standard-library `sqlite3`, `hashlib`, `json`, `pathlib`, `threading`, Pydantic 2, pytest 8, uv.

## Mandatory Pre-Read

Before writing a test or production line, read and cite the governing sections
in these binding documents:

1. `docs/architecture/README.md:73-89` for the mandatory implementation order.
2. `docs/architecture/data-contracts.md:5-29, 214-231` for construction
   validation, opaque exact IDs, and persistence restrictions.
3. `docs/architecture/storage-and-transactions.md:10-159` for the SQLite WAL,
   atomic batch, recovery, idempotency, lease, blob, and redaction mandates.
4. `docs/architecture/component-contracts.md:12-31, 102-127` for `config.py`
   and `storage.py` ownership and dependency boundaries.
5. `docs/architecture/orchestration-lifecycle.md:86-127` for the barrier and
   failure outcomes storage must support without implementing orchestration.

At the start of each task, record its governing document and section in the
task's work note. Before review, cross-check every changed public interface and
test against those citations. Phase 1 is incomplete because invalid Pydantic
records currently surface generic `ValidationError` rather than the typed domain
errors mandated by `data-contracts.md`. Task 0 remediates and verifies Phase 1
before the first Phase-2 production change; Phase-2 work does not otherwise
expand Phase-1 scope.

## Global Constraints

- `docs/architecture/data-contracts.md` and `docs/architecture/storage-and-transactions.md` are binding; do not substitute schemas or storage strategy.
- `src/agent_evolve/core/` remains agent-neutral and must not import CUGA, Gaia, or concrete adapters.
- Do not edit `orchestrator.py`, `parallel.py`, `entropy.py`, `issues.py`, adapter modules, or CUGA-facing modules in this increment.
- All persistence is recursive, content-aware redaction that fails closed; never persist credentials, expected answers, evaluator internals, labels, regexes, raw prompts/responses, or raw unapproved trace bodies.
- SQLite is the reference backend and must use `journal_mode=WAL`, `synchronous=FULL`, `foreign_keys=ON`, a configured busy timeout, and one coordinator-owned writer connection.
- Metadata publication uses `BEGIN IMMEDIATE`; a transaction publishes every staged record category or none.
- Blob files are content-addressed and written before the metadata transaction. Orphan blobs are harmless and recovered later; a committed blob reference missing its file is a corruption failure.
- All IDs are opaque, non-empty, exact full strings. Do not slice, hash, normalize, or derive grouping keys from them.
- Write tests before implementation. Capture every test and verification command with `2>&1 | tee terminal_output/step-2-config-storage/<name>.log`.
- Do not commit unless the user explicitly asks for a commit. At each normal commit checkpoint, inspect `git diff --check` and report the intended commit instead.

---

## File Structure

- Create `src/agent_evolve/core/config.py`: immutable named-profile resolver, feature gates, budgets, manifest-safe serialization, and typed configuration failures.
- Create `src/agent_evolve/core/storage.py`: SQLite schema/setup, coordinator-owned transaction API, blob staging, recursive redaction, atomic batch commit, idempotency, and recovery.
- Modify `src/agent_evolve/core/errors.py`: add only storage/config/redaction typed domain errors required by the new public APIs.
- Create `tests/test_config.py`: profile resolution, feature compatibility, validation, and serialization tests.
- Create `tests/test_storage_acid.py`: pragma, atomicity, statement-failure injection, idempotency, and crash-window tests.
- Create `tests/test_storage_recovery.py`: orphan collection, expired lease cleanup, interrupted batch recording, derived-state rebuilding, and committed-blob corruption tests.
- Create `tests/test_storage_redaction.py`: nested mapping/sequence/object/string redaction failures and sanitized persistence test.

No existing prototype module is modified except `errors.py`. In particular, `memory.py` remains prototype state until Phase 4; storage owns the binding persistence gateway in this phase.

## Public Interfaces

```python
# core/config.py
ExperimentProfile = Literal[
    "minimal", "research_sequential", "research_parallel", "full_ablation"
]

@dataclass(frozen=True, slots=True)
class FeatureGates:
    causal_blame: bool
    edit_memory: bool
    feedback_validation: bool
    semantic_rag: bool
    merge: bool
    parallel_execution: bool
    all_ablation_controls: bool

@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_attempts: int | None
    max_accepted_edits: int | None
    max_model_tokens: int | None
    max_rollouts: int | None
    max_judge_verdicts: int | None
    edit_max_retries: int
    max_wall_seconds: int | None
    max_pool_candidates: int | None
    max_history_records: int | None
    max_rag_context_tokens: int | None

@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    profile: ExperimentProfile
    features: FeatureGates
    budgets: BudgetLimits

    def manifest_payload(self) -> dict[str, object]: ...

def resolve_config(
    profile: ExperimentProfile,
    *,
    budget_overrides: Mapping[str, int | None] | None = None,
    feature_overrides: Mapping[str, bool] | None = None,
) -> ResolvedConfig: ...

# core/storage.py
@dataclass(frozen=True, slots=True)
class RedactionReport:
    rule_hits: tuple[str, ...]
    truncations: int

@dataclass(frozen=True, slots=True)
class SanitizedPayload:
    value: object
    report: RedactionReport

@dataclass(frozen=True, slots=True)
class BlobReference:
    digest: str
    byte_length: int

@dataclass(frozen=True, slots=True)
class BatchRecord:
    record_type: Literal[
        "candidate", "score_cell", "attempt", "validation", "merge",
        "memory", "retry_state", "budget", "lease", "snapshot", "manifest"
    ]
    record_id: str
    payload: Mapping[str, object]
    blob_payloads: Mapping[str, bytes] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class BatchCommit:
    transaction_id: str
    snapshot_version: str
    records: tuple[BatchRecord, ...]

class SQLiteRunStore:
    def __init__(self, root: Path, *, busy_timeout_ms: int = 5_000) -> None: ...
    def close(self) -> None: ...
    def begin_batch(self, transaction_id: str, snapshot_version: str) -> "StagedBatch": ...
    def read_record(self, record_type: str, record_id: str) -> Mapping[str, object] | None: ...
    def list_records(self, record_type: str) -> tuple[Mapping[str, object], ...]: ...
    def recover(self, *, now_epoch_seconds: int) -> "RecoveryReport": ...

class StagedBatch:
    def stage(self, record: BatchRecord) -> None: ...
    def commit(self) -> None: ...
    def rollback(self, reason: str) -> None: ...

def sanitize_for_persistence(value: object) -> SanitizedPayload: ...
```

`BatchRecord` deliberately accepts opaque JSON-like payloads. The typed Phase-1 records will be serialized by later consumers; storage must not import pool, editor, evaluation, merge, or adapter modules to decode them.

The producing module validates every domain record against its exact
`docs/architecture/data-contracts.md` schema before it calls storage. Phase 2
storage validates only its own record envelope, JSON compatibility, redaction,
idempotency, and blob references. It must not import or duplicate Phase-1 or
later domain schemas.

### Task 0: Remediate Phase-1 Typed Construction Failures

**Files:**
- Modify: `src/agent_evolve/core/contracts.py`
- Modify: `src/agent_evolve/core/errors.py`
- Test: `tests/test_contracts_validation.py`

**Governing architecture:** `docs/architecture/README.md:78-80` and
`docs/architecture/data-contracts.md:5-12, 31-59, 105-159, 178-231`.

**Consumes:** Existing `EvolutionContractError`, `ScoreProvenanceError`,
`ScoreRangeError`, `AttemptRecordError`, and `WriteAuthorizationError`.

**Produces:** Typed construction errors for every existing persisted boundary
model rejection rule, without changing legacy runtime dataclasses or importing
later-phase modules.

- [ ] **Step 1: Write failing typed-error tests for the mandated rejection mappings**

```python
from agent_evolve.core.errors import AttemptRecordError, ScoreProvenanceError, ScoreRangeError, WriteAuthorizationError


def test_score_cell_raises_typed_provenance_error_for_zero_rollouts() -> None:
    with pytest.raises(ScoreProvenanceError, match="rollout_count"):
        _score_cell(rollout_count=0, rollout_ids=())


def test_score_cell_raises_typed_range_error_for_out_of_range_score() -> None:
    with pytest.raises(ScoreRangeError, match="score"):
        _score_cell(score=1.1)


def test_accepted_attempt_raises_typed_attempt_record_error_without_result() -> None:
    with pytest.raises(AttemptRecordError, match="result_candidate_id"):
        _attempt_record(status="accepted", validation_result_ref="validation-1")


def test_edit_plan_raises_typed_write_authorization_error_for_unauthorized_target() -> None:
    with pytest.raises(WriteAuthorizationError, match="authorized_writes"):
        EditPlan(
            attempt_id="attempt-1",
            issue_fingerprint="issue-1",
            read_requests=("artifact-read",),
            authorized_writes=("artifact-write",),
            edit_targets=("artifact-read",),
            rationale="bounded sanitized rationale",
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
mkdir -p terminal_output/phase-1-verification terminal_output/step-2-config-storage
uv run --extra dev pytest tests/test_contracts_validation.py -v 2>&1 | tee terminal_output/phase-1-verification/typed-errors-red.log
```

Expected: FAIL because current Pydantic models emit `ValidationError` rather
than the specified typed domain errors.

- [ ] **Step 3: Map Pydantic construction failures to typed domain errors**

```python
class _TypedBoundaryModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    def __init__(self, /, **data: object) -> None:
        try:
            super().__init__(**data)
        except ValidationError as error:
            raise _map_validation_error(error) from error
```

Implement `_map_validation_error()` in `contracts.py`. Inspect only Pydantic's
structured `error.errors()` entries and map: score/severity/confidence/stability
range errors to `ScoreRangeError`; `ScoreCell` provenance fields and relations
to `ScoreProvenanceError`; `AttemptRecord` terminal-state relations to
`AttemptRecordError`; and `EditPlan` authorization relations to
`WriteAuthorizationError`. Preserve ordinary Pydantic validation for fields not
assigned a binding typed error. Make `ScoreCell`, `AttemptRecord`, `EditPlan`,
`ArtifactMergeDecision`, and `MergeProvenance` inherit `_TypedBoundaryModel`.
Keep error messages bounded and do not embed any rejected payload values.

- [ ] **Step 4: Run focused Phase-1 tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_contracts_validation.py tests/test_contracts.py -v 2>&1 | tee terminal_output/phase-1-verification/contracts-focused.log
```

Expected: all focused contract tests PASS, including typed-error assertions.

- [ ] **Step 5: Inspect the checkpoint**

Run:

```bash
git diff --check 2>&1 | tee terminal_output/phase-1-verification/diff-check.log
```

Expected: exit status 0. Do not commit unless explicitly requested.

### Task 1: Resolve Named Profiles, Typed Storage Errors, And Validated Budgets

**Files:**
- Create: `src/agent_evolve/core/config.py`
- Test: `tests/test_config.py`

**Governing architecture:** `docs/architecture/target-rho-parallel-gepa.md:168-182` and `docs/architecture/component-contracts.md:18-19`.

**Consumes:** Phase-1 typed contract errors from Task 0 and the named profile definitions in the binding target architecture.

**Produces:** `ConfigurationError`, `StorageError`, `StorageCorruptionError`,
`TransactionStateError`, `RedactionError`, `FeatureGates`, `BudgetLimits`,
`ResolvedConfig`, and `resolve_config()`.

- [ ] **Step 1: Write failing profile-resolution tests**

```python
import pytest

from agent_evolve.core.config import resolve_config
from agent_evolve.core.errors import ConfigurationError


def test_minimal_profile_disables_causal_memory_merge_and_parallel_features() -> None:
    config = resolve_config("minimal")

    assert config.profile == "minimal"
    assert config.features.causal_blame is False
    assert config.features.edit_memory is False
    assert config.features.feedback_validation is False
    assert config.features.merge is False
    assert config.features.parallel_execution is False


def test_research_parallel_inherits_sequential_features_and_enables_barrier_execution() -> None:
    config = resolve_config("research_parallel")

    assert config.features.causal_blame is True
    assert config.features.edit_memory is True
    assert config.features.feedback_validation is True
    assert config.features.parallel_execution is True


def test_full_ablation_allows_explicit_feature_override_and_records_it() -> None:
    config = resolve_config("full_ablation", feature_overrides={"merge": False})

    assert config.features.all_ablation_controls is True
    assert config.features.merge is False
    assert config.manifest_payload()["features"]["merge"] is False


def test_non_ablation_profile_rejects_feature_override() -> None:
    with pytest.raises(ConfigurationError, match="full_ablation"):
        resolve_config("minimal", feature_overrides={"causal_blame": True})


def test_budget_override_rejects_unknown_or_non_positive_values() -> None:
    with pytest.raises(ConfigurationError, match="unknown budget"):
        resolve_config("minimal", budget_overrides={"not_a_budget": 1})
    with pytest.raises(ConfigurationError, match="must be positive"):
        resolve_config("minimal", budget_overrides={"max_rollouts": 0})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_config.py -v 2>&1 | tee terminal_output/step-2-config-storage/config-red.log
```

Expected: FAIL because profile resolver symbols do not exist.

- [ ] **Step 3: Implement immutable resolved configuration**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

from agent_evolve.core.errors import ConfigurationError

ExperimentProfile = Literal[
    "minimal", "research_sequential", "research_parallel", "full_ablation"
]


@dataclass(frozen=True, slots=True)
class FeatureGates:
    causal_blame: bool
    edit_memory: bool
    feedback_validation: bool
    semantic_rag: bool
    merge: bool
    parallel_execution: bool
    all_ablation_controls: bool


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_attempts: int | None = None
    max_accepted_edits: int | None = None
    max_model_tokens: int | None = None
    max_rollouts: int | None = None
    max_judge_verdicts: int | None = None
    edit_max_retries: int = 3
    max_wall_seconds: int | None = None
    max_pool_candidates: int | None = None
    max_history_records: int | None = None
    max_rag_context_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    profile: ExperimentProfile
    features: FeatureGates
    budgets: BudgetLimits

    def manifest_payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "features": asdict(self.features),
            "budgets": asdict(self.budgets),
        }
```

Add these storage/config errors to `errors.py` before creating `config.py`:

```python
class ConfigurationError(EvolutionContractError):
    """Raised when named-profile or budget configuration is invalid."""


class StorageError(RuntimeError):
    """Base class for transactional storage failures."""


class StorageCorruptionError(StorageError):
    """Raised when committed metadata references unavailable durable content."""


class TransactionStateError(StorageError):
    """Raised when a staged transaction is used outside its valid lifecycle."""


class RedactionError(StorageError):
    """Raised when content cannot be safely persisted."""
```

Implement profile defaults exactly as follows: `minimal` enables none of the seven optional gates; `research_sequential` enables causal blame, edit memory, and feedback validation; `research_parallel` adds parallel execution; `full_ablation` starts with every optional gate enabled and permits only recognized boolean gate overrides. Accept recognized budget overrides for every `BudgetLimits` field. `None` means an operator has not set that hard ceiling; every integer override must be positive, including `edit_max_retries`. The resolver validates complexity-budget values such as `max_pool_candidates` and `max_history_records`, but does not define candidate eviction, history archiving, or stopping behavior: those policies belong to later pool, memory, and orchestration phases.

- [ ] **Step 4: Run configuration tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_config.py -v 2>&1 | tee terminal_output/step-2-config-storage/config-green.log
```

Expected: all `tests/test_config.py` tests PASS.

- [ ] **Step 5: Inspect the checkpoint**

Run:

```bash
git diff --check 2>&1 | tee terminal_output/step-2-config-storage/task-2-diff-check.log
```

Expected: exit status 0. Do not commit unless explicitly requested.

### Task 2: Implement Recursive Fail-Closed Redaction And Blob Staging

**Files:**
- Create: `src/agent_evolve/core/storage.py`
- Test: `tests/test_storage_redaction.py`

**Consumes:** `RedactionError` from Task 1.

**Governing architecture:** `docs/architecture/storage-and-transactions.md:118-133` and `docs/architecture/data-contracts.md:214-231`.

**Produces:** `RedactionReport`, `SanitizedPayload`, `BlobReference`, `sanitize_for_persistence()`, and private content-addressed blob staging helper.

- [ ] **Step 1: Write failing redaction tests**

```python
import pytest

from agent_evolve.core.errors import RedactionError
from agent_evolve.core.storage import sanitize_for_persistence


@dataclass
class NestedRecord:
    metadata: object


@pytest.mark.parametrize(
    "payload",
    (
        {"outer": {"expected_answer": "secret"}},
        {"items": [{"api_key": "secret"}]},
        NestedRecord(metadata=("safe", "contains password=secret")),
        {"summary": "the expected_answer is present"},
        {"raw_prompt": "do not persist"},
    ),
)
def test_recursive_redaction_rejects_prohibited_material(payload: object) -> None:
    with pytest.raises(RedactionError):
        sanitize_for_persistence(payload)


def test_recursive_redaction_returns_new_clean_json_shape_and_report() -> None:
    original = {"outer": ["safe", {"count": 2}], "status": "accepted"}

    result = sanitize_for_persistence(original)

    assert result.value == original
    assert result.value is not original
    assert result.report.rule_hits == ()
    assert result.report.truncations == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_storage_redaction.py -v 2>&1 | tee terminal_output/step-2-config-storage/redaction-red.log
```

Expected: FAIL because `core.storage` does not exist.

- [ ] **Step 3: Implement the recursive sanitizer and staged blob writer**

Use `collections.abc.Mapping` and `Sequence`, but treat `str`, `bytes`, and `bytearray` as scalar values. Recursively inspect mappings, non-string sequences, dataclasses via `dataclasses.fields`, and Pydantic models via `model_dump(mode="python")`. Reject, rather than remove, prohibited field names at any depth after case-folding and reject string values containing case-insensitive markers for `expected_answer`, `expected_label`, `expected_regex`, `api_key`, `password`, `secret`, `token`, `regex`, `raw_prompt`, `raw_response`, `raw_trace`, `evaluator_internal`, `label`, or `credential`.

```python
def sanitize_for_persistence(value: object) -> SanitizedPayload:
    sanitized = _sanitize(value, path=())
    return SanitizedPayload(
        value=sanitized,
        report=RedactionReport(rule_hits=(), truncations=0),
    )


def _write_blob(blob_root: Path, payload: bytes) -> BlobReference:
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    target = blob_root / digest.split(":", 1)[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
    return BlobReference(digest=digest, byte_length=len(payload))
```

Keep raw bytes out of metadata. The caller may persist only the returned digest and length. Validate every input produces a JSON-compatible sanitized structure before it reaches SQLite.

- [ ] **Step 4: Run redaction tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_storage_redaction.py -v 2>&1 | tee terminal_output/step-2-config-storage/redaction-green.log
```

Expected: all redaction tests PASS.

- [ ] **Step 5: Inspect the checkpoint**

Run:

```bash
git diff --check 2>&1 | tee terminal_output/step-2-config-storage/task-3-diff-check.log
```

Expected: exit status 0. Do not commit unless explicitly requested.

### Task 3: Establish SQLite WAL Schema And Read Interface

**Files:**
- Modify: `src/agent_evolve/core/storage.py`
- Test: `tests/test_storage_acid.py`

**Consumes:** Task 2 sanitizer and blob writer.

**Governing architecture:** `docs/architecture/storage-and-transactions.md:10-37, 39-67` and `docs/architecture/component-contracts.md:18-19`.

**Produces:** `SQLiteRunStore`, schema initialization, strict coordinator ownership, `read_record()`, and `list_records()`.

- [ ] **Step 1: Write failing initialization and schema tests**

```python
import sqlite3

from agent_evolve.core.storage import SQLiteRunStore


def test_store_configures_required_sqlite_pragmas(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path, busy_timeout_ms=1_234)
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1_234
    finally:
        connection.close()
        store.close()


def test_store_reads_no_records_before_a_transaction_commits(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path)
    try:
        assert store.list_records("candidate") == ()
        assert store.read_record("candidate", "candidate-1") is None
    finally:
        store.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_storage_acid.py::test_store_configures_required_sqlite_pragmas tests/test_storage_acid.py::test_store_reads_no_records_before_a_transaction_commits -v 2>&1 | tee terminal_output/step-2-config-storage/storage-init-red.log
```

Expected: FAIL because `SQLiteRunStore` is not implemented.

- [ ] **Step 3: Implement schema and single-writer ownership**

Initialize the store under `root` with `metadata.sqlite3` and `blobs/`. Open exactly one `sqlite3.Connection` with `check_same_thread=False`, retain its creating thread identifier, and raise `TransactionStateError` if a mutating API is called by another thread. Use one generic immutable metadata table rather than parallel hand-written tables because all persisted domain records are opaque until later phases:

```sql
CREATE TABLE IF NOT EXISTS records (
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    snapshot_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    blob_refs_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (record_type, record_id),
    UNIQUE (transaction_id, record_type, record_id)
);
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    snapshot_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('committed', 'rolled_back', 'interrupted')),
    reason TEXT,
    created_at INTEGER NOT NULL,
    committed_at INTEGER
);
CREATE TABLE IF NOT EXISTS leases (
    artifact_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    transaction_id TEXT NOT NULL
);
```

Store sanitized payload JSON using `json.dumps(..., sort_keys=True, separators=(",", ":"))`. Read APIs return `json.loads` dictionaries and must never return mutable internal state.

- [ ] **Step 4: Run initialization tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_storage_acid.py::test_store_configures_required_sqlite_pragmas tests/test_storage_acid.py::test_store_reads_no_records_before_a_transaction_commits -v 2>&1 | tee terminal_output/step-2-config-storage/storage-init-green.log
```

Expected: both tests PASS.

- [ ] **Step 5: Inspect the checkpoint**

Run:

```bash
git diff --check 2>&1 | tee terminal_output/step-2-config-storage/task-4-diff-check.log
```

Expected: exit status 0. Do not commit unless explicitly requested.

### Task 4: Atomically Publish Or Roll Back a Complete Batch

**Files:**
- Modify: `src/agent_evolve/core/storage.py`
- Test: `tests/test_storage_acid.py`

**Consumes:** `SQLiteRunStore`, `BatchRecord`, sanitizer, and blob staging from Tasks 2-3.

**Governing architecture:** `docs/architecture/storage-and-transactions.md:39-90` and `docs/architecture/orchestration-lifecycle.md:86-115`.

**Produces:** `StagedBatch`, `begin_batch()`, all-or-nothing `commit()`, explicit `rollback()`, and idempotent record publication.

- [ ] **Step 1: Write failing atomicity and idempotency tests**

```python
import pytest

from agent_evolve.core.storage import BatchRecord, SQLiteRunStore


def _record(record_type: str, record_id: str) -> BatchRecord:
    return BatchRecord(
        record_type=record_type,  # type: ignore[arg-type]
        record_id=record_id,
        payload={"id": record_id, "status": "accepted"},
        blob_payloads={"summary": f"summary:{record_id}".encode()},
    )


def test_batch_commit_publishes_all_required_record_categories(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path)
    try:
        batch = store.begin_batch("tx-1", "snapshot-2")
        for record_type in (
            "candidate", "score_cell", "attempt", "validation", "merge",
            "memory", "retry_state", "budget", "lease", "snapshot", "manifest",
        ):
            batch.stage(_record(record_type, f"{record_type}-1"))
        batch.commit()

        assert all(store.read_record(kind, f"{kind}-1") is not None for kind in (
            "candidate", "score_cell", "attempt", "validation", "merge",
            "memory", "retry_state", "budget", "lease", "snapshot", "manifest",
        ))
    finally:
        store.close()


@pytest.mark.parametrize("failure_index", range(11))
def test_statement_failure_publishes_no_partial_records(tmp_path, failure_index: int) -> None:
    store = SQLiteRunStore(tmp_path)
    store._failure_after_statement = failure_index  # test-only injection seam
    try:
        batch = store.begin_batch("tx-failure", "snapshot-2")
        batch.stage(_record("candidate", "candidate-1"))
        batch.stage(_record("attempt", "attempt-1"))

        with pytest.raises(RuntimeError, match="injected statement failure"):
            batch.commit()

        assert store.list_records("candidate") == ()
        assert store.list_records("attempt") == ()
    finally:
        store.close()


def test_replaying_an_identical_idempotency_key_does_not_duplicate_rows(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path)
    try:
        for transaction_id in ("tx-1", "tx-2"):
            batch = store.begin_batch(transaction_id, "snapshot-2")
            batch.stage(_record("candidate", "candidate-1"))
            batch.commit()

        assert len(store.list_records("candidate")) == 1
    finally:
        store.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_storage_acid.py -v 2>&1 | tee terminal_output/step-2-config-storage/storage-atomic-red.log
```

Expected: FAIL because batch staging/commit behavior is absent.

- [ ] **Step 3: Implement all-or-nothing batch commit**

`begin_batch()` must reject an empty transaction ID or snapshot version and create an in-memory `StagedBatch`; it must not write a transaction row yet. `stage()` must reject duplicate `(record_type, record_id)` entries in the same batch, invalid record type, empty record ID, or a finalized batch.

`commit()` must:

```python
def commit(self) -> None:
    self._store._assert_writer_thread()
    staged = tuple(self._records)
    blob_refs = _stage_blobs_before_metadata_transaction(staged)
    try:
        with self._store._connection:
            self._store._connection.execute("BEGIN IMMEDIATE")
            self._store._insert_transaction_pending_commit(...)
            for record, refs in zip(staged, blob_refs, strict=True):
                self._store._insert_record_idempotently(record, refs, ...)
            self._store._mark_transaction_committed(...)
    except BaseException:
        self._store._connection.rollback()
        self._finalized = True
        raise
    self._finalized = True
```

Do not use callback compensation as the persistence boundary. The database transaction is the only publication boundary. For an existing `(record_type, record_id)`, load its canonical JSON and blob references: accept only exact equality, otherwise raise `StorageError` and roll back the complete batch. Implement the test-only `_failure_after_statement` hook inside the statement execution helper so every injection point runs before one SQL mutation and causes the transaction to roll back.

- [ ] **Step 4: Run atomicity tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_storage_acid.py -v 2>&1 | tee terminal_output/step-2-config-storage/storage-atomic-green.log
```

Expected: all storage atomicity and idempotency tests PASS.

- [ ] **Step 5: Inspect the checkpoint**

Run:

```bash
git diff --check 2>&1 | tee terminal_output/step-2-config-storage/task-5-diff-check.log
```

Expected: exit status 0. Do not commit unless explicitly requested.

### Task 5: Recover Durable State And Validate Transactional-Backend Eligibility

**Files:**
- Modify: `src/agent_evolve/core/storage.py`
- Test: `tests/test_storage_recovery.py`
- Test: `tests/test_config.py`

**Consumes:** Config from Task 1 and durable transaction/record tables from Task 4.

**Governing architecture:** `docs/architecture/storage-and-transactions.md:92-146` and `docs/architecture/orchestration-lifecycle.md:118-127`.

**Produces:** `RecoveryReport`, `SQLiteRunStore.recover()`, committed-blob integrity checking, orphan cleanup, expired lease recovery, interrupted-transaction records, and `ResolvedConfig` validation that parallel execution only accepts a transactional store.

- [ ] **Step 1: Write failing recovery and eligibility tests**

```python
import pytest

from agent_evolve.core.config import resolve_config
from agent_evolve.core.errors import ConfigurationError, StorageCorruptionError
from agent_evolve.core.storage import BatchRecord, SQLiteRunStore


def test_recovery_removes_orphan_blobs_and_expired_leases(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path)
    try:
        orphan = store.blob_root / "orphan"
        orphan.write_bytes(b"orphan")
        store._connection.execute(
            "INSERT INTO leases VALUES (?, ?, ?, ?)",
            ("artifact-1", "worker-1", 10, "tx-old"),
        )
        store._connection.commit()

        report = store.recover(now_epoch_seconds=11)

        assert report.orphan_blobs_removed == 1
        assert report.expired_leases_released == 1
    finally:
        store.close()


def test_recovery_marks_interrupted_batch_without_publishing_records(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path)
    try:
        store._connection.execute(
            "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
            ("tx-interrupted", "snapshot-1", "interrupted", "crash", 1, None),
        )
        store._connection.commit()

        report = store.recover(now_epoch_seconds=2)

        assert report.interrupted_transactions == ("tx-interrupted",)
        assert store.list_records("candidate") == ()
    finally:
        store.close()


def test_committed_missing_blob_is_corruption(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path)
    try:
        batch = store.begin_batch("tx-1", "snapshot-1")
        batch.stage(BatchRecord("candidate", "candidate-1", {"id": "candidate-1"}, {"body": b"x"}))
        batch.commit()
        blob_path = next(store.blob_root.iterdir())
        blob_path.unlink()

        with pytest.raises(StorageCorruptionError, match="missing blob"):
            store.recover(now_epoch_seconds=2)
    finally:
        store.close()


def test_parallel_profile_rejects_a_non_transactional_store() -> None:
    config = resolve_config("research_parallel")

    with pytest.raises(ConfigurationError, match="transactional"):
        config.require_storage_capabilities(transactional=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_storage_recovery.py tests/test_config.py -v 2>&1 | tee terminal_output/step-2-config-storage/recovery-red.log
```

Expected: FAIL because recovery reports and transactional eligibility validation are absent.

- [ ] **Step 3: Implement recovery and storage eligibility**

Add this method to `ResolvedConfig`:

```python
def require_storage_capabilities(self, *, transactional: bool) -> None:
    if self.features.parallel_execution and not transactional:
        raise ConfigurationError(
            "research_parallel requires a transactional storage backend"
        )
```

Implement a frozen `RecoveryReport` containing `orphan_blobs_removed`, `expired_leases_released`, `interrupted_transactions`, and `derived_state_rebuilt`. `recover()` must start `BEGIN IMMEDIATE`, reject a committed record whose referenced blob digest does not exist, delete expired leases, preserve the identifier of every interrupted transaction in the report, and commit the cleanup as one metadata transaction. Rebuild only a deterministic count/index from `records`; do not create a new entropy, clustering, pool, or orchestrator cache in this phase. After commit, delete unreferenced blob files. An unreferenced blob cleanup failure may be reported but must not corrupt committed metadata.

- [ ] **Step 4: Run recovery and eligibility tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_storage_recovery.py tests/test_config.py -v 2>&1 | tee terminal_output/step-2-config-storage/recovery-green.log
```

Expected: all recovery and configuration tests PASS.

- [ ] **Step 5: Inspect the checkpoint**

Run:

```bash
git diff --check 2>&1 | tee terminal_output/step-2-config-storage/task-6-diff-check.log
```

Expected: exit status 0. Do not commit unless explicitly requested.

### Task 6: Run Phase-2 Verification And Record the Unlock Decision

**Files:**
- Modify: `docs/superpowers/specs/2026-08-12-architecture-enforcement-design.md`
- Test: `tests/test_config.py`
- Test: `tests/test_storage_acid.py`
- Test: `tests/test_storage_recovery.py`
- Test: `tests/test_storage_redaction.py`

**Consumes:** All prior Phase-2 APIs and tests.

**Governing architecture:** `docs/architecture/README.md:73-89`, `docs/architecture/storage-and-transactions.md:148-160`, and `AGENTS.md:29-34`.

**Produces:** Verification evidence and a factually bounded Phase-2 completion note. It must not claim Phase 3 or later implementation exists.

- [ ] **Step 1: Run the Phase-2 focused suite**

Run:

```bash
uv run --extra dev pytest tests/test_config.py tests/test_storage_acid.py tests/test_storage_recovery.py tests/test_storage_redaction.py -v 2>&1 | tee terminal_output/step-2-config-storage/focused-suite.log
```

Expected: all focused tests PASS.

- [ ] **Step 2: Run the complete regression suite**

Run:

```bash
uv run --extra dev pytest 2>&1 | tee terminal_output/step-2-config-storage/full-suite.log
```

Expected: all existing tests and the new Phase-2 tests PASS. If an existing prototype test fails, use systematic debugging and do not loosen a Phase-2 invariant to preserve prototype behavior.

- [ ] **Step 3: Verify prohibited imports and prohibited module edits**

Run:

```bash
uv run python -c "from pathlib import Path; files = [Path('src/agent_evolve/core/config.py'), Path('src/agent_evolve/core/storage.py')]; assert all('import cuga' not in path.read_text().lower() for path in files)" 2>&1 | tee terminal_output/step-2-config-storage/core-neutrality.log
git diff --check 2>&1 | tee terminal_output/step-2-config-storage/final-diff-check.log
```

Expected: both commands exit status 0.

- [ ] **Step 4: Add an evidence-only unlock note**

Append the following section only after Steps 1-3 succeed:

```markdown
## Phase-2 Verification Record

Phase 2 is verified only by the tee-captured focused and full-suite logs under
`terminal_output/step-2-config-storage/`. The verified foundation is limited to
validated named-profile resolution, SQLite WAL transactional metadata storage,
atomic batch rollback, idempotency, recovery, blob-reference integrity, and
recursive fail-closed redaction. This does not claim that Phase-3 evidence and
diagnosis modules, later selection/editing modules, parallel services,
orchestration, or a CUGA adapter are implemented.
```

- [ ] **Step 5: Inspect final status without committing**

Run:

```bash
git status --short 2>&1 | tee terminal_output/step-2-config-storage/final-status.log
```

Expected: only intended Phase-2 files and pre-existing user changes appear. Do not commit unless explicitly requested.

## Plan Self-Review

- Binding Step-2 coverage: config/profile resolution is Task 2; SQLite WAL, one coordinator writer, blob ordering, atomic metadata transactions, idempotency, and failure injection are Tasks 4-5; recovery and transactional eligibility are Task 6; recursive redaction is Task 3.
- Architecture-order compliance: this plan makes no production edits to Phase 3-8 modules and explicitly holds qf7’s orchestration concerns until Phase 6.
- Negative-constraint coverage: full opaque IDs, no CUGA imports, no filesystem-only parallel backend, no partial batch publication, no field-name-only redaction, and no raw sensitive payload persistence each have explicit tests or verification steps.
- Placeholder scan: no `TODO`, `TBD`, deferred implementation instruction, or unspecified test behavior remains in implementation tasks.
- Type consistency: all public config and storage types used in later tasks are declared in the Public Interfaces section and introduced by their producing task.
