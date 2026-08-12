# Critical Review: Step-2 Plan & Architecture Enforcement Design

## Context: The Agent Was Blind to `docs/architecture/`

The implementation agent (qf7) has been working from `docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md` (the older target architecture) and was **not aware** of the `docs/architecture/` folder containing the binding specifications:

- `data-contracts.md`
- `selection-algorithms.md`
- `merge-resolution.md`
- `storage-and-transactions.md`
- `component-contracts.md`
- `orchestration-lifecycle.md`
- `persistence-and-provenance.md`

The enforcement design document was created to close this gap. Below is a critical review of whether it succeeds, and whether the Step-2 plan is safe to hand to the agent.

---

## I. Architecture Enforcement Design: Assessment

### What It Gets Right

| Aspect | Verdict | Notes |
|---|---|---|
| Declares `docs/architecture/` as binding | ✅ Correct | Overrides prototype code, docstrings, previous plans |
| Establishes 8-phase hard gate | ✅ Correct | Prevents the agent from jumping ahead |
| Prohibits orchestrator/parallel/entropy/issues edits until Phase 2 passes | ✅ Correct | Directly prevents the qf7 pattern of editing everything at once |
| Requires TDD evidence before phase unlock | ✅ Correct | Prevents docstring-driven deception |
| Gives explicit example of preserving mandates over findings | ✅ Correct | The synthetic blame node example directly addresses qf7's known bug |
| States prototype code is "prototype state, not evidence of phase completion" | ✅ Correct | Prevents qf7 from claiming existing code satisfies the architecture |

### What It Gets Wrong or Leaves Ambiguous

#### 1. It does not specify HOW the agent reads `docs/architecture/`

The enforcement design says `docs/architecture/` is binding, but does not specify:
- Whether the agent must read ALL files in the folder before starting any task
- Whether the agent must cite specific sections from `docs/architecture/` in its implementation plan
- Whether the agent must cross-reference its code against the binding specs before submitting

**Risk:** The agent may read the enforcement design, acknowledge it, but still not internalize the architecture folder's detailed mandates. The enforcement design is a *meta-rule* about precedence, not a *reading instruction*.

**Recommendation:** Add an explicit instruction to the Step-2 plan:

```
MANDATORY PRE-READ: Before writing any code, read and acknowledge the following
binding documents from docs/architecture/:
1. data-contracts.md (record schemas, validation rules)
2. storage-and-transactions.md (SQLite WAL, atomicity, redaction)
3. component-contracts.md (module ownership, public boundaries)
4. orchestration-lifecycle.md (state machine, batch barrier)

For each Task in this plan, cite the specific section of docs/architecture/
that governs the implementation. If no section applies, state that explicitly.
```

#### 2. It does not define what "passing tests for binding requirements" means for Phase 1

Phase 1 is `core/contracts.py` and `core/errors.py`. The enforcement design says no production changes to Phase 2+ modules until Phase 1 passes. But it does not specify:
- What tests must exist for Phase 1
- Whether the existing `contracts.py` (which already has Pydantic models) satisfies Phase 1
- Whether Phase 1 is already considered "passed" or needs re-verification

**Risk:** The agent may skip Phase 1 entirely, assuming the existing `contracts.py` is sufficient, and jump directly to Phase 2.

**Recommendation:** Explicitly state whether Phase 1 is already satisfied or requires re-verification. If the existing `contracts.py` already has `ScoreCell`, `AttemptRecord`, `EditPlan`, `MergeProvenance` with correct validation, state that Phase 1 is provisionally passed pending test verification.

#### 3. The phase list does not mention `docs/architecture/README.md` implementation order

The architecture folder's README specifies an implementation order:
1. contracts.py and errors.py
2. config.py and storage.py
3. pool.py, analysis.py, blame.py, clustering.py
4. entropy.py, issues.py, memory.py, editor.py, evaluation.py
5. merge.py, then parallel.py
6. orchestrator.py
7. cuga_wrapper
8. adapters/cuga.py

The enforcement design's phase list matches this, which is good. But it should explicitly reference the README as the source of truth for ordering, not just restate it.

---

## II. Step-2 Config Storage Foundation Plan: Assessment

### What It Gets Right

| Aspect | Verdict | Notes |
|---|---|---|
| Limits scope to config.py and storage.py only | ✅ Correct | Aligns with Phase 2 gate |
| Prohibits edits to orchestrator, parallel, entropy, issues, adapters | ✅ Correct | Prevents scope creep |
| Requires SQLite WAL with correct pragmas | ✅ Correct | journal_mode=WAL, synchronous=FULL, foreign_keys=ON |
| Requires one coordinator-owned writer connection | ✅ Correct | Prevents worker writes |
| Requires blobs before metadata, BEGIN IMMEDIATE, all-or-nothing commit | ✅ Correct | Matches storage-and-transactions.md |
| Requires recursive fail-closed redaction | ✅ Correct | Matches redaction gateway mandate |
| Requires idempotency keys | ✅ Correct | Prevents duplicate records |
| Requires recovery on startup | ✅ Correct | Matches storage-and-transactions.md |
| Requires test-only failure injection seam | ✅ Correct | Enables atomicity testing |
| Requires parallel profile to reject non-transactional store | ✅ Correct | Matches storage-and-transactions.md |
| Uses opaque record payloads (no domain interpretation) | ✅ Correct | Storage must not import pool/editor/merge modules |
| Requires `git diff --check` before commit | ✅ Correct | Prevents whitespace errors |
| Requires tee-captured logs for all verification | ✅ Correct | Matches AGENTS.md |

### Critical Gaps

#### 1. BudgetLimits is missing two mandated fields

The architecture's budget object (Section 3.2 of the target architecture) requires:

```
GEPA_MAX_ATTEMPTS=...
GEPA_MAX_ACCEPTED_EDITS=...
GEPA_MAX_MODEL_TOKENS=...
GEPA_MAX_ROLLOUTS=...
GEPA_MAX_JUDGE_VERDICTS=...
GEPA_EDIT_MAX_RETRIES=3
GEPA_MAX_WALL_SECONDS=...
GEPA_MAX_POOL_CANDIDATES=...
GEPA_MAX_HISTORY_RECORDS=...
GEPA_MAX_RAG_CONTEXT_TOKENS=...
```

The Step-2 plan's `BudgetLimits` dataclass defines:

```python
max_model_tokens, max_rollouts, max_judge_verdicts, edit_max_retries,
max_wall_seconds, max_pool_candidates, max_history_records, max_rag_context_tokens
```

**Missing:** `max_attempts` and `max_accepted_edits`.

These are not optional. The architecture says "A deployment must expose at least" these fields. The Step-2 plan must add them.

#### 2. The plan references `docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md:206-227` instead of `docs/architecture/`

Task 2 says:

> Consumes: `ConfigurationError` from Task 1 and the profile definitions in `docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md:206-227`.

But the enforcement design says `docs/architecture/` is binding. The plan should reference `docs/architecture/data-contracts.md` and `docs/architecture/storage-and-transactions.md` instead.

**Risk:** The agent reads the older document, which has different specifications (e.g., fixed 13-category taxonomy instead of dynamic causal blame graphs, `(N+1) × k × G` initial pool rule instead of `k × G + N × k + N × k`).

#### 3. The plan does not reference `docs/architecture/data-contracts.md` for record schemas

The architecture folder's `data-contracts.md` defines exact schemas for `ScoreCell`, `AttemptRecord`, `EditPlan`, `MergeProvenance`, `ArtifactMergeDecision`, `ValidationResult`, `MemoryRecord`, `CausalFinding`, `BlameGraph`. The Step-2 plan's storage layer accepts "opaque JSON-like payloads" and does not validate against these schemas.

This is **intentionally correct for Phase 2** (storage should not interpret domain records), but the plan should explicitly state:

> Storage accepts opaque payloads in Phase 2. Typed validation of these payloads against `docs/architecture/data-contracts.md` schemas is the responsibility of the producing module (pool.py, editor.py, merge.py, etc.) in later phases. Storage must not import or reference those schemas.

Without this clarification, the agent may either:
- (a) Import the schemas into storage.py, violating the "opaque payload" principle, or
- (b) Skip schema validation entirely in later phases, assuming storage handles it.

#### 4. The redaction denylist may be incomplete

The Step-2 plan rejects string values containing case-insensitive markers for:
```
expected_answer, expected_label, expected_regex, api_key, password,
secret, token, regex, raw_prompt, raw_response, evaluator_internal
```

The architecture's `data-contracts.md` forbids:
```
raw editor payloads, raw prompts, raw model responses, raw trace bodies,
expected answers, evaluator internals, labels, regexes, and credentials
```

The Step-2 plan's denylist covers most of these, but:
- "raw trace bodies" is not explicitly covered (the plan mentions `raw_prompt` and `raw_response` but not `raw_trace`)
- "labels" as a standalone category is not covered (only `expected_label` is)
- "credentials" as a standalone category is not covered (only `api_key`, `password`, `token`, `secret` are)

**Recommendation:** Add `raw_trace`, `label`, and `credential` to the string marker list.

#### 5. The plan does not specify what happens when `GEPA_MAX_POOL_CANDIDATES` or `GEPA_MAX_HISTORY_RECORDS` is reached

The architecture says these are complexity budgets that "bound persistent pool, edit-memory, and editor-context growth across outer iterations." The Step-2 plan's `BudgetLimits` includes these fields but does not specify:
- What happens when the pool reaches `max_pool_candidates` (reject new candidates? evict oldest? evict dominated?)
- What happens when history reaches `max_history_records` (truncate? archive? reject?)

This is not a Phase 2 problem (the orchestrator handles eviction policy), but the config resolver should validate that these values are positive integers when set, and the plan should note that eviction policy is a later-phase concern.

#### 6. The plan's `ResolvedConfig.manifest_payload()` does not include enough metadata

The architecture requires that "the profile name and all resolved flags" are persisted in the experiment manifest. The plan's `manifest_payload()` returns profile, features, and budgets. But it should also include:
- The architecture version or commit hash that the config was resolved against
- Whether the config was resolved from a named profile or from explicit overrides
- The timestamp of resolution

This is a minor gap but matters for reproducibility.

#### 7. Task 6's `require_storage_capabilities` method is added to `ResolvedConfig` but not tested in Task 2

Task 6 adds a method to `ResolvedConfig`:
```python
def require_storage_capabilities(self, *, transactional: bool) -> None:
    if self.features.parallel_execution and not transactional:
        raise ConfigurationError(...)
```

But Task 2's tests do not test this method. It is only tested in Task 6. This is fine for TDD ordering, but the plan should explicitly note that this method is added in Task 6, not Task 2, to avoid confusion.

---

## III. Risk Assessment for the Agent

| Risk | Severity | Mitigation |
|---|---|---|
| Agent reads older `docs/rho_evolution/` doc instead of `docs/architecture/` | **HIGH** | Add mandatory pre-read instruction citing architecture folder |
| Agent skips Phase 1, assuming existing contracts.py is sufficient | **MEDIUM** | Explicitly state Phase 1 status |
| Agent adds domain schema imports to storage.py | **MEDIUM** | State explicitly that storage accepts opaque payloads |
| Agent misses `max_attempts` and `max_accepted_edits` in BudgetLimits | **HIGH** | Add these fields to the plan |
| Agent uses incomplete redaction denylist | **MEDIUM** | Add missing markers |
| Agent edits orchestrator.py or other prohibited modules | **LOW** | Enforcement design already prohibits this |
| Agent claims Phase 2 is complete without running all tests | **LOW** | Plan requires tee-captured logs |

---

## IV. Verdict

### Architecture Enforcement Design: ✅ APPROVED with one addition

The enforcement design is sound and correctly establishes `docs/architecture/` as binding. The phase gate is correct. The TDD requirement is correct. The prohibition on editing later-phase modules is correct.

**Add:** An explicit instruction that the agent must read and cite `docs/architecture/` documents before implementing any task.

### Step-2 Config Storage Foundation Plan: ⚠️ APPROVED with 4 required fixes

The plan is well-structured and mostly correct. But before handing it to the agent:

1. **Add `max_attempts` and `max_accepted_edits` to `BudgetLimits`**
2. **Change all references from `docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md` to `docs/architecture/data-contracts.md` and `docs/architecture/storage-and-transactions.md`**
3. **Add `raw_trace`, `label`, and `credential` to the redaction string marker list**
4. **Add a mandatory pre-read instruction requiring the agent to read `docs/architecture/data-contracts.md`, `docs/architecture/storage-and-transactions.md`, `docs/architecture/component-contracts.md`, and `docs/architecture/orchestration-lifecycle.md` before writing any code**

After these fixes, the plan is safe to hand to the agent.