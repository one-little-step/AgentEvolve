# Honest Assessment: Doc 18/19 Are Architecture, Not Implementation Design

You are correct. Documents 18 and 19 are **not specific enough to code from reliably**. They are architecture-level specifications — they define *what* the system does and *why*, but they do not define *how* each module works internally. The gap between these documents and working code is where the `dev1` branch already went wrong: an LLM coding agent read the architecture, made up the internal implementation, and produced the 10 high-priority flaws listed in `f1.md`.

Here is exactly what is missing and what a module-level design document must contain.

---

## I. What Doc 18/19 Specify vs. What They Do Not

| Aspect | Doc 18/19 Coverage | Missing for Implementation |
|---|---|---|
| System topology | ✅ Flowchart, execution hierarchy | ❌ Module dependency graph, call sequence |
| Data contracts | ✅ JSON schemas for verdict, score cell, artifact descriptor | ❌ Python class definitions, validation rules, serialization format |
| Feature gates | ✅ Variable names listed | ❌ Resolution order, defaults, validation rules, conflict handling |
| Pareto selection | ✅ Dominance rules in prose | ❌ Function signatures, data structures, edge cases |
| DPP selection | ✅ Kernel formula | ❌ Sampling algorithm choice, fallback for singular matrices, complexity bounds |
| Merge/crossover | ✅ Rules in prose | ❌ Pseudocode, edge cases (deleted artifacts, missing ancestor), provenance format |
| Batch/locking | ✅ Conceptual sequence diagram | ❌ Concurrency primitive, failure detection, rollback protocol, lease representation |
| Storage | ❌ Not specified | ❌ Directory structure, file formats, manifest schema, versioning |
| Error handling | ❌ Not specified | ❌ Malformed LLM output, rollout timeout, embedding unavailability |
| Adapter contract | ✅ 7 method names listed | ❌ Python signatures, return types, exception contracts |
| Test contract | ❌ Not specified | ❌ Required unit tests, integration tests, mock strategy |

**The core problem:** An LLM coding agent reading Doc 18 will *invent* all of the missing details. That is exactly what happened in `dev1`. The `pool.py` author invented `task_id[0]` as an aggregation key because the document never specified the aggregation key. The `entropy.py` author inverted the DPP kernel because the document gave the formula but not the algorithm. The `parallel.py` author skipped rollback because the document described the happy path but not the failure path.

---

## II. What a Module-Level Design Must Contain

You need a **Document 20: Module-Level Implementation Specification**. Below is the exact structure it must follow, module by module.

### 20.1 — Dependency Graph and Initialization Order

```
contracts.py          (no dependencies — pure data types)
    ↑
config.py             (depends on: contracts)
    ↑
storage.py            (depends on: contracts, config)
    ↑
pool.py               (depends on: contracts, config, storage)
    ↑
clustering.py         (depends on: contracts, config)
    ↑
entropy.py            (depends on: contracts, config, pool, clustering)
    ↑
issues.py             (depends on: contracts, config, pool, entropy)
    ↑
blame.py              (depends on: contracts)
    ↑
judging.py            (depends on: contracts, blame, clustering)
    ↑
analysis.py           (depends on: contracts)
    ↑
memory.py             (depends on: contracts, config, storage)
    ↑
editor.py             (depends on: contracts, config, memory)
    ↑
evaluation.py         (depends on: contracts, config, pool)
    ↑
merge.py              (depends on: contracts, config, pool)
    ↑
parallel.py           (depends on: contracts, config, storage)
    ↑
orchestrator.py       (depends on: ALL of the above)
```

**Why this matters:** In `dev1`, the orchestrator imported and called modules in an ad-hoc order. The dependency graph must be explicit so that circular dependencies are caught at design time, not runtime.

### 20.2 — Every Module's Public Interface

For each module, specify **every public function** with:
- Exact Python signature
- Input types (Pydantic models or primitives)
- Output type
- Exceptions raised
- Side effects (what does it write to storage?)
- Preconditions (what must be true before calling?)
- Postconditions (what is guaranteed after calling?)

**Example for `pool.py`:**

```python
class ParetoPool:
    """Persistent candidate pool with causal-blame Pareto selection."""
    
    def __init__(self, config: GEPAConfig, storage: StorageBackend):
        """Initialize pool from storage. Loads all admitted candidates."""
        
    def admit(self, candidate: PoolCandidate, score_tensor: ScoreTensor) -> None:
        """Add a candidate to the pool.
        
        Preconditions:
            - candidate.candidate_id is unique within this iteration
            - score_tensor has at least one cell with rollout_count >= 1
            - candidate.artifact_hashes covers all writable artifacts
            
        Postconditions:
            - candidate is persisted to storage
            - score_tensor is persisted and linked via score_tensor_ref
            - pool size is checked against GEPA_MAX_POOL_CANDIDATES
            
        Raises:
            DuplicateCandidateError: if candidate_id already exists
            PoolCapacityError: if pool is at max capacity
            SchemaValidationError: if score_tensor fails validation
        """
        
    def pareto_frontier(self, task_id: str | None = None) -> list[ParetoEntry]:
        """Return non-dominated candidates.
        
        Args:
            task_id: If provided, restrict to this task. If None, all tasks.
            
        Returns:
            List of (candidate_id, task_id, mechanism_cluster_id, is_winner) tuples.
            
        Dominance rule (from Doc 18 §6.2):
            a dominates b on task t iff:
            - both have comparable score provenance for applicable mechanisms
            - a >= b on every comparable severity-weighted mechanism
            - a > b on at least one
            - no protected floor regression
            Missing mechanisms are EXCLUDED, not treated as zero.
        """
        
    def sample_parents(self, k: int, rng: Random) -> list[str]:
        """Sample k parent candidate IDs using weighted objective coverage.
        
        Weight formula (from Doc 18 §6.3):
            frequency(c) = sum over (t,m):
                severity(t,m) * confidence(t,m) * 1[c wins (t,m)]
                
        Returns candidate_ids, not full candidates.
        """
        
    def mean_score_per_task(self, task_id: str) -> dict[str, float]:
        """Compute mean score per candidate for a specific task.
        
        CRITICAL: Aggregation key is the FULL task_id string,
        not a prefix, substring, or hash.
        
        Returns: {candidate_id: weighted_mean_score}
        """
        
    def select_champion(self, config: ChampionConfig) -> ChampionResult:
        """Select final champion after budget exhaustion.
        
        Aggregate: alpha*Outcome + beta*Process + gamma*Stability - delta*Regression
        Defaults: alpha=0.55, beta=0.20, gamma=0.15, delta=0.10
        
        Protected floors OVERRIDE aggregate. A candidate violating
        any protected floor is disqualified regardless of score.
        
        Returns: ChampionResult with full breakdown for manifest.
        """
```

**This level of detail is required for every function in every module.** Without it, the coding agent will invent the semantics.

### 20.3 — State Machine for the Orchestrator

The orchestrator is not a linear pipeline. It is a state machine with branching, retries, and failure paths. Document 18's flowchart shows the happy path. The module-level design must specify:

```
States:
    INIT
    CORESET_SELECTED
    INITIAL_POOL_EVALUATED
    GEPA_LOOP_ACTIVE
    GEPA_BUDGET_EXHAUSTED
    CHAMPION_SELECTED
    COMPLETED
    FAILED

Transitions:
    INIT -> CORESET_SELECTED
        when: DPP coreset selection succeeds
        on_failure: FAILED (log reason)
        
    CORESET_SELECTED -> INITIAL_POOL_EVALUATED
        when: base G rollouts + N candidate rollouts complete
        on_partial_failure: retry failed rollouts up to 2x, then mark
            incomplete and proceed with available evidence
            
    INITIAL_POOL_EVALUATED -> GEPA_LOOP_ACTIVE
        when: all candidates have score tensors
        
    GEPA_LOOP_ACTIVE -> GEPA_LOOP_ACTIVE
        when: attempt completed (accepted or rejected)
        guard: budget not exhausted
        
    GEPA_LOOP_ACTIVE -> GEPA_BUDGET_EXHAUSTED
        when: any hard budget limit reached
        OR: no-improvement stall threshold reached
        
    GEPA_BUDGET_EXHAUSTED -> CHAMPION_SELECTED
        when: champion selection completes
        
    CHAMPION_SELECTED -> COMPLETED
        when: manifest written and validated
```

**Why this matters:** In `dev1`, the orchestrator had no explicit state machine. It ran a sequence of function calls and hoped nothing failed. When something did fail, there was no recovery path.

### 20.4 — Error Handling Contract

Every LLM call can fail. Every rollout can timeout. Every embedding call can return garbage. The design must specify:

```python
class LLMCallError(Exception):
    """Base for all LLM-related failures."""
    
class MalformedVerdictError(LLMCallError):
    """Judge returned invalid JSON or missing required fields."""
    # Policy: retry once with same input. If second failure,
    # mark verdict as 'judge_parse_failure' and skip this
    # evidence unit. Do NOT fabricate a verdict.
    
class RolloutTimeoutError(Exception):
    """Agent rollout exceeded wall-clock limit."""
    # Policy: mark task as 'rollout_timeout' for this candidate.
    # Do NOT mark as failure. Exclude from score tensor.
    # Retry once if budget allows.
    
class EmbeddingUnavailableError(Exception):
    """Embedding model not reachable or returned wrong dimensions."""
    # Policy: fall back to lexical similarity for DPP.
    # Log 'embedding_fallback' in manifest.
    
class WriteLeaseConflictError(Exception):
    """Two workers attempted to write the same artifact."""
    # Policy: this should NEVER happen if lease management is correct.
    # If it does, abort the batch, log a critical error, and
    # roll back all uncommitted changes.
```

**Why this matters:** In `dev1`, none of these error paths existed. The `floors_violated()` function had a `pass` statement because nobody specified what it should do when a floor is violated.

### 20.5 — Storage Layout

The documents never specify where data lives. This must be explicit:

```
runs/
  {experiment_id}/
    manifest.json                    # run metadata, config, resolved flags
    coreset.json                     # DPP-selected task IDs + fingerprints
    pool/
      {candidate_id}/
        candidate.json               # PoolCandidate record
        score_tensor.json            # full score tensor with provenance
        artifacts/                   # snapshot of all artifact files
          wisdom/reAct.md
          skills/...
          memory/...
    attempts/
      {attempt_id}.json              # full attempt record (Doc 18 §8.2)
    verdicts/
      {verdict_id}.json              # causal-blame verdict (Doc 18 §4.3)
    merges/
      {merge_id}.json                # merge provenance record
    history/
      edit_memory.json               # worked/failed/retry sets
    clusters/
      {iteration}/
        centroids.json               # mechanism cluster centroids
        assignments.json             # mechanism -> cluster mapping
    probes/
      {probe_id}.json                # generalization probe results
    logs/
      budget.json                    # budget consumption tracker
      errors.jsonl                   # append-only error log
```

**Why this matters:** Without this, every module will invent its own storage format. In `dev1`, the memory module stored raw editor payloads because nobody specified what should and should not be persisted.

### 20.6 — Adapter Contract (Python Signatures)

Doc 18 lists 7 method names. The module-level design must specify exact signatures:

```python
class AdapterProtocol(Protocol):
    """Every adapter must implement this protocol."""
    
    @property
    def adapter_name(self) -> str:
        """Human-readable adapter identifier, e.g. 'gaia', 'cuga'."""
        
    @property
    def supports_counterfactual_replay(self) -> bool:
        """True only if adapter has verified checkpoint/replay capability."""
        
    def artifact_inventory(self, candidate_version: str) -> list[ArtifactDescriptor]:
        """Return all artifacts for this candidate version.
        
        Must be deterministic: same version -> same inventory.
        Must include version_hash for every artifact.
        
        Raises:
            CandidateNotFoundError: if version does not exist
        """
        
    def read_artifacts(self, candidate_version: str, artifact_ids: list[str]) -> dict[str, str]:
        """Read content of specified artifacts.
        
        Returns: {artifact_id: content_string}
        
        Raises:
            ArtifactNotFoundError: if any requested artifact is missing
            ReadPermissionError: if artifact is not readable
        """
        
    def materialize_candidate(self, parent_version: str, attempt_id: str) -> str:
        """Create a mutable workspace copy for editing.
        
        Returns: workspace_id (string)
        
        The workspace is isolated. Edits in this workspace do NOT
        affect the parent or any other workspace.
        """
        
    def apply_structured_edit(self, workspace_id: str, edit_plan: EditPlan) -> ApplyResult:
        """Apply edits to the workspace.
        
        Validates:
            - every edit target is in the workspace
            - every edit target is in the declared write_set
            - operation is valid for the artifact format
            
        Raises:
            WritePermissionError: if edit targets artifact outside write_set
            InvalidOperationError: if operation is not valid for format
        """
        
    def run_full_rollout(self, workspace_id: str, task: TaskSpec, rollout_id: str) -> RolloutResult:
        """Execute a full agent rollout.
        
        Returns: RolloutResult with trajectory, final_output, wall_clock, status
        
        Raises:
            RolloutTimeoutError: if wall-clock limit exceeded
        """
        
    def capture_trace(self, rollout_result: RolloutResult) -> TraceRecord:
        """Extract structured trace from rollout result."""
        
    def evaluate_trace(self, trace: TraceRecord, task_contract: TaskContract) -> EvalResult:
        """Evaluate trace against task contract."""
        
    # Optional — only if supports_counterfactual_replay is True
    
    def discover_checkpoints(self, trace: TraceRecord) -> list[CheckpointDescriptor]:
        """Find valid replay boundaries in a completed trace."""
        
    def replay_from_checkpoint(self, checkpoint: CheckpointDescriptor, 
                                updated_artifacts: dict[str, str]) -> RolloutResult:
        """Resume execution from checkpoint with modified artifacts."""
```

### 20.7 — Test Contract

Every module must specify its required tests:

```
pool.py tests:
    test_admit_unique_candidate
    test_admit_duplicate_raises
    test_pareto_dominance_basic
    test_pareto_dominance_missing_mechanism_excluded
    test_pareto_dominance_protected_floor_blocks
    test_mean_score_per_task_uses_full_task_id    ← prevents the dev1 bug
    test_sample_parents_weighted
    test_champion_protected_floor_disqualifies

entropy.py tests:
    test_dpp_penalizes_similarity                ← prevents the dev1 bug
    test_dpp_quality_weighting
    test_entropy_floor_insufficient_candidates
    test_entropy_floor_insufficient_rollouts
    test_entropy_floored_multiplication
    test_entropy_two_tier_classification

editor.py tests:
    test_floors_violated_detects_safety_violation  ← prevents the dev1 bug
    test_floors_violated_detects_privacy_violation
    test_write_set_enforcement                     ← prevents the dev1 bug
    test_sanitization_rejects_nested_sensitive     ← prevents the dev1 bug

parallel.py tests:
    test_commit_barrier_all_or_nothing             ← prevents the dev1 bug
    test_commit_barrier_rollback_on_failure
    test_write_lease_conflict_detected
    test_workers_cannot_write_shared_state

merge.py tests:
    test_disjoint_artifact_merge_deterministic
    test_same_artifact_conflict_evidence_resolution
    test_same_artifact_conflict_llm_fallback
    test_merge_provenance_recorded
    test_merge_eligibility_no_direct_ancestry
```

---

## III. Recommended Document Structure

Write **Document 20** with these sections:

```
20-rho-parallel-gepa-module-design.md

20.0  Purpose and Relationship to Doc 18/19
20.1  Module Dependency Graph
20.2  contracts.py — Data Types and Validation
20.3  config.py — Configuration Resolution
20.4  storage.py — Persistence Layer
20.5  pool.py — Persistent Pool and Pareto Selection
20.6  clustering.py — Mechanism Alignment
20.7  entropy.py — Cross-Candidate Entropy and DPP
20.8  issues.py — Issue Construction and Selection
20.9  blame.py — Causal Blame Graph
20.10 judging.py — Analyzer+Judge Contract
20.11 analysis.py — RHO Analyzer
20.12 memory.py — Structured Edit Memory
20.13 editor.py — Mutator Protocol
20.14 evaluation.py — Focused Validation
20.15 merge.py — Deterministic Crossover
20.16 parallel.py — Locking, Snapshots, Batch Barrier
20.17 orchestrator.py — Lifecycle State Machine
20.18 adapters/base_adapter.py — Adapter Protocol
20.19 adapters/gaia_adapter.py — Gaia Mapping
20.20 Error Handling Contract
20.21 Storage Layout
20.22 Test Contract
20.23 Manifest Schema
```

Each section must contain:
1. **Purpose** (one paragraph)
2. **Public interface** (every function signature)
3. **Internal state** (what data does this module own?)
4. **Dependencies** (what does it import? what calls it?)
5. **Error handling** (what can go wrong? what is the policy?)
6. **Edge cases** (explicit list)
7. **Required tests** (named test cases)

---

## IV. The Honest Bottom Line

| Document | Level | Sufficient for Coding? |
|---|---|---|
| Doc 18 (Target Architecture) | System architecture | ❌ No — specifies *what* and *why*, not *how* |
| Doc 19 (Research Hypotheses) | Experimental design | ❌ No — specifies *what to measure*, not *how to build* |
| **Doc 20 (Module Design)** | **Implementation specification** | **Must be written before any code** |

The `dev1` branch is proof of what happens when you skip Doc 20. The coding agent had the right architecture but invented every internal detail, producing 10 high-priority flaws. Writing Doc 20 first will take 2-3 days of focused design work. It will save weeks of debugging and rewriting.

**Do not write another line of implementation code until Doc 20 exists and has been reviewed.**