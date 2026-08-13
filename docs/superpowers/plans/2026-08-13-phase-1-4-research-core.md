# Phase 1-4 Research Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove the agent-neutral research core for B0-B4 experiments using deterministic fake-adapter evidence, JSON persistence, and the mandated pool, entropy, DPP, memory, editor, and evaluation algorithms.

**Architecture:** Phase 2 uses the explicitly approved single-threaded `JSONFileStorage` research exception with recursive fail-closed redaction and immutable resolved configuration. Phase 3 moves evidence, causal diagnosis, task-local clustering, and provenance-aware persistent-pool behavior to focused core services. Phase 4 implements entropy, hierarchical DPP issue selection, memory, editor authorization, and validation in dependency order. CUGA, merge, parallel execution, and orchestration integration are excluded.

**Tech Stack:** Python 3.12, standard-library `dataclasses`, `json`, `os`, `pathlib`, `random`, `time`, Pydantic 2, pytest 8, NumPy 2; deterministic `examples.fake_adapter.FakeAdapter`.

## Global Constraints

- Cite and follow `docs/architecture/README.md:73-89`, `data-contracts.md`, `component-contracts.md`, `selection-algorithms.md`, and `target-rho-parallel-gepa.md` for every task.
- The approved JSON storage exception applies only to single-threaded fake-adapter Phase 1-4 runs with `parallel_execution=False`; Phase 5 requires SQLite WAL or equivalent transactional storage.
- `src/agent_evolve/core/` must never import CUGA, Gaia, or a concrete adapter implementation.
- Use only opaque, complete identifier strings as aggregation keys; never truncate, slice, prefix, or hash IDs for grouping.
- Every persistence write passes through one recursive fail-closed redaction gateway. Never persist credentials, expected answers, evaluator internals, labels, regexes, raw prompts/responses, or unapproved trace bodies.
- The `.env` values `OLLAMA_EMBEDDING_URL` and `OLLAMA_EMBEDDING_MODEL` are endpoint/model configuration. Phase 1-4 tests make no HTTP call and use recorded lexical fallback.
- Tests precede implementation. Every test, smoke run, and verification command is captured as `command 2>&1 | tee terminal_output/<topic>/<name>.log`.
- Do not modify `orchestrator.py`, `parallel.py`, `merge.py`, `cuga_wrapper/`, or CUGA adapter code in this plan.
- Do not commit unless the user explicitly requests a commit.

## Contract Simplification (Research Exception)

This plan uses standard Pydantic construction and `ValidationError` for
persisted-model validation instead of the typed construction factories proposed
by the earlier Phase-1 contract-completion design. This is a deliberate
research-path simplification: `Field(min_length=1)` and `model_validator`
checks provide sufficient construction validation for deterministic research
runs. Typed `PersistenceSafetyError`, `BudgetExceededError`, and
`WriteAuthorizationError` remain mandatory for their operational boundaries.

This exception does not waive immutable nested state, complete-ID aggregation,
provenance-bearing score cells, or explicit unavailable coverage states.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/agent_evolve/core/errors.py` | Typed validation, persistence, budget, and authorization errors. |
| `src/agent_evolve/core/contracts.py` | Immutable, adapter-neutral validated records. |
| `src/agent_evolve/core/config.py` | Named profile resolution, feature gates, budgets, embedding configuration. |
| `src/agent_evolve/core/storage.py` | JSON record store and recursive persistence gateway. |
| `src/agent_evolve/core/analysis.py` | Sanitized analyzer/judge exchange records and protocol. |
| `src/agent_evolve/core/blame.py` | Trace-backed causal findings and graph validation. |
| `src/agent_evolve/core/clustering.py` | Task-local mechanism clustering and lexical fallback observability. |
| `src/agent_evolve/core/pool.py` | Candidate registry, comparability, Pareto, parent, and champion selection. |
| `src/agent_evolve/core/entropy.py` | Incremental comparable-evidence entropy statistics only. |
| `src/agent_evolve/core/issues.py` | Work-item construction, constraints, selection reports, hierarchical DPP. |
| `src/agent_evolve/core/memory.py` | Redacted append-only attempt memory and retry state. |
| `src/agent_evolve/core/editor.py` | Editor request/response boundary and bounded repair protocol. |
| `src/agent_evolve/core/evaluation.py` | Four-category validation planning and acceptance decision. |
| `tests/test_*.py` | Focused behavioral and rejection evidence for each module. |

### Task 1: Close Phase-1 Contract Gate

**Files:**
- Modify: `src/agent_evolve/core/errors.py`
- Modify: `src/agent_evolve/core/contracts.py`
- Modify: `tests/test_contracts_validation.py`
- Modify: `tests/test_contracts_immutability.py`

**Governing contract:** `docs/architecture/data-contracts.md:5-29, 31-230`; `docs/architecture/component-contracts.md:33-100`.

**Consumes:** Existing Pydantic persisted models and runtime adapter dataclasses.

**Produces:** Validated immutable boundary records that downstream Phase 2-4 modules consume without creating duplicate schemas.

- [ ] **Step 1: Write failing tests for strict IDs, content hashes, and unavailable evidence**

```python
def test_score_cell_rejects_blank_candidate_id() -> None:
    with pytest.raises(ValidationError, match="candidate_id"):
        ScoreCell(**score_cell_values(candidate_id=""))

def test_execution_trace_preserves_exact_candidate_and_task_ids() -> None:
    trace = ExecutionTrace(
        trace_id="trace-1", candidate_id="candidate-alpha", task_id="task-alpha",
        events=(), final_output="", status="completed",
    )
    assert (trace.candidate_id, trace.task_id) == ("candidate-alpha", "task-alpha")
```

- [ ] **Step 2: Run Phase-1 focused tests and confirm the new assertions fail**

Run: `pytest tests/test_contracts_validation.py tests/test_contracts_immutability.py -q 2>&1 | tee terminal_output/phase_1/contracts_before.log`

Expected: New rejection or immutability tests fail before implementation changes.

- [ ] **Step 3: Add only missing construction validation and typed error classes**

```python
class PersistenceSafetyError(EvolutionContractError):
    """A value cannot be safely persisted after recursive redaction."""

class BudgetExceededError(EvolutionContractError):
    """A requested operation exceeds a resolved experiment budget."""

class WriteAuthorizationError(EvolutionContractError):
    """An edit targets an artifact outside its explicit authorization."""
```

Use Pydantic `Field(min_length=1)` and `model_validator` checks for every persisted ID/relation required by `data-contracts.md`. Do not alter ordinary Pydantic coercion accepted by existing tests.

- [ ] **Step 4: Run Phase-1 focused verification**

Run: `pytest tests/test_contracts_validation.py tests/test_contracts_immutability.py -q 2>&1 | tee terminal_output/phase_1/contracts_after.log`

Expected: PASS.

### Task 2: Add Resolved Research Configuration And Budgets

**Files:**
- Create: `src/agent_evolve/core/config.py`
- Create: `tests/test_config.py`
- Modify: `src/agent_evolve/core/__init__.py`

**Governing contract:** `docs/architecture/component-contracts.md:14-31`; `docs/architecture/target-rho-parallel-gepa.md:168-182`; `docs/superpowers/specs/2026-08-12-phase-1-4-research-core-design.md:37-91`.

**Consumes:** Environment mapping, profile name, optional overrides.

**Produces:** `FeatureGates`, `BudgetLimits`, `BudgetUsage`, `EmbeddingConfig`, `ResolvedConfig`, and `resolve_profile()`.

- [ ] **Step 1: Write failing profile and budget tests**

```python
def test_research_parallel_is_resolved_but_inactive_for_json_storage() -> None:
    config = resolve_profile("research_parallel", environ={})
    assert config.features.parallel_execution is False
    assert config.deferred_features == ("parallel_execution",)

def test_budget_refuses_operation_above_limit() -> None:
    limits = BudgetLimits(max_rollouts=1)
    usage = BudgetUsage(rollouts=1)
    with pytest.raises(BudgetExceededError):
        usage.reserve(limits, rollouts=1)

def test_embedding_config_reads_ollama_values_without_network_call() -> None:
    config = resolve_profile("minimal", environ={
        "OLLAMA_EMBEDDING_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "embeddinggemma",
    })
    assert config.embedding.url == "http://localhost:11434"
    assert config.embedding.model == "embeddinggemma"
```

- [ ] **Step 2: Run config tests and confirm failure**

Run: `pytest tests/test_config.py -q 2>&1 | tee terminal_output/phase_2/config_before.log`

Expected: FAIL because `core.config` does not exist.

- [ ] **Step 3: Implement immutable configuration records and resolver**

```python
@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_attempts: int | None = None
    max_accepted_edits: int | None = None
    max_model_tokens: int | None = None
    max_rollouts: int | None = None
    max_judge_verdicts: int | None = None
    edit_max_retries: int = 3
    max_wall_seconds: float | None = None
    max_pool_candidates: int | None = None
    max_history_records: int | None = None
    max_rag_context_tokens: int | None = None

@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    profile_name: Literal["minimal", "research_sequential", "research_parallel", "full_ablation"]
    features: FeatureGates
    budgets: BudgetLimits
    embedding: EmbeddingConfig
    dpp_max_items: int = 100
    dpp_theta: float = 0.7
    dpp_score_floor: float = 0.1
    dpp_min_gain: float = 1e-12
    entropy_refresh_mode: Literal["outer_iteration", "accepted_edits", "pool_growth"] = "outer_iteration"
    entropy_score_floor: float = 0.15
    entropy_recombination_score_threshold: float = 0.30
    entropy_frontier_weight: float = 0.30
    entropy_min_comparable_candidates: int = 3
    entropy_min_rollouts_per_candidate: int = 2
    cluster_similarity_threshold: float = 0.80
    max_clusters_per_task: int = 12
    generalization_probe_mode: Literal["deferred", "enabled"] = "deferred"
    probe_budget_fraction: float = 0.15
    champion_alpha: float = 0.55
    champion_beta: float = 0.20
    champion_gamma: float = 0.15
    champion_delta: float = 0.10
```

```python
@dataclass(slots=True)
class BudgetUsage:
    rollouts: int = 0
    analyzer_judge_calls: int = 0
    editor_calls: int = 0
    validation_calls: int = 0
    embedding_calls: int = 0
    model_tokens: int = 0
    attempts: int = 0
    accepted_edits: int = 0

    def reserve(self, limits: BudgetLimits, **increments: int) -> None:
        limit_fields = {
            "rollouts": "max_rollouts",
            "analyzer_judge_calls": "max_judge_verdicts",
            "model_tokens": "max_model_tokens",
            "attempts": "max_attempts",
            "accepted_edits": "max_accepted_edits",
        }
        for field, increment in increments.items():
            if increment < 0:
                raise ValueError("budget increments must be non-negative")
            limit = getattr(limits, limit_fields.get(field, ""), None)
            if limit is not None and getattr(self, field) + increment > limit:
                raise BudgetExceededError(f"{field} budget exceeded")
        for field, increment in increments.items():
            setattr(self, field, getattr(self, field) + increment)
```

Include every default in the approved design, validate ranges and champion
weights, expose a JSON-safe `manifest_payload()`, and never contact Ollama from
the resolver. Phase 1-4 activates only `outer_iteration`; the other refresh
modes are validated configuration for later lifecycle work.

- [ ] **Step 4: Run configuration verification**

Run: `pytest tests/test_config.py -q 2>&1 | tee terminal_output/phase_2/config_after.log`

Expected: PASS.

### Task 3: Build Recursive Redaction And JSON Research Storage

**Files:**
- Create: `src/agent_evolve/core/storage.py`
- Create: `tests/test_storage.py`

**Governing contract:** `docs/architecture/storage-and-transactions.md:118-133`; approved exception in `docs/superpowers/specs/2026-08-12-architecture-enforcement-design.md`; `docs/architecture/data-contracts.md:214-231`.

**Consumes:** JSON-safe mappings, `ResolvedConfig.features.parallel_execution`.

**Produces:** `StorageBackend`, `JSONFileStorage`, `sanitize_for_persistence()`, and `RedactedValue`.

- [ ] **Step 1: Write failing storage and redaction tests**

```python
def test_storage_writes_and_reads_one_redacted_record(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    store.write_record("attempts", "attempt-1", {"summary": "safe"})
    assert store.read_record("attempts", "attempt-1") == {"summary": "safe"}

def test_storage_rejects_nested_expected_answer(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    with pytest.raises(PersistenceSafetyError):
        store.write_record("attempts", "attempt-1", {"nested": {"expected_answer": "x"}})

def test_json_storage_rejects_active_parallel_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parallel_execution"):
        JSONFileStorage(tmp_path, parallel_execution=True)
```

- [ ] **Step 2: Run storage tests and confirm failure**

Run: `pytest tests/test_storage.py -q 2>&1 | tee terminal_output/phase_2/storage_before.log`

Expected: FAIL because `core.storage` does not exist.

- [ ] **Step 3: Implement path-contained atomic record storage**

```python
class StorageBackend(Protocol):
    def write_record(self, record_type: str, record_id: str, payload: Mapping[str, object]) -> RedactedValue: ...
    def read_record(self, record_type: str, record_id: str) -> Mapping[str, object] | None: ...
    def list_records(self, record_type: str) -> tuple[Mapping[str, object], ...]: ...
    def close(self) -> None: ...

def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("record type and ID must be safe opaque path components")
    return value
```

Serialize with `json.dumps(..., sort_keys=True, separators=(",", ":"))`, write to `path.with_suffix(".tmp")`, then `Path.replace(path)`. Implement recursive mapping/sequence/string inspection and reject prohibited field/category matches. Do not create a transaction API.

This JSON implementation deliberately defers SQLite WAL pragmas, atomic
multi-record barriers, idempotency keys, interrupted-run recovery,
content-addressed blob staging, orphan cleanup, and durable leases to Phase 5.
Per-record temp-write-then-rename atomicity is the complete persistence
guarantee for the single-threaded research path.

- [ ] **Step 4: Run storage verification**

Run: `pytest tests/test_storage.py -q 2>&1 | tee terminal_output/phase_2/storage_after.log`

Expected: PASS, including path traversal, nested sensitive content, list ordering, and atomic replacement cases.

### Task 4: Define Analysis And Trace-Backed Causal Findings

**Files:**
- Create: `src/agent_evolve/core/analysis.py`
- Modify: `src/agent_evolve/core/blame.py`
- Create: `tests/test_analysis.py`
- Modify: `tests/test_blame.py`

**Governing contract:** `docs/architecture/data-contracts.md:81-104`; `docs/architecture/component-contracts.md:45-74`; `docs/architecture/target-rho-parallel-gepa.md:82-105`.

**Consumes:** `EvolutionTask`, `ExecutionTrace`, artifact inventory IDs.

**Produces:** Analyzer protocol, bounded group-report exchange record, and trace-backed `CausalFinding` statuses.

- [ ] **Step 1: Write failing tests for status and trace attribution**

```python
def test_observed_finding_requires_trace_backed_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence_refs"):
        CausalFinding(status="observed", mechanism_description="bad retrieval", evidence_refs=())

def test_insufficient_evidence_is_not_coerced_to_blame() -> None:
    finding = CausalFinding(status="insufficient_evidence", rationale="trace lacks causal link")
    assert finding.mechanism_cluster_id is None
```

- [ ] **Step 2: Run analysis/blame tests and confirm failure**

Run: `pytest tests/test_analysis.py tests/test_blame.py -q 2>&1 | tee terminal_output/phase_3/analysis_before.log`

Expected: FAIL for missing exchange records or insufficient status validation.

- [ ] **Step 3: Implement neutral analyzer boundary without artifact mutation**

```python
class AnalyzerJudge(Protocol):
    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]: ...

@dataclass(frozen=True, slots=True)
class RolloutGroupReport:
    candidate_id: str
    task_id: str
    trace_refs: tuple[str, ...]
    rollout_ids: tuple[str, ...]
    sanitized_evidence: tuple[Mapping[str, object], ...]
```

Reject graph nodes lacking matching trace evidence. Preserve `observed`, `uncertain`, `insufficient_evidence`, and `malformed`; never manufacture a graph node to continue an execution path.

- [ ] **Step 4: Run analysis/blame verification**

Run: `pytest tests/test_analysis.py tests/test_blame.py -q 2>&1 | tee terminal_output/phase_3/analysis_after.log`

Expected: PASS.

### Task 5: Complete Task-Local Clustering With Recorded Fallback

**Files:**
- Modify: `src/agent_evolve/core/clustering.py`
- Modify: `tests/test_clustering.py`

**Governing contract:** `docs/architecture/target-rho-parallel-gepa.md:108-132`; `docs/architecture/selection-algorithms.md:17-65`.

**Consumes:** Observed causal findings and `ResolvedConfig` clustering/embedding settings.

**Produces:** Stable task-local assignments, freshness, barrier refresh records, and explicit lexical fallback metadata.

- [ ] **Step 1: Write failing tests for task locality and fallback observability**

```python
def test_same_mechanism_text_in_two_tasks_never_shares_cluster() -> None:
    registry = ClusterRegistry(embedder_factory=LexicalEmbedder)
    assert registry.assign("task-a", finding("stale schema")).cluster_id != registry.assign("task-b", finding("stale schema")).cluster_id

def test_lexical_fallback_is_recorded_when_provider_unavailable() -> None:
    assignment = MechanismClusterer(embedder=UnavailableEmbedder()).assign(finding("stale schema"))
    assert assignment.embedding_fallback_reason == "provider_unavailable"
```

- [ ] **Step 2: Run clustering tests and confirm failure**

Run: `pytest tests/test_clustering.py -q 2>&1 | tee terminal_output/phase_3/clustering_before.log`

Expected: New fallback or task-locality assertions fail.

- [ ] **Step 3: Implement barrier-only lifecycle and lexical fallback reporting**

```python
@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    task_id: str
    cluster_id: str
    similarity: float
    freshness_iteration: int
    embedding_fallback_reason: str | None = None
```

Keep assignments task-local. Defer create/merge/split changes until `refresh_at_barrier(iteration)` and enforce the resolved threshold and max clusters per task.

- [ ] **Step 4: Run clustering verification**

Run: `pytest tests/test_clustering.py -q 2>&1 | tee terminal_output/phase_3/clustering_after.log`

Expected: PASS.

### Task 6: Replace Pool Comparison, Parent, And Champion Semantics

**Files:**
- Modify: `src/agent_evolve/core/pool.py`
- Modify: `tests/test_pool.py`
- Modify: `tests/test_pool_aggregation.py`

**Governing contract:** `docs/architecture/data-contracts.md:31-79`; `docs/architecture/selection-algorithms.md:282-338`.

**Consumes:** Validated score cells with coverage/evaluator provenance and configured epsilon/weights/seed.

**Produces:** `is_comparable`, comparison coverage reports, Pareto frontier, seeded parent selection, and champion report.

- [ ] **Step 1: Write failing tests for mandated weighted Pareto and selection formulas**

```python
def test_pareto_uses_score_times_severity_times_confidence() -> None:
    pool = populated_pool_with_comparable_cells()
    assert pool.dominates("candidate-high-weighted", "candidate-low-weighted")

def test_parent_frequency_awards_all_tied_winners() -> None:
    frequencies = pool.parent_frequencies()
    assert frequencies["candidate-a"] == frequencies["candidate-b"]

def test_protected_floor_disqualifies_champion_despite_high_aggregate() -> None:
    assert pool.select_champion(protected_floor_violations={"candidate-a"}).candidate_id == "candidate-b"
```

- [ ] **Step 2: Run pool tests and confirm failure**

Run: `pytest tests/test_pool.py tests/test_pool_aggregation.py -q 2>&1 | tee terminal_output/phase_3/pool_before.log`

Expected: New mandated formula tests fail against prototype mean-only behavior.

- [ ] **Step 3: Implement explicit comparability, Pareto, parent, and champion reports**

```python
def weighted_score(cell: ScoreCell) -> float:
    return cell.score * cell.severity * cell.confidence

def parent_frequency(candidate_id: str) -> float:
    return sum(cell.severity * cell.confidence for cell in winning_cells(candidate_id))

def champion_aggregate(
    outcome: float, coverage: float, stability: float, regression_risk: float,
    config: ResolvedConfig,
) -> float:
    return (
        config.champion_alpha * outcome
        + config.champion_beta * coverage
        + config.champion_gamma * stability
        - config.champion_delta * regression_risk
    )
```

Expose excluded-cell reasons and enforce complete task IDs. Missing or unavailable cells are excluded rather than treated as zero.

- [ ] **Step 4: Run pool verification**

Run: `pytest tests/test_pool.py tests/test_pool_aggregation.py -q 2>&1 | tee terminal_output/phase_3/pool_after.log`

Expected: PASS.

### Task 7: Narrow Entropy To Comparable-Evidence Statistics

**Files:**
- Modify: `src/agent_evolve/core/entropy.py`
- Modify: `tests/test_entropy.py`

**Governing contract:** `docs/architecture/selection-algorithms.md:17-65`.

**Consumes:** Comparable score updates and cluster freshness at sequential barriers.

**Produces:** Entropy cell statistics, availability reasons, tier classification, and deterministic priority retrieval.

- [ ] **Step 1: Write failing tests for all entropy tiers**

```python
def test_entropy_classifies_recombination_target() -> None:
    tracker = tracker_with_three_candidates_two_rollouts(max_score=0.8, variance=0.2)
    assert tracker.classify("task-1", "cluster-1") == "recombination_target"

def test_entropy_classifies_frontier_exploration() -> None:
    tracker = tracker_with_three_candidates_two_rollouts(max_score=0.2, variance=0.2)
    assert tracker.classify("task-1", "cluster-1") == "frontier_exploration"

def test_entropy_classifies_skip_when_evidence_is_insufficient() -> None:
    assert EntropyTracker().classify("task-1", "cluster-1") == "skip"
```

- [ ] **Step 2: Run entropy tests and confirm failure**

Run: `pytest tests/test_entropy.py -q 2>&1 | tee terminal_output/phase_4/entropy_before.log`

Expected: Tier tests fail if prototype selector ownership remains or tiers are absent.

- [ ] **Step 3: Remove selector ownership and implement statistics only**

```python
def entropy(self, task_id: str, cluster_id: str) -> float | None:
    cell = self._cells[(task_id, cluster_id)]
    if not self._meets_evidence_floor(cell):
        return None
    variance = max(0.0, cell.sum_of_squares / cell.count - (cell.sum / cell.count) ** 2)
    return variance * max(cell.max_score, self.score_floor)
```

Move all `Issue` and DPP selector definitions to Task 8. Retain only statistics and a deterministic heap API from this module.

- [ ] **Step 4: Run entropy verification**

Run: `pytest tests/test_entropy.py -q 2>&1 | tee terminal_output/phase_4/entropy_after.log`

Expected: PASS.

### Task 8: Implement Trace-Backed Issues And Hierarchical DPP

**Files:**
- Create: `src/agent_evolve/core/issues.py`
- Create: `tests/test_issues.py`
- Modify: `tests/test_dpp_math.py`
- Modify: `src/agent_evolve/core/entropy.py`

**Governing contract:** `docs/architecture/selection-algorithms.md:67-280`.

**Consumes:** Entropy statistics, pool relevance, task-local embeddings, retry state, and writable artifact inventory.

**Produces:** `Issue`, `IssueSelectionReport`, deterministic mode selectors, hierarchical DPP selections, and explicit fallbacks.

- [ ] **Step 1: Write the four mandated DPP behavior tests and constraint tests**

```python
def test_dpp_penalizes_similarity_and_promotes_diversity() -> None:
    selected = selector.select((near_duplicate_a, near_duplicate_b, dissimilar), k=2)
    assert dissimilar.issue_id in {issue.issue_id for issue in selected.items}
    assert not {near_duplicate_a.issue_id, near_duplicate_b.issue_id} <= {issue.issue_id for issue in selected.items}

def test_dpp_prefers_quality_among_equally_diverse_items() -> None:
    assert selector.select((low, high), k=1).items == (high,)

def test_dpp_theta_shifts_quality_diversity_balance() -> None:
    assert low_theta_selection != high_theta_selection

def test_dpp_is_deterministic() -> None:
    assert selector.select(issues, k=2) == selector.select(issues, k=2)

def test_issue_without_trace_backed_writable_artifact_is_rejected() -> None:
    assert build_issue(unattributed_finding, inventory) is None
```

- [ ] **Step 2: Run DPP and issue tests and confirm failure**

Run: `pytest tests/test_issues.py tests/test_dpp_math.py -q 2>&1 | tee terminal_output/phase_4/issues_before.log`

Expected: FAIL because `core.issues` does not exist.

- [ ] **Step 3: Implement quality, constraints, modes, and Schur-complement MAP**

```python
def quality(issue: Issue, maximum_raw_quality: float, theta: float, score_floor: float) -> float:
    floored = max(issue.raw_quality, score_floor)
    normalized = floored / maximum_raw_quality
    alpha = theta / (2 * max(1 - theta, 1e-6)) if theta < 1.0 else 1.0
    return normalized ** alpha

def greedy_map(kernel: np.ndarray, ids: tuple[str, ...], k: int, min_gain: float) -> tuple[int, ...]:
    gains = np.diag(kernel).copy()
    factors: list[list[float]] = [[] for _ in ids]
    selected: list[int] = []
    while len(selected) < k:
        remaining = sorted(
            (i for i in range(len(ids)) if i not in selected),
            key=lambda i: (-gains[i], ids[i]),
        )
        j = remaining[0]
        if gains[j] <= min_gain:
            break
        selected.append(j)
        d_j = math.sqrt(gains[j])
        for i in remaining[1:]:
            projection = sum(left * right for left, right in zip(factors[i], factors[j]))
            e = (kernel[i, j] - projection) / d_j
            factors[i].append(e)
            gains[i] = max(0.0, gains[i] - e * e)
    return tuple(selected)
```

Import `math` and `numpy as np`. Build `L = Q S Q + jitter * I`; prefilter at 100; record all settings and fallback reason. Implement `severity_rank`, `coverage` as deterministic farthest-first extension, and seeded `random` modes distinctly from DPP.

- [ ] **Step 4: Run issues/DPP verification**

Run: `pytest tests/test_issues.py tests/test_dpp_math.py -q 2>&1 | tee terminal_output/phase_4/issues_after.log`

Expected: PASS, including no silent fallback and deterministic full-ID tie-breaking.

### Task 9: Replace Memory With Redacted Append-Only Records

**Files:**
- Modify: `src/agent_evolve/core/memory.py`
- Modify: `tests/test_memory.py`

**Governing contract:** `docs/architecture/data-contracts.md:214-231`; `docs/architecture/component-contracts.md:26-28`.

**Consumes:** `MemoryRecord`, `JSONFileStorage`, bounded context budget.

**Produces:** Append-only memory writer, bounded retrieval, and issue/artifact/lineage retry state.

- [ ] **Step 1: Write failing tests for recursive persistence and bounded retrieval**

```python
def test_memory_persists_only_redacted_reference_record(tmp_path: Path) -> None:
    memory = EditMemory(JSONFileStorage(tmp_path), max_records=2)
    memory.append(valid_memory_record())
    assert memory.retrieve(issue_fingerprint="issue-1", max_records=1) == (valid_memory_record(),)

def test_memory_rejects_raw_nested_editor_response(tmp_path: Path) -> None:
    with pytest.raises(PersistenceSafetyError):
        EditMemory(JSONFileStorage(tmp_path)).append_payload({"editor": {"raw_response": "secret"}})
```

- [ ] **Step 2: Run memory tests and confirm failure**

Run: `pytest tests/test_memory.py -q 2>&1 | tee terminal_output/phase_4/memory_before.log`

Expected: New storage-backed redaction tests fail.

- [ ] **Step 3: Implement append-only memory and scoped retry accounting**

```python
@dataclass(slots=True)
class RetryState:
    attempts_by_scope: dict[tuple[str, tuple[str, ...], str], int]

    def exhausted(self, issue: str, artifacts: tuple[str, ...], lineage: str, limit: int) -> bool:
        return self.attempts_by_scope.get((issue, artifacts, lineage), 0) >= limit
```

Route every persisted object through `JSONFileStorage`; never preserve raw editor/task/evaluator data in memory. Enforce `max_history_records` and bounded retrieval.

- [ ] **Step 4: Run memory verification**

Run: `pytest tests/test_memory.py -q 2>&1 | tee terminal_output/phase_4/memory_after.log`

Expected: PASS.

### Task 10: Narrow Editor To Authorization And One Repair Attempt

**Files:**
- Modify: `src/agent_evolve/core/editor.py`
- Modify: `src/agent_evolve/core/fake_editor.py`
- Modify: `tests/test_editor.py`

**Governing contract:** `docs/architecture/data-contracts.md:105-128`; `docs/architecture/component-contracts.md:54-74`.

**Consumes:** Trace-backed `Issue`, readable artifact inventory, bounded memory refs, explicit authorized write set.

**Produces:** `EditorRequest`, validated `EditPlan`, repair request/result, and no promotion decision for repeated malformed output.

- [ ] **Step 1: Write failing editor authorization and repair tests**

```python
def test_editor_plan_rejects_edit_outside_authorized_write_set() -> None:
    with pytest.raises(WriteAuthorizationError):
        validate_editor_plan(plan_targeting("artifact-outside"), authorized_writes=("artifact-inside",))

def test_second_malformed_response_returns_recordable_non_promotion() -> None:
    result = repair_once_then_classify(MalformedEditor(), request)
    assert result.status == "malformed"
    assert result.correction_requests == 1
```

- [ ] **Step 2: Run editor tests and confirm failure**

Run: `pytest tests/test_editor.py -q 2>&1 | tee terminal_output/phase_4/editor_before.log`

Expected: New explicit authorization/repair semantics fail.

- [ ] **Step 3: Implement protocol-only editor behavior**

```python
def validate_editor_plan(plan: EditPlan, readable: frozenset[str], authorized_writes: frozenset[str]) -> EditPlan:
    if not set(plan.read_requests) <= readable:
        raise WriteAuthorizationError("editor requested a non-readable artifact")
    if not set(plan.authorized_writes) <= authorized_writes:
        raise WriteAuthorizationError("editor declared an unauthorized write")
    if any(edit.artifact_id not in authorized_writes for edit in plan.edits):
        raise WriteAuthorizationError("editor edit is outside authorization")
    return plan
```

Do not calculate protected floors or acceptance in this module; Task 11 owns evaluation. Keep adapter-side authorization independently required.

- [ ] **Step 4: Run editor verification**

Run: `pytest tests/test_editor.py -q 2>&1 | tee terminal_output/phase_4/editor_after.log`

Expected: PASS.

### Task 11: Add Four-Category Evaluation And Acceptance Rules

**Files:**
- Create: `src/agent_evolve/core/evaluation.py`
- Create: `tests/test_evaluation.py`

**Governing contract:** `docs/architecture/data-contracts.md:161-176`; `docs/architecture/component-contracts.md:76-82`; `docs/architecture/target-rho-parallel-gepa.md:134-155`.

**Consumes:** Origin cases, written artifacts, memory history, protected floors, deferred probe mode.

**Produces:** `ValidationPlan`, `ValidationResult`, `AcceptanceDecision`, and explicit `generalization_unverified` status.

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_deferred_generalization_is_explicitly_unverified() -> None:
    plan = build_validation_plan(origin_case, written_artifacts=("artifact-1",), probe_mode="deferred")
    assert plan.generalization_status == "generalization_unverified"

def test_protected_floor_forces_rejection_despite_positive_gain() -> None:
    decision = decide_acceptance(result(primary_gain=0.4, weighted_net_gain=0.2, protected_floor_outcome="violated"))
    assert decision.decision == "reject"

def test_unavailable_case_is_not_counted_as_passing() -> None:
    assert summarize_cases((ValidationCase(case_id="x", outcome="unavailable"),)).passed == 0
```

- [ ] **Step 2: Run evaluation tests and confirm failure**

Run: `pytest tests/test_evaluation.py -q 2>&1 | tee terminal_output/phase_4/evaluation_before.log`

Expected: FAIL because `core.evaluation` does not exist.

- [ ] **Step 3: Implement four-category plans and decision function**

```python
def decide_acceptance(result: ValidationResult) -> AcceptanceDecision:
    if result.protected_floor_outcome == "violated":
        return AcceptanceDecision("reject", "protected_floor_violated")
    if result.primary_gain <= 0.0:
        return AcceptanceDecision("reject", "primary_gain_not_positive")
    if result.weighted_net_gain <= 0.0:
        return AcceptanceDecision("reject", "weighted_net_gain_not_positive")
    return AcceptanceDecision("accept", "validated_gain")
```

Always populate origin/worked/regression/generalization categories. Under deferred mode, create zero executed generalization cases with `generalization_unverified`; do not silently drop the category.

- [ ] **Step 4: Run evaluation verification**

Run: `pytest tests/test_evaluation.py -q 2>&1 | tee terminal_output/phase_4/evaluation_after.log`

Expected: PASS.

### Task 12: Run Phase-Gate Verification And B0/B1 Smoke Harness

**Files:**
- Create: `examples/run_phase_1_4_smoke.py`
- Create: `tests/test_phase_1_4_fake_adapter.py`
- Modify: `docs/superpowers/specs/2026-08-12-phase-1-4-research-core-design.md`

**Governing contract:** `docs/research/hypotheses-and-validation.md:5-56`; `AGENTS.md` verification and safety rules.

**Consumes:** Phase 1-4 services and `FakeAdapter`; no orchestrator, CUGA, merge, or parallel service.

**Produces:** Deterministic B0/B1 smoke evidence and a documented follow-up experiment protocol.

- [ ] **Step 1: Write failing end-to-end fake-adapter test**

```python
def test_b1_retains_all_initial_candidates_while_b0_discards_non_winners(tmp_path: Path) -> None:
    outcome = run_fixed_budget_comparison(seed=7, storage_root=tmp_path)
    assert outcome.b0_retained_candidate_count == 1
    assert outcome.b1_retained_candidate_count > 1
    assert outcome.storage_records_are_redacted is True
```

- [ ] **Step 2: Run smoke test and confirm failure**

Run: `pytest tests/test_phase_1_4_fake_adapter.py -q 2>&1 | tee terminal_output/phase_1_4/smoke_before.log`

Expected: FAIL because the standalone harness does not exist.

- [ ] **Step 3: Implement a fixed-budget, deterministic smoke harness**

```python
def run_fixed_budget_comparison(seed: int, storage_root: Path) -> ComparisonOutcome:
    config = resolve_profile("minimal", environ={"OLLAMA_EMBEDDING_MODEL": "embeddinggemma"}, seed=seed)
    # B0 keeps only its best initial candidate; B1 stores every initial candidate
    # and derives a Pareto frontier from comparable fake-adapter evidence.
    return ComparisonOutcome(...)
```

Use a fixed local task coreset and fake rollouts only. Persist the resolved manifest, candidate/score records, and budget usage via `JSONFileStorage`. Never persist expected substrings or evaluator internals. This is a smoke harness, not a claim of H1.

This harness uses deterministic fake candidates instead of the full RHO outer
stage. Full RHO coreset selection, proposal generation, and initial-pool seeding
remain later work; controlled fake candidates are sufficient to test pool
preservation versus best-of-N discard. For each of three seeds, report held-out
outcome mean and dispersion, B0/B1 pool sizes, comparison coverage,
accepted/rejected/no-op/exhausted counts, every budget category, and the
resolved configuration manifest.

- [ ] **Step 4: Run focused Phase 1-4 verification**

Run: `pytest tests/test_contracts_validation.py tests/test_config.py tests/test_storage.py tests/test_analysis.py tests/test_blame.py tests/test_clustering.py tests/test_pool.py tests/test_pool_aggregation.py tests/test_entropy.py tests/test_issues.py tests/test_dpp_math.py tests/test_memory.py tests/test_editor.py tests/test_evaluation.py tests/test_phase_1_4_fake_adapter.py -q 2>&1 | tee terminal_output/phase_1_4/focused.log`

Expected: PASS.

- [ ] **Step 5: Run full regression suite**

Run: `pytest -q 2>&1 | tee terminal_output/phase_1_4/full_suite.log`

Expected: PASS, or document only failures caused by intentionally superseded prototype assumptions before changing them.

- [ ] **Step 6: Run the deterministic smoke script**

Run: `python examples/run_phase_1_4_smoke.py 2>&1 | tee terminal_output/phase_1_4/b0_b1_smoke.log`

Expected: Three fixed-seed B0/B1 smoke records, with resolved configuration, redacted persistence, matched budgets, and no network/CUGA access.

## Plan Self-Review

- Spec coverage: Tasks 1-3 implement contracts, profiles, complete budgets, approved JSON storage, and redaction. Tasks 4-6 implement evidence, blame, clustering, pool comparability, Pareto, parent sampling, and champion formulas. Tasks 7-11 implement every Phase-4 dependency in the approved order. Task 12 verifies the phase gate and B0/B1 smoke behavior.
- Scope check: No task modifies Phase 5-8 modules. `research_parallel` is configuration-only and inactive.
- Type consistency: `ResolvedConfig` feeds storage/clustering/selection; `CausalFinding` feeds clustering/issues; score evidence feeds pool/entropy; `Issue` feeds editor; `EditPlan` feeds evaluation only through validated adapter results.
- Safety check: All persistence routes through Task 3 storage; no task writes secrets or expected answers to storage or logs.
- Placeholder scan: No deferred implementation action is left inside the Phase 1-4 scope. Later-phase work is deliberately excluded.
