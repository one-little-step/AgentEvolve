# Phase 1-4 Research Core Design

## Status

Approved research-path design for contracts, research persistence, evidence,
diagnosis, selection, and editing. It implements H1-H4 algorithm prerequisites
against a deterministic fake adapter. CUGA, merge, parallel execution, and
orchestration lifecycle changes are out of scope.

## Goals

- Produce a reproducible, agent-neutral core capable of B0-B4 experiments.
- Use a deterministic fake adapter, never CUGA, for all Phase 1-4 tests.
- Preserve data safety through one recursive fail-closed persistence gateway.
- Implement the mandated entropy and hierarchical DPP algorithms rather than
  approximating them with severity ranking.
- Persist sufficient experiment evidence without building production storage
  infrastructure.

## Research Storage Exception

This design uses `JSONFileStorage` despite the production SQLite WAL mandate in
`docs/architecture/storage-and-transactions.md`. The exception is formally
recorded in `2026-08-12-architecture-enforcement-design.md`.

The JSON backend is only valid for single-threaded Phase 1-4 research runs. It
provides per-record temp-write-then-rename atomicity, but no multi-record
transactions, recovery, blobs, garbage collection, or persistent leases.
`parallel_execution` is rejected while this backend is active.

## Phase 1: Contract Consolidation

`core/contracts.py` and `core/errors.py` are the sole core schema boundary.
Phase 3-4 modules consume their validated records rather than introducing a
third semantic record layer. Records remain immutable where the contract
requires it, validate at construction, use complete opaque IDs, and retain
explicit unavailable states.

The fake adapter remains outside `core` and is the only concrete runtime used
through this design. `core` must not import CUGA, Gaia, or any other concrete
agent implementation.

## Phase 2: Configuration And Research Persistence

### Configuration

`core/config.py` resolves immutable named profiles:

- `minimal`: persistent pool, outcome Pareto, severity-directed editing.
- `research_sequential`: causal blame, edit memory, focused validation.
- `research_parallel`: resolves parallel feature configuration but marks the
  feature inactive until Phase 5.
- `full_ablation`: exposes each Phase 1-4 feature gate independently.

`ResolvedConfig` validates and records budgets, feature gates, random seed,
and all algorithm defaults required by downstream modules:

```text
GEPA_DPP_MAX_ITEMS=100
GEPA_DPP_THETA=0.7
GEPA_DPP_SCORE_FLOOR=0.1
GEPA_DPP_MIN_GAIN=1e-12
GEPA_ENTROPY_REFRESH_MODE=outer_iteration
GEPA_ENTROPY_SCORE_FLOOR=0.15
GEPA_ENTROPY_RECOMBINATION_SCORE_THRESHOLD=0.30
GEPA_ENTROPY_FRONTIER_WEIGHT=0.30
GEPA_ENTROPY_MIN_COMPARABLE_CANDIDATES=3
GEPA_ENTROPY_MIN_ROLLOUTS_PER_CANDIDATE=2
GEPA_CLUSTER_SIMILARITY_THRESHOLD=0.80
GEPA_MAX_CLUSTERS_PER_TASK=12
GEPA_MECHANISM_EMBEDDING_MODEL=embeddinggemma
GEPA_GENERALIZATION_PROBE_MODE=deferred
```

The resolver reads `OLLAMA_EMBEDDING_URL` and `OLLAMA_EMBEDDING_MODEL` from the
environment. They are endpoint/model configuration, not secrets. The manifest
records the resolved model and whether deterministic lexical fallback was used,
including the fallback reason. Phase 1-4 tests use lexical embeddings and make
no Ollama network call.

Budget limits match the architecture mandate:

```text
GEPA_MAX_ATTEMPTS              total edit attempts
GEPA_MAX_ACCEPTED_EDITS        accepted edits
GEPA_MAX_MODEL_TOKENS          total model tokens across roles
GEPA_MAX_ROLLOUTS              agent rollout executions
GEPA_MAX_JUDGE_VERDICTS        analyzer/judge calls
GEPA_EDIT_MAX_RETRIES=3        per-issue retry ceiling
GEPA_MAX_WALL_SECONDS          wall-clock limit
GEPA_MAX_POOL_CANDIDATES       persistent-pool size limit
GEPA_MAX_HISTORY_RECORDS       edit-memory size limit
GEPA_MAX_RAG_CONTEXT_TOKENS    editor-context limit
```

Usage additionally records embedding and validation calls. Validation rollouts
also consume `GEPA_MAX_ROLLOUTS`; embedding calls are reported separately so
research cost accounting is complete.

### Storage And Redaction

`StorageBackend` exposes `write_record(record_type, record_id, payload)`,
`read_record(record_type, record_id)`, `list_records(record_type)`, and
`close()`.

`JSONFileStorage` writes one deterministic JSON object per `(record_type,
record_id)` below a validated storage root. Record types and IDs are validated
as safe opaque path components. A temporary sibling file is written and renamed
to publish one complete record.

Every write passes through one recursive sanitizer. It walks mappings,
sequences, nested models, and strings; rejects credentials, expected answers,
evaluator internals, labels, regexes, raw prompts/model responses, and
unapproved raw trace bodies. It produces a `RedactionReport`; failure to obtain
a safe representation aborts the write. No caller may bypass this gateway.

## Phase 3: Evidence And Diagnosis

### Analysis And Blame

`analysis.py` owns sanitized rollout-group reports and the analyzer/judge
exchange boundary. `blame.py` owns dynamic causal findings and trace-backed
graphs. Unknown, malformed, uncertain, and insufficient evidence remain
explicit findings; no synthetic blame nodes or fixed taxonomy are allowed.

### Clustering

`clustering.py` aligns free-form mechanisms only within a task. Base-harness
observations anchor the task-local clusters. Assignment uses configured semantic
embedding when available or recorded deterministic lexical fallback otherwise.
Creation, merge, and split occur only at refresh barriers; each cluster records
freshness and assignment lineage.

### Pool

`pool.py` owns immutable candidate registry, score tensor, comparability,
Pareto, parent sampling, and champion selection. It compares only evaluated
`ScoreCell` records matching exact task ID, cluster ID, compatible evaluator
family, and the operation's rollout floor. Missing evidence is excluded, never
zero-filled.

Pareto applies the mandated weighted score:

```text
weighted(candidate, mechanism) = score * severity * confidence
```

It excludes mechanisms unavailable for either candidate, records comparison
coverage, respects protected floors, and retains every initial RHO proposal and
accepted child in the persistent pool.

Parent selection uses weighted objective coverage:

```text
frequency(candidate) = sum over (task, mechanism) of
  severity(task, mechanism) * confidence(candidate, task, mechanism)
  * indicator[candidate wins (task, mechanism)]
```

Winning means the strict maximum comparable weighted score; ties award every
tied winner. Sampling is seeded and proportional to frequency.

Champion selection is transparent:

```text
aggregate(candidate) =
  0.55 * Outcome(candidate)
  + 0.20 * ProcessCoverage(candidate)
  + 0.15 * Stability(candidate)
  - 0.10 * RegressionRisk(candidate)
```

Protected critical floors disqualify a candidate before this aggregate. The
manifest records every component, coverage, disqualification, and tie-breaker.

## Phase 4: Selection And Editing

Implementation order is mandatory. Each item has a separate TDD cycle and
focused verification before work proceeds to its dependent module.

### 1. Entropy

`entropy.py` owns incremental statistics only. For each comparable
`(task_id, mechanism_cluster_id)` cell it maintains count, sum, sum of squares,
candidate score map, maximum score/owner, rollout counts, and cluster freshness.
It uses population variance and:

```text
entropy = variance * max(max_score, GEPA_ENTROPY_SCORE_FLOOR)
```

Entropy is unavailable until three comparable candidates and two rollouts per
candidate exist. Frontier cells below the recombination threshold retain the
configured frontier weight. Heap operations are deterministic and run only at a
sequential barrier in this phase.

Cells are explicitly classified as `recombination_target` when evidence and
variance are sufficient and maximum score exceeds the recombination threshold,
`frontier_exploration` when evidence and variance are sufficient but maximum
score is below it, or `skip` when evidence or variance is insufficient.

### 2. Issues And DPP

`issues.py` creates only trace-backed issues with inventory-declared writable
artifact attribution. Hard constraints reject duplicate parent/write-set pairs,
overlapping write sets, exhausted retry contexts, overrepresented mechanism
clusters, and invalid embeddings before selection.

Raw issue quality is the configured weighted sum of severity, confidence,
normalized entropy, coverage need, and Pareto relevance. The selector supports
and records `dpp`, `severity_rank`, `coverage`, and seeded `random` modes.

The `dpp` mode is mandatory hierarchical selection: tasks first, then mechanism
clusters inside selected tasks. It prefilters to at most
`GEPA_DPP_MAX_ITEMS` items using entropy and quality ranking before any dense
kernel. Selection reports record prefilter threshold, candidate counts,
retained counts, theta, alpha, score floor, selected IDs, and fallback reason.

Embeddings are L2-normalized; cosine and optional structural similarity are
clamped to `[0, 1]`. The kernel is:

```text
L = Q S Q + JITTER * I
```

Greedy MAP selection uses Cholesky-style Schur-complement updates, stops at
`GEPA_DPP_MIN_GAIN`, clamps only floating-point drift, and resolves ties by
ascending stable issue ID. Missing/malformed/incompatible embeddings, unstable
kernels, and selection exceptions fall back to deterministic quality ordering
with an explicit reason.

### 3. Memory

`memory.py` persists append-only, redacted references to attempt outcomes,
worked/failed/regression state, and scoped retry exhaustion. It never persists
raw prompts, raw editor responses, expected answers, evaluator internals,
labels, regexes, or raw traces. Retrieval is bounded and references only the
sanitized records.

### 4. Editor

`editor.py` owns the editor protocol and response validation. It receives an
issue, declared readable inventory, bounded memory references, and an explicit
authorized write set. It may request reads, but read requests never grant write
authority. The adapter independently rejects edits outside the authorized set.
A malformed response receives one schema-defect correction opportunity, then
generates a recorded malformed non-promotion outcome.

### 5. Evaluation

`evaluation.py` constructs validation plans and produces the validated
`ValidationResult`. Every plan has origin, worked, regression, and
generalization categories. Generalization is deferred by default: the result
explicitly records `generalization_unverified`; it is neither omitted nor
treated as passing. Later phases execute cluster-completion probes under the
configured probe budget.

Acceptance requires positive primary gain, positive weighted net gain, and no
protected-floor violation. Unavailable evidence remains unavailable and is not
counted as a pass or a failure.

## Prototype Migration Boundaries

Phase 1-4 replaces prototype behavior, rather than extending it:

- broad `entropy.py` selection ownership moves to `issues.py`;
- shallow caller-trusted sanitization becomes one recursive storage gateway;
- editor-owned floor/validation behavior moves to `evaluation.py`;
- direct orchestration edits without target-level authorization are replaced by
  editor validation plus adapter-enforced write scopes.

`orchestrator.py`, `parallel.py`, `merge.py`, the CUGA wrapper, and the CUGA
adapter are not Phase 1-4 implementation targets.

The task-local rollout diversity sampler for judge-budget allocation and
real-model role configuration (rollout, analyzer/judge, editor) are deferred to
later phases. The Phase 1-4 fake adapter uses deterministic local substitutes
and does not make model or network calls.

## Test And Verification Requirements

Tests precede each implementation unit. Focused tests cover contract rejection,
profile incompatibility, JSON record atomicity and path containment, recursive
redaction, every budget category, fake-adapter evidence flow, full-ID isolation,
comparability exclusions, Pareto weighting, cluster freshness, entropy floors,
all mandated DPP behaviors, DPP fallback recording, redacted memory, write
authorization, one-repair malformed editor handling, protected floors, and
deferred generalization status.

Every test, smoke run, and verification command is captured with:

```text
2>&1 | tee terminal_output/<topic>/<name>.log
```

## Initial Experiment

The first research result compares B0 and B1 on a fixed coreset with matched
rollout budgets and at least three deterministic seeds. It reports held-out
outcome mean and dispersion, pool size, comparison coverage, accepted/rejected
attempts, and every budget category. One smoke run is not evidence for H1.

### Phase-Gate Smoke Protocol

`examples/run_phase_1_4_smoke.py` is the phase-gate verification harness. For a
fixed local task coreset it builds a deterministic base plus two token-injected
fake candidates (`c1` satisfies task A, `c2` satisfies task B, base satisfies
neither), then contrasts:

- B0 (best-of-N): evaluate every candidate on every task, retain only the single
  highest-scoring candidate (`b0_retained == 1`).
- B1 (persistent pool): retain base plus every candidate, record comparable score
  evidence under one fixed mechanism cluster id, and derive a Pareto frontier
  (`b1_retained > 1`).

It persists a redacted manifest, one candidate record per candidate, and one
score record, then reads them back to confirm the expected-substring tokens
never reached storage. This is a deterministic offline smoke harness, not an H1
claim.
