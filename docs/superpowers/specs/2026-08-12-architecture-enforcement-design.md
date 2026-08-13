# Architecture Enforcement And Step-2 Recovery Design

## Status

Approved recovery approach: strict architecture reset without reverting existing
prototype work. This document is the execution gate for future changes.

## Binding Precedence

`docs/architecture/` is the binding implementation contract for AgentEvolve.
The documents named by `docs/architecture/README.md` as binding specifications
control implementation details, algorithms, record fields, storage strategy,
and negative constraints. Existing prototype code, previous plans, docstrings,

When a reviewer finding conflicts with a binding architecture mandate, preserve
the mandate and translate the finding into an architecture-compliant test and
implementation task. For example, a score tensor cell without trace-backed
causal attribution must remain insufficient evidence; it must not receive a
synthetic blame node merely to make an orchestration path proceed.

## Mandatory Architecture Pre-Read And Citation

Before planning, testing, or changing a phase, an implementation agent must read
the phase-relevant binding documents in `docs/architecture/`. For Phase 2, the
mandatory pre-read set is:

1. `README.md`, including the implementation order at lines 73-89.
2. `data-contracts.md`, to preserve validation-at-construction and opaque-ID
   requirements at the storage boundary.
3. `storage-and-transactions.md`, which governs SQLite, atomicity, recovery,
   blobs, leases, idempotency, and redaction.
4. `component-contracts.md`, which assigns `config.py` and `storage.py`
   ownership and prohibits their selection or causal-reasoning responsibilities.
5. `orchestration-lifecycle.md`, which defines the durable barrier outcome that
   storage must support without implementing orchestration.

Every implementation task must cite the governing architecture document and
section before code changes begin. If no binding section governs a task, the plan
must say so explicitly. Before a task is submitted for review, the implementer
must cross-check the changed public interfaces and tests against those cited
sections.

## Phase-1 Verification Status

Phase 1 is in final remediation: current Pydantic boundary models require the
remaining global non-empty ID and merge content-hash validation before the phase
can unlock Phase 2. By explicit user decision, standard Pydantic construction,
coercion, `ValidationError`, and nested mutability are accepted. The prior
verification record is superseded until this remediation passes its focused and
full suites. This is a verification gate, not authorization to expand or migrate
unrelated prototype callers during Phase 2.

## Hard Phase Gate

No production changes may be made to a phase's modules until every preceding
phase has passing tests for its binding requirements.

1. Contracts and errors: `core/contracts.py`, `core/errors.py`.
2. Configuration and storage: `core/config.py`, `core/storage.py`.
3. Evidence and diagnosis: `pool.py`, `analysis.py`, `blame.py`,
   `clustering.py`.
4. Selection and editing: `entropy.py`, `issues.py`, `memory.py`,
   `editor.py`, `evaluation.py`.
5. Merge and parallel execution: `merge.py`, `parallel.py`.
6. Coordination: `orchestrator.py`.
7. CUGA wrapper.
8. CUGA adapter after official SDK verification.

This sequence is the source-of-truth order from
`docs/architecture/README.md:73-89`, not a restatement that can independently
drift from the architecture README.

Until Phase 2 passes its required tests, edits to `orchestrator.py`,
`parallel.py`, `entropy.py`, `issues.py`, adapter code, and CUGA-facing code
are prohibited. Existing changes in those modules are prototype state, not
evidence of phase completion. They are neither expanded nor reverted unless a
user explicitly requests it.

## Step-2 Scope

The next implementation increment is limited to the Phase-2 foundation.

`core/config.py` must resolve profiles and feature gates through validated,
agent-neutral configuration. It must not hardcode profile behavior in the
orchestrator.

`core/storage.py` must provide the mandated SQLite transactional metadata
store. Its reference configuration is WAL mode, FULL synchronous writes,
foreign keys enabled, configured busy timeout, and one coordinator-owned writer
connection. It owns atomic persistence for candidates, score cells, attempts,
validation results, provenance, memory records, retry state, budgets, leases,
snapshots, manifests, and transaction audit rows.

All persistence must pass a recursive redaction gateway that fails closed for
credentials, expected answers, evaluator internals, labels, regexes, raw model
payloads, and raw unapproved trace bodies. Blob writes precede metadata writes;
the metadata transaction commits last.

## Research Storage Exception (Approved)

For Phase 1-4 research runs using single-threaded execution and deterministic
fake adapters, `JSONFileStorage` is approved as a substitute for the SQLite WAL
mandate in `docs/architecture/storage-and-transactions.md`. This is an approved,
bounded exception, not a replacement for the production storage architecture.

Justification:

- Phase 1-4 runs are single-threaded; WAL concurrency is unused.
- Runs are short-lived and reproducible; crash recovery is not needed to test
  H1-H5.
- The research contribution is the selection and editing algorithms, not
  storage durability.

The exception applies only when `parallel_execution` is disabled, all runs are
single-threaded, no concurrent writers exist, and the `StorageBackend` protocol
is satisfied. It is revoked when Phase 5 begins; SQLite WAL or an equivalent
transactional backend is then required.

The research backend preserves recursive fail-closed redaction, budget tracking
by cost category, profile resolution, manifest-safe feature-gate serialization,
opaque full-string IDs, and temp-write-then-rename atomicity for each individual
record.

It does not provide cross-record transactions, recovery, content-addressed
blobs, orphan cleanup, or durable leases. `research_parallel` is resolved and
validated as a profile but its `parallel_execution` feature remains inactive;
the JSON backend rejects any active parallel-execution configuration.

Production upgrade work remains required before parallel execution or a
production deployment: SQLite WAL with `journal_mode=WAL`, `synchronous=FULL`,
and `foreign_keys=ON`; content-addressed blob staging; `BEGIN IMMEDIATE`
barrier commits; interrupted-transaction recovery; orphan cleanup; and durable
lease management. `StorageBackend` preserves this as a backend substitution,
not a core redesign.

## Required TDD Evidence Before Phase-3 Unlock

Tests are written before each implementation unit and must prove:

- Config rejects invalid profile and incompatible feature-gate combinations.
- SQLite pragmas and coordinator ownership match the storage mandate.
- A successful commit atomically publishes every staged record category.
- Failure injection at every metadata write leaves no published partial state.
- A failure after blob staging and before commit leaves no metadata rows.
- Idempotency keys prevent duplicate records.
- Startup recovery rebuilds derived state, releases expired leases, and marks
  interrupted batches without publishing partial state.
- A non-transactional test backend is rejected when parallel execution is
  enabled.
- Recursive redaction rejects prohibited nested keys and string-embedded
  sensitive material.

Every command used for test, smoke, or migration verification is captured with
`2>&1 | tee terminal_output/<topic>/<name>.log`, as required by `AGENTS.md`.

## Enforcement Method

Before changing a production module, the implementation plan must name the
architecture phase, the binding source sections, the required rejection and
failure tests, and the explicit phase-unlock condition. A matching test must
exist for every negative constraint relied upon by the module. A copied
docstring, a passing happy-path test, or a reviewer assertion alone is never
evidence of compliance.

The next approved work after this design review is a detailed Step-2 TDD plan.
It must not include orchestrator, parallel, selector, adapter, or CUGA changes.
