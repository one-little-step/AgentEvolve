# Current-To-Target Implementation Mapping

## Purpose

The current package is a prototype. This map prevents a human or AI agent from
mistaking existing names for completed target behavior. Do not perform broad
refactors before the relevant target contract and tests are approved.

| Current location | Target disposition | Required change |
| --- | --- | --- |
| `core/contracts.py` | retain and expand | Add exact neutral records, provenance, error/adapter boundaries; preserve no-CUGA rule. |
| `core/blame.py` | retain and strengthen | Represent uncertainty and trace-backed artifact attribution. |
| `core/analyzer.py` | split | Move exchange records to `analysis.py`; keep fake analyzers test-only. |
| `core/clustering.py` | complete | Implement barrier-only create/merge/split lifecycle and fallback observability. |
| `core/pool.py` | replace/partition | Separate candidate registry, score tensor, comparability, Pareto, parent/champion concerns. |
| `core/entropy.py` | narrow | Retain entropy statistics only; move issue selection/DPP into `issues.py`. |
| `core/memory.py` | replace | Enforce recursive sanitization and append-only redacted state. |
| `core/editor.py` | split | Keep editor protocol/authorization; move floors and validation to `evaluation.py`. |
| `core/merge.py` | complete | Add ancestor-aware no-op handling, evidence selection, and scoped conflict refinement. |
| `core/parallel.py` | replace/partition | Implement prepare/commit/rollback semantics, not callback-by-callback pseudo-atomicity. |
| `core/orchestrator.py` | refactor | Make state transitions explicit and delegate all component behavior. |
| `adapters/base.py` | expand | Validate contract behavior/capabilities without importing concrete runtimes. |
| `examples/fake_adapter.py` | retain | Keep as reference test fixture; make its artificial evaluator semantics explicit. |
| `adapters/cuga.py` | new/deferred | Create only after pinned-SDK verification record and adapter contract tests. |
| `cuga_wrapper/` | new/deferred | Implement manifest/workspace layer before concrete CUGA adapter logic. |

## Migration Order

1. Approve these architecture contracts and define the source-of-truth test
   matrix for contracts, redaction, atomicity, and CUGA wrapper behavior.
2. Write `core/contracts.py` and `core/errors.py` to
   [Data Contracts](data-contracts.md) with every rejection rule tested. Lock
   these schemas before any consuming module is written.
3. Add `config.py` and a transactional `storage.py` per
   [Storage And Transactions](storage-and-transactions.md), including
   failure-injection and recovery tests.
4. Correct pool/evaluation/memory invariants before adding feature depth.
5. Split analysis, issue selection, and evaluation responsibilities from the
   current broad modules, implementing
   [Selection Algorithms](selection-algorithms.md) exactly.
6. Implement [Merge Resolution](merge-resolution.md), then transactional parallel
   services with rollback tests.
7. Refactor the orchestrator into an explicit coordinator.
8. Implement `cuga_wrapper` against fake runtime factories and immutable
   manifests.
9. Complete development-time CUGA SDK inspection and adapter tests.
10. Add `CUGAAdapter` only for verified, pinned SDK behavior.
11. Run matched-budget profile ablations before making research claims.

## Explicit Non-Claims

The current prototype does not yet prove persistent-pool selection, causal
