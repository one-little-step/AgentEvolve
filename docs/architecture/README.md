# Architecture Documentation

This directory defines the intended AgentEvolve system before implementation.
It is written for both human reviewers and AI implementation agents. Treat the
documents as a design contract: they distinguish target behavior from current
prototype behavior and make ownership, dependencies, failure policies, and
verification obligations explicit.

## Reading Order

1. [Target RHO-Parallel-GEPA Architecture](target-rho-parallel-gepa.md) states
   the approved research objective and feature-gated system behavior.
2. [Component Architecture](component-architecture.md) shows the target package
   topology, component responsibilities, and component interactions.
3. [Component Contracts](component-contracts.md) specifies module ownership,
   public boundaries, dependencies, and required behavior.
4. [Data Contracts](data-contracts.md) locks the record schemas and validation
   rules that all mathematics depends on.
5. [Selection Algorithms](selection-algorithms.md) mandates the entropy, DPP,
   dominance, sampling, and champion algorithms.
6. [Merge Resolution](merge-resolution.md) mandates three-way inheritance and
   evidence-based conflict resolution.
7. [Storage And Transactions](storage-and-transactions.md) mandates the
   transactional backend, barrier semantics, and redaction gateway.
8. [Orchestration Lifecycle](orchestration-lifecycle.md) defines lifecycle
   states, attempt processing, LLM recovery, and atomic batch commits.
9. [Persistence And Provenance](persistence-and-provenance.md) defines durable
   run records, immutable candidate versions, redaction, and transactional
   writes.
10. [Implementation Mapping](implementation-mapping.md) maps the current
    prototype to the target package layout and lists the required migration order.
11. [CUGA Wrapper And Adapter](cuga-adapter/README.md) defines the internal
    `cuga_wrapper` package, future `CUGAAdapter`, and development-time SDK
    verification process.

Documents 4 through 7 are binding implementation specifications. An
implementation agent must not substitute an alternative formula, algorithm,
schema field, or storage strategy for the mandates they contain.

## Status Vocabulary

Every target component and feature is labelled using one of these terms:

| Status | Meaning |
| --- | --- |
| `target` | Approved design; not evidence that code exists. |
| `prototype` | Current partial implementation; may be replaced. |
| `new` | Required target module absent from the repository. |
| `deferred` | Deliberately postponed pending research evidence or SDK inspection. |
| `verified` | A pinned dependency, official source, and tests prove the stated behavior. |

## Global Rules

- `src/agent_evolve/core/` remains agent-neutral and never imports CUGA, Gaia,
  or a concrete adapter.
- The system evolves immutable, externally stored harness versions. A parent
  candidate is never mutated in place.
- Adapters declare artifact units and merge granularity. The generic core never
  assumes Markdown, filenames, headings, or a particular agent runtime.
- LLM output is stochastic. Machine validation consumes only minimal structured
  fields; rationale and uncertainty remain available as bounded natural language.
- Unknown, malformed, and insufficient-evidence outcomes are explicit states,
  never silently coerced into a fixed failure class.
- Raw credentials, expected answers, evaluator internals, labels, and regexes
  must not enter edit memory, embeddings, manifests, prompts, or terminal logs.
- Records validate at construction. An invalid record raises a typed error; it is
  never silently defaulted or coerced.
- Every aggregation uses complete identifier strings as keys. Prefixes, slices,
  first characters, and derived hashes of IDs are defects.
- CUGA integration requires development-time inspection of a pinned official SDK
  and adapter tests before implementation. Do not infer APIs from these documents.

## Implementation Order

Build in this order. Do not begin the orchestrator, selection algorithms, or the
CUGA adapter first.

1. `core/contracts.py` and `core/errors.py`, per
   [Data Contracts](data-contracts.md), with complete validation test coverage
   including every rejection rule.
2. `core/config.py` and `core/storage.py`, per
   [Storage And Transactions](storage-and-transactions.md), including
   failure-injection and recovery tests.
3. Evidence and diagnosis modules: `pool.py`, `analysis.py`, `blame.py`,
   `clustering.py`.
4. Selection and editing modules: `entropy.py`, `issues.py`, `memory.py`,
   `editor.py`, `evaluation.py`.
5. `merge.py`, then `parallel.py` with rollback tests.
6. `orchestrator.py` as a coordinator over the above.
7. `cuga_wrapper` against fake runtime factories.
8. `adapters/cuga.py` only after the SDK verification record exists.
