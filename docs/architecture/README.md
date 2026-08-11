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
4. [Orchestration Lifecycle](orchestration-lifecycle.md) defines lifecycle
   states, attempt processing, LLM recovery, and atomic batch commits.
5. [Persistence And Provenance](persistence-and-provenance.md) defines durable
   run records, immutable candidate versions, redaction, and transactional
   writes.
6. [Implementation Mapping](implementation-mapping.md) maps the current
   prototype to the target package layout and lists the required migration order.
7. [CUGA Wrapper And Adapter](cuga-adapter/README.md) defines the internal
   `cuga_wrapper` package, future `CUGAAdapter`, and development-time SDK
   verification process.

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
- CUGA integration requires development-time inspection of a pinned official SDK
  and adapter tests before implementation. Do not infer APIs from these documents.
