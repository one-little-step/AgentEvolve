# Phase-1 Standard Pydantic Contract Completion Design

## Status

Approved user architecture decision: persisted contracts use standard Pydantic
behavior. This replaces the prior factory-only, typed-error, strict-construction,
and deep-immutability proposal.

## Governing Sources

- `docs/architecture/README.md:73-89` defines the implementation order.
- `docs/architecture/data-contracts.md` defines the required neutral record
  fields and cross-field relationships.
- `docs/architecture/component-contracts.md:12-31` assigns neutral records and
  provenance references to `contracts.py`.

## Decision Override

Persisted records directly subclass standard Pydantic `BaseModel` and declare
`ConfigDict(frozen=True)`. Direct constructors and all normal Pydantic validation
APIs remain supported. Pydantic owns parsing, coercion, error reporting, and
exception chaining. Invalid construction raises Pydantic `ValidationError`, not
project-specific wrapped errors.

`frozen=True` prevents model attribute reassignment. Nested mappings and
sequences retain standard Pydantic mutability. Do not create `_TypedBoundaryModel`,
`from_payload()`, `from_json()`, typed `ValidationError` mapping, exception-chain
suppression, recursive freezing, or Pydantic API blocking.

Existing domain error classes remain available for later runtime services, but
are not construction errors for persisted Pydantic records.

## Phase-1 Scope

Complete the binding neutral record schemas and ordinary Pydantic validators:

- `ScoreCell`: fields, ranges, coverage reason, rollout cardinality/uniqueness,
  stability relation, and lexical content-hash validation.
- `AttemptRecord`: terminal status, resulting-candidate and validation-reference
  relations, evidence-reference requirements, and sealed-workspace hash state.
- `ArtifactEdit`, `ExpectedEffect`, and `EditPlan`: actual edit targets must be
  contained in authorized writes; edit payload remains opaque.
- `ValidationCase` and `ValidationResult`: required validation case/gain/floor
  fields and `protected_floor_outcome == violated` implies rejection.
- `ArtifactMergeDecision` and `MergeProvenance`: inheritance, refinement,
  no-op, and admitted-child relationships.
- `RedactionReport` and `MemoryRecord`: reference-only record shape and Pydantic
  rejection of undeclared raw fields through `extra="forbid"`.

All failed constraints use ordinary `ValueError` from validators and therefore
surface as Pydantic `ValidationError`.

## Ollama Configuration Boundary

After Phase 1 passes, Phase 2 configuration loads optional `.env` metadata:

```text
OLLAMA_EMBEDDING_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=embeddinggemma
```

It validates and persists URL/model metadata only. No network probe, embedding,
vector storage, clustering, or semantic retrieval is part of Phase 1 or Phase 2.

## Test Strategy

Tests are written first and verify normal Pydantic construction and
`ValidationError` for documented field and cross-field failures. They also prove
frozen attribute reassignment is rejected and nested mapping mutation remains
allowed by explicit decision. The focused and full test suites must pass before
the Phase-2 configuration/storage plan resumes.

## Scope Boundary

Only `core/contracts.py`, contract tests, and minimal error cleanup if needed
are changed. No config, storage, evidence, selection, editing, parallel,
orchestrator, adapter, or CUGA implementation is permitted.
