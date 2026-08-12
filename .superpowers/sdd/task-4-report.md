# Phase-1 Task 4 Report: Merge and Memory Contracts

## Status

Completed without a commit. Production and test edits are confined to the task
scope: `src/agent_evolve/core/contracts.py` and
`tests/test_contracts_validation.py`. This report and the required command logs
are task evidence.

## Governing Requirements

- `data-contracts.md:178-190` defines merge provenance, the three distinct
  parent IDs, an optional child candidate only for an admitted child, non-empty
  artifact decisions, non-negative complementarity, and eligibility checks.
- `data-contracts.md:191-213` defines artifact decision fields and the shared,
  ancestor, refined, and no-operation relationships.
- `data-contracts.md:214-231` defines an append-only, sanitized,
  reference-based memory record and prohibits raw prompt/payload/response/trace
  categories.
- `task-4-brief.md:42-48` requires ordinary `ValueError` model validators and
  frozen `RedactionReport` and `MemoryRecord`, with `extra="forbid"` on the
  latter and no recursive sanitation implementation.
- The explicit user decision accepts standard Pydantic construction, coercion,
  nested mutability, exception chains, and `ValidationError`.

## Tests First (RED)

Added test-first helpers and relation tests to
`tests/test_contracts_validation.py`:

- refined inheritance requires `refinement_request_ref`;
- shared inheritance requires equal parent hashes;
- an ancestor-resulting hash cannot emit an operation;
- an admitted merge requires `child_candidate_id` and an unadmitted merge
  rejects one;
- `MemoryRecord` rejects undeclared `raw_prompt` data; and
- a valid memory record accepts the declared outcome and redaction report.

RED command:

```bash
uv run --extra dev pytest tests/test_contracts_validation.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/merge-memory-red.log
```

Result: failed during collection because `MemoryRecord` and `RedactionReport`
did not yet exist in `agent_evolve.core.contracts`, proving the requested schema
surface was absent. Evidence:
`terminal_output/phase-1-contract-completion/merge-memory-red.log`.

## Implementation

Updated `src/agent_evolve/core/contracts.py`:

- `ArtifactMergeDecision` now declares `refinement_request_ref` and
  `operation_emitted`. Its ordinary `ValueError` validator enforces refined
  refinement provenance, disallows a refinement reference for other inheritance
  modes, retains the documented shared/ancestor checks, and prevents an
  operation where `resulting_hash == ancestor_hash`.
- `MergeProvenance` now declares `child_admitted` and enforces the exact
  admitted-child/child-ID relation with ordinary `ValueError`, preserving the
  existing distinct-parent validation.
- Added frozen `RedactionReport` with `rule_hits` and non-negative
  `truncations`, reflecting the report requirements in
  `storage-and-transactions.md:118-129`.
- Added frozen `MemoryRecord` with `extra="forbid"` and all reference-only
  fields listed in `data-contracts.md:218-227`. Its `outcome` uses the existing
  terminal attempt-status enum.

No custom error classes, strict mode, sanitation scanner, recursive freezing,
factories, or Pydantic API overrides were introduced.

## GREEN Verification

The first GREEN run exposed two existing merge fixtures that lacked the newly
required `operation_emitted` field. Those fixtures were updated in the scoped
test file, then the exact required command passed.

```bash
uv run --extra dev pytest tests/test_contracts_validation.py tests/test_contracts_immutability.py -v 2>&1 | tee terminal_output/phase-1-contract-completion/merge-memory-green.log
git diff --check 2>&1 | tee terminal_output/phase-1-contract-completion/task-4-diff-check.log
```

Results:

- `66 passed in 0.15s`; evidence:
  `terminal_output/phase-1-contract-completion/merge-memory-green.log`.
- `git diff --check` exited successfully with no output; evidence:
  `terminal_output/phase-1-contract-completion/task-4-diff-check.log`.

## Self-Review

- Verified the changed models remain agent-neutral and add no CUGA, Gaia,
  adapter, persistence, or orchestration import.
- Confirmed all new cross-field checks raise ordinary `ValueError`, which
  Pydantic reports as `ValidationError`.
- Confirmed `MemoryRecord` uses `extra="forbid"`, so the representative
  prohibited `raw_prompt` field fails closed at the boundary.
- Confirmed scope did not modify unrelated dirty files. Existing unrelated
  worktree modifications remain preserved.

## Concerns

- Task 4 deliberately provides a reference-only memory schema and field-level
  extra rejection. It does not inspect nested values for sensitive content; the
  governing brief explicitly defers recursive sanitation to a later phase.
- The focused suite was run as required. No full repository suite was run for
  this task because the brief only requires the two contract test modules.
