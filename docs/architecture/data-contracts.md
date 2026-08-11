# Data Contracts

## Purpose

This document locks the data schemas so downstream mathematics cannot be
corrupted by invented fields or permissive defaults. Every record below is
validated at construction. An invalid record raises a typed error from
`core/errors.py`; it is never silently normalized, coerced, or defaulted.

Implementation note: validation-at-construction is mandatory. `pydantic` is
already a declared dependency and is acceptable; frozen dataclasses with explicit
`__post_init__` validation are equally acceptable. What is not acceptable is a
record type that can hold an invalid state.

## Identity And Reference Types

```text
ExperimentId, CandidateId, AttemptId, WorkspaceId, TaskId, RolloutId,
TraceId, AnalysisId, VerdictId, MergeId, ProbeId, MemoryRecordId,
TransactionId, SnapshotVersion, MechanismClusterId, ArtifactId, ContentHash
```

Rules:

- All IDs are non-empty strings, stable for the lifetime of a run, and compared
  by exact full value. Prefixes, substrings, first characters, truncations, and
  hashes of IDs are forbidden as aggregation or grouping keys.
- `ContentHash` is `"<algorithm>:<hexdigest>"` with a declared algorithm.
- Cross-record references are stored as IDs, never as embedded mutable objects.

## ScoreCell

The atomic unit of comparable evidence.

| Field | Type | Validation |
| --- | --- | --- |
| `candidate_id` | `CandidateId` | required, non-empty |
| `task_id` | `TaskId` | required, non-empty, used whole |
| `mechanism_cluster_id` | `MechanismClusterId` | required, non-empty |
| `mechanism_ids` | tuple of str | may be empty; free-form judge mechanisms |
| `score` | float | required, `0.0 <= score <= 1.0` |
| `severity` | float | required, `0.0 <= severity <= 1.0` |
| `confidence` | float | required, `0.0 <= confidence <= 1.0` |
| `stability` | float or `None` | if present, `0.0 <= stability <= 1.0`; `None` only when `rollout_count == 1` |
| `rollout_count` | int | required, `>= 1` |
| `rollout_ids` | tuple of `RolloutId` | required, length equals `rollout_count`, unique |
| `verdict_refs` | tuple of `VerdictId` | required, non-empty |
| `artifact_versions` | mapping `ArtifactId -> ContentHash` | required, non-empty |
| `evaluator_id` | str | required; analyzer/judge/evaluator identity and version |
| `coverage` | `Coverage` | required; see below |

Rejection rules:

- `rollout_count < 1` raises `ScoreProvenanceError`.
- Missing or empty `mechanism_cluster_id` raises `ScoreProvenanceError`.
- Empty `verdict_refs` or empty `artifact_versions` raises `ScoreProvenanceError`.
- A score outside `[0, 1]` raises `ScoreRangeError`.
- `stability` supplied with `rollout_count == 1` raises `ScoreProvenanceError`;
  single-rollout stability is unknown, not perfect.

`Coverage` records `evaluated`, `unavailable`, or `excluded`, plus a reason code
when not `evaluated`. An `unavailable` cell never participates in dominance,
entropy, merge evidence, or champion aggregation.

## Comparability

Two `ScoreCell` values are comparable only when all of the following match:

```text
task_id
mechanism_cluster_id
evaluator_id family declared compatible by config
coverage == evaluated
rollout_count >= configured minimum for the requested operation
```

`core/pool.py` must expose comparability as an explicit predicate. Any operation
that consumes score evidence takes comparable cells only. Non-comparable evidence
is excluded and the exclusion reason is recorded.

## CausalFinding And BlameGraph

| Field | Type | Validation |
| --- | --- | --- |
| `verdict_id` | `VerdictId` | required |
| `candidate_id`, `task_id`, `trace_id` | IDs | required |
| `status` | enum | `observed \| uncertain \| insufficient_evidence \| malformed` |
| `mechanism_description` | str | required when `status == observed` |
| `mechanism_cluster_id` | `MechanismClusterId` or `None` | required when `observed`; assigned by clustering |
| `severity`, `confidence` | float or `None` | required when `observed`, in `[0, 1]` |
| `blame_graph` | `BlameGraph` | may be empty; nodes must be trace-backed |
| `evidence_refs` | tuple of str | required when `observed`, non-empty |
| `rationale` | str | bounded free text, required |
| `counterfactual_notes` | tuple of str | optional, bounded |

`BlameGraph` validation:

- Node blame values are in `[0, 1]`.
- Every edge endpoint references an existing node.
- `artifact_candidates` may contain only inventory-declared `ArtifactId` values
  observed in the trace or read set.
- A node with no trace evidence is invalid. Synthetic placeholder nodes are
  forbidden; absence of evidence must be expressed as `insufficient_evidence`.

## EditPlan

The editor exchange strictly separates reads from authorized writes.

| Field | Type | Validation |
| --- | --- | --- |
| `attempt_id` | `AttemptId` | required |
| `issue_fingerprint` | str | required |
| `read_requests` | tuple of `ArtifactId` | each must be inventory-declared readable |
| `authorized_writes` | tuple of `ArtifactId` | non-empty, each lease-held and declared writable |
| `edits` | tuple of `ArtifactEdit` | non-empty; every target in `authorized_writes` |
| `rationale` | str | bounded free text, required |
| `risks` | tuple of str | bounded, may be empty |
| `expected_effect` | `ExpectedEffect` | mechanism cluster references only |

Rules:

- `read_requests` never grants write permission.
- An edit targeting an artifact outside `authorized_writes` raises
  `WriteAuthorizationError` before any workspace mutation.
- `ArtifactEdit` carries `artifact_id`, adapter-declared `operation`, and opaque
  `payload`. The core does not interpret payload structure.
- The adapter re-validates authorization independently of the editor and the
  orchestrator.

## AttemptRecord

Terminal statuses are exactly:

```text
accepted | rejected | no_op | malformed | exhausted | unavailable
```

| Field | Type | Validation |
| --- | --- | --- |
| `attempt_id` | `AttemptId` | required |
| `snapshot_version` | `SnapshotVersion` | required |
| `parent_candidate_id` | `CandidateId` | required |
| `result_candidate_id` | `CandidateId` or `None` | required iff `status == accepted`, otherwise must be `None` |
| `status` | enum above | required |
| `issue_fingerprint` | str | required |
| `task_refs`, `mechanism_cluster_refs` | tuples | required |
| `read_set`, `write_set` | tuples of `ArtifactId` | required |
| `hashes_before`, `hashes_after` | mappings | `hashes_after` required iff a workspace was sealed |
| `analysis_refs`, `verdict_refs`, `memory_refs` | tuples of IDs | required, may be empty only for `unavailable` |
| `validation_result_ref` | ID or `None` | required for `accepted` and `rejected` |
| `rationale_summary`, `risk_summary` | str | sanitized, bounded |
| `budget_usage` | `BudgetUsage` | required |
| `retry_state` | `RetryState` | required |
| `timestamps` | `AttemptTimestamps` | required |

`accepted` without a validation result or resulting candidate raises
`AttemptRecordError`. `malformed` and `exhausted` records are mandatory, not
optional bookkeeping; the orchestrator must persist them through the same
transaction as accepted results.

## ValidationResult

| Field | Type | Validation |
| --- | --- | --- |
| `origin_cases` | tuple of `ValidationCase` | non-empty |
| `worked_cases`, `regression_cases` | tuples | required for every written artifact with existing state |
| `generalization_cases` | tuple | may be empty; deferred by default |
| `primary_gain` | float | required |
| `weighted_net_gain` | float | required |
| `protected_floor_outcome` | enum | `satisfied \| violated \| unavailable` |
| `decision` | enum | `accept \| reject` |
| `decision_reason` | str | required |
| `unavailable_cases` | tuple | cases whose evidence could not be collected |

An `unavailable` case is never counted as a pass. `protected_floor_outcome ==
violated` forces `decision == reject` regardless of gains.

## MergeProvenance

Per-artifact three-way provenance is mandatory.

| Field | Type | Validation |
| --- | --- | --- |
| `merge_id` | `MergeId` | required |
| `ancestor_candidate_id`, `left_candidate_id`, `right_candidate_id` | IDs | required, distinct |
| `child_candidate_id` | `CandidateId` or `None` | required iff merge produced an admitted child |
| `artifact_decisions` | tuple of `ArtifactMergeDecision` | non-empty |
| `complementarity` | float | required, `>= 0` |
| `eligibility_checks` | mapping check name to bool | required |

`ArtifactMergeDecision` requires:

```text
artifact_id
ancestor_hash
left_hash
right_hash
resulting_hash
inheritance: ancestor | left | right | shared | refined
evidence_score_left
evidence_score_right
decision_reason
```

Rules:

- `inheritance == shared` requires `left_hash == right_hash`.
- `inheritance == ancestor` requires `resulting_hash == ancestor_hash`.
- `inheritance == refined` requires a recorded conflict-refinement request scoped
  to that single artifact unit.
- A decision whose `resulting_hash` equals `ancestor_hash` must not emit an edit
  operation.

## MemoryRecord

Append-only, sanitized, and reference-based.

| Field | Type | Validation |
| --- | --- | --- |
| `memory_record_id` | ID | required |
| `attempt_id` | `AttemptId` | required |
| `artifact_ids` | tuple | required |
| `issue_fingerprint` | str | required |
| `outcome` | enum | mirrors attempt status |
| `summary` | str | sanitized, bounded |
| `evidence_refs` | tuple of IDs | required |
| `redaction_report` | `RedactionReport` | required |

Forbidden fields: raw editor payloads, raw prompts, raw model responses, raw
trace bodies, expected answers, evaluator internals, labels, regexes, and
credentials. A record that cannot be sanitized fails closed and is not written.
