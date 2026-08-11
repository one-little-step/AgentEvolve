# Orchestration Lifecycle And Failure Policies

## Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> Initializing
    Initializing --> CorpusReady: config, storage, adapter, manifest valid
    Initializing --> Failed: invalid configuration or unavailable required service

    CorpusReady --> InitialPoolEvaluating: coreset and initial candidates ready
    InitialPoolEvaluating --> PoolReady: evidence persisted with provenance
    InitialPoolEvaluating --> Failed: no viable evidence or hard budget reached

    PoolReady --> SelectingWork: inner budget available
    SelectingWork --> AttemptExecuting: compatible issue selected
    SelectingWork --> MergeExecuting: eligible parent pair selected
    SelectingWork --> ChampionSelecting: budget or stop policy reached

    AttemptExecuting --> Validating: authorized workspace sealed
    AttemptExecuting --> AttemptRecorded: no-op, malformed, rejected, exhausted
    MergeExecuting --> Validating: child workspace sealed
    MergeExecuting --> AttemptRecorded: merge ineligible or unresolved

    Validating --> AttemptRecorded: terminal result assembled
    AttemptRecorded --> BarrierCommitting: sequential result or completed batch
    BarrierCommitting --> PoolReady: all staged records publish atomically
    BarrierCommitting --> Failed: durable transaction failure; no publication

    ChampionSelecting --> Completed: champion and manifest durable
    ChampionSelecting --> Failed: no eligible candidate
    Completed --> [*]
    Failed --> [*]
```

The state machine controls deterministic orchestration, budgets, and durable
records. It must not assume that an LLM response is deterministic or force model
reasoning into fixed branches.

## Attempt Lifecycle

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant I as Issue selector
    participant M as Edit memory
    participant E as Editor
    participant A as Adapter
    participant V as Evaluator
    participant S as Storage

    O->>I: select compatible evidence-backed work item
    I-->>O: read/write set, evidence refs, snapshot version
    O->>M: retrieve bounded redacted related history
    O->>E: issue, inventory, history refs, authorized scope
    E-->>O: reads plus proposed edit or abstention
    O->>A: materialize isolated parent workspace
    O->>A: apply authorized structured edits
    O->>V: validate origin, worked, regression cases
    V-->>O: provenance-bearing validation result
    O->>S: stage attempt and any candidate evidence
    O->>S: atomically commit or record rejection
```

The editor receives current artifact content only after requesting readable
artifact IDs. The adapter enforces the write set independently of the editor.

## LLM Output Recovery

```mermaid
flowchart TD
    R["Analyzer/editor response"] --> V{"Required machine fields valid?"}
    V -->|yes| A["Retain bounded rationale\nand continue authorized action"]
    V -->|repairable no| C["One correction request\nwith validation defect"]
    C --> V2{"Valid after correction?"}
    V2 -->|yes| A
    V2 -->|no| M["Record malformed response\ndiscard workspace"]
    V -->|uncertain or abstains| U["Record uncertainty\nno fabricated finding/edit"]
    M --> X["Apply scoped retry/exhaustion policy"]
    U --> X
```

The correction request contains no expected answer, evaluator internals, or
hidden labels. It describes only the missing/invalid interface field.

## Parallel Batch Protocol

```mermaid
sequenceDiagram
    participant C as Coordinator
    participant Snap as Immutable snapshot
    participant W1 as Worker 1
    participant W2 as Worker 2
    participant Tx as Storage transaction

    C->>Snap: create pool/history/budget snapshot v42
    C->>C: choose compatible issues and acquire write leases
    C->>W1: immutable work item + isolated workspace + leases
    C->>W2: immutable work item + isolated workspace + leases
    W1-->>C: immutable staged result
    W2-->>C: immutable staged result
    C->>C: verify leases, budgets, evidence, deterministic order
    C->>Tx: prepare all records and candidate snapshots
    alt every record validates and persists
        Tx-->>C: commit v43
        C->>C: release leases and refresh derived state at barrier
    else any record fails
        Tx-->>C: rollback all staged writes
        C->>C: release leases and record batch failure
    end
```

Workers may write only their isolated workspace. They never write shared pool,
history, score tensor, budgets, or manifest state. Entropy/clustering refreshes

## Failure Policy

| Failure | Required response |
| --- | --- |
| Malformed analyzer/editor result | One repair request, then record non-promotion; never fabricate a verdict/edit. |
| Rollout timeout/cancellation | Record status and provenance; retry only within budget; exclude unavailable score cells. |
| Evaluator unavailable | Record unavailable evidence; do not call it a task success/failure. |
| Embedding unavailable | Use configured deterministic lexical fallback only if profile permits; record fallback in manifest. |
| Write lease conflict | Reject selection before execution; a conflict during execution aborts the batch transaction. |
| Protected floor violation | Reject candidate regardless of aggregate gain; persist validation evidence. |
| Storage transaction failure | Publish no candidate/history/tensor update from that barrier. |
| Retry exhaustion | Retain scoped exhaustion state; reopen only after materially changed evidence. |
