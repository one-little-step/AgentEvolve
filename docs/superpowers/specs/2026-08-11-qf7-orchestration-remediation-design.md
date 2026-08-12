# QF7 Orchestration Remediation Design

## Goal

Remove the verified qf7 control-flow and redaction defects before resuming the
separate DPP architecture-completion work.

## Scope

This increment addresses exactly five defects:

1. Parallel candidate admission must record real validation evidence in the
   candidate score tensor.
2. Reconstructed issues without retained trace evidence must not fabricate
   blame nodes or reach the editor.
3. Both sequential and parallel edit paths must reject edits outside the
   `EditorRequest.write_set` before workspace mutation or lease acquisition.
4. Minimal evaluation without a declared expected substring must not be scored
   as a success.
5. Edit-memory sanitization must recursively reject denylisted nested keys and
   sensitive string markers.

`core/storage.py`, `core/config.py`, adapter behavior, and full unavailable
attempt persistence are outside this increment. The current runtime records
cannot represent a durable unavailable evaluation status; this increment makes
missing minimal contracts fail closed with score `0.0` and no fabricated blame.

## Design

### Parallel Evidence

Each accepted parallel worker already has a focused validation report, a task,
an issue ID, and an actual post-edit rollout trace. Preserve those values in
the staged worker result. At barrier admission, create the candidate and record
one score cell per validation result using the actual task, trace ID, score,
and mechanism cluster parsed from the issue ID. The associated analysis has an
empty blame graph and represents only measured validation output; it does not
claim causal attribution.

### Evidence-Free Issues

The prior code reconstructed failed issues from score tensors after discarding
their trace analysis, then created a fake `agent` blame node. Instead, the
orchestrator records that evidence is insufficient by leaving the graph empty
and excludes the issue from editor dispatch. It never constructs an artifact
target from missing causal data.

### Write Authorization

Immediately after `editor.propose_edit(request)`, validate every returned
`ArtifactEdit.artifact_id` against `request.write_set`. Raise
`WriteAuthorizationError` before `apply_structured_edits()` in the sequential
path and before clash detection or lease acquisition in the parallel path.

### Minimal Evaluation

Minimal scoring uses `expected_substring` only when it is explicitly present.
Absent contracts yield score `0.0`, an empty blame graph, and an
`insufficient_evidence` mechanism marker. This prevents the empty string from
being interpreted as proof of success.

### Recursive Redaction

`sanitize_payload` accepts arbitrary payload values and recursively traverses
mappings and non-string sequences. It fails closed on denylisted keys at any
depth and on case-insensitive sensitive markers in string values. Clean values
are reconstructed without mutating their input.

## Tests

- A parallel accepted candidate has a non-empty score tensor with at least one
  recorded rollout.
- An editor edit outside the declared write set raises `WriteAuthorizationError`
  before the adapter observes a mutation, for both sequential and parallel
  paths.
- Score-tensor reconstruction creates no synthetic blame nodes and does not
  invoke the editor for evidence-free issues.
- Minimal rollout with no expected substring is not a success.
- Nested mappings, sequences, and sensitive strings are rejected by the memory
  sanitizer; nested clean values retain their shape.

## Constraints

- No synthetic causal events or artifact attribution.
- All new behavior uses tests first.
- Capture verification commands with `2>&1 | tee terminal_output/...`.
- Do not persist expected answers, evaluator internals, labels, regexes, or
  credentials.
