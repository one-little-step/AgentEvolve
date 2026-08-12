# CUGA Causal Tracing Design

## Status And Phase Gate

**Phase:** 7, CUGA wrapper.

**Prerequisites:** Phases 1 through 6 must have passing tests for their binding
requirements before any production implementation changes begin. This design
does not authorize implementation until that gate is satisfied. It follows the
hard phase gate in `docs/superpowers/specs/2026-08-12-architecture-enforcement-design.md`.

This design specifies a wrapper-owned, persisted causal trace for CUGA runs. It
does not infer CUGA capabilities from generic LangGraph knowledge, and it does
not authorize an adapter to claim replay merely because trace data exists.

## Goals

- Persist ordered, inspectable rollout evidence for causal analysis.
- Record raw observations at supported tool boundaries for deterministic
  recorded-environment validation replay.
- Provide both split per-rollout records and an optional portable single-file
  export.
- Keep the generic adapter contract minimal and agent-neutral.
- Make disabled, unavailable, captured, truncated, and withheld data explicit.
- Preserve the existing wrapper return shape when tracing is disabled.

## Non-Goals

- Attaching a LangGraph checkpointer or claiming checkpoint reconstruction.
- Treating `track_tool_calls=True` as a complete causal trace.
- Treating recorded-environment replay as counterfactual checkpoint replay.
- Establishing causal effects without LLM determinism controls or repeated
  rollout evidence.
- Making tracker, Langfuse, or OpenLit a required local service.

## Architecture Boundary

`src/agent_evolve/core/contracts.py` remains the minimal adapter-facing
contract. Its `ExecutionTrace`, `TraceEvent`, `CheckpointDescriptor`, and
`EvolutionAdapter` methods remain unchanged.

`src/agent_evolve/core/trace.py` will define the versioned persisted schema:

```text
CausalTrace
  manifest and provenance
  ordered persisted events
  state snapshots, when genuinely available
  tool observations
  trace capabilities
  persistence and redaction reports
```

The wrapper writes `CausalTrace`. A mapping from the persisted trace to the
existing `ExecutionTrace` returns only the minimal adapter view. The CUGA
adapter must not return `CausalTrace` through `capture_trace()`. It may include
only replay-safe checkpoint IDs in `ExecutionTrace.checkpoint_ids`.

## Trace Configuration

`TraceConfig` belongs in `agent_evolve.cuga_wrapper` and is set by the
hardcoded runner configuration. Its defaults favor internal causal analysis
without silently storing unapproved raw data.

```text
enabled: false
output_root: data/traces
write_split_files: true
write_self_contained_export: true
capture_stream_events: true
capture_graph_final_state: true
capture_graph_history: true
capture_tool_observations: true
capture_external_correlation: true
payload_level: causal_sufficient
max_observation_bytes: 1048576
high_risk_tool_allowlist: ()
recorded_environment_replay: fail_closed
```

Each facility has an independent setting. The persisted capability report must
distinguish `disabled_by_config` from unavailable SDK/runtime facilities.

### Payload Levels

All levels pass through the recursive fail-closed redaction gateway. The table
defines the maximum payload that can be considered for persistence; prohibited
content remains rejected at every level.

| Field | `structural` | `causal_sufficient` | `raw_opt_in` |
| --- | --- | --- | --- |
| Event kind, sequence, timing, status | Persist | Persist | Persist |
| Node and phase identifiers | Persist | Persist | Persist |
| Tool name and canonicalized arguments | Persist | Persist | Persist |
| Tool result | Omit | Persist after approved redaction | Persist after approved redaction |
| Model decision/reasoning content | Omit | Persist after approved redaction | Persist after approved redaction |
| Model prompts | Omit | Omit | Persist after approved redaction |
| Graph-state values | Omit | Persist after approved redaction | Persist after approved redaction |
| Raw event and trace bodies | Omit | Omit | Persist after approved redaction |

`raw_opt_in` requires explicit operator configuration. `causal_sufficient` is
failures while still applying the persistence boundary.

## Redaction And Tool Safety

Every trace payload, manifest value, filename-derived metadata value, and
self-contained export passes through the recursive, fail-closed redaction
gateway required by `docs/architecture/storage-and-transactions.md`.

The gateway recursively rejects credentials, expected answers, evaluator
internals, labels, regexes, unapproved raw model payloads, and unapproved raw
trace bodies. A rejected payload aborts the trace write; it must not be silently
removed or represented as a successful sanitized trace. The trace includes a
redaction report only when a permitted bounded transformation was completed.

Tool arguments and results are collected raw for normal supported tools, as
requested for recorded-environment replay. A tool categorized as a grader,
verifier, evaluator, or other high-risk source may persist results only when its
identity appears in `high_risk_tool_allowlist`. Otherwise its result is recorded
as `withheld_high_risk_tool_output` and is unavailable for replay.

### Observation Size Limit

`max_observation_bytes` defaults to 1 MiB per tool observation. The byte budget
applies after serialization to UTF-8 JSON and before persistence. An
over-limit observation records:

```text
truncated: true
original_bytes: <full serialized byte length>
retained_bytes: <persisted byte length>
content_digest: <digest of the complete serialized value>
```

Only the bounded prefix permitted by the payload policy is persisted. A
truncated or withheld result is not eligible for recorded-environment replay.
A replay that reaches such an observation fails closed before live tool I/O.

## Run Identity And Runtime Capability Evidence

The wrapper generates a unique `run_id` and uses the same value as the
wrapper-owned CUGA `thread_id`. Every public CUGA stream, invoke, and graph
configuration operation receives that ID consistently. The manifest records:

```text
run_id
thread_id
thread_id_source: wrapper_generated
task_id
harness_version
model configuration without credentials
CUGA package version
wrapper schema version
```

`TraceCapabilities` is detected for each run and persisted rather than inferred
from tests. It records public stream availability and observed behavior, graph
final-state availability, graph-history availability, active-checkpointer
evidence, tracked-tool-call availability, tracker state, observability
correlation IDs, and each facility's status and reason.

The wrapper records an external trace or span ID only when a verified public
CUGA, Langfuse, or OpenLit surface exposes a joinable ID for that run. Otherwise
the correlation facility is `unavailable_no_sdk_surface` or its observed runtime
failure reason.

## Collection Flow

1. Construct the runtime `TraceCapabilities` snapshot and establish `run_id` /
   `thread_id`.
2. Wrap each supported declared tool in `ToolObservationRecorder` before the
   CUGA agent is constructed.
3. Run the task through a verified public CUGA observation surface. Stream
   events are captured only after focused tests establish the installed SDK's
   public semantics and ordering.
4. Supplement observations with `InvokeResult.tool_calls` only when available;
   these are never represented as a complete causal trace.
5. Query verified public graph methods only for final state or history they
   actually expose.
6. Filter, bound, validate, and persist the trace. Trace persistence failure
   fails an enabled traced rollout rather than publishing incomplete evidence.

This phase does not attach a checkpointer. Graph state history is captured only
when CUGA's running graph already has an active checkpointer and the installed
SDK proves public history access. When absent, the manifest contains:

```text
graph_history: unavailable_no_checkpointer
checkpoints: []
```

Attaching a `MemorySaver`, SQLite saver, or other checkpointer is a separate
SDK-verification task governed by
`docs/architecture/cuga-adapter/sdk-verification-matrix.md`. A final graph state
is not a checkpoint history.

## Persisted Layout

When tracing is enabled, one rollout produces a directory under
`<output_root>/<run-id>/`:

```text
manifest.json
events.jsonl
checkpoints/<checkpoint-id>.json
observations/<sequence>-<tool-name>.json
causal-trace.json
```

`manifest.json` is always written and references each enabled split component.
`causal-trace.json` is produced only when `write_self_contained_export` is true.
Disabled output forms are declared in the manifest. Empty checkpoint directories
or sets are explicit when graph history is unavailable; they are never silently
omitted. When tracing is disabled, no trace directory is written and the
wrapper's existing normal return shape remains unchanged.

The phase-7 file layout is a worker-local trace staging format, not the future
transactional metadata barrier. Once Phase 2 storage exists and the phase gate
allows integration, these files become sanitized blobs written before the
coordinator commits their references as required by
`storage-and-transactions.md`.

## Recorded-Environment Replay

`ToolObservationRecorder` exposes a wrapper/runtime capability named
`supports_recorded_environment_replay()`. This capability is distinct from the
generic adapter's `supports_counterfactual_replay()`.

In record mode it stores the tool call sequence, tool identity, canonicalized
arguments, result/error, timing, and replay eligibility. In replay mode it
consumes one persisted observation at the expected sequence. A match requires:

```text
same sequence position
same tool name
same canonicalized arguments
replay-eligible complete recorded result
```

Canonicalization recursively normalizes mappings by sorted string keys,
preserves sequence order, normalizes scalar JSON values, rejects non-finite
floats, and serializes finite floats using the standard deterministic JSON
representation. It rejects unsupported values rather than creating an unstable
comparison key.

Missing, duplicate, exhausted, truncated, withheld, or mismatched observations
raise a replay failure before the underlying live tool is called. No fallback to
live I/O is allowed in fail-closed mode.

Recorded-environment replay fixes tool and external-observation variance only.
It does not eliminate LLM sampling variance and is not checkpoint replay.
Deterministic validation or causal counterfactual claims additionally require
verified LLM determinism controls, such as a supported fixed seed, temperature
control, or prompt-cache reuse, or multi-rollout averaging through the existing
rollout design. These controls belong to rollout/adapter configuration, not this
tracing layer.

`CugaAdapter.supports_counterfactual_replay()` remains `False` until a focused
test proves checkpoint reconstruction, state restoration, and branch execution
for the installed official SDK version.

## Error Handling

- A facility disabled by configuration records `disabled_by_config`.
- A missing public SDK surface records `unavailable_no_sdk_surface`.
- Missing active graph checkpointing records `unavailable_no_checkpointer`.
- A rejected payload or serialization failure aborts an enabled traced rollout.
- A tool error records a tool observation and affects terminal rollout status.
- A high-risk or oversized tool result is explicit but replay-ineligible.
- Trace files never contain credentials or other prohibited persistence
  categories.

## Verification Plan

Tests must be written before implementation and all commands must be captured
with `2>&1 | tee terminal_output/cuga-tracing/<name>.log`.

1. Validate persisted causal-trace models and their mapping to the minimal
   `ExecutionTrace` contract.
2. Test configuration switches, per-rollout directory creation, split-file
   output, optional self-contained export, and disabled-tracing compatibility.
3. Test every payload level against its explicit preservation table and test the
   explicit raw opt-in requirement.
4. Test recursive fail-closed rejection for nested and string-embedded
   prohibited material.
5. Test normal raw tool observation capture, high-risk withholding, size
   truncation metadata, and replay ineligibility for truncated/withheld values.
6. Test exact sequence/name/canonical-argument replay matching and each
   fail-closed mismatch path without real tool I/O.
7. Test wrapper-generated thread ID injection and manifest provenance.
8. Test runtime capability reporting for disabled stream capture, unavailable
   SDK surfaces, and `unavailable_no_checkpointer` with explicit empty
   checkpoint output.
9. Test that the CUGA adapter continues to report
   `supports_counterfactual_replay() is False` and exports no unverified
   checkpoint IDs.
10. After source/signature tests establish public behavior for the installed
    pinned SDK, run a live CUGA smoke test and record only facilities actually
    observed.

## Governing Sources

- `AGENTS.md`: core neutrality, SDK-only integration, no invented replay,
  redaction limits, test-first changes, and captured verification commands.
- `docs/architecture/target-rho-parallel-gepa.md`: optional replay and adapter
  capability boundary.
- `docs/architecture/storage-and-transactions.md`: recursive fail-closed
  redaction, future blob ordering, and transaction ownership.
- `docs/architecture/cuga-adapter/sdk-verification-matrix.md`: proof required
  before relying on streaming, graph, tracker, or checkpoint surfaces.
- `docs/superpowers/specs/2026-08-12-architecture-enforcement-design.md`:
  phase order and hard implementation gate.
