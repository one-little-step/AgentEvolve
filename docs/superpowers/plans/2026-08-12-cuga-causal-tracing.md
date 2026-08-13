# CUGA Causal Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable, capability-honest causal trace persistence and fail-closed recorded-environment tool replay to the CUGA wrapper.

**Architecture:** Persist a rich, agent-neutral `CausalTrace` separately from the existing minimal adapter-facing `ExecutionTrace`. The wrapper owns trace collection, payload filtering, tool recording, and per-rollout output; the CUGA adapter retains its minimal trace mapping and does not claim checkpoint replay without independently verified state reconstruction.

**Tech Stack:** Python 3.12+, Pydantic v2, official CUGA SDK, LangChain tools, pytest, JSON/JSONL filesystem artifacts.

## Research Capability Boundary

This Phase-7 tracing work enables ordered event evidence, captured supported
tool observations, tool/environment-fixed validation re-execution, and richer
inputs for a later Phase-3 analyzer/judge when the real adapter is introduced.
It does not enable branching from arbitrary agent state, changing a skill or
memory at a mid-execution decision point, or counterfactual replay from a
LangGraph checkpoint. Those require a separately verified active checkpointer,
state restoration, and branch execution protocol.

`CausalTrace` is persisted for later diagnosis by reference. This plan does not
implement analyzer/judge ingestion, score-tensor provenance, mechanism
embeddings, Ollama configuration, or verdict generation; those remain core
Phase-3 concerns tested first against a fake adapter.

### Payload Preservation Table

All payload levels pass through the binding recursive redaction gateway. The
table is the precise preservation policy used by Tasks 1-4.

| Field | `structural` | `causal_sufficient` | `raw_opt_in` |
| --- | --- | --- | --- |
| Event type, sequence, timestamps, status | Persist | Persist | Persist |
| Node and phase identifiers | Persist | Persist | Persist |
| Tool name and canonical arguments | Persist | Persist | Persist |
| Tool results | Omit | Persist after approved redaction | Persist after approved redaction |
| LLM reasoning content | Omit | Persist after approved redaction | Persist after approved redaction |
| LLM prompts | Omit | Omit | Persist after approved redaction |
| Graph-state values | Omit | Persist after approved redaction | Persist after approved redaction |
| Raw event and trace bodies | Omit | Omit | Persist after approved redaction |

## Global Constraints

- **Phase:** 7, CUGA wrapper. Do not begin production implementation until Phases 1-6 have passing tests for their binding requirements; record the gate evidence before Task 1.
- Read and cite `AGENTS.md`, `docs/architecture/README.md`, `docs/architecture/storage-and-transactions.md`, `docs/architecture/target-rho-parallel-gepa.md`, and `docs/architecture/cuga-adapter/sdk-verification-matrix.md` before implementation.
- `src/agent_evolve/core/` must remain agent-neutral and must never import CUGA, LangChain, or an adapter implementation.
- Use only verified public CUGA SDK behavior. Do not attach a LangGraph checkpointer in this increment.
- The existing CUGA adapter is Phase 8 prototype state. Do not modify it in this Phase-7 plan; a later Phase-8 plan must prove its SDK mapping and keep `supports_counterfactual_replay()` false until checkpoint reconstruction is verified.
- Every test, smoke run, and verification command must use `2>&1 | tee terminal_output/cuga-tracing/<name>.log`.
- Trace persistence must pass the recursive fail-closed redaction boundary. Never persist credentials, expected answers, evaluator internals, labels, regexes, or unapproved raw trace bodies.
- Tool arguments and normal tool results are captured for replay subject to the redaction boundary, high-risk tool withholding, and `max_observation_bytes` defaulting to `1048576`.
- `max_events_per_trace` defaults to `10000`. Event collection is append-only up to that cap; later events are not persisted, and the manifest records `events_truncated`, `captured_event_count`, and `dropped_event_count`. Any trace with dropped events is incomplete evidence and cannot support a complete causal trajectory claim.
- Truncated or withheld observations are replay-ineligible. Recorded-environment replay matches `(sequence, tool name, canonicalized arguments)` and fails closed before live tool I/O.
- `payload_level` defaults to `causal_sufficient`; `raw_opt_in` must require explicit configuration.
- Preserve the existing wrapper return shape and write no trace directory when tracing is disabled.

---

## File Structure

- Create: `src/agent_evolve/core/trace.py`
  - Agent-neutral persisted models, deterministic payload canonicalization, and conversion to the existing minimal `ExecutionTrace`.
- Create: `tests/test_trace.py`
  - Validation and conversion tests for the persisted trace schema only.
- Modify: `src/agent_evolve/cuga_wrapper/__init__.py`
  - `TraceConfig`, runtime capability reporting, recorder/replay machinery, CUGA collection hooks, and wrapper result trace reference.
- Modify: `tests/test_cuga_wrapper.py`
  - Fake-runtime and fake-agent tests for runtime tracing, storage, payload policy, tool replay, and CUGA capability honesty.
- Modify: `scripts/inference_run.py`
  - One hardcoded top-level tracing configuration block passed into the wrapper.
- Modify: `tests/test_inference_run.py`
  - Runner configuration and tracing-disabled compatibility tests.
- Modify: `docs/architecture/cuga-adapter/sdk-verification-matrix.md`
  - Record only actual verified public SDK tracing observations and their focused test evidence after a successful smoke run.

### Task 0: Verify the Phase Gate and SDK Baseline

**Files:**
- Read: `docs/superpowers/specs/2026-08-12-architecture-enforcement-design.md`
- Read: `docs/architecture/README.md`
- Read: `docs/architecture/storage-and-transactions.md`
- Read: `docs/architecture/cuga-adapter/sdk-verification-matrix.md`
- Create: `terminal_output/cuga-tracing/phase-gate.log`

**Interfaces:**
- Consumes: binding phase ordering and existing CUGA verification matrix.
- Produces: an evidence-backed go/no-go decision for Tasks 1-7.

- [ ] **Step 1: Verify every preceding phase's required focused suites and full suite pass**

Run:

```bash
mkdir -p terminal_output/cuga-tracing
uv run --extra dev pytest 2>&1 | tee terminal_output/cuga-tracing/phase-gate.log
```

Expected: all suite tests pass, and the log provides evidence that Phases 1-6 are unlocked. If the phase gate cannot be established from existing architecture-required tests, stop; do not edit production files.

- [ ] **Step 2: Record the installed SDK surface before relying on it**

Run:

```bash
uv run python -c "import cuga, inspect; from cuga import CugaAgent; print(getattr(cuga, '__version__', 'unknown')); print(inspect.signature(CugaAgent.invoke)); print(inspect.signature(CugaAgent.stream)); print(isinstance(getattr(CugaAgent, 'graph', None), property))" 2>&1 | tee terminal_output/cuga-tracing/sdk-surface-baseline.log
```

Expected: the installed version and exact public `invoke`, `stream`, and `graph` surface are recorded. If a surface is absent, later tasks must report it as unavailable rather than emulate it.

- [ ] **Step 3: Record the thread-ID injection decision from the inspected signature**

If `thread_id` is present in the inspected public `CugaAgent.invoke` signature,
inject the wrapper-generated `run_id` as `thread_id` and persist
`thread_id_source: "wrapper_generated_injected"`. If it is absent, do not pass
an unsupported keyword; persist `thread_id_source:
"wrapper_generated_not_injected"` and set the graph/checkpoint correlation
facility to `unavailable_no_sdk_surface`.

- [ ] **Step 4: Commit no code for a failed gate**

Do not create a commit if the gate is not proven. Record the blocking test output and return the unmet prerequisite to the user.

### Task 1: Add Agent-Neutral Persisted Trace Models

**Files:**
- Create: `src/agent_evolve/core/trace.py`
- Create: `tests/test_trace.py`

**Interfaces:**
- Consumes: `ExecutionTrace` and `TraceEvent` from `agent_evolve.core.contracts`.
- Produces: `PayloadLevel`, `CaptureStatus`, `FacilityCapability`, `TraceCapabilities`, `StateSnapshot`, `ToolObservation`, `CausalTrace`, `canonical_json()`, and `CausalTrace.to_execution_trace()`.

- [ ] **Step 1: Write failing validation and conversion tests**

```python
from agent_evolve.core.trace import CausalTrace, FacilityCapability, ToolObservation


def test_causal_trace_maps_only_minimal_adapter_fields():
    trace = CausalTrace(
        run_id="run-1",
        task_id="task-1",
        thread_id="run-1",
        thread_id_source="wrapper_generated",
        harness_version="h1",
        status="success",
        final_output="answer",
        events=(),
        checkpoints=(),
        tool_observations=(),
        capabilities={"graph_history": FacilityCapability(status="unavailable_no_checkpointer")},
    )

    minimal = trace.to_execution_trace(candidate_id="candidate-1", trace_id="rollout-1")

    assert minimal.trace_id == "rollout-1"
    assert minimal.candidate_id == "candidate-1"
    assert minimal.checkpoint_ids == ()


def test_tool_observation_rejects_replay_eligible_truncation():
    with pytest.raises(ValidationError, match="truncated"):
        ToolObservation(
            sequence=0,
            tool_name="lookup",
            canonical_arguments='{"q":"x"}',
            result="partial",
            truncated=True,
            original_bytes=2_000_000,
            retained_bytes=1_048_576,
            content_digest="sha256:abc",
            replay_eligible=True,
        )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_trace.py -q 2>&1 | tee terminal_output/cuga-tracing/trace-models-red.log
```

Expected: FAIL because `agent_evolve.core.trace` does not exist.

- [ ] **Step 3: Implement immutable Pydantic models and deterministic canonical JSON**

```python
class FacilityCapability(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal[
        "captured",
        "disabled_by_config",
        "unavailable_no_sdk_surface",
        "unavailable_no_checkpointer",
        "runtime_failure",
    ]
    reason: str | None = None


def canonical_json(value: object) -> str:
    normalized = _normalize_json_value(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
```

Define `ToolObservation` so `truncated`, `withheld_reason`, and `error` always make `replay_eligible=False`. Define `CausalTrace.to_execution_trace()` to convert persisted events into existing `TraceEvent` values and include only snapshots whose `replay_safe` field is true.

- [ ] **Step 4: Run focused tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_trace.py -q 2>&1 | tee terminal_output/cuga-tracing/trace-models-green.log
```

Expected: PASS.

- [ ] **Step 5: Inspect the schema increment without committing**

Run:

```bash
git diff --check
git status --short
```

Expected: only intended Phase-7 files and pre-existing user changes appear. Do not commit unless explicitly requested.

### Task 2: Add Trace Configuration, Payload Policy, and Trace Writer

**Files:**
- Modify: `src/agent_evolve/cuga_wrapper/__init__.py`
- Modify: `tests/test_cuga_wrapper.py`

**Interfaces:**
- Consumes: `CausalTrace`, `PayloadLevel`, and `FacilityCapability` from `agent_evolve.core.trace`.
- Produces: `TraceConfig`, `TraceWriter.write(trace) -> Path`, and an explicit wrapper result `causal_trace_path` only when tracing is enabled.

- [ ] **Step 1: Write failing storage and configuration tests**

```python
def test_disabled_tracing_writes_no_rollout_directory(tmp_path):
    wrapper = CugaWrapper(
        InMemoryRuntime(),
        RuntimeSettings(model="test-model"),
        trace_config=TraceConfig(enabled=False, output_root=tmp_path),
    )

    trace = wrapper.run_task("task-1", {"input": "hello"})

    assert "causal_trace_path" not in trace
    assert list(tmp_path.iterdir()) == []


def test_enabled_tracing_writes_manifest_split_files_and_export(tmp_path):
    wrapper = CugaWrapper(
        InMemoryRuntime(),
        RuntimeSettings(model="test-model"),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    )

    result = wrapper.run_task("task-1", {"version": "h1", "input": "hello"})
    output = Path(result["causal_trace_path"])

    assert (output / "manifest.json").is_file()
    assert (output / "events.jsonl").is_file()
    assert (output / "causal-trace.json").is_file()
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_cuga_wrapper.py -q 2>&1 | tee terminal_output/cuga-tracing/trace-writer-red.log
```

Expected: FAIL because `TraceConfig` and trace file output do not exist.

- [ ] **Step 3: Implement configuration and atomic per-rollout writing**

```python
@dataclass(frozen=True, slots=True)
class TraceConfig:
    enabled: bool = False
    output_root: Path = Path("data/traces")
    write_split_files: bool = True
    write_self_contained_export: bool = True
    capture_stream_events: bool = True
    capture_graph_final_state: bool = True
    capture_graph_history: bool = True
    capture_tool_observations: bool = True
    capture_external_correlation: bool = True
    payload_level: PayloadLevel = PayloadLevel.CAUSAL_SUFFICIENT
    max_observation_bytes: int = 1_048_576
    max_events_per_trace: int = 10_000
    high_risk_tool_allowlist: frozenset[str] = frozenset()
```

Reject non-positive `max_observation_bytes` and `max_events_per_trace`; reject `raw_opt_in` unless an explicit `allow_raw_payloads=True` field is supplied. Write JSON with sorted keys and UTF-8. Write into a sibling temporary directory, rename it only after every configured file validates, and remove the temporary directory on error. The writer always emits `manifest.json`; it emits `events.jsonl`, `checkpoints/`, `observations/`, and `causal-trace.json` only as configuration requires. Manifest capabilities must report disabled outputs as `disabled_by_config`.

- [ ] **Step 4: Add failing redaction policy tests, then implement the policy boundary**

```python
def test_trace_write_rejects_nested_credentials(tmp_path):
    trace = make_trace_with_payload({"nested": {"api_key": "secret"}})

    with pytest.raises(TracePersistenceError, match="credential"):
        TraceWriter(TraceConfig(enabled=True, output_root=tmp_path)).write(trace)

    assert list(tmp_path.iterdir()) == []
```

Implement an explicit fail-closed wrapper boundary that rejects prohibited values recursively in mappings, sequences, and strings. If a shared Phase-2 gateway exists when this plan executes, call it instead of duplicating its implementation. If it does not exist, stop implementation: Phase 7 cannot replace the binding Phase-2 persistence requirement with a local approximation.

- [ ] **Step 5: Run focused tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_trace.py tests/test_cuga_wrapper.py -q 2>&1 | tee terminal_output/cuga-tracing/trace-writer-green.log
```

Expected: PASS.

- [ ] **Step 6: Inspect the configuration and writer increment without committing**

Run:

```bash
git diff --check
git status --short
```

Expected: only intended Phase-7 files and pre-existing user changes appear. Do not commit unless explicitly requested.

### Task 3: Add Tool Observation Recording and Fail-Closed Replay

**Files:**
- Modify: `src/agent_evolve/cuga_wrapper/__init__.py`
- Modify: `tests/test_cuga_wrapper.py`

**Interfaces:**
- Consumes: `TraceConfig.max_observation_bytes`, `ToolObservation`, and `canonical_json()`.
- Produces: `ToolObservationRecorder.wrap(tool)`, `ToolObservationRecorder.replay_tool_call(...)`, and `supports_recorded_environment_replay() -> bool`.

- [ ] **Step 1: Write failing recorder, cap, and replay tests**

```python
def test_recorder_persists_raw_normal_tool_result_with_sequence():
    recorder = ToolObservationRecorder(TraceConfig(enabled=True))
    wrapped = recorder.wrap(FakeTool(name="lookup", handler=lambda query: {"value": query}))

    assert wrapped.invoke({"query": "Paris"}) == {"value": "Paris"}
    observation = recorder.observations[0]
    assert observation.sequence == 0
    assert observation.canonical_arguments == '{"query":"Paris"}'
    assert observation.result == {"value": "Paris"}
    assert observation.replay_eligible is True


def test_recorder_marks_oversized_result_truncated_and_replay_ineligible():
    recorder = ToolObservationRecorder(TraceConfig(enabled=True, max_observation_bytes=4))
    wrapped = recorder.wrap(FakeTool(name="lookup", handler=lambda _: "abcdefgh"))

    wrapped.invoke({"query": "Paris"})

    assert recorder.observations[0].truncated is True
    assert recorder.observations[0].replay_eligible is False


def test_replay_fails_closed_on_sequence_name_or_argument_mismatch():
    recorder = ToolObservationRecorder.replay([make_observation(sequence=0, tool_name="lookup")])

    with pytest.raises(RecordedEnvironmentReplayError, match="mismatch"):
        recorder.replay_tool_call(sequence=0, tool_name="other", arguments={})
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_cuga_wrapper.py -q 2>&1 | tee terminal_output/cuga-tracing/tool-recorder-red.log
```

Expected: FAIL because the recorder and replay error do not exist.

- [ ] **Step 3: Implement recorder wrapping without assuming CUGA internals**

```python
class ToolObservationRecorder:
    def replay_tool_call(self, *, sequence: int, tool_name: str, arguments: object) -> object:
        observation = self._next_replay_observation(sequence)
        if (
            observation.tool_name != tool_name
            or observation.canonical_arguments != canonical_json(arguments)
            or not observation.replay_eligible
        ):
            raise RecordedEnvironmentReplayError("recorded tool observation mismatch")
        return observation.result
```

Wrap only tool types whose documented invocation surface can be exercised by a fake. Preserve tool metadata, call the original exactly once in record mode, and record error/duration before re-raising. Categorize high-risk tools by an explicit recorder argument or a configured tool-name registry; if not allowlisted, store the withholding status and no replayable result. Do not wrap unknown tool representations; report them as uninstrumented in capabilities.

- [ ] **Step 4: Add and run canonicalization edge-case tests**

```python
def test_canonical_json_sorts_mapping_keys_and_preserves_list_order():
    assert canonical_json({"b": [2, 1], "a": {"y": 2, "x": 1}}) == '{"a":{"x":1,"y":2},"b":[2,1]}'


def test_canonical_json_rejects_non_finite_float():
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"value": float("nan")})
```

Run:

```bash
uv run --extra dev pytest tests/test_trace.py tests/test_cuga_wrapper.py -q 2>&1 | tee terminal_output/cuga-tracing/tool-recorder-green.log
```

Expected: PASS.

- [ ] **Step 5: Inspect the recorder increment without committing**

Run:

```bash
git diff --check
git status --short
```

Expected: only intended Phase-7 files and pre-existing user changes appear. Do not commit unless explicitly requested.

### Task 4: Integrate Capability-Honest CUGA Runtime Collection

**Files:**
- Modify: `src/agent_evolve/cuga_wrapper/__init__.py`
- Modify: `tests/test_cuga_wrapper.py`

**Interfaces:**
- Consumes: `TraceConfig`, `ToolObservationRecorder`, `CausalTrace`, and installed CUGA public method signatures established in Task 0.
- Produces: trace-aware `CugaSdkRuntime.run_task()` and wrapper-level `supports_recorded_environment_replay()`.

- [ ] **Step 1: Write failing fake-agent tests for thread provenance and capabilities**

```python
def test_sdk_runtime_injects_wrapper_thread_id_and_reports_no_checkpointer(tmp_path):
    captured = {}

    class FakeAgent:
        graph = object()

        async def invoke(self, message, *, thread_id, track_tool_calls):
            captured["thread_id"] = thread_id
            return FakeResult(answer="done", error=None, tool_calls=[])

        async def aclose(self):
            pass

    runtime = CugaSdkRuntime(lambda _: FakeAgent(), trace_config=TraceConfig(enabled=True, output_root=tmp_path))
    result = runtime.run_task("task-1", {"input": "answer"})
    manifest = json.loads((Path(result["causal_trace_path"]) / "manifest.json").read_text())

    assert captured["thread_id"] == manifest["thread_id"]
    assert manifest["thread_id_source"] == "wrapper_generated"
    assert manifest["capabilities"]["graph_history"]["status"] == "unavailable_no_checkpointer"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_cuga_wrapper.py -q 2>&1 | tee terminal_output/cuga-tracing/runtime-collection-red.log
```

Expected: FAIL because the runtime does not generate/inject thread IDs or write capability manifests.

- [ ] **Step 3: Implement only verified SDK paths and explicit unavailable states**

```python
run_id = str(uuid.uuid4())
thread_id = run_id
invoke_kwargs = {"track_tool_calls": True}
thread_id_source = "wrapper_generated_not_injected"
if "thread_id" in inspect.signature(agent.invoke).parameters:
    invoke_kwargs["thread_id"] = thread_id
    thread_id_source = "wrapper_generated_injected"
result = asyncio.run(agent.invoke(message, **invoke_kwargs))

capabilities["graph_history"] = FacilityCapability(
    status="unavailable_no_checkpointer",
    reason="no verified active checkpointer exposed by this runtime",
)
```

Use `agent.stream()` only after its fake-agent test proves the expected yielded shape and ordering. Normalize each supported streamed item into `{event_id, sequence, kind, actor_id, parent_event_id, timestamp, payload}`. `kind` may be `node_update`, `model_update`, `tool_update`, or `runtime_update` only when the public item identifies that category; otherwise use `runtime_update`. Preserve only payload fields permitted by the selected payload level. If stream semantics cannot be reduced to this stable normalized event, set `stream_events` to `unavailable_no_sdk_surface` or `runtime_failure` and retain terminal/invocation evidence. Stop accepting events after `max_events_per_trace`, set `events_truncated=true`, increment `dropped_event_count`, and retain the terminal result. Inspect `agent.graph` only through verified public methods. Capture a final graph state only when its value is safely serializable and payload policy allows it. Never synthesize checkpoint IDs, attach a checkpointer, call undocumented graph internals, or call graph history a replay-safe checkpoint chain.

- [ ] **Step 4: Test disabled versus unavailable facilities and failure propagation**

```python
def test_disabled_stream_capture_is_distinct_from_missing_sdk_stream(tmp_path):
    config = TraceConfig(enabled=True, output_root=tmp_path, capture_stream_events=False)
    result = CugaSdkRuntime(lambda _: FakeAgentWithoutStream(), trace_config=config).run_task("task-1", {"input": "x"})
    manifest = read_manifest(result)

    assert manifest["capabilities"]["stream_events"]["status"] == "disabled_by_config"
```

Run:

```bash
uv run --extra dev pytest tests/test_cuga_wrapper.py -q 2>&1 | tee terminal_output/cuga-tracing/runtime-collection-green.log
```

Expected: PASS.

- [ ] **Step 5: Inspect the runtime integration increment without committing**

Run:

```bash
git diff --check
git status --short
```

Expected: only intended Phase-7 files and pre-existing user changes appear. Do not commit unless explicitly requested.

### Task 5: Wire the Hardcoded Runner Configuration

**Files:**
- Modify: `scripts/inference_run.py`
- Modify: `tests/test_inference_run.py`

**Interfaces:**
- Consumes: `TraceConfig` from `agent_evolve.cuga_wrapper`.
- Produces: a top-level `TRACE_CONFIG` and construction of `CugaWrapper` with that configuration.

- [ ] **Step 1: Write failing runner configuration tests**

```python
def test_mock_runner_can_enable_trace_output_from_hardcoded_config(tmp_path):
    setattr(CONFIG, "TRACE_CONFIG", RUNNER.TraceConfig(enabled=True, output_root=tmp_path / "traces"))
    setattr(CONFIG, "USE_MOCK_RUNTIME", True)

    assert RUNNER.main() == 0
    assert list((tmp_path / "traces").iterdir())
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_inference_run.py -q 2>&1 | tee terminal_output/cuga-tracing/runner-config-red.log
```

Expected: FAIL because the runner does not define or pass `TRACE_CONFIG`.

- [ ] **Step 3: Add the configuration block without CLI arguments**

```python
TRACE_CONFIG = TraceConfig(
    enabled=False,
    output_root=Path("data/traces"),
    payload_level=PayloadLevel.CAUSAL_SUFFICIENT,
    max_observation_bytes=1_048_576,
)
```

Pass `trace_config=TRACE_CONFIG` to both mock and live wrapper construction. Do not add command-line parsing or read credentials into the configuration or output path.

- [ ] **Step 4: Run runner tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_inference_run.py -q 2>&1 | tee terminal_output/cuga-tracing/runner-config-green.log
```

Expected: PASS.

- [ ] **Step 5: Inspect the runner wiring without committing**

Run:

```bash
git diff --check
git status --short
```

Expected: only intended Phase-7 files and pre-existing user changes appear. Do not commit unless explicitly requested.

### Task 6: Verify Official SDK Behavior and Run End-to-End Checks

**Files:**
- Modify: `docs/architecture/cuga-adapter/sdk-verification-matrix.md`
- Create: `terminal_output/cuga-tracing/live-smoke.log`
- Create: `terminal_output/cuga-tracing/full-suite.log`

**Interfaces:**
- Consumes: completed wrapper implementation and installed CUGA SDK public surface.
- Produces: SDK verification record limited to actually observed behavior and captured verification evidence.

- [ ] **Step 1: Add a focused fake test for every public SDK method used**

```python
def test_sdk_runtime_uses_public_invoke_with_thread_id_and_tool_tracking():
    agent = FakeAgent()
    runtime = CugaSdkRuntime(lambda _: agent, trace_config=TraceConfig(enabled=True))

    runtime.run_task("task-1", {"input": "answer"})

    assert agent.invocations == [("answer", agent.thread_id, True)]
```

Run:

```bash
uv run --extra dev pytest tests/test_cuga_wrapper.py -q 2>&1 | tee terminal_output/cuga-tracing/sdk-contract-green.log
```

Expected: PASS.

- [ ] **Step 2: Run a live trace smoke test only if required environment variables are present**

Run:

```bash
uv run python -m scripts.inference_run 2>&1 | tee terminal_output/cuga-tracing/live-smoke.log
```

Expected: either a successful trace directory whose manifest reflects only observed SDK facilities, or a captured environmental/runtime failure. Do not print environment variable values or credentials. If stream/graph/checkpointer support is absent, retain the explicit unavailable capability state.

- [ ] **Step 3: Update the SDK verification matrix from observed evidence only**

Add a dated row or evidence entry containing:

```text
feature name
installed CUGA package version
public SDK call used
focused test file and test name
live smoke command log path
observed behavior
known limits
```

Do not state that graph history, checkpointer replay, tracker, Langfuse, or OpenLit work unless the smoke evidence proves the exact public behavior.

- [ ] **Step 4: Run the full suite and diff validation**

Run:

```bash
uv run --extra dev pytest 2>&1 | tee terminal_output/cuga-tracing/full-suite.log
git diff --check 2>&1 | tee terminal_output/cuga-tracing/diff-check.log
```

Expected: all tests pass and `git diff --check` exits zero.

- [ ] **Step 5: Inspect verified implementation and evidence documentation without committing**

Run:

```bash
git diff --check
git status --short
```

Expected: only intended Phase-7 files, generated verification logs, and pre-existing user changes appear. Do not commit unless explicitly requested. Do not stage credentials, live trace artifacts containing prohibited content, or unrelated worktree changes.

## Plan Self-Review

### Spec Coverage

- Phase gate: Task 0 verifies it before any production task. Phases 1-6 are implemented and tested against fakes; this Phase-7 plan cannot start earlier.
- Agent-neutral rich trace versus minimal adapter trace: Task 1; the later Phase-8 adapter plan will consume the reference.
- Configurable split and portable output: Task 2 and Task 5.
- Causal-sufficient/raw payload policy and explicit raw opt-in: Task 2. The exact payload table is in the governing design specification.
- Recursive fail-closed redaction: Task 2, with a stop condition if Phase-2 shared gateway is absent.
- Raw normal tool capture, high-risk withholding, 1 MiB cap, and truncation: Task 3.
- Sequence-aware deterministic replay and fail-closed behavior: Task 3.
- Wrapper-owned thread provenance and runtime capability state: Task 4.
- No checkpointer attachment and no checkpoint replay claim: Task 4. The CUGA adapter is deliberately deferred to a separate Phase-8 plan.
- LLM variance limit: global constraint and trace model documentation; no false determinism mechanism is planned.
- Official SDK proof and full verification: Task 6.

### Placeholder Scan

The plan contains no deferred implementation instructions. Where an existing Phase-2 redaction gateway is required but unavailable, it explicitly requires stopping rather than inventing a substitute.

### Type Consistency

- Task 1 defines the persisted models and `canonical_json()` consumed by Tasks 2-4.
- Task 2 defines `TraceConfig` consumed by Tasks 3-5.
- Task 3 defines `ToolObservationRecorder` consumed by Task 4.
- Task 4 produces the optional wrapper `causal_trace_path` for a later Phase-8 adapter plan.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-12-cuga-causal-tracing.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using `executing-plans`, batch execution with checkpoints.
