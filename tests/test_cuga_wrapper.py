from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.core.errors import PersistenceSafetyError
from agent_evolve.core.trace import (
    CausalEvent,
    CausalTrace,
    PayloadLevel,
    ToolObservation,
)
from agent_evolve.cuga_wrapper import (
    CugaSdkRuntime,
    CugaWrapper,
    InMemoryRuntime,
    MockHarnessRuntime,
    RecordedEnvironmentReplayError,
    RuntimeSettings,
    ToolObservationRecorder,
    TraceConfig,
    TraceWriter,
)


def test_wrapper_runs_task_and_returns_json_serializable_trace():
    runtime = InMemoryRuntime(artifacts={"skills/default": "Use concise answers."})
    wrapper = CugaWrapper(runtime, RuntimeSettings(model="test-model"))

    trace = wrapper.run_task("task-1", {"input": "What is 2 + 2?"})

    assert trace == {
        "task_id": "task-1",
        "status": "success",
        "model": "test-model",
        "final_output": "Use concise answers.\n\nWhat is 2 + 2?",
        "events": [
            {"event_id": "task-1:started", "kind": "run_started"},
            {"event_id": "task-1:completed", "kind": "run_completed"},
        ],
    }


def test_wrapper_reads_and_updates_opaque_artifacts():
    runtime = InMemoryRuntime(artifacts={"skills/default": "original"})
    wrapper = CugaWrapper(runtime, RuntimeSettings(model="test-model"))

    assert wrapper.get_artifacts() == {"skills/default": "original"}
    wrapper.update_artifact("skills/default", "updated")

    assert wrapper.get_artifacts() == {"skills/default": "updated"}


def test_runtime_settings_load_litellm_environment_without_exposing_api_key(monkeypatch):
    monkeypatch.delenv("CUGA_MODEL", raising=False)
    monkeypatch.delenv("CUGA_BASE_URL", raising=False)
    monkeypatch.delenv("CUGA_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_MODEL", "test/model")
    monkeypatch.setenv("LITELLM_API_KEY", "secret")

    settings = RuntimeSettings.from_env()

    assert settings.base_url == "http://localhost:4000"
    assert settings.model == "test/model"
    assert settings.api_key == "secret"
    assert "secret" not in settings.public_config()


def test_runtime_settings_configures_cuga_openai_compatibility_environment(monkeypatch):
    settings = RuntimeSettings(
        model="openai/azure/test-model",
        base_url="https://gateway.example/v1",
        api_key="secret",
    )

    settings.configure_cuga_environment()

    assert __import__("os").environ["AGENT_SETTING_CONFIG"] == "settings.openai.toml"
    assert __import__("os").environ["MODEL_NAME"] == "azure/test-model"
    assert __import__("os").environ["OPENAI_BASE_URL"] == "https://gateway.example/v1"
    assert __import__("os").environ["OPENAI_API_KEY"] == "secret"


def test_cuga_sdk_runtime_captures_invoke_result_as_json_trace():
    class FakeResult:
        answer = "four"
        error = None
        thread_id = "thread-1"
        tool_calls = [{"name": "calculator", "result": "4"}]

    class FakeAgent:
        async def invoke(self, message, *, track_tool_calls):
            assert message == "What is 2 + 2?"
            assert track_tool_calls is True
            return FakeResult()

        async def aclose(self):
            return None

    runtime = CugaSdkRuntime(agent_factory=lambda config, workspace_dir=None: FakeAgent())

    trace = runtime.run_task("task-1", {"input": "What is 2 + 2?"})

    assert trace == {
        "task_id": "task-1",
        "status": "success",
        "final_output": "four",
        "harness_version": "unversioned",
        "active_artifacts": {"instructions": [], "skills": [], "memory": [], "tools": [], "policies": []},
        "unavailable_artifacts": {},
        "events": [
            {"event_id": "task-1:tool:0", "kind": "tool_call", "tool_call": {"name": "calculator", "result": "4"}},
        ],
    }


def test_cuga_sdk_runtime_does_not_relabel_opaque_artifacts_as_instructions():
    received_configs = []

    class FakeResult:
        answer = "done"
        error = None
        tool_calls = []

    class FakeAgent:
        async def invoke(self, message, *, track_tool_calls):
            return FakeResult()

        async def aclose(self):
            return None

    runtime = CugaSdkRuntime(agent_factory=lambda config, workspace_dir=None: received_configs.append(config) or FakeAgent())

    runtime.run_task("task-1", {"input": "answer"})

    assert runtime.get_artifacts() == {}
    assert received_configs == [{}]


def test_mock_harness_runtime_reports_active_artifacts_and_applies_mock_tool():
    wrapper = CugaWrapper(MockHarnessRuntime(), RuntimeSettings(model="mock-model"))

    trace = wrapper.run_task(
        "task-1",
        {
            "version": "b1-v2",
            "instructions": "Answer in one sentence.",
            "skills": {"retrieval": "Use the catalog."},
            "memory": {"city": "Paris"},
            "tools": [{"name": "lookup", "when": "lookup", "result": "catalog-result"}],
            "policies": {"style": "concise"},
            "input": "lookup the catalog",
        },
    )

    assert trace["harness_version"] == "b1-v2"
    assert trace["active_artifacts"] == {
        "instructions": ["instructions"],
        "skills": ["retrieval"],
        "memory": ["city"],
        "tools": ["lookup"],
        "policies": ["style"],
    }
    assert trace["final_output"] == "catalog-result"
    assert trace["events"][-2]["tool_call"]["name"] == "lookup"


def test_mock_harness_runtime_exposes_configured_memory_to_recall_tasks():
    wrapper = CugaWrapper(MockHarnessRuntime(), RuntimeSettings(model="mock-model"))

    trace = wrapper.run_task(
        "task-1",
        {
            "version": "b1-v2",
            "instructions": "",
            "skills": {},
            "memory": {"capital": "Paris"},
            "tools": [],
            "policies": {},
            "input": "recall capital",
        },
    )

    assert trace["final_output"] == "Paris"
    assert trace["events"][-2] == {"event_id": "task-1:memory:capital", "kind": "memory_recalled"}


def test_cuga_sdk_runtime_materializes_full_harness_and_reports_active(tmp_path):
    received = []

    class FakeResult:
        answer = "done"
        error = None
        tool_calls = []

    class FakeAgent:
        async def invoke(self, message, *, track_tool_calls):
            return FakeResult()

        async def aclose(self):
            return None

    tool = object()
    runtime = CugaSdkRuntime(
        agent_factory=lambda config, workspace_dir=None: received.append((config, workspace_dir)) or FakeAgent(),
        workspace_root=tmp_path,
    )

    trace = runtime.run_task(
        "task-1",
        {
            "version": "b1-v2",
            "instructions": "Use exact language.",
            "skills": {"retrieval": "Use the catalog."},
            "memory": {"fact": "known"},
            "tools": [tool],
            "policies": {"style": "concise"},
            "input": "answer",
        },
    )

    config, workspace_dir = received[0]
    assert config == {
        "instructions": "Use exact language.",
        "tools": [tool],
        "skills": {"retrieval": "Use the catalog."},
        "memory": {"fact": "known"},
        "policies": {"style": "concise"},
    }
    assert workspace_dir is not None
    assert (Path(workspace_dir) / "skills" / "retrieval" / "SKILL.md").exists()
    assert (Path(workspace_dir) / "playbooks" / "style.md").exists()
    assert (Path(workspace_dir) / "memory" / "fact.md").exists()
    assert trace["active_artifacts"]["instructions"] == ["instructions"]
    assert trace["active_artifacts"]["tools"] == ["tool-0"]
    assert trace["active_artifacts"]["skills"] == ["retrieval"]
    assert trace["active_artifacts"]["memory"] == ["fact"]
    assert trace["active_artifacts"]["policies"] == ["style"]
    assert trace["unavailable_artifacts"] == {}


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

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["task_id"] == "task-1"
    assert manifest["status"] == "success"


def _make_trace_with_payload(payload):
    return CausalTrace(
        run_id="run-1",
        task_id="task-1",
        status="success",
        final_output="answer",
        events=(CausalEvent(event_id="e1", sequence=0, kind="tool_update", payload=payload),),
    )


def test_trace_write_rejects_nested_credentials(tmp_path):
    trace = _make_trace_with_payload({"nested": {"api_key": "secret"}})

    with pytest.raises(PersistenceSafetyError, match="credential"):
        TraceWriter(TraceConfig(enabled=True, output_root=tmp_path)).write(trace)

    assert list(tmp_path.iterdir()) == []


def test_trace_write_rejects_secret_value_pattern(tmp_path):
    trace = _make_trace_with_payload({"note": "token sk-abcdefghijklmnopqrstuvwxyz123456"})

    with pytest.raises(PersistenceSafetyError):
        TraceWriter(TraceConfig(enabled=True, output_root=tmp_path)).write(trace)

    assert list(tmp_path.iterdir()) == []


def test_trace_config_rejects_non_positive_max_observation_bytes():
    with pytest.raises(ValueError, match="max_observation_bytes"):
        TraceConfig(enabled=True, max_observation_bytes=0)


def test_trace_config_rejects_non_positive_max_events_per_trace():
    with pytest.raises(ValueError, match="max_events_per_trace"):
        TraceConfig(enabled=True, max_events_per_trace=-1)


def test_trace_config_rejects_raw_opt_in_without_explicit_allow():
    with pytest.raises(ValueError, match="raw_opt_in"):
        TraceConfig(enabled=True, payload_level=PayloadLevel.RAW_OPT_IN)


def test_trace_config_accepts_raw_opt_in_with_explicit_allow():
    config = TraceConfig(
        enabled=True,
        payload_level=PayloadLevel.RAW_OPT_IN,
        allow_raw_payloads=True,
    )
    assert config.payload_level is PayloadLevel.RAW_OPT_IN


class FakeTool:
    def __init__(self, name, handler):
        self.name = name
        self._handler = handler

    def invoke(self, input_obj):
        try:
            return self._handler(**input_obj)
        except TypeError:
            return self._handler(input_obj)


def make_observation(sequence=0, tool_name="lookup"):
    return ToolObservation(
        sequence=sequence,
        tool_name=tool_name,
        canonical_arguments='{"q":"x"}',
        result={"value": "x"},
        original_bytes=10,
        retained_bytes=10,
        content_digest="sha256:abc",
        replay_eligible=True,
    )


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


def test_replay_returns_recorded_result_on_exact_match():
    recorder = ToolObservationRecorder.replay([make_observation(sequence=0, tool_name="lookup")])

    assert recorder.replay_tool_call(sequence=0, tool_name="lookup", arguments={"q": "x"}) == {"value": "x"}


def test_replay_fails_closed_on_argument_mismatch():
    recorder = ToolObservationRecorder.replay([make_observation(sequence=0, tool_name="lookup")])

    with pytest.raises(RecordedEnvironmentReplayError, match="mismatch"):
        recorder.replay_tool_call(sequence=0, tool_name="lookup", arguments={"q": "y"})


def test_recorder_records_error_and_re_raises():
    recorder = ToolObservationRecorder(TraceConfig(enabled=True))

    def failing(query):
        raise RuntimeError("boom")

    wrapped = recorder.wrap(FakeTool(name="lookup", handler=failing))

    with pytest.raises(RuntimeError, match="boom"):
        wrapped.invoke({"query": "Paris"})

    assert recorder.observations[0].error is not None
    assert recorder.observations[0].replay_eligible is False


class FakeResult:
    def __init__(self, answer="done", error=None, tool_calls=None):
        self.answer = answer
        self.error = error
        self.tool_calls = tool_calls or []


def read_manifest(result):
    return json.loads((Path(result["causal_trace_path"]) / "manifest.json").read_text())


def test_sdk_runtime_injects_wrapper_thread_id_and_reports_no_checkpointer(tmp_path):
    captured = {}

    class FakeAgent:
        graph = object()

        async def invoke(self, message, *, thread_id, track_tool_calls):
            captured["thread_id"] = thread_id
            return FakeResult(answer="done")

        async def aclose(self):
            pass

    runtime = CugaSdkRuntime(
        lambda config, workspace_dir=None: FakeAgent(),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    )
    result = runtime.run_task("task-1", {"input": "answer"})
    manifest = read_manifest(result)

    assert captured["thread_id"] == manifest["thread_id"]
    assert manifest["thread_id_source"] == "wrapper_generated_injected"
    assert manifest["capabilities"]["graph_history"]["status"] == "unavailable_no_checkpointer"


def test_disabled_stream_capture_is_distinct_from_missing_sdk_stream(tmp_path):
    class FakeAgentWithoutStream:
        async def invoke(self, message, *, track_tool_calls):
            return FakeResult(answer="done")

        async def aclose(self):
            pass

    config = TraceConfig(enabled=True, output_root=tmp_path, capture_stream_events=False)
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: FakeAgentWithoutStream(),
        trace_config=config,
    ).run_task("task-1", {"input": "x"})
    manifest = read_manifest(result)

    assert manifest["capabilities"]["stream_events"]["status"] == "disabled_by_config"


def test_sdk_runtime_reports_missing_stream_as_unavailable_no_sdk_surface(tmp_path):
    class FakeAgentWithoutStream:
        async def invoke(self, message, *, track_tool_calls):
            return FakeResult(answer="done")

        async def aclose(self):
            pass

    config = TraceConfig(enabled=True, output_root=tmp_path, capture_stream_events=True)
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: FakeAgentWithoutStream(),
        trace_config=config,
    ).run_task("task-1", {"input": "x"})
    manifest = read_manifest(result)

    assert manifest["capabilities"]["stream_events"]["status"] == "unavailable_no_sdk_surface"


def test_sdk_runtime_uses_public_invoke_with_thread_id_and_tool_tracking():
    class RecordingAgent:
        def __init__(self):
            self.invocations = []
            self.thread_id = None

        async def invoke(self, message, *, thread_id=None, track_tool_calls=False):
            self.thread_id = thread_id
            self.invocations.append((message, thread_id, track_tool_calls))
            return FakeResult(answer="done")

        async def aclose(self):
            pass

    agent = RecordingAgent()
    runtime = CugaSdkRuntime(
        lambda config, workspace_dir=None: agent,
        trace_config=TraceConfig(enabled=True),
    )

    runtime.run_task("task-1", {"input": "answer"})

    assert agent.thread_id is not None
    assert agent.invocations == [("answer", agent.thread_id, True)]
