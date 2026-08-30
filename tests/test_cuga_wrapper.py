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
    GraphEventCollector,
    InMemoryRuntime,
    MockHarnessRuntime,
    PACKAGED_MODEL_SETTINGS_PATH,
    RecordedEnvironmentReplayError,
    RuntimeSettings,
    ToolObservationRecorder,
    TraceConfig,
    TraceWriter,
    materialize_harness,
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

    assert __import__("os").environ["AGENT_SETTING_CONFIG"] == str(PACKAGED_MODEL_SETTINGS_PATH)
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


def test_materialized_playbook_uses_matchable_triggers(tmp_path):
    """A materialized playbook must carry triggers CUGA actually evaluates.

    Verified against cuga 0.3.1: ``PolicyAgent.match_policy`` builds candidates
    only from the keyword evaluator (which filters ``KeywordTrigger``) and the
    natural-language evaluator. Nothing selects an ``AlwaysTrigger``, so a
    playbook written with only ``always: true`` loads and deserializes but can
    never match, making the policy artifact inert and unoptimizable.

    Live evidence: ``always`` -> ``matched=False``; ``keywords`` and
    ``natural_language`` -> ``matched=True`` with guidance injected.
    """
    workspace_dir = materialize_harness(
        {"policies": {"style": "Always answer in one sentence."}},
        tmp_path / "workspace",
    )
    assert workspace_dir is not None

    body = (Path(workspace_dir) / "playbooks" / "style.md").read_text(encoding="utf-8")

    # Required by CUGA's filesystem_sync, which deletes policies whose
    # frontmatter id is absent from disk.
    assert "id: playbook_style" in body

    # The frontmatter key is plural; ``keyword:`` yields zero triggers and CUGA
    # rejects the file with "must have at least one trigger".
    assert "keywords:" in body or "natural_language:" in body

    # An always-only playbook is inert, so it must not be the sole trigger.
    trigger_section = body.split("---")[1]
    assert not (
        "always: true" in trigger_section
        and "keywords:" not in trigger_section
        and "natural_language:" not in trigger_section
    ), "playbook relies solely on an always trigger, which never matches"


def test_materialized_playbook_frontmatter_is_valid_yaml_with_colons(tmp_path):
    """Policy text containing ``:`` must not corrupt the playbook frontmatter.

    Regression guard: policy bodies routinely contain a colon (for example
    "end your reply with the exact line: MARKER"). When that text is copied into
    a YAML trigger as an unquoted scalar, CUGA rejects the entire file with
    "Invalid YAML in frontmatter: mapping values are not allowed here" and the
    policy is silently dropped, so the artifact appears configured but has no
    effect on the agent.
    """
    yaml = pytest.importorskip("yaml")

    workspace_dir = materialize_harness(
        {
            "policies": {
                "status-format": (
                    "When reporting status, you MUST end your reply with the "
                    "exact line: POLICY-MARKER: ABC-123"
                )
            },
        },
        tmp_path / "workspace",
    )
    assert workspace_dir is not None

    body = (Path(workspace_dir) / "playbooks" / "status-format.md").read_text(encoding="utf-8")
    frontmatter = body.split("---")[1]

    parsed = yaml.safe_load(frontmatter)
    assert parsed["id"] == "playbook_status-format"
    assert parsed["triggers"]["natural_language"], "NL trigger must survive parsing"


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


# --------------------------------------------------------------------------- #
# CUGA settings snapshot at the SDK boundary (S4-11 follow-up)
# --------------------------------------------------------------------------- #
def test_settings_snapshot_reads_live_cuga_settings(monkeypatch):
    """The snapshot reads the LIVE cuga settings object, not os.environ:
    .env presence does not prove CUGA consumed the value."""
    import agent_evolve.cuga_wrapper as cw

    class _FakeAdvanced:
        force_autonomous_mode = True
        enable_shell_tool = True

    class _FakeSkills:
        enabled = True

    class _FakeKnowledge:
        enabled = True

    class _FakeSettings:
        advanced_features = _FakeAdvanced()
        skills = _FakeSkills()
        knowledge = _FakeKnowledge()

    import types as _types
    fake = _types.SimpleNamespace()
    fake.settings = _FakeSettings()
    monkeypatch.setattr(
        cw, "_import_cuga_config", lambda: _FakeSettings()
    )
    snap = cw.cuga_settings_snapshot()
    assert snap == {
        "force_autonomous_mode": True,
        "enable_shell_tool": True,
        "skills_enabled": True,
        "knowledge_enabled": True,
    }


def test_settings_snapshot_tolerates_missing_surfaces(monkeypatch):
    """A settings object without a knowledge section snapshots what exists."""
    import agent_evolve.cuga_wrapper as cw

    class _FakeSettings:
        class advanced_features:
            force_autonomous_mode = False
            enable_shell_tool = True

        class skills:
            enabled = False
        # no knowledge attribute at all

    monkeypatch.setattr(cw, "_import_cuga_config", lambda: _FakeSettings())
    snap = cw.cuga_settings_snapshot()
    assert snap["force_autonomous_mode"] is False
    assert snap["skills_enabled"] is False
    assert snap["knowledge_enabled"] is None  # explicit absence, not False


def test_manifest_carries_settings_snapshot(tmp_path):
    """Every trace manifest binds the env it ran under."""
    wrapper = CugaWrapper(
        InMemoryRuntime(),
        RuntimeSettings(model="test-model"),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    )
    result = wrapper.run_task("task-1", {"version": "h1", "input": "hello"})
    manifest = json.loads(
        (Path(result["causal_trace_path"]) / "manifest.json").read_text()
    )
    snap = manifest["cuga_settings"]
    assert set(snap) == {
        "force_autonomous_mode", "enable_shell_tool", "skills_enabled", "knowledge_enabled",
    }


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


# --------------------------------------------------------------------------- #
# Windows rename-retry guard (run 3, task-04 lost to WinError 5)
# --------------------------------------------------------------------------- #
def test_trace_write_survives_a_transient_rename_lock(tmp_path, monkeypatch):
    """First rename raises Windows 'access denied', retry succeeds.

    Measured live 2026-08-30: the atomic staging->final rename failed with
    PermissionError [WinError 5] (AV/indexer lock) and destroyed a PAID
    rollout's measurement. A momentary lock must not cost a rollout.
    """
    from agent_evolve.cuga_wrapper import _RENAME_MAX_ATTEMPTS

    writer = TraceWriter(TraceConfig(enabled=True, output_root=tmp_path))
    trace = _make_trace_with_payload({"ok": True})
    real_replace = type(tmp_path).replace
    calls = {"n": 0}

    def flaky_replace(self, target):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(type(tmp_path), "replace", flaky_replace)
    import agent_evolve.cuga_wrapper as cw_pkg

    monkeypatch.setattr(cw_pkg, "_RENAME_BACKOFF_S", 0.0)

    out = writer.write(trace)

    assert calls["n"] == 2
    assert (out / "manifest.json").is_file()


def test_trace_write_rename_exhaustion_uses_recovery_fallback(
    tmp_path, monkeypatch
):
    """If the lock never clears, salvage the payload instead of losing it."""
    from agent_evolve.cuga_wrapper import _RENAME_MAX_ATTEMPTS

    writer = TraceWriter(TraceConfig(enabled=True, output_root=tmp_path))
    trace = _make_trace_with_payload({"ok": True})
    real_replace = type(tmp_path).replace

    def always_locked(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(type(tmp_path), "replace", always_locked)
    import agent_evolve.cuga_wrapper as cw_pkg

    monkeypatch.setattr(cw_pkg, "_RENAME_BACKOFF_S", 0.0)

    out = writer.write(trace)

    # The fallback: content salvaged (renamed, or staging left in place when
    # even that is locked) -- never deleted, never raised.
    assert (out / "manifest.json").is_file()
    assert list(tmp_path.iterdir())


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


class _StubStateSnapshot:
    """Mimic the LangGraph ``StateSnapshot`` shape returned by ``graph.get_state``."""

    def __init__(self, values, *, next_nodes=(), checkpoint_id="cp-1", metadata=None):
        self.values = values
        self.next = next_nodes
        self.config = {"configurable": {"checkpoint_id": checkpoint_id}}
        self.metadata = metadata or {"source": "loop", "step": 2}


class _CallbackAgent:
    """Fake agent that drives supplied callbacks the way LangGraph nodes do."""

    def __init__(self, nodes=("PlanControllerAgent", "CodeAgent"), final_values=None):
        self._nodes = nodes
        self._final_values = final_values if final_values is not None else {"final_answer": "done"}
        self.invoke_calls = 0
        self.stream_calls = 0
        self.get_state_configs = []

    @property
    def graph(self):
        agent = self

        class _Graph:
            def get_state(self_, config):
                agent.get_state_configs.append(config)
                return _StubStateSnapshot(agent._final_values)

        return _Graph()

    async def invoke(self, message, *, thread_id=None, track_tool_calls=False, config=None):
        self.invoke_calls += 1
        handlers = list((config or {}).get("callbacks") or ())
        # Nest the nodes the way LangGraph does: each node is a child of the
        # previous one, so ``parent_run_id`` forms a real chain. A flat sibling
        # list would not exercise edge construction at all.
        for index, node in enumerate(self._nodes):
            for handler in handlers:
                handler.on_chain_start(
                    {"name": node},
                    {"input": message},
                    run_id=f"run-{index}",
                    parent_run_id=f"run-{index - 1}" if index else None,
                    metadata={"langgraph_node": node, "langgraph_step": index},
                )
        for index in reversed(range(len(self._nodes))):
            for handler in handlers:
                handler.on_chain_end(
                    {"ok": True},
                    run_id=f"run-{index}",
                    parent_run_id=f"run-{index - 1}" if index else None,
                )
        return FakeResult(answer="done")

    async def stream(self, *args, **kwargs):  # pragma: no cover - must never run
        self.stream_calls += 1
        raise AssertionError("stream() must not be invoked during a traced invoke() run")

    async def aclose(self):
        pass


def test_sdk_runtime_captures_callback_node_events_as_stream_events(tmp_path):
    agent = _CallbackAgent()
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: agent,
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})
    manifest = read_manifest(result)

    assert manifest["capabilities"]["stream_events"]["status"] == "captured"
    assert agent.invoke_calls == 1
    assert agent.stream_calls == 0

    events = [
        json.loads(line)
        for line in (Path(result["causal_trace_path"]) / "events.jsonl").read_text().splitlines()
    ]
    node_events = [event for event in events if event["kind"] == "graph_node_start"]
    assert [event["payload"]["node"] for event in node_events] == [
        "PlanControllerAgent",
        "CodeAgent",
    ]


def test_sdk_runtime_captures_graph_final_state_without_second_execution(tmp_path):
    agent = _CallbackAgent(final_values={"final_answer": "done", "step": 7})
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: agent,
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})
    manifest = read_manifest(result)

    assert manifest["capabilities"]["graph_final_state"]["status"] == "captured"
    assert agent.invoke_calls == 1
    assert agent.stream_calls == 0

    trace = json.loads((Path(result["causal_trace_path"]) / "causal-trace.json").read_text())
    snapshots = trace["checkpoints"]
    assert len(snapshots) == 1
    assert snapshots[0]["checkpoint_id"] == "cp-1"
    assert snapshots[0]["replay_safe"] is False


def test_sdk_runtime_reports_stream_events_runtime_failure_when_no_events_arrive(tmp_path):
    agent = _CallbackAgent(nodes=())
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: agent,
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})
    manifest = read_manifest(result)

    stream_events = manifest["capabilities"]["stream_events"]
    assert stream_events["status"] == "runtime_failure"
    assert stream_events["reason"] == "callback handler attached but emitted no events"


def test_sdk_runtime_records_model_in_trace_manifest(tmp_path):
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: _CallbackAgent(),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
        model="openai/azure/gpt-5.6-luna",
    ).run_task("task-1", {"input": "x"})

    assert read_manifest(result)["model"] == "openai/azure/gpt-5.6-luna"


def test_graph_event_collector_satisfies_langchain_callback_contract():
    """The handler must be a real ``BaseCallbackHandler``.

    LangChain's async dispatch reads handler attributes such as ``run_inline``
    (``langchain_core/callbacks/manager.py``). A duck-typed handler raises
    ``AttributeError`` mid-run, which previously surfaced only as an opaque
    ``status="error"`` with no captured events.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    from agent_evolve.cuga_wrapper import build_graph_callback_handler

    collector = GraphEventCollector(max_events=10)
    handler = build_graph_callback_handler(collector)

    assert isinstance(handler, BaseCallbackHandler)
    assert hasattr(handler, "run_inline")

    handler.on_chain_start(
        {"name": "prepare"},
        {},
        run_id="r1",
        metadata={"langgraph_node": "prepare", "langgraph_step": 0},
    )
    assert [event["kind"] for event in collector.events] == ["graph_node_start"]
    assert collector.events[0]["node"] == "prepare"


def test_sdk_runtime_persists_invoke_exception_as_trace_evidence(tmp_path):
    """A failed run must record why it failed, not just ``status="error"``."""

    class ExplodingAgent:
        graph = None

        async def invoke(self, message, *, thread_id=None, track_tool_calls=False, config=None):
            raise RuntimeError("boom-3391")

        async def aclose(self):
            pass

    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: ExplodingAgent(),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})

    assert result["status"] == "error"
    assert "boom-3391" in str(result["error"])

    trace = json.loads((Path(result["causal_trace_path"]) / "causal-trace.json").read_text())
    assert "boom-3391" in str(trace["error"])


class _ToolCallAgent:
    """Fake agent returning the verified CUGA ``InvokeResult.tool_calls`` shape.

    ``InvokeResult.tool_calls`` is typed ``List[Dict[str, Any]]`` (cuga/sdk.py:110)
    and ``track_tool_calls=True`` populates each entry with ``name``, ``arguments``,
    ``result``, ``app_name``, ``operation_id``, ``timestamp``, ``duration_ms`` and
    ``error``. This is a real observation surface, so the manifest must not report
    it as absent.
    """

    graph = None

    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    async def invoke(self, message, *, thread_id=None, track_tool_calls=False, config=None):
        return FakeResult(tool_calls=list(self._tool_calls))

    async def aclose(self):
        pass


def _run_with_tool_calls(tmp_path, tool_calls, **trace_kwargs):
    return CugaSdkRuntime(
        lambda config, workspace_dir=None: _ToolCallAgent(tool_calls),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path, **trace_kwargs),
    ).run_task("task-1", {"input": "x"})


def test_tool_observations_reported_captured_when_sdk_returns_tool_calls(tmp_path):
    """Real returned tool calls must not be reported ``unavailable_no_sdk_surface``.

    Regression: a live run executed three chained tools and persisted all of
    them, while the manifest still claimed no SDK surface existed. That
    understates real provenance and would make the evidence look unusable.
    """
    result = _run_with_tool_calls(
        tmp_path,
        [
            {
                "name": "fetch_alpha_token",
                "arguments": {},
                "result": "ALPHA-1",
                "duration_ms": 0.5,
                "error": None,
            }
        ],
    )

    assert read_manifest(result)["capabilities"]["tool_observations"]["status"] == "captured"


def test_tool_observations_absent_when_no_tool_calls_returned(tmp_path):
    """No tool calls must stay honestly unavailable rather than claim capture."""
    result = _run_with_tool_calls(tmp_path, [])

    assert read_manifest(result)["capabilities"]["tool_observations"]["status"] == (
        "unavailable_no_sdk_surface"
    )


def test_tool_observations_persisted_with_arguments_and_results(tmp_path):
    """The trace must carry per-call provenance, not just a count.

    Argument/result linkage is what proves a dependency chain really executed
    (tool 2 received tool 1's exact output) rather than being hallucinated.
    """
    result = _run_with_tool_calls(
        tmp_path,
        [
            {"name": "fetch_alpha", "arguments": {}, "result": "ALPHA-1", "duration_ms": 0.5},
            {
                "name": "exchange_alpha",
                "arguments": {"alpha_token": "ALPHA-1"},
                "result": "BETA-2",
                "duration_ms": 0.25,
            },
        ],
    )

    trace = json.loads((Path(result["causal_trace_path"]) / "causal-trace.json").read_text())
    observations = trace["tool_observations"]

    assert [obs["tool_name"] for obs in observations] == ["fetch_alpha", "exchange_alpha"]
    assert [obs["sequence"] for obs in observations] == [0, 1]
    assert observations[0]["result"] == "ALPHA-1"
    assert json.loads(observations[1]["canonical_arguments"]) == {"alpha_token": "ALPHA-1"}
    assert observations[0]["content_digest"].startswith("sha256:")


def test_tool_observation_records_sdk_reported_error_as_replay_ineligible(tmp_path):
    """A failed tool call must be retained as evidence and never replay-eligible."""
    result = _run_with_tool_calls(
        tmp_path,
        [{"name": "flaky", "arguments": {"q": "x"}, "result": None, "error": "boom-77"}],
    )

    trace = json.loads((Path(result["causal_trace_path"]) / "causal-trace.json").read_text())
    observation = trace["tool_observations"][0]

    assert "boom-77" in str(observation["error"])
    assert observation["replay_eligible"] is False


def test_tool_observations_respect_disabled_by_config(tmp_path):
    """Operators must be able to turn the facility off without a false claim."""
    result = _run_with_tool_calls(
        tmp_path,
        [{"name": "fetch_alpha", "arguments": {}, "result": "ALPHA-1"}],
        capture_tool_observations=False,
    )

    manifest = read_manifest(result)
    trace = json.loads((Path(result["causal_trace_path"]) / "causal-trace.json").read_text())

    assert manifest["capabilities"]["tool_observations"]["status"] == "disabled_by_config"
    assert trace["tool_observations"] == []


def test_oversized_tool_result_is_truncated_and_replay_ineligible(tmp_path):
    """Large results must be bounded, keeping a digest for provenance."""
    result = _run_with_tool_calls(
        tmp_path,
        [{"name": "dump", "arguments": {}, "result": "x" * 5000}],
        max_observation_bytes=64,
    )

    trace = json.loads((Path(result["causal_trace_path"]) / "causal-trace.json").read_text())
    observation = trace["tool_observations"][0]

    assert observation["truncated"] is True
    assert observation["replay_eligible"] is False
    assert observation["result"] is None
    assert observation["content_digest"].startswith("sha256:")


# --- Task 1: real graph edges from callback run identity ---------------------
#
# LangChain supplies a genuine ``run_id``/``parent_run_id`` on every callback
# (verified live: 24/26 populated, correctly nested). Recording them turns the
# flat event list into a traversable DAG. Synthesising parents is forbidden by
# docs/architecture/data-contracts.md:103, so edges appear only where the
# runtime actually reported one.


def test_collector_records_run_identity_for_graph_edges():
    from agent_evolve.cuga_wrapper import build_graph_callback_handler

    collector = GraphEventCollector(max_events=10)
    handler = build_graph_callback_handler(collector)

    handler.on_chain_start(
        {"name": "CugaLiteSubgraph"},
        {},
        run_id="run-parent",
        parent_run_id=None,
        metadata={"langgraph_node": "CugaLiteSubgraph", "langgraph_step": 0},
    )
    handler.on_chain_start(
        {"name": "call_model"},
        {},
        run_id="run-child",
        parent_run_id="run-parent",
        metadata={"langgraph_node": "call_model", "langgraph_step": 1},
    )

    root, child = collector.events
    assert root["run_id"] == "run-parent"
    assert root.get("parent_run_id") is None
    assert child["run_id"] == "run-child"
    assert child["parent_run_id"] == "run-parent"


def test_chain_end_carries_node_identity_for_pairing():
    """``on_chain_end`` must be pairable with its start.

    LangGraph does not put ``langgraph_node`` on end callbacks, so pairing by
    name is impossible; ``run_id`` is the only reliable key (verified 10/10).
    """
    from agent_evolve.cuga_wrapper import build_graph_callback_handler

    collector = GraphEventCollector(max_events=10)
    handler = build_graph_callback_handler(collector)

    handler.on_chain_start(
        {"name": "call_model"},
        {},
        run_id="run-1",
        parent_run_id="run-0",
        metadata={"langgraph_node": "call_model", "langgraph_step": 3},
    )
    handler.on_chain_end({"ok": True}, run_id="run-1", parent_run_id="run-0")

    start, end = collector.events
    assert end["run_id"] == start["run_id"] == "run-1"
    assert end["node"] == "call_model"


def test_repeated_node_names_at_different_depths_are_distinct_events():
    """Nested same-name nodes are real structure, not duplicates.

    A live run showed ``prepare`` four times and ``CugaLiteSubgraph`` twice with
    distinct run ids at different nesting depths. Collapsing them would destroy
    the parent/child relation the blame graph depends on.
    """
    from agent_evolve.cuga_wrapper import build_graph_callback_handler

    collector = GraphEventCollector(max_events=10)
    handler = build_graph_callback_handler(collector)

    for run_id, parent in (("r1", "r0"), ("r2", "r1")):
        handler.on_chain_start(
            {"name": "prepare"},
            {},
            run_id=run_id,
            parent_run_id=parent,
            metadata={"langgraph_node": "prepare"},
        )

    assert [event["run_id"] for event in collector.events] == ["r1", "r2"]
    assert [event["parent_run_id"] for event in collector.events] == ["r0", "r1"]


def test_events_from_dicts_preserves_edges_as_top_level_fields():
    """``parent_event_id``/``actor_id``/``timestamp`` must not sink into payload.

    They are graph fields on ``CausalEvent``; burying them in the opaque payload
    leaves the analyzer unable to traverse the trajectory.
    """
    from agent_evolve.cuga_wrapper import _events_from_dicts

    events = _events_from_dicts(
        [
            {
                "event_id": "graph:1",
                "kind": "graph_node_start",
                "parent_event_id": "graph:0",
                "actor_id": "call_model",
                "timestamp": "2026-08-15T01:00:00Z",
                "node": "call_model",
            }
        ]
    )

    event = events[0]
    assert event.parent_event_id == "graph:0"
    assert event.actor_id == "call_model"
    assert event.timestamp == "2026-08-15T01:00:00Z"
    assert "parent_event_id" not in event.payload
    assert "actor_id" not in event.payload
    assert "timestamp" not in event.payload
    assert event.payload["node"] == "call_model"


def test_runtime_emits_traversable_parent_child_chain(tmp_path):
    """End to end: a nested run must persist real edges, not nulls."""
    agent = _CallbackAgent(nodes=("CugaLiteSubgraph", "call_model"))
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: agent,
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})

    events = [
        json.loads(line)
        for line in (Path(result["causal_trace_path"]) / "events.jsonl").read_text().splitlines()
    ]
    starts = [event for event in events if event["kind"] == "graph_node_start"]

    assert starts, "expected node start events"
    # The child node must point at the parent node's event, forming an edge.
    by_actor = {event.get("actor_id"): event for event in starts}
    child = by_actor.get("call_model")
    assert child is not None
    assert child["parent_event_id"] == by_actor["CugaLiteSubgraph"]["event_id"]


def test_edges_resolve_when_parent_callback_arrives_after_its_children():
    """A parent recorded AFTER its children must still receive the edges.

    LangGraph's outermost chain ends last, so the root run's own callback is the
    final event while its children fire first. Resolving parents at record time
    silently dropped those edges and split one trajectory into several apparent
    roots. Resolution must therefore happen once every run id is known.
    """
    from agent_evolve.cuga_wrapper import build_graph_callback_handler

    collector = GraphEventCollector(max_events=10)
    handler = build_graph_callback_handler(collector)

    # Child first, naming a parent run that has not been seen yet.
    handler.on_chain_start(
        {"name": "CugaLiteSubgraph"},
        {},
        run_id="child",
        parent_run_id="root",
        metadata={"langgraph_node": "CugaLiteSubgraph"},
    )
    handler.on_chain_start(
        {"name": "FinalAnswerAgent"},
        {},
        run_id="sibling",
        parent_run_id="root",
        metadata={"langgraph_node": "FinalAnswerAgent"},
    )
    # The root's own callback lands last.
    handler.on_chain_end({"ok": True}, run_id="root", parent_run_id=None)

    events = collector.resolved_events()
    by_run = {event["run_id"]: event for event in events}
    root_event_id = by_run["root"]["event_id"]

    assert by_run["child"]["parent_event_id"] == root_event_id
    assert by_run["sibling"]["parent_event_id"] == root_event_id
    assert by_run["root"].get("parent_event_id") is None


def test_unreported_parent_yields_no_edge_rather_than_a_guess():
    """An unresolvable parent must stay empty; never invent a link."""
    from agent_evolve.cuga_wrapper import build_graph_callback_handler

    collector = GraphEventCollector(max_events=10)
    handler = build_graph_callback_handler(collector)

    handler.on_chain_start(
        {"name": "orphan"},
        {},
        run_id="only",
        parent_run_id="never-observed",
        metadata={"langgraph_node": "orphan"},
    )

    event = collector.resolved_events()[0]
    assert event.get("parent_event_id") is None


def test_runtime_persists_single_connected_trajectory(tmp_path):
    """The whole run must form ONE tree, not several detached fragments."""
    agent = _RootLastCallbackAgent()
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: agent,
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})

    events = [
        json.loads(line)
        for line in (Path(result["causal_trace_path"]) / "events.jsonl").read_text().splitlines()
    ]
    graph_events = [event for event in events if str(event["kind"]).startswith("graph_node")]
    roots = [event for event in graph_events if event.get("parent_event_id") is None]

    # Exactly one event may lack a parent: the graph invocation itself.
    assert len(roots) == 1, f"expected one root, got {[r.get('actor_id') for r in roots]}"


class _RootLastCallbackAgent:
    """Fake agent whose root chain callback fires last, as LangGraph's does."""

    graph = None

    async def invoke(self, message, *, thread_id=None, track_tool_calls=False, config=None):
        handlers = list((config or {}).get("callbacks") or ())
        for handler in handlers:
            for node, run_id in (("CugaLiteSubgraph", "r1"), ("FinalAnswerAgent", "r2")):
                handler.on_chain_start(
                    {"name": node},
                    {},
                    run_id=run_id,
                    parent_run_id="root",
                    metadata={"langgraph_node": node},
                )
                handler.on_chain_end({"ok": True}, run_id=run_id, parent_run_id="root")
            handler.on_chain_end({"ok": True}, run_id="root", parent_run_id=None)
        return FakeResult(answer="done")

    async def aclose(self):
        pass


# --- Task 3: full-fidelity payload capture -----------------------------------
#
# Reconstructing a subagent's pre/post state (for isolated simulation) needs the
# real node inputs/outputs and the real prompt/response, not summaries. Payloads
# are stored verbatim as content-addressed blobs so events.jsonl stays scannable
# and identical state dicts collapse to one blob.


class _PayloadAgent:
    """Fake agent that drives chain + chat-model callbacks with real payloads."""

    graph = None

    def __init__(self, *, inputs=None, outputs=None, messages=None, response=None):
        self.inputs = inputs if inputs is not None else {"chat_messages": [{"role": "user"}]}
        self.outputs = outputs if outputs is not None else {"final_answer": "done"}
        self.messages = messages
        self.response = response

    async def invoke(self, message, *, thread_id=None, track_tool_calls=False, config=None):
        for handler in list((config or {}).get("callbacks") or ()):
            handler.on_chain_start(
                {"name": "call_model"},
                self.inputs,
                run_id="r1",
                parent_run_id=None,
                metadata={"langgraph_node": "call_model"},
            )
            if self.messages is not None:
                handler.on_chat_model_start(
                    {"name": "chat"}, self.messages, run_id="r2", parent_run_id="r1"
                )
            if self.response is not None:
                handler.on_llm_end(self.response, run_id="r2", parent_run_id="r1")
            handler.on_chain_end(self.outputs, run_id="r1", parent_run_id=None)
        return FakeResult(answer="done")

    async def aclose(self):
        pass


def _raw_trace_config(tmp_path, **kwargs):
    """Full-fidelity config: payload blobs are stored verbatim."""
    from agent_evolve.core.trace import PayloadLevel

    return TraceConfig(
        enabled=True,
        output_root=tmp_path,
        payload_level=PayloadLevel.RAW_OPT_IN,
        allow_raw_payloads=True,
        **kwargs,
    )


def _run_payload_agent(tmp_path, agent, **kwargs):
    return CugaSdkRuntime(
        lambda config, workspace_dir=None: agent,
        trace_config=_raw_trace_config(tmp_path, **kwargs),
    ).run_task("task-1", {"input": "x"})


def test_node_inputs_and_outputs_persisted_as_payload_blobs(tmp_path):
    """Before/after state must be recoverable per node."""
    agent = _PayloadAgent(
        inputs={"chat_messages": [{"role": "user", "content": "hi"}], "step_count": 1},
        outputs={"chat_messages": [{"role": "user", "content": "hi"}], "final_answer": "42"},
    )
    result = _run_payload_agent(tmp_path, agent)
    trace_dir = Path(result["causal_trace_path"])

    events = [
        json.loads(line) for line in (trace_dir / "events.jsonl").read_text().splitlines()
    ]
    start = next(e for e in events if e["kind"] == "graph_node_start")
    end = next(e for e in events if e["kind"] == "graph_node_end")

    assert start["payload"]["state_before_ref"]
    assert end["payload"]["state_after_ref"]

    before = json.loads(
        (trace_dir / "payloads" / f"{start['payload']['state_before_ref']}.json").read_text()
    )
    after = json.loads(
        (trace_dir / "payloads" / f"{end['payload']['state_after_ref']}.json").read_text()
    )
    assert before["step_count"] == 1
    assert after["final_answer"] == "42"


def test_large_payload_is_stored_untruncated(tmp_path):
    """A 39KB prompt must survive whole; the default 2000-char cap would gut it."""
    big = "x" * 40_000
    agent = _PayloadAgent(inputs={"prepared_prompt": big}, outputs={"ok": True})
    result = _run_payload_agent(tmp_path, agent)
    trace_dir = Path(result["causal_trace_path"])

    events = [
        json.loads(line) for line in (trace_dir / "events.jsonl").read_text().splitlines()
    ]
    ref = next(e for e in events if e["kind"] == "graph_node_start")["payload"]["state_before_ref"]
    stored = json.loads((trace_dir / "payloads" / f"{ref}.json").read_text())

    assert len(stored["prepared_prompt"]) == 40_000


def test_identical_payloads_share_one_content_addressed_blob(tmp_path):
    """The same state repeated across nodes must not be written twice."""
    same = {"chat_messages": [{"role": "user", "content": "hi"}]}
    agent = _PayloadAgent(inputs=same, outputs=same)
    result = _run_payload_agent(tmp_path, agent)
    trace_dir = Path(result["causal_trace_path"])

    events = [
        json.loads(line) for line in (trace_dir / "events.jsonl").read_text().splitlines()
    ]
    start = next(e for e in events if e["kind"] == "graph_node_start")
    end = next(e for e in events if e["kind"] == "graph_node_end")

    assert start["payload"]["state_before_ref"] == end["payload"]["state_after_ref"]
    assert len(list((trace_dir / "payloads").glob("*.json"))) == 1


def test_llm_prompt_and_response_captured_verbatim(tmp_path):
    """Simulating a subagent needs its exact prompt and its exact response."""
    agent = _PayloadAgent(
        messages=[[{"type": "SystemMessage", "content": "sys"}, {"type": "HumanMessage", "content": "ask"}]],
        response={"generations": [[{"text": "the answer"}]], "llm_output": {"model": "m"}},
    )
    result = _run_payload_agent(tmp_path, agent)
    trace_dir = Path(result["causal_trace_path"])

    events = [
        json.loads(line) for line in (trace_dir / "events.jsonl").read_text().splitlines()
    ]
    prompt_event = next(e for e in events if e["kind"] == "llm_call_start")
    response_event = next(e for e in events if e["kind"] == "llm_call_end")

    prompt = json.loads(
        (trace_dir / "payloads" / f"{prompt_event['payload']['messages_ref']}.json").read_text()
    )
    response = json.loads(
        (trace_dir / "payloads" / f"{response_event['payload']['response_ref']}.json").read_text()
    )
    assert prompt[0][1]["content"] == "ask"
    assert response["generations"][0][0]["text"] == "the answer"


def test_llm_events_link_to_their_issuing_node(tmp_path):
    """An LLM call must be attributable to the node that made it."""
    agent = _PayloadAgent(
        messages=[[{"type": "HumanMessage", "content": "ask"}]],
        response={"generations": [[{"text": "ok"}]]},
    )
    result = _run_payload_agent(tmp_path, agent)
    events = [
        json.loads(line)
        for line in (Path(result["causal_trace_path"]) / "events.jsonl").read_text().splitlines()
    ]

    node_start = next(e for e in events if e["kind"] == "graph_node_start")
    llm_start = next(e for e in events if e["kind"] == "llm_call_start")
    assert llm_start["parent_event_id"] == node_start["event_id"]


def test_payload_capture_can_be_disabled(tmp_path):
    """Structural-only capture must remain possible, with no blob directory."""
    agent = _PayloadAgent(inputs={"secretish": "x" * 100})
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: agent,
        trace_config=TraceConfig(enabled=True, output_root=tmp_path, capture_node_payloads=False),
    ).run_task("task-1", {"input": "x"})
    trace_dir = Path(result["causal_trace_path"])

    events = [
        json.loads(line) for line in (trace_dir / "events.jsonl").read_text().splitlines()
    ]
    start = next(e for e in events if e["kind"] == "graph_node_start")

    assert "state_before_ref" not in start["payload"]
    assert not (trace_dir / "payloads").exists()
    assert read_manifest(result)["capabilities"]["node_payloads"]["status"] == "disabled_by_config"


def test_lazy_state_reader_returns_before_and_after(tmp_path):
    """The reader must resolve blobs on demand, without loading the whole trace."""
    from agent_evolve.cuga_wrapper import load_node_state

    agent = _PayloadAgent(
        inputs={"step_count": 1, "chat_messages": []},
        outputs={"step_count": 2, "final_answer": "done"},
    )
    result = _run_payload_agent(tmp_path, agent)

    before, after = load_node_state(Path(result["causal_trace_path"]), node="call_model")
    assert before["step_count"] == 1
    assert after["step_count"] == 2
    assert after["final_answer"] == "done"


# --- Task 3b: routing objects and derived post-state -------------------------
#
# Most CUGA nodes return a LangGraph ``Command`` (goto + update) rather than a
# full state dict, because routing happens inside nodes. ``Command`` uses
# __slots__, so a vars()-based projection silently yields {} and the routing
# decision - the thing a blame graph needs most - is lost. And a Command must
# never be presented as if it were a post-state.


class _SlottedCommand:
    """Stand-in with ``Command`` semantics: dataclass fields, empty ``vars()``."""

    __slots__ = ("graph", "update", "resume", "goto")

    def __init__(self, *, goto=None, update=None):
        self.graph = None
        self.update = update
        self.resume = None
        self.goto = goto


def test_json_safe_preserves_slotted_routing_fields():
    """A __slots__ object must not collapse to an empty mapping."""
    from agent_evolve.cuga_wrapper import _json_safe

    projected = _json_safe(_SlottedCommand(goto="FinalAnswerAgent", update={"final_answer": "x"}))

    assert projected["goto"] == "FinalAnswerAgent"
    assert projected["update"] == {"final_answer": "x"}


def test_routing_decision_recorded_on_node_end(tmp_path):
    """The chosen branch must be queryable without opening a blob."""
    agent = _PayloadAgent(
        inputs={"step_count": 1},
        outputs=_SlottedCommand(goto="FinalAnswerAgent", update={"final_answer": "x"}),
    )
    result = _run_payload_agent(tmp_path, agent)
    events = [
        json.loads(line)
        for line in (Path(result["causal_trace_path"]) / "events.jsonl").read_text().splitlines()
    ]

    end = next(e for e in events if e["kind"] == "graph_node_end")
    assert end["payload"]["routed_to"] == "FinalAnswerAgent"


def test_command_output_is_not_reported_as_post_state(tmp_path):
    """A routing object is not a state; saying otherwise would mislead analysis."""
    from agent_evolve.cuga_wrapper import load_node_state

    agent = _PayloadAgent(
        inputs={"step_count": 1, "final_answer": None},
        outputs=_SlottedCommand(goto="END", update={"final_answer": "done"}),
    )
    result = _run_payload_agent(tmp_path, agent)

    before, after, provenance = load_node_state(
        Path(result["causal_trace_path"]), node="call_model", with_provenance=True
    )

    assert before["step_count"] == 1
    # The node returned a Command, so the post-state is derived from its update,
    # never the raw routing object.
    assert provenance["after_source"] == "command_update"
    assert after["final_answer"] == "done"
    assert after["step_count"] == 1, "unchanged keys must carry over from the pre-state"


def test_state_dict_output_is_reported_as_direct_post_state(tmp_path):
    """When a node does return full state, provenance must say so."""
    from agent_evolve.cuga_wrapper import load_node_state

    agent = _PayloadAgent(inputs={"step_count": 1}, outputs={"step_count": 2, "extra": True})
    result = _run_payload_agent(tmp_path, agent)

    before, after, provenance = load_node_state(
        Path(result["causal_trace_path"]), node="call_model", with_provenance=True
    )

    assert provenance["after_source"] == "chain_end_outputs"
    assert before["step_count"] == 1
    assert after["step_count"] == 2


# --- Tasks 4 & 5: static topology + chronological ordering --------------------


class _TopologyAgent:
    """Fake agent exposing a LangGraph-style drawable graph."""

    def __init__(self):
        class _Edge:
            def __init__(self, source, target, conditional):
                self.source = source
                self.target = target
                self.conditional = conditional

        class _Drawable:
            nodes = {"__start__": object(), "call_model": object(), "__end__": object()}
            edges = (
                _Edge("__start__", "call_model", False),
                _Edge("call_model", "__end__", True),
            )

        class _Graph:
            def get_graph(self_, xray=False):
                return _Drawable()

            def get_state(self_, config):
                return _StubStateSnapshot({"final_answer": "done"})

        self.graph = _Graph()

    async def invoke(self, message, *, thread_id=None, track_tool_calls=False, config=None):
        return FakeResult(answer="done")

    async def aclose(self):
        pass


def test_static_graph_topology_persisted(tmp_path):
    """The declared topology says which transitions are even legal.

    Observed adjacency alone cannot distinguish a legal transition from an
    anomaly, so blame attribution needs the compiled graph's edges.
    """
    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: _TopologyAgent(),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})

    topology = json.loads((Path(result["causal_trace_path"]) / "graph-topology.json").read_text())

    assert sorted(topology["nodes"]) == ["__end__", "__start__", "call_model"]
    assert {"source": "__start__", "target": "call_model", "conditional": False} in topology["edges"]
    assert {"source": "call_model", "target": "__end__", "conditional": True} in topology["edges"]
    assert read_manifest(result)["capabilities"]["graph_topology"]["status"] == "captured"


def test_graph_topology_absent_is_reported_honestly(tmp_path):
    """No graph surface must not be reported as a captured topology."""

    class _NoGraphAgent:
        graph = None

        async def invoke(self, message, *, thread_id=None, track_tool_calls=False, config=None):
            return FakeResult(answer="done")

        async def aclose(self):
            pass

    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: _NoGraphAgent(),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})

    assert not (Path(result["causal_trace_path"]) / "graph-topology.json").exists()
    assert read_manifest(result)["capabilities"]["graph_topology"]["status"] == (
        "unavailable_no_sdk_surface"
    )


def test_events_are_sequenced_chronologically(tmp_path):
    """``sequence`` must reflect real time, not which list an event came from.

    SDK tool calls are appended after callback events, so before this fix every
    tool_call sat at sequence 0..n ahead of the nodes that actually invoked it.
    """
    agent = _CallbackAgent(nodes=("CugaLiteSubgraph", "call_model"))
    agent_tool_calls = [
        {
            "name": "late_tool",
            "arguments": {},
            "result": "v",
            "timestamp": "2099-01-01T00:00:00",
        }
    ]

    class _AgentWithLateTool(_CallbackAgent):
        async def invoke(self, message, *, thread_id=None, track_tool_calls=False, config=None):
            await super().invoke(
                message, thread_id=thread_id, track_tool_calls=track_tool_calls, config=config
            )
            return FakeResult(answer="done", tool_calls=agent_tool_calls)

    result = CugaSdkRuntime(
        lambda config, workspace_dir=None: _AgentWithLateTool(
            nodes=("CugaLiteSubgraph", "call_model")
        ),
        trace_config=TraceConfig(enabled=True, output_root=tmp_path),
    ).run_task("task-1", {"input": "x"})

    events = [
        json.loads(line)
        for line in (Path(result["causal_trace_path"]) / "events.jsonl").read_text().splitlines()
    ]

    assert [event["sequence"] for event in events] == list(range(len(events)))
    # The tool call carries a far-future timestamp, so it must sort last rather
    # than leading the trace as list-append order would put it.
    assert events[-1]["kind"] == "tool_call"


# --- Task 5b: tool-call attribution ------------------------------------------
#
# Known and ACCEPTED limitation (user decision, 2026-08-15): CUGA reports tool
# timestamps as naive LOCAL time while callback events are UTC with a Z suffix.
# Timestamps are therefore NOT normalized to a common zone, so tool_call events
# sort after the graph events they actually occurred within. Ordering within each
# source is correct; cross-source ordering is not. Do not "fix" by string
# comparison - that silently compares local against UTC.
#
# SDK tool reports also carry no run_id, so they cannot be linked to the issuing
# node from identity alone. They stay unattributed rather than guessing.


def test_unattributable_tool_call_keeps_no_parent(tmp_path):
    """With no enclosing node, the tool call must stay unlinked, not guess one."""
    result = _run_with_tool_calls(
        tmp_path, [{"name": "orphan", "arguments": {}, "result": "v"}]
    )
    events = [
        json.loads(line)
        for line in (Path(result["causal_trace_path"]) / "events.jsonl").read_text().splitlines()
    ]
    tool_event = next(e for e in events if e["kind"] == "tool_call")

    assert tool_event.get("parent_event_id") is None
