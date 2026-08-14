from __future__ import annotations

from pathlib import Path

from agent_evolve.cuga_wrapper import (
    CugaSdkRuntime,
    CugaWrapper,
    InMemoryRuntime,
    MockHarnessRuntime,
    RuntimeSettings,
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
