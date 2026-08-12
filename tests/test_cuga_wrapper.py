from __future__ import annotations

from agent_evolve.cuga_wrapper import (
    CugaSdkRuntime,
    CugaWrapper,
    InMemoryRuntime,
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

    runtime = CugaSdkRuntime(agent_factory=lambda _: FakeAgent())

    trace = runtime.run_task("task-1", {"input": "What is 2 + 2?"})

    assert trace == {
        "task_id": "task-1",
        "status": "success",
        "final_output": "four",
        "events": [
            {"event_id": "task-1:tool:0", "kind": "tool_call", "tool_call": {"name": "calculator", "result": "4"}},
        ],
    }


def test_cuga_sdk_runtime_injects_updated_wrapper_artifacts_as_instructions():
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

    runtime = CugaSdkRuntime(
        agent_factory=lambda config: received_configs.append(config) or FakeAgent(),
        artifacts={"skills/default": "Prefer exact calculations."},
    )
    runtime.update_artifact("skills/default", "Use citations when possible.")

    runtime.run_task("task-1", {"input": "answer", "special_instructions": "Be concise."})

    assert runtime.get_artifacts() == {"skills/default": "Use citations when possible."}
    assert received_configs == [
        {
            "input": "answer",
            "special_instructions": "Be concise.\n\nUse citations when possible.",
        }
    ]
