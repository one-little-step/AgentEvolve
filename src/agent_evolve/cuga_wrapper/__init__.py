"""Thin runtime boundary for collecting CUGA-shaped baseline observations.

The wrapper owns no evolution policy. A verified CUGA SDK integration supplies a
runtime implementation; ``InMemoryRuntime`` keeps the collection path runnable
and deterministic until that dependency is available.
"""
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import os
from typing import Callable, Mapping, Protocol


class CugaRuntime(Protocol):
    """Minimal runtime surface required by the baseline collector."""

    def run_task(self, task_id: str, harness_config: Mapping[str, object]) -> dict[str, object]: ...

    def get_artifacts(self) -> dict[str, str]: ...

    def update_artifact(self, artifact_id: str, content: str) -> None: ...


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Connection settings kept configurable outside versioned source."""

    model: str
    base_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        model = os.environ.get("LITELLM_MODEL")
        if not model:
            raise RuntimeError("LITELLM_MODEL is required for a live inference run")
        return cls(
            model=model,
            base_url=os.environ.get("LITELLM_BASE_URL"),
            api_key=os.environ.get("LITELLM_API_KEY"),
        )

    def public_config(self) -> dict[str, str | None]:
        """Return trace-safe configuration; credentials never enter output."""
        return {"model": self.model, "base_url": self.base_url}

    def configure_cuga_environment(self) -> None:
        """Map the project's LiteLLM settings onto CUGA's documented OpenAI mode."""
        os.environ["AGENT_SETTING_CONFIG"] = "settings.openai.toml"
        # CUGA's OpenAI integration adds its own ``openai/`` platform prefix.
        model_name = self.model.removeprefix("openai/")
        os.environ["MODEL_NAME"] = model_name
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key


class CugaWrapper:
    """Expose a compact, JSON-oriented interface over an injected runtime."""

    def __init__(self, runtime: CugaRuntime, settings: RuntimeSettings) -> None:
        self._runtime = runtime
        self._settings = settings

    @classmethod
    def from_cuga(cls, settings: RuntimeSettings) -> "CugaWrapper":
        return cls(CugaSdkRuntime.from_settings(settings), settings)

    def run_task(self, task_id: str, harness_config: Mapping[str, object]) -> dict[str, object]:
        trace = self._runtime.run_task(task_id, harness_config)
        result = {
            "task_id": trace["task_id"],
            "status": trace["status"],
            "model": self._settings.model,
            "final_output": trace["final_output"],
            "events": trace["events"],
        }
        for field_name in ("harness_version", "active_artifacts", "unavailable_artifacts"):
            if field_name in trace:
                result[field_name] = trace[field_name]
        return result

    def get_artifacts(self) -> dict[str, str]:
        return self._runtime.get_artifacts()

    def update_artifact(self, artifact_id: str, content: str) -> None:
        self._runtime.update_artifact(artifact_id, content)


class InMemoryRuntime:
    """Deterministic local runtime used until official CUGA APIs are verified."""

    def __init__(self, artifacts: Mapping[str, str] | None = None) -> None:
        self._artifacts = dict(artifacts or {})

    def run_task(self, task_id: str, harness_config: Mapping[str, object]) -> dict[str, object]:
        input_text = str(harness_config.get("input", ""))
        artifact_text = "\n\n".join(self._artifacts.values())
        final_output = "\n\n".join(part for part in (artifact_text, input_text) if part)
        return {
            "task_id": task_id,
            "status": "success",
            "final_output": final_output,
            "events": [
                {"event_id": f"{task_id}:started", "kind": "run_started"},
                {"event_id": f"{task_id}:completed", "kind": "run_completed"},
            ],
        }

    def get_artifacts(self) -> dict[str, str]:
        return dict(self._artifacts)

    def update_artifact(self, artifact_id: str, content: str) -> None:
        self._artifacts[artifact_id] = content


def _artifact_metadata(harness_config: Mapping[str, object], *, available: set[str]) -> dict[str, object]:
    """Describe exactly which declared harness artifacts reached a runtime."""
    active = {"instructions": [], "skills": [], "memory": [], "tools": [], "policies": []}
    unavailable: dict[str, list[str]] = {}
    for field_name in ("skills", "memory", "policies"):
        entries = harness_config.get(field_name, {})
        names = list(entries) if isinstance(entries, Mapping) else []
        if field_name in available:
            active[field_name] = names
        elif names:
            unavailable[field_name] = names
    instructions = harness_config.get("instructions")
    if instructions:
        if "instructions" in available:
            active["instructions"] = ["instructions"]
        else:
            unavailable["instructions"] = ["instructions"]
    tools = harness_config.get("tools", [])
    tool_names = [
        (
            str(tool.get("name", f"tool-{index}"))
            if isinstance(tool, Mapping)
            else str(getattr(tool, "name", f"tool-{index}"))
        )
        for index, tool in enumerate(tools if isinstance(tools, list) else [])
    ]
    if tool_names:
        if "tools" in available:
            active["tools"] = tool_names
        else:
            unavailable["tools"] = tool_names
    return {
        "harness_version": str(harness_config.get("version", "unversioned")),
        "active_artifacts": active,
        "unavailable_artifacts": unavailable,
    }


class MockHarnessRuntime:
    """Deterministic structured-harness runtime for evolution-loop testing.

    It simulates every configured component locally; it does not claim an SDK
    mapping for CUGA-only skills, memory, or policies.
    """

    def run_task(self, task_id: str, harness_config: Mapping[str, object]) -> dict[str, object]:
        metadata = _artifact_metadata(
            harness_config,
            available={"instructions", "skills", "memory", "tools", "policies"},
        )
        input_text = str(harness_config.get("input", ""))
        tools = harness_config.get("tools", [])
        events = [{"event_id": f"{task_id}:started", "kind": "run_started"}]
        for tool in tools if isinstance(tools, list) else []:
            if isinstance(tool, Mapping) and str(tool.get("when", "")) in input_text:
                result = str(tool.get("result", ""))
                events.append(
                    {
                        "event_id": f"{task_id}:tool:0",
                        "kind": "tool_call",
                        "tool_call": {"name": str(tool.get("name", "tool")), "result": result},
                    }
                )
                events.append({"event_id": f"{task_id}:completed", "kind": "run_completed"})
                return {"task_id": task_id, "status": "success", "final_output": result, "events": events, **metadata}
        memory = harness_config.get("memory", {})
        if input_text.startswith("recall ") and isinstance(memory, Mapping):
            memory_id = input_text.removeprefix("recall ")
            if memory_id in memory:
                events.append({"event_id": f"{task_id}:memory:{memory_id}", "kind": "memory_recalled"})
                events.append({"event_id": f"{task_id}:completed", "kind": "run_completed"})
                return {
                    "task_id": task_id,
                    "status": "success",
                    "final_output": str(memory[memory_id]),
                    "events": events,
                    **metadata,
                }
        events.append({"event_id": f"{task_id}:completed", "kind": "run_completed"})
        return {"task_id": task_id, "status": "success", "final_output": input_text, "events": events, **metadata}

    def get_artifacts(self) -> dict[str, str]:
        return {}

    def update_artifact(self, artifact_id: str, content: str) -> None:
        raise NotImplementedError("MockHarnessRuntime receives artifacts through HARNESS")


class CugaSdkRuntime:
    """Official-SDK runtime for one-shot vanilla trajectory collection.

    CUGA exposes no generic mutable-artifact API that is appropriate to assume,
    so artifact operations remain unavailable until a specific SDK surface is
    verified. The baseline runner only needs task invocation and observation.
    """

    def __init__(
        self,
        agent_factory: Callable[[Mapping[str, object]], object],
        artifacts: Mapping[str, str] | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._artifacts = dict(artifacts or {})

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "CugaSdkRuntime":
        settings.configure_cuga_environment()

        def build_agent(harness_config: Mapping[str, object]) -> object:
            from cuga import CugaAgent

            instructions = harness_config.get("instructions")
            tools = harness_config.get("tools")
            return CugaAgent(
                tools=tools if isinstance(tools, list) else None,
                special_instructions=str(instructions) if instructions else None,
                auto_load_policies=False,
                filesystem_sync=False,
                enable_knowledge=False,
                enable_skills=False,
            )

        return cls(build_agent)

    def run_task(self, task_id: str, harness_config: Mapping[str, object]) -> dict[str, object]:
        config = {key: harness_config[key] for key in ("instructions", "tools") if key in harness_config}
        agent = self._agent_factory(config)
        message = str(harness_config["input"])
        result = asyncio.run(agent.invoke(message, track_tool_calls=True))
        asyncio.run(agent.aclose())
        tool_calls = list(getattr(result, "tool_calls", ()) or ())
        events = [
            {
                "event_id": f"{task_id}:tool:{index}",
                "kind": "tool_call",
                "tool_call": tool_call,
            }
            for index, tool_call in enumerate(tool_calls)
        ]
        error = getattr(result, "error", None)
        metadata = _artifact_metadata(harness_config, available={"instructions", "tools"})
        return {
            "task_id": task_id,
            "status": "error" if error else "success",
            "final_output": str(getattr(result, "answer", "")),
            "events": events,
            **metadata,
        }

    def get_artifacts(self) -> dict[str, str]:
        return dict(self._artifacts)

    def update_artifact(self, artifact_id: str, content: str) -> None:
        self._artifacts[artifact_id] = content
