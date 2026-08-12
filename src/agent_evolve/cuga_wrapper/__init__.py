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
        return {
            "task_id": trace["task_id"],
            "status": trace["status"],
            "model": self._settings.model,
            "final_output": trace["final_output"],
            "events": trace["events"],
        }

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

            instructions = harness_config.get("special_instructions")
            return CugaAgent(
                special_instructions=str(instructions) if instructions else None,
                auto_load_policies=False,
                filesystem_sync=False,
                enable_knowledge=False,
                enable_skills=False,
            )

        return cls(build_agent)

    def run_task(self, task_id: str, harness_config: Mapping[str, object]) -> dict[str, object]:
        config = dict(harness_config)
        artifact_instructions = "\n\n".join(self._artifacts.values())
        existing_instructions = config.get("special_instructions")
        if artifact_instructions:
            config["special_instructions"] = "\n\n".join(
                part
                for part in (str(existing_instructions) if existing_instructions else "", artifact_instructions)
                if part
            )
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
        return {
            "task_id": task_id,
            "status": "error" if error else "success",
            "final_output": str(getattr(result, "answer", "")),
            "events": events,
        }

    def get_artifacts(self) -> dict[str, str]:
        return dict(self._artifacts)

    def update_artifact(self, artifact_id: str, content: str) -> None:
        self._artifacts[artifact_id] = content
