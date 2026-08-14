"""Thin runtime boundary for collecting CUGA-shaped baseline observations.

The wrapper owns no evolution policy. A verified CUGA SDK integration supplies a
runtime implementation; ``InMemoryRuntime`` keeps the collection path runnable
and deterministic until that dependency is available.

CUGA is never imported at module import time. All CUGA imports are deferred into
``CugaSdkRuntime`` methods so the environment (``.env``, ``DYNACONF_*``,
``AGENT_SETTING_CONFIG``, ``SKILLS_ROOT``) is fully resolved before the SDK reads
its configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import os
import re
from pathlib import Path
from typing import Callable, Mapping, Protocol

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_SKILLS_ROOT = PROJECT_ROOT / ".cuga" / "skills"
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "data" / "workspaces"

DEFAULT_SPECIAL_INSTRUCTIONS = (
    "You are an autonomous agent. Solve the user's task carefully and accurately. "
    "Use the available tools when they are useful. Do not claim to have performed "
    "an action or accessed information unless you actually did so."
)


class CugaRuntime(Protocol):
    """Minimal runtime surface required by the baseline collector."""

    def run_task(self, task_id: str, harness_config: Mapping[str, object]) -> dict[str, object]: ...

    def get_artifacts(self) -> dict[str, str]: ...

    def update_artifact(self, artifact_id: str, content: str) -> None: ...


def normalize_cuga_configuration_directory() -> None:
    """Treat a blank optional CUGA configuration directory as unset.

    CUGA reads ``CUGA_CONFIGURATIONS_DIR`` with ``os.environ.get``, so an empty
    string resolves model files as relative paths and breaks the SDK import.
    """
    value = os.getenv("CUGA_CONFIGURATIONS_DIR")
    if value is not None and not value.strip():
        os.environ.pop("CUGA_CONFIGURATIONS_DIR", None)


def resolve_skills_root() -> str:
    """Resolve the CUGA skills root, mapping ``cuga`` to the project's directory."""
    skills_root = os.getenv("SKILLS_ROOT", "cuga")
    if skills_root == "cuga":
        skills_root = str(DEFAULT_SKILLS_ROOT)
    path = Path(skills_root).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def prepare_cuga_environment() -> None:
    """Load ``.env`` and normalize optional CUGA variables before SDK import."""
    load_dotenv(DOTENV_PATH)
    normalize_cuga_configuration_directory()


def _require_autonomous_mode() -> None:
    """Fail fast when CUGA autonomous mode is not enabled."""
    from cuga.config import settings

    if not settings.advanced_features.force_autonomous_mode:
        raise RuntimeError(
            "CUGA autonomous mode is disabled. "
            "Set DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true."
        )


def _construct_agent(
    harness_config: Mapping[str, object],
    default_tools: list,
    default_instructions: str | None,
    workspace_dir: str | None = None,
) -> object:
    """Build a CUGA agent using only the verified constructor surface."""
    from cuga import CugaAgent

    instructions = harness_config.get("instructions")
    config_tools = harness_config.get("tools")
    has_skills = bool(harness_config.get("skills"))
    has_policies = bool(harness_config.get("policies"))
    return CugaAgent(
        tools=config_tools if isinstance(config_tools, list) and config_tools else default_tools,
        special_instructions=str(instructions) if instructions else default_instructions,
        enable_knowledge=True,
        enable_skills=has_skills,
        skills_folder=workspace_dir if has_skills else None,
        cuga_folder=workspace_dir if has_policies else None,
        auto_load_policies=has_policies,
    )


def _safe_segment(name: str) -> str:
    """Sanitize a harness artifact name into a single safe path segment."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    sanitized = re.sub(r"\.{2,}", "_", sanitized)
    return sanitized or "artifact"


def _derive_description(body: str) -> str:
    """Derive a short skill/policy description from the first body line."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return "Harness artifact"


def materialize_harness(
    harness_config: Mapping[str, object],
    workspace_dir: Path | str,
) -> str | None:
    """Write harness skills/policies/memory into a fresh CUGA-style workspace.

    Returns the workspace directory when any editable artifact is present,
    otherwise ``None``.
    """
    skills = harness_config.get("skills") or {}
    policies = harness_config.get("policies") or {}
    memory = harness_config.get("memory") or {}
    has_editable = bool(skills) or bool(policies) or bool(memory)
    if not has_editable:
        return None

    workspace = Path(workspace_dir)

    if isinstance(skills, Mapping):
        for name, body in skills.items():
            segment = _safe_segment(str(name))
            skill_dir = workspace / "skills" / segment
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {segment}\ndescription: {_derive_description(str(body))}\n---\n{body}\n",
                encoding="utf-8",
            )

    if isinstance(policies, Mapping):
        policy_dir = workspace / "playbooks"
        policy_dir.mkdir(parents=True, exist_ok=True)
        for name, content in policies.items():
            segment = _safe_segment(str(name))
            (policy_dir / f"{segment}.md").write_text(
                f"---\nname: {segment}\nid: playbook_{segment}\ntriggers:\n  always: true\n---\n{content}\n",
                encoding="utf-8",
            )

    if isinstance(memory, Mapping):
        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        for key, value in memory.items():
            segment = _safe_segment(str(key))
            (memory_dir / f"{segment}.md").write_text(f"# {key}\n\n{value}\n", encoding="utf-8")

    return str(workspace)


def _memory_doc_paths(
    harness_config: Mapping[str, object],
    workspace_dir: str | None,
) -> list[str]:
    memory = harness_config.get("memory") or {}
    if not workspace_dir or not isinstance(memory, Mapping) or not memory:
        return []
    return [
        str(Path(workspace_dir) / "memory" / f"{_safe_segment(str(key))}.md")
        for key in memory
    ]


async def _execute(agent: object, message: str, memory_docs: list[str]) -> object:
    """Ingest any memory documents, then invoke the agent once."""
    knowledge = getattr(agent, "knowledge", None)
    for doc in memory_docs:
        if knowledge is None:
            break
        await knowledge.ingest(doc)
    return await agent.invoke(message, track_tool_calls=True)


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Connection settings kept configurable outside versioned source."""

    model: str
    base_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        model = os.environ.get("CUGA_MODEL") or os.environ.get("LITELLM_MODEL")
        if not model:
            raise RuntimeError("CUGA_MODEL or LITELLM_MODEL is required for a live inference run")
        return cls(
            model=model,
            base_url=os.environ.get("CUGA_BASE_URL") or os.environ.get("LITELLM_BASE_URL"),
            api_key=os.environ.get("CUGA_API_KEY") or os.environ.get("LITELLM_API_KEY"),
        )

    def public_config(self) -> dict[str, str | None]:
        """Return trace-safe configuration; credentials never enter output."""
        return {"model": self.model, "base_url": self.base_url}

    def configure_cuga_environment(self) -> None:
        """Map the project's model settings onto CUGA's documented OpenAI mode."""
        os.environ["AGENT_SETTING_CONFIG"] = os.getenv("AGENT_SETTING_CONFIG", "settings.openai.toml")
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
        agent_factory: Callable[..., object],
        artifacts: Mapping[str, str] | None = None,
        workspace_root: Path | str | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._artifacts = dict(artifacts or {})
        self._workspace_root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "CugaSdkRuntime":
        settings.configure_cuga_environment()
        os.environ["SKILLS_ROOT"] = resolve_skills_root()
        _require_autonomous_mode()

        from agent_evolve.cuga_wrapper.tools import build_tools

        default_tools = build_tools()

        def build_agent(
            harness_config: Mapping[str, object],
            workspace_dir: str | None = None,
        ) -> object:
            return _construct_agent(harness_config, default_tools, DEFAULT_SPECIAL_INSTRUCTIONS, workspace_dir)

        return cls(build_agent)

    def run_task(self, task_id: str, harness_config: Mapping[str, object]) -> dict[str, object]:
        workspace_dir = materialize_harness(harness_config, self._workspace_root / task_id)
        config = {
            key: harness_config[key]
            for key in ("instructions", "tools", "skills", "memory", "policies")
            if key in harness_config
        }
        agent = self._agent_factory(config, workspace_dir)
        message = str(harness_config["input"])
        memory_docs = _memory_doc_paths(harness_config, workspace_dir)
        result = asyncio.run(_execute(agent, message, memory_docs))
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
        available = {"instructions", "tools"}
        for field_name in ("skills", "memory", "policies"):
            if harness_config.get(field_name):
                available.add(field_name)
        metadata = _artifact_metadata(harness_config, available=available)
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
