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
import hashlib
import inspect
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from dotenv import load_dotenv

from agent_evolve.core.storage import sanitize_for_persistence
from agent_evolve.core.trace import (
    CausalEvent,
    CausalTrace,
    FacilityCapability,
    PayloadLevel,
    ToolObservation,
    canonical_json,
)

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


async def _execute(
    agent: object,
    message: str,
    memory_docs: list[str],
    invoke_kwargs: dict[str, object],
) -> object:
    """Ingest any memory documents, then invoke the agent once."""
    knowledge = getattr(agent, "knowledge", None)
    for doc in memory_docs:
        if knowledge is None:
            break
        await knowledge.ingest(doc)
    return await agent.invoke(message, **invoke_kwargs)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Configuration for causal trace persistence and recorded-environment replay."""

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
    allow_raw_payloads: bool = False

    def __post_init__(self) -> None:
        if self.max_observation_bytes <= 0:
            raise ValueError("max_observation_bytes must be positive")
        if self.max_events_per_trace <= 0:
            raise ValueError("max_events_per_trace must be positive")
        if self.payload_level is PayloadLevel.RAW_OPT_IN and not self.allow_raw_payloads:
            raise ValueError("raw_opt_in payload level requires allow_raw_payloads=True")


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _facility(config_flag: bool, *, available: bool = False, captured: bool = False) -> FacilityCapability:
    if not config_flag:
        return FacilityCapability(status="disabled_by_config")
    if captured:
        return FacilityCapability(status="captured")
    if not available:
        return FacilityCapability(status="unavailable_no_sdk_surface")
    return FacilityCapability(status="runtime_failure", reason="surface present but not collected")


def _events_from_dicts(events: Sequence[object]) -> tuple[CausalEvent, ...]:
    result: list[CausalEvent] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        payload = {
            str(key): value for key, value in event.items() if key not in ("event_id", "kind")
        }
        result.append(
            CausalEvent(
                event_id=str(event.get("event_id", f"event-{index}")),
                sequence=index,
                kind=str(event.get("kind", "runtime_update")),
                payload=payload,
            )
        )
    return tuple(result)


def _generic_capabilities(config: TraceConfig) -> dict[str, FacilityCapability]:
    return {
        "stream_events": _facility(config.capture_stream_events),
        "graph_final_state": _facility(config.capture_graph_final_state),
        "graph_history": (
            FacilityCapability(status="disabled_by_config")
            if not config.capture_graph_history
            else FacilityCapability(status="unavailable_no_checkpointer", reason="no checkpointer attached")
        ),
        "tool_observations": _facility(config.capture_tool_observations),
        "external_correlation": _facility(config.capture_external_correlation),
    }


def _build_generic_causal_trace(
    *,
    run_id: str,
    task_id: str,
    runtime_result: Mapping[str, object],
    model: str | None,
    config: TraceConfig,
) -> CausalTrace:
    events = _events_from_dicts(runtime_result.get("events", ()) or ())  # type: ignore[arg-type]
    return CausalTrace(
        run_id=run_id,
        task_id=task_id,
        thread_id=run_id,
        thread_id_source="wrapper_generated_not_injected",
        harness_version=str(runtime_result.get("harness_version", "unversioned")),
        status=str(runtime_result.get("status", "success")),
        final_output=str(runtime_result.get("final_output", "")),
        model=model,
        events=events,
        capabilities=_generic_capabilities(config),
        captured_event_count=len(events),
    )


class TraceWriter:
    """Write a :class:`CausalTrace` atomically into a per-rollout directory.

    Every persisted byte passes through the shared recursive redaction gateway;
    a prohibited value raises before any output directory is created.
    """

    def __init__(self, config: TraceConfig) -> None:
        self._config = config

    def write(self, trace: CausalTrace) -> Path:
        if not self._config.enabled:
            raise ValueError("TraceWriter.write requires an enabled TraceConfig")

        data = trace.model_dump(mode="json")
        redacted = sanitize_for_persistence(
            data, max_string_length=self._config.max_observation_bytes
        ).value

        output_root = Path(self._config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = output_root / trace.run_id
        staging = output_root / f".{trace.run_id}.tmp-{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            self._write_files(staging, redacted)
            staging.replace(run_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return run_dir

    def _write_files(self, staging: Path, data: Mapping[str, object]) -> None:
        files: dict[str, bool] = {}

        events = data.get("events") or ()
        if self._config.write_split_files and events:
            lines = [_json_dumps(event) for event in events]
            (staging / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            files["events.jsonl"] = True

        observations = data.get("tool_observations") or ()
        if self._config.write_split_files and self._config.capture_tool_observations and observations:
            observations_dir = staging / "observations"
            observations_dir.mkdir(parents=True, exist_ok=True)
            for observation in observations:
                sequence = int(observation["sequence"])  # type: ignore[index]
                (observations_dir / f"{sequence:06d}.json").write_text(
                    _json_dumps(observation), encoding="utf-8"
                )
            files["observations/"] = True

        checkpoints = data.get("checkpoints") or ()
        if self._config.write_split_files and self._config.capture_graph_history and checkpoints:
            checkpoints_dir = staging / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            for checkpoint in checkpoints:
                sequence = int(checkpoint["sequence"])  # type: ignore[index]
                (checkpoints_dir / f"{sequence:06d}.json").write_text(
                    _json_dumps(checkpoint), encoding="utf-8"
                )
            files["checkpoints/"] = True

        if self._config.write_self_contained_export:
            (staging / "causal-trace.json").write_text(_json_dumps(data), encoding="utf-8")
            files["causal-trace.json"] = True

        manifest = self._manifest(data, files)
        (staging / "manifest.json").write_text(_json_dumps(manifest), encoding="utf-8")

    def _manifest(self, data: Mapping[str, object], files: Mapping[str, bool]) -> dict[str, object]:
        return {
            "run_id": data.get("run_id"),
            "task_id": data.get("task_id"),
            "thread_id": data.get("thread_id"),
            "thread_id_source": data.get("thread_id_source"),
            "harness_version": data.get("harness_version"),
            "status": data.get("status"),
            "model": data.get("model"),
            "payload_level": self._config.payload_level.value,
            "events_truncated": data.get("events_truncated", False),
            "captured_event_count": data.get("captured_event_count", 0),
            "dropped_event_count": data.get("dropped_event_count", 0),
            "capabilities": data.get("capabilities", {}),
            "started_at": data.get("started_at"),
            "completed_at": data.get("completed_at"),
            "files": dict(files),
        }


class RecordedEnvironmentReplayError(RuntimeError):
    """Raised when recorded-environment tool replay fails closed on a mismatch."""


class ToolObservationRecorder:
    """Sequence-aware tool recorder and fail-closed recorded-environment replay."""

    def __init__(
        self,
        config: TraceConfig,
        *,
        high_risk_tool_names: frozenset[str] = frozenset(),
    ) -> None:
        self._config = config
        self._high_risk_tool_names = frozenset(high_risk_tool_names)
        self.observations: list[ToolObservation] = []
        self.uninstrumented_tools: list[str] = []
        self._sequence = 0

    @classmethod
    def replay(cls, observations: Sequence[ToolObservation]) -> "ToolObservationRecorder":
        recorder = cls(TraceConfig())
        recorder.observations = list(observations)
        return recorder

    def wrap(self, tool: object) -> object:
        name = getattr(tool, "name", None)
        invoke = getattr(tool, "invoke", None)
        if not name or not callable(invoke):
            self.uninstrumented_tools.append(str(name or type(tool).__name__))
            return tool

        recorder = self

        class _WrappedTool:
            def invoke(self_, input_obj, *args, **kwargs):
                return recorder._record(str(name), invoke, input_obj, *args, **kwargs)

            def __getattr__(self_, item):
                return getattr(tool, item)

        return _WrappedTool()

    def _record(self, name: str, invoke: Callable, input_obj: object, *args, **kwargs) -> object:
        sequence = self._sequence
        self._sequence += 1
        canonical_arguments = canonical_json(input_obj)
        start = time.perf_counter()
        try:
            result = invoke(input_obj, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            self.observations.append(
                ToolObservation(
                    sequence=sequence,
                    tool_name=name,
                    canonical_arguments=canonical_arguments,
                    error=repr(exc),
                    replay_eligible=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        self._append_recorded(name, sequence, canonical_arguments, result, duration_ms)
        return result

    def _append_recorded(
        self,
        name: str,
        sequence: int,
        canonical_arguments: str,
        result: object,
        duration_ms: float,
    ) -> None:
        if name in self._high_risk_tool_names and name not in self._config.high_risk_tool_allowlist:
            self.observations.append(
                ToolObservation(
                    sequence=sequence,
                    tool_name=name,
                    canonical_arguments=canonical_arguments,
                    withheld_reason="high_risk_tool",
                    replay_eligible=False,
                    duration_ms=duration_ms,
                )
            )
            return

        serialized = canonical_json(result)
        original_bytes = len(serialized.encode("utf-8"))
        truncated = original_bytes > self._config.max_observation_bytes
        content_digest = f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"
        self.observations.append(
            ToolObservation(
                sequence=sequence,
                tool_name=name,
                canonical_arguments=canonical_arguments,
                result=result if not truncated else None,
                truncated=truncated,
                original_bytes=original_bytes,
                retained_bytes=0 if truncated else original_bytes,
                content_digest=content_digest,
                replay_eligible=not truncated,
                duration_ms=duration_ms,
            )
        )

    def supports_recorded_environment_replay(self) -> bool:
        return any(observation.replay_eligible for observation in self.observations)

    def replay_tool_call(self, *, sequence: int, tool_name: str, arguments: object) -> object:
        if sequence < 0 or sequence >= len(self.observations):
            raise RecordedEnvironmentReplayError("recorded tool observation mismatch")
        observation = self.observations[sequence]
        if (
            observation.tool_name != tool_name
            or observation.canonical_arguments != canonical_json(arguments)
            or not observation.replay_eligible
        ):
            raise RecordedEnvironmentReplayError("recorded tool observation mismatch")
        return observation.result


class CugaWrapper:
    """Expose a compact, JSON-oriented interface over an injected runtime."""

    def __init__(
        self,
        runtime: CugaRuntime,
        settings: RuntimeSettings,
        trace_config: TraceConfig | None = None,
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._trace_config = trace_config if trace_config is not None else TraceConfig()

    @classmethod
    def from_cuga(
        cls, settings: RuntimeSettings, trace_config: TraceConfig | None = None
    ) -> "CugaWrapper":
        return cls(CugaSdkRuntime.from_settings(settings, trace_config), settings, trace_config)

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
        causal_trace_path = self._maybe_write_trace(task_id, trace)
        if causal_trace_path is not None:
            result["causal_trace_path"] = str(causal_trace_path)
        return result

    def _maybe_write_trace(
        self, task_id: str, trace: Mapping[str, object]
    ) -> Path | None:
        if not self._trace_config.enabled:
            return None
        existing = trace.get("causal_trace_path")
        if existing:
            return Path(str(existing))
        causal = _build_generic_causal_trace(
            run_id=str(uuid.uuid4()),
            task_id=task_id,
            runtime_result=trace,
            model=self._settings.model,
            config=self._trace_config,
        )
        return TraceWriter(self._trace_config).write(causal)

    def supports_recorded_environment_replay(self) -> bool:
        return self._trace_config.enabled

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
        trace_config: TraceConfig | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._artifacts = dict(artifacts or {})
        self._workspace_root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
        self._trace_config = trace_config if trace_config is not None else TraceConfig()

    @classmethod
    def from_settings(
        cls, settings: RuntimeSettings, trace_config: TraceConfig | None = None
    ) -> "CugaSdkRuntime":
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

        return cls(build_agent, trace_config=trace_config)

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

        run_id = str(uuid.uuid4())
        thread_id = run_id
        invoke_kwargs: dict[str, object] = {"track_tool_calls": True}
        thread_id_source = "wrapper_generated_not_injected"
        try:
            params = inspect.signature(agent.invoke).parameters
        except (TypeError, ValueError):
            params = {}
        if "thread_id" in params:
            invoke_kwargs["thread_id"] = thread_id
            thread_id_source = "wrapper_generated_injected"

        started_at = _now_iso()
        try:
            result = asyncio.run(_execute(agent, message, memory_docs, invoke_kwargs))
            error = getattr(result, "error", None)
            tool_calls = list(getattr(result, "tool_calls", ()) or ())
            final_output = str(getattr(result, "answer", ""))
        except Exception as exc:  # noqa: BLE001 - captured as runtime evidence
            error = repr(exc)
            tool_calls = []
            final_output = ""
        finally:
            try:
                asyncio.run(agent.aclose())
            except Exception:  # noqa: BLE001 - cleanup must not mask evidence
                pass
        completed_at = _now_iso()

        events = [
            {
                "event_id": f"{task_id}:tool:{index}",
                "kind": "tool_call",
                "tool_call": tool_call,
            }
            for index, tool_call in enumerate(tool_calls)
        ]
        available = {"instructions", "tools"}
        for field_name in ("skills", "memory", "policies"):
            if harness_config.get(field_name):
                available.add(field_name)
        metadata = _artifact_metadata(harness_config, available=available)

        result_dict: dict[str, object] = {
            "task_id": task_id,
            "status": "error" if error else "success",
            "final_output": final_output,
            "events": events,
            **metadata,
        }

        if self._trace_config.enabled:
            causal_trace_path = self._write_trace(
                run_id=run_id,
                task_id=task_id,
                thread_id=thread_id,
                thread_id_source=thread_id_source,
                status=str(result_dict["status"]),
                final_output=final_output,
                events=events,
                metadata=metadata,
                agent=agent,
                started_at=started_at,
                completed_at=completed_at,
            )
            result_dict["causal_trace_path"] = str(causal_trace_path)

        return result_dict

    def _compute_capabilities(self, agent: object) -> dict[str, FacilityCapability]:
        config = self._trace_config
        has_stream = callable(getattr(agent, "stream", None))
        has_graph = getattr(agent, "graph", None) is not None
        return {
            "stream_events": _facility(config.capture_stream_events, available=has_stream),
            "graph_final_state": _facility(config.capture_graph_final_state, available=has_graph),
            "graph_history": (
                FacilityCapability(status="disabled_by_config")
                if not config.capture_graph_history
                else FacilityCapability(
                    status="unavailable_no_checkpointer",
                    reason="no verified active checkpointer exposed by this runtime",
                )
            ),
            "tool_observations": _facility(config.capture_tool_observations),
            "external_correlation": _facility(config.capture_external_correlation),
        }

    def _write_trace(
        self,
        *,
        run_id: str,
        task_id: str,
        thread_id: str,
        thread_id_source: str,
        status: str,
        final_output: str,
        events: list[dict[str, object]],
        metadata: Mapping[str, object],
        agent: object,
        started_at: str,
        completed_at: str,
    ) -> Path:
        causal = CausalTrace(
            run_id=run_id,
            task_id=task_id,
            thread_id=thread_id,
            thread_id_source=thread_id_source,
            harness_version=str(metadata.get("harness_version", "unversioned")),
            status=status,
            final_output=final_output,
            events=_events_from_dicts(events),
            capabilities=self._compute_capabilities(agent),
            captured_event_count=len(events),
            started_at=started_at,
            completed_at=completed_at,
        )
        return TraceWriter(self._trace_config).write(causal)

    def get_artifacts(self) -> dict[str, str]:
        return dict(self._artifacts)

    def update_artifact(self, artifact_id: str, content: str) -> None:
        self._artifacts[artifact_id] = content
