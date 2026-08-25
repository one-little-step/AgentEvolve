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
    StateSnapshot,
    ToolObservation,
    canonical_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_SKILLS_ROOT = PROJECT_ROOT / ".cuga" / "skills"
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "data" / "workspaces"

# Packaged CUGA per-agent model profile (?08 token-budget audit). The shipped
# settings.openai.toml caps agent.action at 400 max_tokens, which starves a
# reasoning model silently (reasoning consumes the budget before content).
# Absolute path on purpose: cuga's config loader does join(MODELS_DIR, value),
# and an absolute value routes here instead of into site-packages.
_PACKAGED_MODEL_SETTINGS = Path(__file__).with_name("settings.agentevolve.toml")
PACKAGED_MODEL_SETTINGS_PATH: Path | None = (
    _PACKAGED_MODEL_SETTINGS if _PACKAGED_MODEL_SETTINGS.is_file() else None
)

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


#: Body parameters that disable the upstream gateway's response cache.
#:
#: MUST travel inside ``extra_body``. Verified live on 2026-08-18 against both
#: ``azure/gpt-5.6-luna`` and ``gcp/gemini-3.6-flash``: four identical requests
#: returned ONE shared response ``id`` and identical text by default, and four
#: distinct ids with four distinct completions once these keys were sent.
#:
#: Two alternatives fail silently and must not be substituted:
#:
#: * ``extra_params={"caching": False}`` in model settings -- CUGA merges it into
#:   ``litellm_params`` but the langchain client is a pydantic model with no
#:   ``caching`` field and no extras, so it is discarded before the wire.
#: * a bare ``model_kwargs={"caching": False}`` -- reaches litellm (confirmed by
#:   wire capture) but is consumed as litellm's own client-side cache setting and
#:   never forwarded upstream; output stays byte-identical.
#:
#: Both forms are sent because the gateway honors either, and a gateway upgrade
#: that drops one should not silently re-enable caching.
CACHE_BYPASS_EXTRA_BODY: dict[str, object] = {
    "caching": False,
    "cache": {"no-cache": True},
}


def apply_response_cache_policy(model: object, *, disable_cache: bool) -> None:
    """Disable the upstream response cache on a constructed chat client.

    Rollout diversity is the evidence that RHO's ``G`` group and the genetic
    path's ``R`` repeats exist to gather. A cached repeat returns one observation
    N times, so variance collapses to zero while every counter still reports N
    rollouts -- evidence that looks abundant and is not.

    Never raises. ``model_kwargs`` is a langchain implementation detail; if a
    future client drops it, a less-diverse rollout is strictly better than a
    crashed run. Merges into any existing ``extra_body`` rather than replacing
    it, since that channel is shared with other provider parameters.
    """
    if not disable_cache:
        return
    existing = getattr(model, "model_kwargs", None)
    if existing is None and not hasattr(model, "model_kwargs"):
        return
    model_kwargs = dict(existing) if isinstance(existing, dict) else {}
    extra_body_raw = model_kwargs.get("extra_body")
    extra_body = dict(extra_body_raw) if isinstance(extra_body_raw, dict) else {}
    extra_body.update(CACHE_BYPASS_EXTRA_BODY)
    model_kwargs["extra_body"] = extra_body
    try:
        model.model_kwargs = model_kwargs  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive, never fail a rollout
        return


#: Marker attribute so repeated installs do not stack wrappers.
_CACHE_POLICY_MARKER = "_agent_evolve_cache_policy_installed"

#: Opt back INTO the upstream response cache. Set by ``--allow-response-cache``.
#:
#: An environment variable rather than a plain argument because ``--isolation
#: process`` executes every rollout in a child process that builds its own
#: wrapper: a constructor value in the parent never reaches it. This is the only
#: channel the serial and parallel paths share.
ALLOW_RESPONSE_CACHE_ENV = "AGENT_EVOLVE_ALLOW_RESPONSE_CACHE"


def response_cache_disabled(default: bool = True) -> bool:
    """Whether to disable the upstream response cache for this process.

    Only an explicit truthy opt-in re-enables caching. ``0``/``false``/empty are
    ignored, so a leftover blank export cannot silently restore cached rollouts
    and collapse the variance the run is trying to measure.
    """
    raw = os.getenv(ALLOW_RESPONSE_CACHE_ENV)
    if raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return default


def install_response_cache_policy(manager: object, *, disable_cache: bool) -> None:
    """Apply the cache policy to every LLM CUGA builds for this process.

    ``LLMManager`` is a process-wide singleton holding its own model cache, and
    ``get_model`` hands back clients through three paths: a pre-instantiated
    model, a cache hit, and a fresh construction. All three call
    ``_update_model_parameters`` on the way out, which makes it the only choke
    point that covers an entire multi-role run.

    Wrapping ``_create_llm_instance`` instead would miss every cache hit, so the
    first agent role would sample correctly and every later one would silently
    fall back to cached completions -- the worst outcome, because the run still
    looks instrumented.
    """
    if not disable_cache:
        return
    if getattr(manager, _CACHE_POLICY_MARKER, False):
        return
    original = getattr(manager, "_update_model_parameters", None)
    if not callable(original):
        return

    def _patched(model, *args, **kwargs):
        updated = original(model, *args, **kwargs)
        target = updated if updated is not None else model
        apply_response_cache_policy(target, disable_cache=True)
        return updated

    try:
        manager._update_model_parameters = _patched  # type: ignore[attr-defined]
        setattr(manager, _CACHE_POLICY_MARKER, True)
    except Exception:  # pragma: no cover - defensive, never fail a rollout
        return


def _construct_agent(
    harness_config: Mapping[str, object],
    default_tools: list,
    default_instructions: str | None,
    workspace_dir: str | None = None,
) -> object:
    """Build a CUGA agent using only the verified constructor surface.

    Candidate isolation requires BOTH the constructor argument and the
    ``CUGA_FOLDER`` environment variable:

    * ``cuga_folder`` must be bound whenever a workspace exists, not only for
      policies. CUGA resolves its skills directory from ``cuga_folder``
      (``skills.loader.get_skill_root``), so leaving it ``None`` makes a
      skills-only candidate fall back to ``<cwd>/.cuga/skills``.
    * ``CUGA_FOLDER`` must be exported because the constructor argument does
      not reach two consumers on this build: ``build_runtime_tools`` calls
      ``create_sandbox_tools(thread_id=...)`` without ``cuga_folder``, and
      ``prepare_node`` reads the env var directly to load playbooks.
    * ``reset_policy_storage`` must be set for candidate runs. CUGA persists
      policies in a process-global store at ``<cuga package>/dbs/cuga.db``
      (``config.DBS_DIR``) that survives every run and ignores
      ``cuga_folder``. Without a reset, a playbook written by any earlier run
      keeps matching for every later candidate. Reset clears the store, then
      ``auto_load_policies`` reloads only this candidate's playbooks.

    Without all three, candidates silently share stale state and behave
    identically -- evolution would measure nothing while appearing to run
    correctly.
    """
    from cuga import CugaAgent

    instructions = harness_config.get("instructions")
    config_tools = harness_config.get("tools")
    has_skills = bool(harness_config.get("skills"))
    has_policies = bool(harness_config.get("policies"))
    if workspace_dir:
        os.environ["CUGA_FOLDER"] = str(workspace_dir)
    else:
        # Never let a previous candidate's workspace leak into this run.
        os.environ.pop("CUGA_FOLDER", None)
    return CugaAgent(
        tools=config_tools if isinstance(config_tools, list) and config_tools else default_tools,
        special_instructions=str(instructions) if instructions else default_instructions,
        enable_knowledge=True,
        enable_skills=has_skills,
        skills_folder=workspace_dir if has_skills else None,
        cuga_folder=workspace_dir,
        auto_load_policies=has_policies,
        reset_policy_storage=bool(workspace_dir),
    )


def _safe_segment(name: str) -> str:
    """Sanitize a harness artifact name into a single safe path segment."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    sanitized = re.sub(r"\.{2,}", "_", sanitized)
    return sanitized or "artifact"


def _derive_description(body: str) -> str:
    """Derive a short skill/policy description from the first body line.

    Markdown heading markers are stripped and the result is safe to emit as a
    quoted YAML scalar. An unquoted value beginning with ``#`` is a YAML
    comment, which silently yields ``description: None`` and makes CUGA's
    skill loader reject the file ("missing name or description") -- the skill
    then never reaches the model even though the file exists on disk.
    """
    for line in body.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return _yaml_scalar(stripped[:120])
    return "Harness artifact"


def _yaml_scalar(text: str) -> str:
    """Make arbitrary text safe as a double-quoted YAML scalar.

    Colons, quotes and backslashes in derived text otherwise produce invalid
    frontmatter, and CUGA drops the whole artifact while the run still
    succeeds -- a silent capability loss.
    """
    return text.replace("\\", " ").replace('"', "'")


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
                f'---\nname: {segment}\ndescription: "{_derive_description(str(body))}"\n---\n{body}\n',
                encoding="utf-8",
            )

    if isinstance(policies, Mapping):
        policy_dir = workspace / "playbooks"
        policy_dir.mkdir(parents=True, exist_ok=True)
        for name, content in policies.items():
            segment = _safe_segment(str(name))
            body = str(content)
            # Trigger choice is load-bearing, not cosmetic. Verified against cuga
            # 0.3.1: ``PolicyAgent.match_policy`` builds candidates only from
            # ``_evaluate_keyword_triggered_policies`` (filters ``KeywordTrigger``)
            # and ``_evaluate_natural_language_policies``. No evaluator selects an
            # ``AlwaysTrigger``, so a playbook carrying only ``always: true``
            # deserializes correctly and then never matches -- the policy artifact
            # would be silently inert and impossible to optimize against.
            #
            # A natural-language trigger derived from the policy text is therefore
            # emitted as the primary matcher (LLM-validated against the user
            # intent), with ``always: true`` retained as forward-compatible intent
            # in case a future CUGA release evaluates it.
            #
            # ``id`` is required: ``filesystem_sync`` reconciles storage against
            # frontmatter ids and deletes any policy it cannot find on disk.
            #
            # The trigger phrase is emitted as a double-quoted YAML scalar: policy
            # text routinely contains ``:`` (e.g. "end with the line: MARKER"),
            # and an unquoted scalar makes CUGA reject the whole file with
            # "Invalid YAML in frontmatter: mapping values are not allowed here",
            # which silently drops the policy.
            trigger_phrase = _derive_description(body).replace("\\", " ").replace('"', "'")
            (policy_dir / f"{segment}.md").write_text(
                "---\n"
                f"name: {segment}\n"
                f"id: playbook_{segment}\n"
                "triggers:\n"
                "  natural_language:\n"
                f'    - "{trigger_phrase}"\n'
                "  target: intent\n"
                "  threshold: 0.5\n"
                "  always: true\n"
                "---\n"
                f"{body}\n",
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
    """Initialize the agent, ingest any memory documents, then invoke once.

    ``initialize()`` must be awaited explicitly: ``CugaAgent.invoke()`` does
    not initialize the policy system on this build (that lazy init lives in
    ``CugaSupervisor.invoke()``, a different class). Since the policy reset
    only runs inside ``initialize()``, skipping it leaves CUGA's
    process-global policy store carrying playbooks from earlier candidates.

    Failures propagate deliberately: a rollout contaminated by another
    candidate's policy is worse than a failed rollout, because it looks
    like valid evidence.
    """
    initialize = getattr(agent, "initialize", None)
    if callable(initialize):
        await initialize()
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
        # ?08: default to the packaged profile (action agent at 16000 tokens,
        # not the shipped 400 that silently starves a reasoning model). An
        # explicit AGENT_SETTING_CONFIG always wins.
        default_profile = (
            str(_PACKAGED_MODEL_SETTINGS)
            if _PACKAGED_MODEL_SETTINGS.is_file()
            else "settings.openai.toml"
        )
        os.environ["AGENT_SETTING_CONFIG"] = os.getenv("AGENT_SETTING_CONFIG") or default_profile
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
    capture_node_payloads: bool = True
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
    """Normalize raw event dicts, keeping graph fields at the top level.

    ``parent_event_id``, ``actor_id`` and ``timestamp`` are the traversal fields
    on :class:`CausalEvent`. Letting them fall into ``payload`` leaves the trace
    a flat list that no analyzer can walk as a DAG.
    """
    reserved = {"event_id", "kind", "parent_event_id", "actor_id", "timestamp", "sequence"}
    result: list[CausalEvent] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        payload = {str(key): value for key, value in event.items() if key not in reserved}
        parent_event_id = event.get("parent_event_id")
        actor_id = event.get("actor_id")
        timestamp = event.get("timestamp")
        result.append(
            CausalEvent(
                event_id=str(event.get("event_id", f"event-{index}")),
                sequence=index,
                kind=str(event.get("kind", "runtime_update")),
                actor_id=str(actor_id) if actor_id is not None else None,
                parent_event_id=str(parent_event_id) if parent_event_id is not None else None,
                timestamp=str(timestamp) if timestamp is not None else None,
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

    def write(
        self,
        trace: CausalTrace,
        *,
        payload_store: "PayloadStore | None" = None,
        topology: Mapping[str, object] | None = None,
    ) -> Path:
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
            self._write_payload_blobs(staging, payload_store)
            self._write_topology(staging, topology)
            staging.replace(run_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return run_dir

    def _write_topology(self, staging: Path, topology: Mapping[str, object] | None) -> None:
        """Write the declared graph topology as a sidecar file.

        Kept out of ``CausalTrace`` (which forbids extra fields) so the
        agent-neutral schema stays unchanged while adapters can still record the
        structure their runtime declares.
        """
        if not topology:
            return
        (staging / "graph-topology.json").write_text(_json_dumps(topology), encoding="utf-8")

    def _write_payload_blobs(self, staging: Path, payload_store: "PayloadStore | None") -> None:
        """Write verbatim payload blobs, bypassing the redaction gateway.

        Deliberate and opt-in: the gateway truncates at 2000 chars and rejects
        field names common in agent state, which would destroy the exact prompts
        and states needed to reconstruct a subagent. Requires
        ``PayloadLevel.RAW_OPT_IN`` with ``allow_raw_payloads=True``.
        """
        if payload_store is None or not self._config.capture_node_payloads:
            return
        blobs = payload_store.blobs
        if not blobs:
            return
        payloads_dir = staging / "payloads"
        payloads_dir.mkdir(parents=True, exist_ok=True)
        for digest, serialized in blobs.items():
            (payloads_dir / f"{digest}.json").write_text(serialized, encoding="utf-8")

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


class SingleCallReplayError(RuntimeError):
    """Raised when a single recorded LLM call cannot be re-issued or decoded."""


# Recorded LangChain message discriminators mapped onto OpenAI chat roles. The
# recorded blobs carry both ``__type__`` (``SystemMessage``) and a lowercase
# ``type`` (``system``); the lowercase discriminator is authoritative here.
_RECORDED_ROLE_BY_TYPE = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "function": "function",
}
_RECORDED_ROLE_BY_CLASS = {
    "SystemMessage": "system",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "AIMessageChunk": "assistant",
    "ToolMessage": "tool",
    "FunctionMessage": "function",
}


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One recorded LLM call, reduced to the inputs needed to re-issue it.

    ``messages`` is a provider-ready ``[{"role": ..., "content": ...}]`` list
    rebuilt from the recorded LangChain message batch, including the system
    prompt. ``baseline_response`` is the text the recorded run produced, when a
    paired ``llm_call_end`` event carried a resolvable ``response_ref``.
    """

    event_id: str
    model: str | None
    messages: list[dict[str, str]]
    baseline_response: str | None = None

    @property
    def has_system_message(self) -> bool:
        return any(message.get("role") == "system" for message in self.messages)

    @property
    def total_content_chars(self) -> int:
        return sum(len(message.get("content", "")) for message in self.messages)


def _trace_document(trace_dir: Path) -> dict[str, object]:
    path = trace_dir / "causal-trace.json"
    if not path.exists():
        raise FileNotFoundError(f"no causal-trace.json under {trace_dir}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"causal-trace.json in {trace_dir} is not an object")
    return document


def _load_payload_blob(trace_dir: Path, ref: str) -> object:
    """Resolve one content-addressed payload blob.

    The on-disk layout written by ``TraceWriter`` is ``payloads/<sha256>.json``
    where ``<sha256>`` is exactly the ``*_ref`` digest recorded in the event
    payload.
    """
    path = trace_dir / "payloads" / f"{ref}.json"
    if not path.exists():
        raise FileNotFoundError(f"payload blob {ref} is missing under {trace_dir / 'payloads'}")
    return json.loads(path.read_text(encoding="utf-8"))


def _recorded_message_batch(blob: object) -> list[Mapping[str, object]]:
    """Flatten a recorded ``messages_ref`` blob into a single message list.

    LangChain's ``on_chat_model_start`` callback receives a list of message
    *batches* (one per prompt), so the recorded blob is ``[[msg, msg, ...]]``.
    Only the first batch is replayable as a single call.
    """
    if isinstance(blob, Mapping):
        return [blob]
    if not isinstance(blob, Sequence) or isinstance(blob, (str, bytes)):
        return []
    items = list(blob)
    if items and isinstance(items[0], Sequence) and not isinstance(items[0], (str, bytes)):
        items = list(items[0])
    return [item for item in items if isinstance(item, Mapping)]


def _recorded_role(message: Mapping[str, object]) -> str:
    explicit = message.get("role")
    if isinstance(explicit, str) and explicit:
        return explicit
    discriminator = message.get("type")
    if isinstance(discriminator, str) and discriminator in _RECORDED_ROLE_BY_TYPE:
        return _RECORDED_ROLE_BY_TYPE[discriminator]
    class_name = message.get("__type__")
    if isinstance(class_name, str) and class_name in _RECORDED_ROLE_BY_CLASS:
        return _RECORDED_ROLE_BY_CLASS[class_name]
    return "user"


def _recorded_content(message: Mapping[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    # Multimodal/blocked content is recorded as a list of parts; keep the text.
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def _baseline_response_text(blob: object) -> str | None:
    """Extract the assistant text from a recorded ``LLMResult`` payload blob."""
    if isinstance(blob, str):
        return blob
    if not isinstance(blob, Mapping):
        return None
    generations = blob.get("generations")
    if not isinstance(generations, Sequence):
        return None
    for group in generations:
        candidates = group if isinstance(group, Sequence) and not isinstance(group, str) else [group]
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            text = candidate.get("text")
            if isinstance(text, str) and text:
                return text
            message = candidate.get("message")
            if isinstance(message, Mapping):
                content = _recorded_content(message)
                if content:
                    return content
    return None


def list_recorded_llm_calls(trace_dir: Path | str) -> tuple[str, ...]:
    """Return the ``event_id`` of every ``llm_call_start`` event, in trace order."""
    directory = Path(trace_dir)
    events = _trace_document(directory).get("events")
    if not isinstance(events, Sequence):
        return ()
    starts = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("kind") == "llm_call_start"
    ]
    starts.sort(key=lambda event: (event.get("sequence") or 0))
    return tuple(str(event.get("event_id")) for event in starts)


def load_recorded_call(trace_dir: Path | str, event_id: str) -> RecordedCall:
    """Load one recorded LLM call's replay inputs from a persisted trace directory.

    Resolves the ``llm_call_start`` event's ``messages_ref`` payload blob into a
    provider-ready message list, and pairs the event with its ``llm_call_end``
    sibling (matched on ``payload["run_id"]``) to recover the baseline response
    text from ``response_ref`` when one was recorded.

    The model is taken from the trace document's top-level ``model`` field, and
    falls back to ``manifest.json`` when the trace omits it. Individual events do
    not record a model in this trace format.
    """
    directory = Path(trace_dir)
    document = _trace_document(directory)
    events = document.get("events")
    if not isinstance(events, Sequence):
        raise KeyError(f"{event_id} not found: trace has no events")

    start: Mapping[str, object] | None = None
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if str(event.get("event_id")) == event_id and event.get("kind") == "llm_call_start":
            start = event
            break
    if start is None:
        raise KeyError(f"{event_id} is not an llm_call_start event in {directory}")

    payload = start.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    messages_ref = payload.get("messages_ref")
    if not isinstance(messages_ref, str) or not messages_ref:
        raise ValueError(f"{event_id} has no messages_ref and carries no messages")

    recorded = _recorded_message_batch(_load_payload_blob(directory, messages_ref))
    if not recorded:
        raise ValueError(f"{event_id} resolved to no messages")
    messages = [
        {"role": _recorded_role(message), "content": _recorded_content(message)}
        for message in recorded
    ]

    baseline: str | None = None
    run_id = payload.get("run_id")
    if run_id is not None:
        for event in events:
            if not isinstance(event, Mapping) or event.get("kind") != "llm_call_end":
                continue
            end_payload = event.get("payload")
            end_payload = end_payload if isinstance(end_payload, Mapping) else {}
            if end_payload.get("run_id") != run_id:
                continue
            response_ref = end_payload.get("response_ref")
            if isinstance(response_ref, str) and response_ref:
                try:
                    baseline = _baseline_response_text(
                        _load_payload_blob(directory, response_ref)
                    )
                except (FileNotFoundError, ValueError):
                    baseline = None
            break

    model = document.get("model")
    if not isinstance(model, str) or not model:
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ValueError:
                manifest = {}
            candidate = manifest.get("model") if isinstance(manifest, Mapping) else None
            model = candidate if isinstance(candidate, str) and candidate else None
        else:
            model = None

    return RecordedCall(
        event_id=event_id,
        model=model,
        messages=messages,
        baseline_response=baseline,
    )


def _completion_choice_texts(response: object) -> tuple[str, ...]:
    """Read assistant texts out of an OpenAI/litellm-shaped completion response."""
    choices = (
        response.get("choices")
        if isinstance(response, Mapping)
        else getattr(response, "choices", None)
    )
    if not choices:
        raise SingleCallReplayError("completion response carried no choices")
    texts: list[str] = []
    for choice in choices:
        message = (
            choice.get("message")
            if isinstance(choice, Mapping)
            else getattr(choice, "message", None)
        )
        content = (
            message.get("content")
            if isinstance(message, Mapping)
            else getattr(message, "content", None)
        )
        if content is None and not isinstance(choice, Mapping):
            content = getattr(choice, "text", None)
        texts.append("" if content is None else str(content))
    return tuple(texts)


def replay_single_llm_call(
    call: RecordedCall,
    *,
    messages: Sequence[Mapping[str, object]] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    n: int = 1,
    completion_fn: Callable[..., object] | None = None,
) -> tuple[str, ...]:
    """Re-issue ONE recorded LLM call. This is NOT agent-state or checkpoint replay.

    This function replays a single LLM call reconstructed from a recorded
    ``messages_ref`` blob. It does not reconstruct agent state, does not restore
    a LangGraph checkpoint, and does not resume a trajectory. Nothing here makes
    counterfactual agent replay available: the recorded checkpoint payloads carry
    no ``channel_values``, so agent state cannot be rebuilt from this trace
    format. ``supports_counterfactual_replay`` must stay ``False``.

    Messages and model default to the recorded values so the baseline call is
    reproduced; pass ``messages``/``model`` to substitute counterfactual prompt
    content or a different model. ``temperature`` is forwarded only when supplied
    so provider defaults are otherwise untouched.

    ``n`` samples are requested with the provider's ``n`` parameter. Providers
    that ignore ``n`` and return a single choice are topped up with additional
    sequential calls until ``n`` texts are collected, so ``n`` may cost up to
    ``n`` separate requests.

    Connection settings come from ``RuntimeSettings.from_env`` (``CUGA_BASE_URL``
    / ``CUGA_API_KEY`` and their ``LITELLM_*`` aliases); the recorded model is
    still used unless overridden. Credentials are never returned or logged. Pass
    ``completion_fn`` to inject a completion callable for offline tests; the
    default performs a live ``litellm.completion`` call.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    payload_messages = [dict(message) for message in (messages or call.messages)]
    if not payload_messages:
        raise ValueError("replay requires at least one message")
    target_model = model or call.model
    if not target_model:
        raise SingleCallReplayError(
            "no model available for replay: trace recorded none and no override was given"
        )

    request: dict[str, object] = {"model": target_model, "messages": payload_messages}
    if temperature is not None:
        request["temperature"] = temperature

    # Reuse the wrapper's configured connection settings. ``RuntimeSettings``
    # requires a model in the environment, but a replay is governed by the
    # recorded model, so an unconfigured model must not block it.
    try:
        settings: RuntimeSettings | None = RuntimeSettings.from_env()
    except RuntimeError:
        settings = None
    base_url = settings.base_url if settings else (
        os.environ.get("CUGA_BASE_URL") or os.environ.get("LITELLM_BASE_URL")
    )
    api_key = settings.api_key if settings else (
        os.environ.get("CUGA_API_KEY") or os.environ.get("LITELLM_API_KEY")
    )
    if base_url:
        request["api_base"] = base_url
    if api_key:
        request["api_key"] = api_key

    invoke = completion_fn if completion_fn is not None else _litellm_completion
    collected: list[str] = []
    attempts = 0
    while len(collected) < n and attempts < n:
        attempts += 1
        remaining = n - len(collected)
        call_request = dict(request)
        if remaining > 1 or n > 1:
            call_request["n"] = remaining
        collected.extend(_completion_choice_texts(invoke(**call_request))[:remaining])
    if len(collected) < n:
        raise SingleCallReplayError(
            f"requested {n} completions but the provider returned {len(collected)}"
        )
    return tuple(collected[:n])


def _litellm_completion(**request: object) -> object:
    """Perform the live single-call completion. Imported lazily to keep tests offline."""
    import litellm

    return litellm.completion(**request)


class PayloadStore:
    """Content-addressed store for verbatim callback payloads.

    Payloads are kept whole and unsanitized: reconstructing a subagent's exact
    pre/post state, prompt and response is the entire point, and the shared
    persistence gateway would truncate strings at 2000 chars and hard-fail on
    field names such as ``token`` or ``label`` that occur naturally in agent
    state. Callers must therefore opt in with
    ``PayloadLevel.RAW_OPT_IN``/``allow_raw_payloads``.

    Content addressing means identical state dicts - common, because every node
    sees a near-identical ``AgentState`` - collapse into one blob.
    """

    def __init__(self) -> None:
        self._blobs: dict[str, str] = {}

    def put(self, value: object) -> str | None:
        """Store ``value`` verbatim and return its digest, or None if unusable."""
        if value is None:
            return None
        try:
            serialized = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self._blobs.setdefault(digest, serialized)
        return digest

    @property
    def blobs(self) -> dict[str, str]:
        return dict(self._blobs)


def _json_safe(value: object, depth: int = 0) -> object:
    """Best-effort JSON projection that preserves scalar content verbatim.

    Unlike the canonicalizer in ``core.trace`` this never raises: a payload that
    cannot be represented is reduced to a typed marker so one exotic object in
    agent state cannot discard the whole trajectory.
    """
    if depth > 12:
        return {"__truncated__": "max_depth"}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in value]
    for attribute in ("model_dump", "dict"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return {"__type__": type(value).__name__, **_json_safe(method(), depth + 1)}
            except Exception:  # noqa: BLE001 - fall through to other strategies
                pass
    # ``__slots__`` classes (notably LangGraph's ``Command``) have an empty
    # ``vars()``, so projecting via ``__dict__`` alone silently discards the
    # routing decision. Read declared slots explicitly first.
    slot_names: list[str] = []
    for klass in type(value).__mro__:
        for name in getattr(klass, "__slots__", ()) or ():
            if name not in slot_names:
                slot_names.append(str(name))
    if slot_names:
        projected: dict[str, object] = {}
        for name in slot_names:
            try:
                projected[name] = _json_safe(getattr(value, name), depth + 1)
            except AttributeError:
                continue
        if projected:
            return {"__type__": type(value).__name__, **projected}
    if hasattr(value, "__dict__"):
        try:
            attributes = vars(value)
        except TypeError:
            attributes = {}
        if attributes:
            try:
                return {"__type__": type(value).__name__, **_json_safe(attributes, depth + 1)}
            except Exception:  # noqa: BLE001
                pass
    return {"__type__": type(value).__name__, "__repr__": repr(value)}


def _routing_target(value: object) -> str | None:
    """Extract the branch a node routed to, when it returned a routing object."""
    goto = getattr(value, "goto", None)
    if goto is None and isinstance(value, Mapping):
        goto = value.get("goto")
    if goto is None:
        return None
    if isinstance(goto, (list, tuple)):
        return ",".join(str(item) for item in goto) or None
    return str(goto)


def load_node_state(
    trace_dir: Path | str,
    *,
    node: str | None = None,
    event_id: str | None = None,
    with_provenance: bool = False,
):
    """Lazily resolve one node's (before, after) state from a persisted trace.

    Reads only ``events.jsonl`` and the referenced blobs, so recovering one
    subagent's state never materializes the whole trajectory.

    The post-state is honest about its origin. Most CUGA nodes return a
    LangGraph ``Command`` rather than full state, so the "after" is derived by
    applying ``Command.update`` onto the pre-state; a raw routing object is never
    returned as though it were a state. Pass ``with_provenance=True`` to receive
    ``(before, after, provenance)`` where ``provenance["after_source"]`` is one
    of ``chain_end_outputs``, ``command_update`` or ``unavailable``.
    """
    directory = Path(trace_dir)
    events_path = directory / "events.jsonl"
    if not events_path.exists():
        return (None, None, {"after_source": "unavailable"}) if with_provenance else (None, None)

    start: Mapping[str, object] | None = None
    end: Mapping[str, object] | None = None
    target_run: str | None = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        payload = event.get("payload") or {}
        if event.get("kind") == "graph_node_start" and start is None:
            if event_id is not None and event.get("event_id") != event_id:
                continue
            if node is not None and event.get("actor_id") != node:
                continue
            start = event
            target_run = payload.get("run_id")
        elif event.get("kind") == "graph_node_end" and start is not None and end is None:
            if target_run is not None and payload.get("run_id") != target_run:
                continue
            end = event

    def resolve(event: Mapping[str, object] | None, key: str) -> object | None:
        if event is None:
            return None
        reference = (event.get("payload") or {}).get(key)
        if not reference:
            return None
        blob = directory / "payloads" / f"{reference}.json"
        if not blob.exists():
            return None
        return json.loads(blob.read_text(encoding="utf-8"))

    before = resolve(start, "state_before_ref")
    raw_after = resolve(end, "state_after_ref")
    before_state = before if isinstance(before, Mapping) else None

    after_state: Mapping[str, object] | None = None
    after_source = "unavailable"
    if isinstance(raw_after, Mapping) and "__type__" not in raw_after:
        # A genuine state mapping: the node returned full state.
        after_state = raw_after
        after_source = "chain_end_outputs"
    elif isinstance(raw_after, Mapping):
        update = raw_after.get("update")
        if isinstance(update, Mapping):
            merged = dict(before_state or {})
            merged.update(update)
            after_state = merged
            after_source = "command_update"

    if with_provenance:
        provenance = {
            "after_source": after_source,
            "after_type": raw_after.get("__type__") if isinstance(raw_after, Mapping) else None,
            "routed_to": (end or {}).get("payload", {}).get("routed_to") if end else None,
        }
        return before_state, after_state, provenance
    return before_state, after_state


def _graph_topology(agent: object) -> dict[str, object] | None:
    """Extract the compiled graph's declared nodes and edges.

    Observed adjacency shows what happened; the declared topology shows what was
    *permitted*. Blame attribution needs both, so a transition can be judged a
    legal branch rather than an anomaly. ``conditional`` marks routing edges.

    Note: on this CUGA build ``get_graph(xray=True)`` returns exactly the same
    10 nodes / 15 edges as ``get_graph()``, so subgraph internals are not
    expanded; the non-xray form is used and no expansion is claimed.
    """
    graph = getattr(agent, "graph", None)
    get_graph = getattr(graph, "get_graph", None)
    if not callable(get_graph):
        return None
    try:
        drawable = get_graph()
    except Exception:  # noqa: BLE001 - tracing must never break a run
        return None
    nodes: list[str] = []
    try:
        nodes = sorted(str(name) for name in getattr(drawable, "nodes", ()) or ())
    except Exception:  # noqa: BLE001
        nodes = []
    edges: list[dict[str, object]] = []
    for edge in getattr(drawable, "edges", ()) or ():
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        if source is None or target is None:
            continue
        edges.append(
            {
                "source": str(source),
                "target": str(target),
                "conditional": bool(getattr(edge, "conditional", False)),
            }
        )
    if not nodes and not edges:
        return None
    return {"nodes": nodes, "edges": edges}


def _event_sort_key(event: Mapping[str, object], index: int) -> tuple[str, int]:
    """Order events by observed time, falling back to arrival order.

    Callback events and the SDK's post-hoc tool-call report are two separate
    lists; concatenating them made ``sequence`` encode "which list" instead of
    "when", placing tool calls before the nodes that issued them.
    """
    timestamp = event.get("timestamp")
    if not timestamp:
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            timestamp = payload.get("timestamp")
            if not timestamp:
                inner = payload.get("tool_call")
                if isinstance(inner, Mapping):
                    timestamp = inner.get("timestamp")
    if not timestamp:
        nested = event.get("tool_call")
        if isinstance(nested, Mapping):
            timestamp = nested.get("timestamp")
    return (_normalize_timestamp(str(timestamp)) if timestamp else "", index)


def _normalize_timestamp(value: str) -> str:
    """Normalize to a lexicographically comparable UTC form."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    return text


def _sorted_events(events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return events in chronological order, preserving ties by arrival."""
    decorated = sorted(
        ((_event_sort_key(event, index), event) for index, event in enumerate(events)),
        key=lambda pair: pair[0],
    )
    return [dict(event) for _, event in decorated]


class GraphEventCollector:
    """Agent-neutral sink for graph node/tool lifecycle events.

    This holds no LangChain types. ``build_graph_callback_handler`` adapts it to
    the LangChain callback contract, which must be satisfied by real inheritance:
    LangChain's async dispatch reads handler attributes such as ``run_inline``
    (``langchain_core/callbacks/manager.py:471``), so a duck-typed handler raises
    ``AttributeError`` mid-run.

    Only structural identifiers are retained. Node inputs and outputs are not
    persisted, because they can carry evaluator internals or expected answers.
    """

    def __init__(self, *, max_events: int, payload_store: "PayloadStore | None" = None) -> None:
        self._max_events = max_events
        self.events: list[dict[str, object]] = []
        self.dropped_event_count = 0
        self.payload_store = payload_store
        # run_id -> (event_id, node) of the start event, so an edge can point at a
        # real recorded event and an end callback can recover its node name.
        # LangGraph omits ``langgraph_node`` on end callbacks and reuses node
        # names across nesting depths, so run_id is the only reliable key.
        self._run_index: dict[str, tuple[str, str | None]] = {}

    def store_payload(self, value: object) -> str | None:
        """Persist a verbatim payload, returning its content digest."""
        if self.payload_store is None:
            return None
        return self.payload_store.put(value)

    def record(
        self,
        kind: str,
        payload: Mapping[str, object],
        *,
        run_id: object = None,
        parent_run_id: object = None,
    ) -> None:
        if len(self.events) >= self._max_events:
            self.dropped_event_count += 1
            return
        event_id = f"graph:{len(self.events)}"
        run_key = str(run_id) if run_id is not None else None
        parent_key = str(parent_run_id) if parent_run_id is not None else None
        event: dict[str, object] = {
            "event_id": event_id,
            "kind": kind,
            "timestamp": _now_iso(),
            **{key: value for key, value in payload.items() if value is not None},
        }
        if run_key is not None:
            event["run_id"] = run_key
        if parent_key is not None:
            event["parent_run_id"] = parent_key
        node = event.get("node")
        if node is not None:
            event["actor_id"] = str(node)
        if run_key is not None and run_key not in self._run_index:
            self._run_index[run_key] = (event_id, str(node) if node is not None else None)
        self.events.append(event)

    def resolved_events(self) -> list[dict[str, object]]:
        """Return events with ``parent_event_id`` resolved from run identity.

        Resolution is deferred to here rather than done in :meth:`record` because
        LangGraph's outermost chain reports last: the root run's own callback is
        the final event while its children fire first. Resolving eagerly dropped
        every edge into the root and split one trajectory into several apparent
        roots. Only genuinely reported parents produce an edge - an unresolvable
        ``parent_run_id`` leaves no link rather than a guess.
        """
        resolved: list[dict[str, object]] = []
        for event in self.events:
            copy = dict(event)
            parent_key = copy.get("parent_run_id")
            entry = self._run_index.get(str(parent_key)) if parent_key else None
            if entry is not None and entry[0] != copy["event_id"]:
                copy["parent_event_id"] = entry[0]
            resolved.append(copy)
        return resolved

    def node_for_run(self, run_id: object) -> str | None:
        """Recover the node name recorded for a run id, if any."""
        if run_id is None:
            return None
        entry = self._run_index.get(str(run_id))
        return entry[1] if entry else None

    @staticmethod
    def node_name(serialized: object, metadata: object) -> str | None:
        if isinstance(metadata, Mapping):
            node = metadata.get("langgraph_node")
            if node:
                return str(node)
        if isinstance(serialized, Mapping):
            name = serialized.get("name")
            if name:
                return str(name)
        return None


def build_graph_callback_handler(collector: GraphEventCollector) -> object:
    """Adapt a :class:`GraphEventCollector` to the LangChain callback contract.

    ``BaseCallbackHandler`` is imported lazily so the module keeps importing
    without LangChain present, and subclassed rather than duck-typed so async
    dispatch finds every attribute it requires.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _GraphCallbackHandler(BaseCallbackHandler):
        # Signatures stay permissive: the caller is LangChain/LangGraph.
        def on_chain_start(self, serialized=None, inputs=None, **kwargs) -> None:  # noqa: ANN001
            node = collector.node_name(serialized, kwargs.get("metadata"))
            if node is None:
                return
            metadata = kwargs.get("metadata")
            step = metadata.get("langgraph_step") if isinstance(metadata, Mapping) else None
            collector.record(
                "graph_node_start",
                {
                    "node": node,
                    "step": step,
                    # Pre-state of this node: the "before" half of a counterfactual.
                    "state_before_ref": collector.store_payload(inputs),
                },
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_chain_end(self, outputs=None, **kwargs) -> None:  # noqa: ANN001
            # End callbacks omit ``langgraph_node``; recover it from the run id so
            # start/end pair exactly (verified 10/10 on a live run).
            node = collector.node_name(None, kwargs.get("metadata")) or collector.node_for_run(
                kwargs.get("run_id")
            )
            collector.record(
                "graph_node_end",
                {
                    "node": node,
                    "state_after_ref": collector.store_payload(outputs),
                    # Most CUGA nodes return Command(goto=...); surfacing the branch
                    # here makes routing queryable without opening a blob.
                    "routed_to": _routing_target(outputs),
                },
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_chat_model_start(self, serialized=None, messages=None, **kwargs) -> None:  # noqa: ANN001
            # Chat models never fire ``on_llm_start``; this is the only hook that
            # exposes the real prompt (verified: 3 calls, up to 39,506 bytes).
            collector.record(
                "llm_call_start",
                {"messages_ref": collector.store_payload(messages)},
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_llm_start(self, serialized=None, prompts=None, **kwargs) -> None:  # noqa: ANN001
            collector.record(
                "llm_call_start",
                {"messages_ref": collector.store_payload(prompts)},
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_llm_end(self, response=None, **kwargs) -> None:  # noqa: ANN001
            collector.record(
                "llm_call_end",
                {"response_ref": collector.store_payload(response)},
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_llm_error(self, error=None, **kwargs) -> None:  # noqa: ANN001
            collector.record(
                "llm_call_error",
                {"error": repr(error)},
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_chain_error(self, error=None, **kwargs) -> None:  # noqa: ANN001
            node = collector.node_name(None, kwargs.get("metadata")) or collector.node_for_run(
                kwargs.get("run_id")
            )
            collector.record(
                "graph_node_error",
                {"node": node, "error": repr(error)},
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_tool_start(self, serialized=None, input_str=None, **kwargs) -> None:  # noqa: ANN001
            name = serialized.get("name") if isinstance(serialized, Mapping) else None
            # ?14 Phase 1: persist the invocation verbatim so replay can tape
            # it. LangChain passes structured ``inputs``; older call sites may
            # only provide the string form.
            inputs = kwargs.get("inputs")
            args_source = inputs if inputs is not None else input_str
            collector.record(
                "graph_tool_start",
                {
                    "tool_name": str(name) if name else None,
                    "args_ref": collector.store_payload(args_source),
                },
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_tool_end(self, output=None, **kwargs) -> None:  # noqa: ANN001
            # ?14 Phase 1: the raw result is the tape for non-deterministic
            # tools during prefix replay (design R3); dropping it made every
            # trace unreplayable across anything externally coupled.
            collector.record(
                "graph_tool_end",
                {"output_ref": collector.store_payload(output)},
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

        def on_tool_error(self, error=None, **kwargs) -> None:  # noqa: ANN001
            collector.record(
                "graph_tool_error",
                {"error": repr(error)},
                run_id=kwargs.get("run_id"),
                parent_run_id=kwargs.get("parent_run_id"),
            )

    return _GraphCallbackHandler()


def _collector_tool_observations_captured(
    collector: "GraphEventCollector | None",
) -> bool:
    """Whether graph-layer capture persisted at least one tool RESULT blob.

    Start events alone do not count: without the recorded output a replay
    cannot tape a non-deterministic tool, which is the entire point (?14).
    """
    if collector is None:
        return False
    # In-memory events are FLAT; payload nesting appears only at
    # serialization time.
    return any(
        event.get("kind") == "graph_tool_end" and event.get("output_ref")
        for event in collector.events
    )


def _final_state_snapshot(agent: object, run_config: Mapping[str, object]) -> StateSnapshot | None:
    """Read the post-invoke graph state without re-executing the graph.

    ``CugaAgent.graph`` compiles with a ``MemorySaver`` checkpointer
    (``sdk.py:2291-2301``), so ``get_state`` reflects the state left by the run
    that just completed. The snapshot is reported ``replay_safe=False``: reading
    a final state is not a verified state-reconstruction capability.
    """
    graph = getattr(agent, "graph", None)
    get_state = getattr(graph, "get_state", None)
    if not callable(get_state):
        return None
    try:
        state = get_state(run_config)
    except Exception:  # noqa: BLE001 - absence of state is not a run failure
        return None
    if state is None:
        return None

    values = getattr(state, "values", None)
    config = getattr(state, "config", None)
    checkpoint_id = None
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        if isinstance(configurable, Mapping):
            checkpoint_id = configurable.get("checkpoint_id")

    try:
        serialized = canonical_json(_state_payload(values))
    except Exception:  # noqa: BLE001 - unserializable state still yields structure
        serialized = None

    return StateSnapshot(
        sequence=0,
        checkpoint_id=str(checkpoint_id) if checkpoint_id else None,
        state_hash=(
            f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"
            if serialized is not None
            else None
        ),
        payload=_state_payload(values),
        replay_safe=False,
    )


def _state_payload(values: object) -> object:
    """Reduce a graph state mapping to trace-safe structural keys."""
    if not isinstance(values, Mapping):
        return None
    return {
        "state_keys": sorted(str(key) for key in values.keys()),
        "next_nodes_pending": False,
    }


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
        duration_ms: float | None,
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

    def ingest_sdk_tool_calls(self, tool_calls: Sequence[object]) -> None:
        """Record observations from CUGA's returned ``InvokeResult.tool_calls``.

        This is the only tool surface a live run actually exposes. CUGA does not
        call ``tool.invoke``: ``prepare_node`` extracts ``tool.coroutine``,
        ``tool.func`` or ``tool._run`` and registers the bare callable through
        ``make_tool_awaitable`` (``cuga_agent_core/execution/code_extraction.py:120``),
        and the sandbox then calls that callable directly. So ``wrap()`` can never
        observe a live call, while ``track_tool_calls=True`` reports every call
        after the fact with name, arguments, result, duration and error.

        The trade-off is explicit: these are post-hoc SDK reports, not intercepted
        invocations, so timings come from CUGA rather than from our own clock.
        """
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            name = tool_call.get("name") or tool_call.get("operation_id")
            if not name:
                continue
            sequence = self._sequence
            self._sequence += 1
            canonical_arguments = canonical_json(tool_call.get("arguments"))
            duration = tool_call.get("duration_ms")
            duration_ms = float(duration) if isinstance(duration, (int, float)) else None
            if duration_ms is not None and duration_ms < 0:
                duration_ms = None
            reported_error = tool_call.get("error")
            if reported_error:
                # Keep failures as evidence, never as replayable ground truth.
                self.observations.append(
                    ToolObservation(
                        sequence=sequence,
                        tool_name=str(name),
                        canonical_arguments=canonical_arguments,
                        error=str(reported_error),
                        replay_eligible=False,
                        duration_ms=duration_ms,
                    )
                )
                continue
            self._append_recorded(
                str(name),
                sequence,
                canonical_arguments,
                tool_call.get("result"),
                duration_ms,
            )

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
        cls,
        settings: RuntimeSettings,
        trace_config: TraceConfig | None = None,
        disable_response_cache: bool = True,
    ) -> "CugaWrapper":
        return cls(
            CugaSdkRuntime.from_settings(
                settings, trace_config, disable_response_cache=disable_response_cache
            ),
            settings,
            trace_config,
        )

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
        model: str | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._artifacts = dict(artifacts or {})
        self._workspace_root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
        self._trace_config = trace_config if trace_config is not None else TraceConfig()
        self._model = model

    @classmethod
    def from_settings(
        cls,
        settings: RuntimeSettings,
        trace_config: TraceConfig | None = None,
        disable_response_cache: bool = True,
    ) -> "CugaSdkRuntime":
        settings.configure_cuga_environment()
        os.environ["SKILLS_ROOT"] = resolve_skills_root()
        _require_autonomous_mode()

        # Rollout diversity is the evidence RHO's ``G`` group and the genetic
        # path's ``R`` repeats exist to gather, so the upstream response cache is
        # disabled by default. Installed on the LLMManager singleton because that
        # is where CUGA builds a client for every agent role.
        from cuga.backend.llm.models import LLMManager

        install_response_cache_policy(
            LLMManager(),
            disable_cache=response_cache_disabled(disable_response_cache),
        )

        from agent_evolve.cuga_wrapper.tools import build_tools

        default_tools = build_tools()

        def build_agent(
            harness_config: Mapping[str, object],
            workspace_dir: str | None = None,
        ) -> object:
            return _construct_agent(harness_config, default_tools, DEFAULT_SPECIAL_INSTRUCTIONS, workspace_dir)

        return cls(
            build_agent,
            trace_config=trace_config,
            model=settings.public_config()["model"],
        )

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

        # Attach the graph event collector to the SAME invoke() call. CUGA merges
        # caller callbacks into that single execution, so node evidence needs no
        # second run; stream() would re-execute the graph and repeat side effects.
        collector: GraphEventCollector | None = None
        payload_store: PayloadStore | None = None
        if self._trace_config.enabled and self._trace_config.capture_stream_events and "config" in params:
            try:
                if self._trace_config.capture_node_payloads:
                    payload_store = PayloadStore()
                candidate = GraphEventCollector(
                    max_events=self._trace_config.max_events_per_trace,
                    payload_store=payload_store,
                )
                handler = build_graph_callback_handler(candidate)
            except Exception:  # noqa: BLE001 - tracing must never break a run
                collector = None
                payload_store = None
            else:
                collector = candidate
                invoke_kwargs["config"] = {
                    "configurable": {"thread_id": thread_id},
                    "callbacks": [handler],
                }

        started_at = _now_iso()
        final_state: StateSnapshot | None = None
        try:
            result = asyncio.run(_execute(agent, message, memory_docs, invoke_kwargs))
            error = getattr(result, "error", None)
            tool_calls = list(getattr(result, "tool_calls", ()) or ())
            final_output = str(getattr(result, "answer", ""))
        except Exception as exc:  # noqa: BLE001 - captured as runtime evidence
            error = repr(exc)
            tool_calls = []
            final_output = ""
        else:
            if self._trace_config.enabled and self._trace_config.capture_graph_final_state:
                final_state = _final_state_snapshot(agent, {"configurable": {"thread_id": thread_id}})
        finally:
            try:
                asyncio.run(agent.aclose())
            except Exception:  # noqa: BLE001 - cleanup must not mask evidence
                pass
        completed_at = _now_iso()

        events: list[dict[str, object]] = [
            {
                "event_id": f"{task_id}:tool:{index}",
                "kind": "tool_call",
                "tool_call": tool_call,
            }
            for index, tool_call in enumerate(tool_calls)
        ]
        if collector is not None:
            events.extend(collector.resolved_events())

        # Order by observed time so ``sequence`` means "when", not "which list".
        # Callback events and the SDK's post-hoc tool report are separate lists;
        # concatenating them put tool calls ahead of the nodes that issued them.
        events = _sorted_events(events)

        # Tool provenance comes from the SDK's post-hoc report, the only surface a
        # live CUGA run exposes (see ToolObservationRecorder.ingest_sdk_tool_calls).
        recorder: ToolObservationRecorder | None = None
        if self._trace_config.enabled and self._trace_config.capture_tool_observations:
            recorder = ToolObservationRecorder(self._trace_config)
            try:
                recorder.ingest_sdk_tool_calls(tool_calls)
            except Exception:  # noqa: BLE001 - tracing must never break a run
                recorder = None
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
        if error:
            # A failed run must say why. Without this, an exception collapses
            # into an opaque status="error" with no diagnosable evidence.
            result_dict["error"] = str(error)

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
                collector=collector,
                final_state=final_state,
                error=str(error) if error else None,
                recorder=recorder,
                payload_store=payload_store,
                topology=_graph_topology(agent),
            )
            result_dict["causal_trace_path"] = str(causal_trace_path)

        return result_dict

    def _compute_capabilities(
        self,
        agent: object,
        *,
        collector: "GraphEventCollector | None" = None,
        final_state: StateSnapshot | None = None,
        recorder: "ToolObservationRecorder | None" = None,
        payload_store: "PayloadStore | None" = None,
        topology: Mapping[str, object] | None = None,
    ) -> dict[str, FacilityCapability]:
        config = self._trace_config
        has_stream = callable(getattr(agent, "stream", None))
        has_graph = getattr(agent, "graph", None) is not None
        return {
            "stream_events": self._stream_events_capability(collector, has_stream=has_stream),
            "graph_final_state": _facility(
                config.capture_graph_final_state,
                available=has_graph,
                captured=final_state is not None,
            ),
            "graph_history": (
                FacilityCapability(status="disabled_by_config")
                if not config.capture_graph_history
                else FacilityCapability(
                    status="unavailable_no_checkpointer",
                    reason="no verified active checkpointer exposed by this runtime",
                )
            ),
            # Only claim capture when observations actually exist. A run where the
            # model never called a tool is not a tracing failure and must not be
            # reported as one, so absence stays "unavailable" rather than
            # "runtime_failure" (which would imply we lost real evidence).
            # ?14: graph-layer results count even when the SDK post-hoc
            # recorder has nothing -- either surface taping a tool is enough.
            "tool_observations": (
                FacilityCapability(status="disabled_by_config")
                if not config.capture_tool_observations
                else FacilityCapability(status="captured")
                if (recorder is not None and recorder.observations)
                or _collector_tool_observations_captured(collector)
                else FacilityCapability(
                    status="unavailable_no_sdk_surface",
                    reason="no tool calls reported by this run",
                )
            ),
            "external_correlation": _facility(config.capture_external_correlation),
            # Node payloads are the substrate for subagent-level simulation, so
            # claim capture only when blobs were actually written.
            # Declared topology: what the graph permits, vs what was observed.
            "graph_topology": (
                FacilityCapability(status="captured")
                if topology
                else FacilityCapability(
                    status="unavailable_no_sdk_surface",
                    reason="compiled graph exposes no drawable topology",
                )
            ),
            "node_payloads": (
                FacilityCapability(status="disabled_by_config")
                if not config.capture_node_payloads
                else FacilityCapability(status="captured")
                if payload_store is not None and payload_store.blobs
                else FacilityCapability(
                    status="unavailable_no_sdk_surface",
                    reason="no callback payloads observed",
                )
            ),
        }

    def _stream_events_capability(
        self,
        collector: "GraphEventCollector | None",
        *,
        has_stream: bool,
    ) -> FacilityCapability:
        """Report stream-event capture honestly, distinguishing every failure mode."""
        config = self._trace_config
        if not config.capture_stream_events:
            return FacilityCapability(status="disabled_by_config")
        if collector is None:
            if not has_stream:
                return FacilityCapability(status="unavailable_no_sdk_surface")
            return FacilityCapability(
                status="unavailable_no_sdk_surface",
                reason="invoke() does not accept a config argument for callbacks",
            )
        if not collector.events:
            return FacilityCapability(
                status="runtime_failure",
                reason="callback handler attached but emitted no events",
            )
        return FacilityCapability(status="captured")

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
        collector: "GraphEventCollector | None" = None,
        final_state: StateSnapshot | None = None,
        error: str | None = None,
        recorder: "ToolObservationRecorder | None" = None,
        payload_store: "PayloadStore | None" = None,
        topology: Mapping[str, object] | None = None,
    ) -> Path:
        causal = CausalTrace(
            run_id=run_id,
            task_id=task_id,
            thread_id=thread_id,
            thread_id_source=thread_id_source,
            harness_version=str(metadata.get("harness_version", "unversioned")),
            status=status,
            final_output=final_output,
            error=error,
            model=self._model,
            events=_events_from_dicts(events),
            checkpoints=(final_state,) if final_state is not None else (),
            tool_observations=tuple(recorder.observations) if recorder else (),
            capabilities=self._compute_capabilities(
                agent,
                collector=collector,
                final_state=final_state,
                recorder=recorder,
                payload_store=payload_store,
                topology=topology,
            ),
            captured_event_count=len(events),
            dropped_event_count=collector.dropped_event_count if collector else 0,
            events_truncated=bool(collector and collector.dropped_event_count),
            started_at=started_at,
            completed_at=completed_at,
        )
        return TraceWriter(self._trace_config).write(
            causal, payload_store=payload_store, topology=topology
        )

    def get_artifacts(self) -> dict[str, str]:
        return dict(self._artifacts)

    def update_artifact(self, artifact_id: str, content: str) -> None:
        self._artifacts[artifact_id] = content
