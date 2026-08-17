"""Bind a real CUGA rollout to the agent-neutral benchmark runner.

Why this module exists
---------------------
:func:`agent_evolve.benchmarks.runner.run_benchmark` knows how to fan tasks out,
isolate failures and keep an honest denominator, but it deliberately knows
nothing about any agent. Its single integration point is ``executor_factory``.
Until now the only factory in the repository was the offline replay factory in
``scripts/run_benchmark.py``, which returns answers a previous run already
recorded. That proves the runner works; it cannot measure a harness.

This module supplies the missing half: a factory that executes each benchmark
task with a **real CUGA agent**, against an **explicitly named harness
version**, and **persists a causal trace per rollout**.

Three properties are load-bearing, and each is enforced rather than documented
and hoped for.

1. **The harness version is explicit and required.**
   :class:`HarnessVersion` is constructed from a named built-in or a JSON file
   that must declare ``version``. That string is placed in the harness config
   under ``"version"``, which is exactly the key
   ``agent_evolve.cuga_wrapper._artifact_metadata`` reads to stamp
   ``harness_version`` onto the persisted trace and its manifest. A run is
   therefore self-describing: the trace on disk names the harness that produced
   it. We are evolving harnesses; a measurement that cannot name its harness is
   not a measurement.

2. **A rollout without a trace is a failed rollout.**
   The analyzer, the blame graph and
   :func:`agent_evolve.cuga_wrapper.load_recorded_call` all consume the
   persisted trace, not the answer string. Answers without traces are useless
   to us, so tracing is checked twice: the wrapper must report tracing enabled
   *before* the run starts (:class:`TracingDisabledError`), and every result
   must carry a trace path that actually exists on disk
   (:class:`MissingTraceError`). Neither check can be silently skipped.

3. **No answer is never a wrong answer.**
   A crashed harness, an ``status="error"`` result, a missing answer field or a
   blank answer all raise, so the runner records ``ok=False`` and excludes the
   task from the scoring denominator. Returning ``""`` instead would put a
   broken harness into the denominator as a wrong answer and quietly deflate
   every pass rate we report.

Concurrency, honestly
--------------------
The runner builds one executor per worker thread, which is the right shape for
a stateful agent. It is *not* sufficient for CUGA, and this was settled by
measurement rather than by reading. Two process-global singletons collide:

1. **The knowledge engine's exclusive lock.** ``KnowledgeEngine.__init__`` takes
   a non-blocking ``flock`` on ``<persist_dir>/.lock`` and raises
   ``RuntimeError("Knowledge engine already running in another process")`` when
   it cannot get it. ``flock`` conflicts between two file objects in one
   process, so threads collide exactly as separate processes do. This one is
   *fixable* in-process: two engines with **distinct** ``persist_dir`` both start
   (verified), and ``persist_dir`` is a supported override -- an empty
   ``[knowledge].persist_dir`` in CUGA's ``knowledge_settings.toml`` documents
   ``<cwd>/.cuga/knowledge`` as merely the default.

2. **``CUGA_FOLDER``, which is not fixable in-process.** ``_construct_agent``
   binds a candidate's workspace by exporting that single environment variable,
   because the constructor argument does not reach two consumers on this build
   (CUGA's sandbox executors and ``prepare_node`` read the env var directly).
   Threads share ``os.environ``. Two threads that each called the real
   ``_construct_agent`` with a different workspace were both left observing the
   *second* thread's workspace (verified). Serializing construction does not
   help: the read happens later, inside ``invoke()``.

Point 2 is disqualifying, and specifically for the case this project exists to
run: two candidates evaluated concurrently. A task could execute against another
candidate's skills while its trace still stamped its own ``harness_version``,
because that field is copied from the harness config and cannot detect the
swap. Silently wrong evidence is worse than slow evidence, so in-process
concurrency stays refused (:class:`ConcurrencyUnsupportedError`) and real
parallelism is per **process** -- see :data:`PROCESS_ISOLATION` and
:class:`WorkerPool`. A subprocess has its own ``os.environ`` and its own
knowledge and policy stores, which makes both globals per-worker by
construction.

A third concurrency hazard was found by actually running this code, not by
reading it: CUGA imports 172 modules lazily from inside ``invoke()``, and two
worker threads racing that import graph hit a CPython import-lock deadlock. See
:func:`warm_up_cuga_imports`, which :func:`preflight` invokes for any run with
``max_workers > 1``.

CUGA is never imported at module import time; every CUGA-facing import is
deferred into a function so this module -- and its tests -- stay importable
without the SDK.
"""

from __future__ import annotations

import json
import itertools
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, cast, runtime_checkable

from .base import BenchmarkTask

__all__ = [
    "ANSWER_KEYS",
    "BUILTIN_HARNESS_NAMES",
    "DEFAULT_TRACE_ROOT",
    "PROCESS_ISOLATION",
    "THREAD_ISOLATION",
    "TRACE_PATH_KEYS",
    "VANILLA_HARNESS",
    "ConcurrencyUnsupportedError",
    "ConcurrentWorkspaceError",
    "CugaEnvironmentError",
    "CugaExecutorError",
    "CugaExecutorFactory",
    "HarnessSpecError",
    "HarnessVersion",
    "MissingTraceError",
    "NoAnswerError",
    "NoQuestionError",
    "RolloutRecord",
    "TraceRecorder",
    "TracingDisabledError",
    "TracingWrapper",
    "WorkerLease",
    "WorkerPool",
    "default_trace_config",
    "make_cuga_executor_factory",
    "missing_trace_task_ids",
    "preflight",
    "prepare_environment",
    "require_model_env",
    "warm_up_cuga_imports",
]

#: In-process concurrency: one wrapper per worker **thread**. Safe for one
#: worker only. Measured unsafe beyond that -- see the module docstring.
THREAD_ISOLATION = "thread"

#: One CUGA **subprocess** per worker, each with its own environment and its own
#: knowledge and policy stores. The only mode in which ``max_workers > 1`` is
#: permitted for real execution.
PROCESS_ISOLATION = "process"

_ISOLATION_MODES = (THREAD_ISOLATION, PROCESS_ISOLATION)

#: Where traces land unless a caller says otherwise. Matches the path every
#: verified live script in ``scripts/`` already writes to.
DEFAULT_TRACE_ROOT = Path("data/traces")

#: Result keys that may carry the answer text, most authoritative first.
#: ``final_output`` is what :meth:`agent_evolve.cuga_wrapper.CugaWrapper.run_task`
#: returns. ``answer`` is the key the recorded Gaia baseline used (see
#: ``datasets/gaia/*/tasks/*/result.json`` and the ``run_task``-shaped JSON in
#: ``stdout.log``), so a recorded-shape dict is understood too.
ANSWER_KEYS: tuple[str, ...] = ("final_output", "answer")

#: Result keys that may carry the trace location, most authoritative first.
#: ``causal_trace_path`` is a trace *directory* written by ``TraceWriter``;
#: ``trace`` is the single-file path the recorded baseline used.
TRACE_PATH_KEYS: tuple[str, ...] = ("causal_trace_path", "trace")

#: Harness keys the CUGA wrapper materializes into a workspace directory.
#: Presence of any of these makes a harness workspace-bound, and therefore
#: unsafe to run concurrently (see the module docstring).
_WORKSPACE_KEYS: tuple[str, ...] = ("skills", "memory", "policies")

#: Serializes wrapper construction. ``CugaSdkRuntime.from_settings`` mutates
#: ``os.environ`` and imports the SDK; two threads doing that simultaneously is
#: needless risk for no gain, since construction is once per thread.
_BUILD_LOCK = threading.Lock()

#: Guards the one-time import warmup below.
_WARMUP_LOCK = threading.Lock()
_WARMED_UP = False

#: Hands out process-unique worker ids. A pool keys each worker's isolated
#: knowledge and policy stores by this id, so two factories sharing a pool must
#: not both mint ``w01`` -- that would put two live rollouts on one knowledge
#: ``persist_dir`` and the second would die on the flock, which is precisely the
#: failure process isolation exists to remove. Evaluating two candidates
#: concurrently is a first-class use case, so uniqueness is global, not
#: per-factory.
_WORKER_SEQUENCE = itertools.count(1)
_WORKER_SEQUENCE_LOCK = threading.Lock()


def _next_worker_id() -> str:
    with _WORKER_SEQUENCE_LOCK:
        return f"w{next(_WORKER_SEQUENCE):04d}"


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


class CugaExecutorError(RuntimeError):
    """Base class for every refusal this module raises.

    Each subclass is raised from inside the executor, so the runner records it
    as ``ok=False`` for that task and the rest of the batch survives.
    """


class CugaEnvironmentError(CugaExecutorError):
    """Model configuration is absent, so no rollout can possibly run."""


class HarnessSpecError(CugaExecutorError):
    """A harness could not be resolved, or does not declare its version."""


class TracingDisabledError(CugaExecutorError):
    """The wrapper would run without persisting a causal trace."""


class MissingTraceError(CugaExecutorError):
    """A rollout finished but no usable trace was written to disk."""


class NoAnswerError(CugaExecutorError):
    """The rollout produced no answer text. Not a wrong answer -- no answer."""


class NoQuestionError(CugaExecutorError):
    """The benchmark task carries no question, so there is nothing to execute."""


class ConcurrentWorkspaceError(CugaExecutorError):
    """A workspace-bound harness was asked to run on more than one thread."""


class ConcurrencyUnsupportedError(CugaExecutorError):
    """Real CUGA execution cannot be parallelized *within one process*.

    Measured, not inferred, and the decisive reason is not the one that shows up
    first in a log. Two process-global singletons collide:

    * **The knowledge engine's exclusive lock.** ``KnowledgeEngine.__init__``
      ``flock``s ``<persist_dir>/.lock`` and raises ``RuntimeError("Knowledge
      engine already running in another process")``. ``flock`` conflicts between
      two file objects in one process (verified), so threads collide like
      processes. But two engines with **distinct** ``persist_dir`` both start
      (verified), and ``persist_dir`` is a supported override -- so this blocker
      alone would be fixable in-process.

    * **``CUGA_FOLDER``, which is not.** It is one environment variable, shared
      by every thread, and CUGA's sandbox executors and ``prepare_node`` read it
      during ``invoke()``. Two threads that each called the real
      ``cuga_wrapper._construct_agent`` with a different workspace both ended up
      observing the *second* thread's workspace (verified). A build lock cannot
      fix it, because the read happens after construction.

    The second one decides it. A task could run against another candidate's
    harness while its trace still stamped its own ``harness_version`` -- that
    field is copied from the harness config, so it cannot detect the swap. Half a
    benchmark silently measuring the wrong harness is worse than a slow run, so
    threaded concurrency is refused and :data:`PROCESS_ISOLATION` is the
    supported way to go parallel.
    """


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #


def prepare_environment() -> None:
    """Load ``.env`` and normalize CUGA's optional variables.

    Delegates to :func:`agent_evolve.cuga_wrapper.prepare_cuga_environment`
    rather than calling ``load_dotenv`` directly: a blank
    ``CUGA_CONFIGURATIONS_DIR`` breaks the SDK import, and that normalization
    lives in the wrapper.
    """
    from agent_evolve.cuga_wrapper import prepare_cuga_environment

    prepare_cuga_environment()


def require_model_env() -> str:
    """Return the configured model, or raise a message that says what is missing.

    Called once before a run starts. Without it, a 42-task batch would fail 42
    times deep inside the SDK with an error that never mentions the actual
    problem.
    """
    prepare_environment()
    model = os.environ.get("CUGA_MODEL") or os.environ.get("LITELLM_MODEL")
    if not model:
        raise CugaEnvironmentError(
            "no model configured: set CUGA_MODEL (or LITELLM_MODEL) in .env or "
            "the environment. A live CUGA benchmark run also needs CUGA_BASE_URL "
            "/ LITELLM_BASE_URL and CUGA_API_KEY / LITELLM_API_KEY. Refusing to "
            "start a run that cannot execute a single task."
        )
    return model


# --------------------------------------------------------------------------- #
# harness version
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HarnessVersion:
    """One named, reproducible harness a benchmark run can execute against.

    ``version`` is mandatory and is not derived from a filename: it is the label
    that ends up in every trace's ``harness_version`` field and in the manifest,
    and a guessed label makes two runs indistinguishable after the fact.
    """

    version: str
    instructions: str | None = None
    skills: Mapping[str, str] = field(default_factory=dict)
    memory: Mapping[str, str] = field(default_factory=dict)
    policies: Mapping[str, str] = field(default_factory=dict)
    source: str = "<inline>"

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise HarnessSpecError(
                "HarnessVersion.version must be a non-empty string: a rollout "
                "whose harness cannot be named is not reproducible"
            )
        for group in _WORKSPACE_KEYS:
            value = getattr(self, group)
            if not isinstance(value, Mapping):
                raise HarnessSpecError(
                    f"harness {self.version!r}: {group!r} must be a mapping of "
                    f"name -> text; got {type(value).__name__}"
                )
            object.__setattr__(
                self, group, {str(k): str(v) for k, v in value.items()}
            )
        if self.instructions is not None:
            object.__setattr__(self, "instructions", str(self.instructions))

    # -- resolution ------------------------------------------------------- #

    @classmethod
    def resolve(cls, spec: str) -> "HarnessVersion":
        """Resolve ``--harness`` from a built-in name or a JSON file path."""
        text = str(spec).strip()
        if not text:
            raise HarnessSpecError(
                "--harness requires a value: a built-in name "
                f"({', '.join(BUILTIN_HARNESS_NAMES)}) or a path to a harness "
                "JSON file"
            )
        if text in _BUILTINS:
            return _BUILTINS[text]
        path = Path(text)
        if path.exists():
            return cls.from_path(path)
        raise HarnessSpecError(
            f"unknown harness {text!r}: not a built-in "
            f"({', '.join(BUILTIN_HARNESS_NAMES)}) and no such file. A harness "
            f"file is JSON with a required 'version' plus optional "
            f"'instructions', 'skills', 'memory' and 'policies'."
        )

    @classmethod
    def from_path(cls, path: Path | str) -> "HarnessVersion":
        """Load a harness from JSON. ``version`` is required, never inferred."""
        file_path = Path(path)
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HarnessSpecError(f"{file_path} is not valid JSON: {exc}") from exc
        except OSError as exc:
            raise HarnessSpecError(f"cannot read harness {file_path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise HarnessSpecError(
                f"{file_path} must contain a JSON object; got "
                f"{type(raw).__name__}"
            )
        version = raw.get("version")
        if not isinstance(version, str) or not version.strip():
            raise HarnessSpecError(
                f"{file_path} does not declare a non-empty string 'version'. The "
                f"harness version is stamped onto every trace this run writes and "
                f"is how a result is attributed to a harness later; it will not be "
                f"guessed from the filename."
            )
        instructions = raw.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise HarnessSpecError(
                f"{file_path}: 'instructions' must be a string when present"
            )
        return cls(
            version=version,
            instructions=instructions,
            skills=raw.get("skills") or {},
            memory=raw.get("memory") or {},
            policies=raw.get("policies") or {},
            source=str(file_path),
        )

    # -- properties ------------------------------------------------------- #

    @property
    def requires_workspace(self) -> bool:
        """True when CUGA must materialize a per-task workspace directory.

        Such a harness cannot run on more than one thread on this build; see the
        module docstring for the ``CUGA_FOLDER`` reason.
        """
        return bool(self.skills or self.memory or self.policies)

    @property
    def artifact_summary(self) -> str:
        parts = [
            f"{group}={len(getattr(self, group))}"
            for group in _WORKSPACE_KEYS
        ]
        parts.append(f"instructions={'yes' if self.instructions else 'default'}")
        return " ".join(parts)

    # -- the config CUGA receives ----------------------------------------- #

    def harness_config(self, task: BenchmarkTask) -> dict[str, object]:
        """Build the exact mapping ``CugaWrapper.run_task`` consumes.

        Follows the established convention in
        :meth:`agent_evolve.adapters.cuga_adapter.CugaAdapter._harness_config`:
        the task text goes under ``input``, editable artifacts go under their
        CUGA group keys, and only keys that are actually present are sent.

        ``tools`` is deliberately omitted so ``_construct_agent`` falls back to
        the wrapper's ``build_tools()`` set -- the same tool surface the recorded
        baseline used. Passing an empty list would instead be treated as "no
        override" by that function anyway, but omitting it makes the intent
        explicit rather than incidental.
        """
        question = task.question
        if not isinstance(question, str) or not question.strip():
            raise NoQuestionError(
                f"task {task.task_id!r} carries no question text; refusing to "
                f"run an agent on an empty prompt and report the output as an "
                f"answer"
            )
        config: dict[str, object] = {
            # Read by ``_artifact_metadata`` -> trace ``harness_version``.
            "version": self.version,
            "input": question,
        }
        if self.instructions:
            config["instructions"] = self.instructions
        for group in _WORKSPACE_KEYS:
            value = getattr(self, group)
            if value:
                config[group] = dict(value)
        return config


#: Neutral task framing for the base harness. Deliberately says only what the
#: agent is for and what shape its answer should take.
#:
#: It exists because ``instructions`` is the strongest editable lever the CUGA
#: harness exposes, and the pipeline can only put an artifact in the editor's
#: write set if the base harness already owns it. A vanilla harness with no
#: instructions collapsed to a single empty skill slot, leaving the prompt
#: unreachable by evolution.
#:
#: It must stay neutral. The measured non-answer mechanism is that the rollout
#: model narrates a plan and never emits a fenced Python block, so CUGA extracts
#: no code and routes straight to the final answer. Naming that remedy here
#: would hand evolution the fix and invalidate every self-improvement
#: measurement taken afterwards. Nothing about code execution, fences, or how to
#: reach a tool belongs in this string; discovering that is the experiment.
#: Pinned by ``test_vanilla_instructions_carry_no_code_execution_directive``.
VANILLA_INSTRUCTIONS = (
    "You are a question-answering agent. Answer the question you are given.\n"
    "Report the final answer only, as briefly as the question allows: a number, "
    "a name, a date, or a short phrase.\n"
    "Do not restate the question and do not include your reasoning in the answer."
)

#: The base harness: neutral instructions, no injected workspace artifacts, and
#: the wrapper's default tool set. This is the configuration the recorded Gaia
#: baseline ran under, so it is the correct control arm and the only sane
#: default for a smoke run. Named ``vanilla`` rather than ``default`` because it
#: is still an explicit choice a caller has to make.
VANILLA_HARNESS = HarnessVersion(
    version="vanilla",
    instructions=VANILLA_INSTRUCTIONS,
    source="<builtin:vanilla>",
)

_BUILTINS: dict[str, HarnessVersion] = {"vanilla": VANILLA_HARNESS}

BUILTIN_HARNESS_NAMES: tuple[str, ...] = tuple(sorted(_BUILTINS))


# --------------------------------------------------------------------------- #
# trace recording
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    """Where one task's evidence landed.

    Carries no answer text and no grading material: only the pointer the
    analyzer needs, plus enough provenance to tell two runs apart. The answer
    itself is returned to the runner, which owns scoring.
    """

    task_id: str
    trace_path: Path
    status: str
    harness_version: str
    answer_chars: int
    thread_name: str


class TraceRecorder:
    """Thread-safe collection of per-task trace pointers, in completion order.

    Executors run on worker threads, so this is the one shared mutable object in
    the design and it is guarded by a lock. It is append-only: nothing here can
    influence execution or scoring.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[RolloutRecord] = []

    def record(self, record: RolloutRecord) -> None:
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> tuple[RolloutRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def trace_path(self, task_id: str) -> Path | None:
        """The trace written for ``task_id``, or ``None`` if it never got one."""
        for record in self.records:
            if record.task_id == task_id:
                return record.trace_path
        return None

    @property
    def trace_paths(self) -> tuple[Path, ...]:
        return tuple(record.trace_path for record in self.records)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# --------------------------------------------------------------------------- #
# wrapper protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class TracingWrapper(Protocol):
    """The slice of :class:`agent_evolve.cuga_wrapper.CugaWrapper` used here.

    Narrow on purpose: tests inject a fake that implements exactly this, so the
    executor's contract is verifiable without the CUGA SDK, a model endpoint or
    a network.
    """

    def run_task(
        self, task_id: str, harness_config: Mapping[str, object]
    ) -> Mapping[str, object]: ...

    def supports_recorded_environment_replay(self) -> bool: ...


#: Builds one wrapper. Called once per worker thread.
WrapperFactory = Callable[[], TracingWrapper]


#: An opaque handle identifying one isolated worker.
#:
#: Deliberately untyped beyond "some object": what a worker needs in order to be
#: isolated is the pool's business. The executor only carries a lease from
#: :meth:`WorkerPool.lease` back into :meth:`WorkerPool.run` and never inspects
#: it, so narrowing this type would constrain pool implementations for no gain.
#: The real pool puts a live subprocess handle plus its per-worker store paths
#: here; a test fake puts a plain dict.
WorkerLease = Any


@runtime_checkable
class WorkerPool(Protocol):
    """Runs rollouts in isolated workers, one leased per calling thread.

    This is the seam that makes ``--max-workers > 1`` safe. The runner still
    fans out over threads -- that part is fine, and stays unchanged -- but each
    thread's rollout happens in a worker with its **own** ``os.environ`` and its
    **own** knowledge and policy stores, which is the only arrangement in which
    ``CUGA_FOLDER`` and the knowledge ``flock`` are not shared (see the module
    docstring for the measurements).

    Two methods, because the two costs are different: leasing is once per worker
    and expensive (a CUGA process start), running is once per task.
    """

    def lease(self, worker_id: str, harness_version: str) -> WorkerLease:
        """Reserve an isolated worker for the calling thread."""
        ...

    def run(
        self,
        lease: WorkerLease,
        task_id: str,
        harness_config: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute one task in ``lease``'s worker, returning a run_task result."""
        ...



def default_trace_config(output_root: Path | str = DEFAULT_TRACE_ROOT) -> object:
    """The trace configuration a benchmark run needs, not merely a working one.

    ``capture_node_payloads`` with ``RAW_OPT_IN`` is required, not optional:
    :func:`agent_evolve.cuga_wrapper.load_recorded_call` resolves an LLM call's
    ``messages_ref`` out of ``payloads/<sha256>.json``, and those blobs are only
    written when payload capture is on. A trace without them records that a call
    happened but not what was sent, which is exactly the evidence the analyzer
    and the proxy validator consume. These are the same settings the verified
    live scripts (``verify_adapter_e2e_live.py``,
    ``verify_complete_trace_graph.py``) use.
    """
    from agent_evolve.core.trace import PayloadLevel
    from agent_evolve.cuga_wrapper import TraceConfig

    return TraceConfig(
        enabled=True,
        output_root=Path(output_root),
        payload_level=PayloadLevel.RAW_OPT_IN,
        allow_raw_payloads=True,
        capture_node_payloads=True,
        max_observation_bytes=4_194_304,
    )


def warm_up_cuga_imports() -> int:
    """Import CUGA's module graph once, on one thread, before any parallel run.

    Measured, not speculative. ``CugaAgent`` imports only 6 ``cuga.*`` modules
    eagerly; a single observed rollout pulled in **172 more** lazily from inside
    ``invoke()``. When two worker threads execute ``run_task`` concurrently they
    race on CPython's per-module import locks, and a live two-worker run on the
    tiny5 dataset failed exactly this way::

        _DeadlockError: deadlock detected by
        _ModuleLock('cuga.backend.cuga_graph.policy.configurable')

    That is not a flake: thread A holds the lock for a partially-initialized
    module that thread B needs while B holds one A needs, and CPython raises
    rather than hanging. It costs a whole task, and which task dies depends on
    scheduling.

    Importing the package graph up-front on a single thread removes the race:
    afterwards every lazy import is a ``sys.modules`` hit that takes no lock.
    Verified to leave zero of the 172 observed lazy modules unimported, in ~10s
    once per process -- cheap next to a ~40s/task run.

    Returns the number of ``cuga.*`` modules resident afterwards. Individual
    module failures are ignored: a handful of CUGA submodules fail to import
    standalone (7 observed) and were never on the run path, so warming them is
    best-effort. A failure here must not fail a run that would otherwise work.

    The package is located via ``importlib`` rather than a literal ``import
    cuga`` so this module keeps no static CUGA import. ``benchmarks/`` is not a
    CUGA boundary -- only ``cuga_wrapper/`` and ``adapters/`` are -- and
    ``test_active_package_has_no_legacy_or_adapter_runtime_imports`` enforces
    that. The agent-neutral rule is worth more than the one-line convenience.
    """
    global _WARMED_UP
    with _WARMUP_LOCK:
        if _WARMED_UP:
            return sum(1 for name in sys.modules if name.startswith("cuga"))
        import importlib
        import pkgutil

        package = importlib.import_module("cuga")
        for module in pkgutil.walk_packages(package.__path__, prefix="cuga."):
            try:
                importlib.import_module(module.name)
            except Exception:  # noqa: BLE001 - best-effort; see docstring
                continue
        _WARMED_UP = True
        return sum(1 for name in sys.modules if name.startswith("cuga"))


def _default_wrapper_factory(
    trace_root: Path | str, trace_config: object | None
) -> WrapperFactory:
    """Build real CUGA wrappers, one per worker thread, with tracing on."""

    def build() -> TracingWrapper:
        from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings

        config = (
            trace_config if trace_config is not None else default_trace_config(trace_root)
        )
        # Construction mutates os.environ and imports the SDK; serialize it.
        with _BUILD_LOCK:
            prepare_environment()
            settings = RuntimeSettings.from_env()
            return CugaWrapper.from_cuga(settings, trace_config=cast(Any, config))

    return build


# --------------------------------------------------------------------------- #
# the executor
# --------------------------------------------------------------------------- #


def _error_from_trace(result: Mapping[str, object]) -> str:
    """Recover the failure reason from the persisted trace.

    ``CugaSdkRuntime.run_task`` puts a real diagnosis in its result's ``error``
    field, but ``CugaWrapper.run_task`` copies only a fixed set of keys onto what
    it returns and ``error`` is not among them, so the caller sees
    ``status="error"`` with no reason. The trace on disk does carry it, and
    reading it back turns an undiagnosable failure into a specific one -- which
    is how the knowledge-engine lock conflict behind
    :class:`ConcurrencyUnsupportedError` was identified at all.

    Best-effort: a missing or unreadable trace yields a plain note rather than
    masking the original failure with a second one. The wrapper should forward
    ``error`` directly; that fix is outside this module's scope.
    """
    for key in TRACE_PATH_KEYS:
        raw = result.get(key)
        if not raw:
            continue
        document = Path(str(raw))
        if document.is_dir():
            document = document / "causal-trace.json"
        try:
            payload = json.loads(document.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if error:
                return f"{error} (recovered from {document})"
    return (
        "no error detail recorded; CugaWrapper.run_task does not forward the "
        "runtime's 'error' field and no trace error was readable"
    )


def _extract_answer(
    result: Mapping[str, object], task_id: str, harness_version: str
) -> str:
    """Pull the answer out of a ``run_task`` result, or refuse.

    Every refusal here becomes ``ok=False`` in the runner, which excludes the
    task from the scoring denominator. That distinction is the point: a harness
    that crashed, returned nothing, or returned whitespace has produced *no
    measurement*, and grading it as a wrong answer would make a broken run look
    like a merely bad one.
    """
    status = str(result.get("status", "")) or "<unset>"
    if status == "error":
        reason = str(result.get("error", "")) or _error_from_trace(result)
        raise NoAnswerError(
            f"task {task_id!r} on harness {harness_version!r} failed inside "
            f"CUGA (status=error): {reason[:400]}"
        )

    for key in ANSWER_KEYS:
        if key not in result:
            continue
        value = result[key]
        if value is None:
            raise NoAnswerError(
                f"task {task_id!r} on harness {harness_version!r} returned "
                f"{key}=None (status={status}); a missing answer is an execution "
                f"failure, not a wrong answer"
            )
        if not isinstance(value, str):
            raise NoAnswerError(
                f"task {task_id!r} on harness {harness_version!r} returned "
                f"{key} of type {type(value).__name__}, not str; refusing to "
                f"stringify a non-answer into an answer"
            )
        if not value.strip():
            raise NoAnswerError(
                f"task {task_id!r} on harness {harness_version!r} returned a "
                f"blank {key} (status={status}); recording no answer rather than "
                f"grading emptiness as a wrong answer"
            )
        return value

    raise NoAnswerError(
        f"task {task_id!r} on harness {harness_version!r} returned no answer "
        f"field (looked for {', '.join(ANSWER_KEYS)}; got keys "
        f"{sorted(str(k) for k in result)})"
    )


def _extract_trace_path(
    result: Mapping[str, object], task_id: str, harness_version: str
) -> Path:
    """Pull the persisted trace location out of a result, or refuse.

    Verifies the trace exists on disk instead of trusting the returned string.
    A directory must contain ``causal-trace.json``, because that file is what
    ``CugaAdapter._rich_events`` and ``load_recorded_call`` open; a directory
    without it would be a trace pointer that nothing can read.
    """
    for key in TRACE_PATH_KEYS:
        raw = result.get(key)
        if raw is None or str(raw) == "":
            continue
        path = Path(str(raw))
        if path.is_dir():
            if not (path / "causal-trace.json").is_file():
                raise MissingTraceError(
                    f"task {task_id!r}: trace directory {path} has no "
                    f"causal-trace.json, so no analyzer can read it"
                )
            return path
        if path.is_file():
            return path
        raise MissingTraceError(
            f"task {task_id!r}: result reported trace at {path} but nothing "
            f"exists there"
        )

    raise MissingTraceError(
        f"task {task_id!r} on harness {harness_version!r} produced an answer but "
        f"no causal trace (looked for {', '.join(TRACE_PATH_KEYS)}). A benchmark "
        f"run exists to produce traces; an untraced rollout is discarded rather "
        f"than scored."
    )


class _CugaExecutor:
    """Executes one task per call. Owned by exactly one worker thread."""

    __slots__ = ("_wrapper", "_harness", "_recorder")

    def __init__(
        self,
        wrapper: TracingWrapper,
        harness: HarnessVersion,
        recorder: TraceRecorder,
    ) -> None:
        self._wrapper = wrapper
        self._harness = harness
        self._recorder = recorder

    def __call__(self, task: BenchmarkTask) -> str:
        config = self._harness.harness_config(task)
        result = self._wrapper.run_task(task.task_id, config)
        return _consume_result(result, task, self._harness, self._recorder)


class _PooledCugaExecutor:
    """Executes one task per call, in this thread's leased isolated worker.

    Structurally identical to :class:`_CugaExecutor` -- same refusals, same
    recording, same answer extraction -- because the correctness properties must
    not depend on which backend produced the result. Only *where* the rollout
    runs differs.
    """

    __slots__ = ("_pool", "_lease", "_harness", "_recorder")

    def __init__(
        self,
        pool: WorkerPool,
        lease: WorkerLease,
        harness: HarnessVersion,
        recorder: TraceRecorder,
    ) -> None:
        self._pool = pool
        self._lease = lease
        self._harness = harness
        self._recorder = recorder

    def __call__(self, task: BenchmarkTask) -> str:
        config = self._harness.harness_config(task)
        result = self._pool.run(self._lease, task.task_id, config)
        return _consume_result(result, task, self._harness, self._recorder)


def _consume_result(
    result: object,
    task: BenchmarkTask,
    harness: HarnessVersion,
    recorder: TraceRecorder,
) -> str:
    """Validate one rollout result, record its evidence, return its answer.

    Shared by both executors so a threaded run and a process-isolated run are
    held to exactly the same standard: no answer is never a wrong answer, and an
    untraced rollout is discarded rather than scored.
    """
    if not isinstance(result, Mapping):
        raise NoAnswerError(
            f"task {task.task_id!r}: run_task returned "
            f"{type(result).__name__}, not a mapping"
        )
    answer = _extract_answer(result, task.task_id, harness.version)
    trace_path = _extract_trace_path(result, task.task_id, harness.version)
    recorder.record(
        RolloutRecord(
            task_id=task.task_id,
            trace_path=trace_path,
            status=str(result.get("status", "")) or "<unset>",
            harness_version=str(result.get("harness_version", harness.version)),
            answer_chars=len(answer),
            thread_name=threading.current_thread().name,
        )
    )
    return answer


class CugaExecutorFactory:
    """The ``executor_factory`` :func:`run_benchmark` calls once per thread.

    A callable object rather than a closure so a caller can inspect what a run
    actually did -- how many executors were built, and where the traces went --
    without reaching into internals.

    Two backends, one contract. With ``worker_pool`` set, each worker thread
    leases an isolated CUGA **process** and the workspace race described in the
    module docstring cannot occur, so a workspace-bound harness is allowed to run
    concurrently. Without one, rollouts happen in this process and the
    single-executor guard stays.
    """

    def __init__(
        self,
        harness: HarnessVersion,
        *,
        trace_root: Path | str = DEFAULT_TRACE_ROOT,
        recorder: TraceRecorder | None = None,
        wrapper_factory: WrapperFactory | None = None,
        trace_config: object | None = None,
        worker_pool: WorkerPool | None = None,
    ) -> None:
        self.harness = harness
        self.trace_root = Path(trace_root)
        self.recorder = recorder if recorder is not None else TraceRecorder()
        self.worker_pool = worker_pool
        self._wrapper_factory = (
            wrapper_factory
            if wrapper_factory is not None or worker_pool is not None
            else _default_wrapper_factory(self.trace_root, trace_config)
        )
        self._lock = threading.Lock()
        self._built = 0

    @property
    def isolation(self) -> str:
        """Which concurrency model this factory actually provides."""
        return PROCESS_ISOLATION if self.worker_pool is not None else THREAD_ISOLATION

    @property
    def executors_built(self) -> int:
        """How many worker threads have built an executor so far."""
        with self._lock:
            return self._built

    def __call__(self) -> Callable[[BenchmarkTask], str]:
        with self._lock:
            if (
                self.worker_pool is None
                and self.harness.requires_workspace
                and self._built >= 1
            ):
                # Refuse rather than corrupt. See the module docstring: a
                # workspace-bound harness is pinned to the process-global
                # CUGA_FOLDER, so a second concurrent agent would read another
                # task's artifacts and the run would silently measure the wrong
                # harness. A pooled run is exempt: each worker is its own
                # process, with its own environment.
                raise ConcurrentWorkspaceError(
                    f"harness {self.harness.version!r} materializes a CUGA "
                    f"workspace ({self.harness.artifact_summary}), which binds "
                    f"the process-global CUGA_FOLDER environment variable. A "
                    f"second concurrent executor would race on it and read "
                    f"another task's artifacts. Run this harness with "
                    f"--max-workers 1, or with process isolation."
                )
            self._built += 1
            worker_id = _next_worker_id()

        if self.worker_pool is not None:
            lease = self.worker_pool.lease(worker_id, self.harness.version)
            return _PooledCugaExecutor(
                self.worker_pool, lease, self.harness, self.recorder
            )

        assert self._wrapper_factory is not None  # set whenever no pool is used
        wrapper = self._wrapper_factory()
        self._verify_tracing(wrapper)
        return _CugaExecutor(wrapper, self.harness, self.recorder)

    def _verify_tracing(self, wrapper: TracingWrapper) -> None:
        """Refuse a wrapper that would run without persisting traces.

        Checked before the first task rather than after: discovering at the end
        of a 42-task run that nothing was traced wastes the entire run.
        """
        probe = getattr(wrapper, "supports_recorded_environment_replay", None)
        if probe is None:
            return
        if not probe():
            raise TracingDisabledError(
                f"{type(wrapper).__name__} reports causal tracing disabled. A "
                f"benchmark run exists to produce traces for the analyzer and "
                f"the proxy validator; refusing to execute a run whose evidence "
                f"would be discarded. Build the wrapper with "
                f"TraceConfig(enabled=True) -- see default_trace_config()."
            )


def make_cuga_executor_factory(
    harness: HarnessVersion,
    *,
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    recorder: TraceRecorder | None = None,
    wrapper_factory: WrapperFactory | None = None,
    trace_config: object | None = None,
    worker_pool: WorkerPool | None = None,
) -> CugaExecutorFactory:
    """Build the factory to hand :func:`run_benchmark`.

    :param harness: the harness version to execute against. Required and
        explicit; there is no default harness.
    :param trace_root: directory traces are written under.
    :param recorder: collects per-task trace pointers; a fresh one is created
        when omitted. Read it after the run to report what evidence exists.
    :param wrapper_factory: builds one wrapper per worker thread, in *this*
        process. Defaults to a real traced CUGA wrapper. Tests inject a fake.
    :param trace_config: overrides :func:`default_trace_config`. Only honoured
        by the default wrapper factory.
    :param worker_pool: run each worker's rollouts in an isolated CUGA process
        instead. Required for ``max_workers > 1`` -- see
        :class:`ConcurrencyUnsupportedError` for why in-process concurrency is
        refused. Mutually exclusive with ``wrapper_factory``.
    """
    if not isinstance(harness, HarnessVersion):
        raise HarnessSpecError(
            f"harness must be a HarnessVersion; got {type(harness).__name__}. "
            f"Use HarnessVersion.resolve('<name-or-path>')."
        )
    if worker_pool is not None and wrapper_factory is not None:
        # Two backends would mean one silently wins, and which one decides
        # whether the run is safely isolated. Refuse instead of picking.
        raise CugaExecutorError(
            "worker_pool and wrapper_factory are mutually exclusive: the first "
            "runs each rollout in an isolated CUGA process, the second runs it "
            "in this process. Supplying both leaves the isolation of the run "
            "ambiguous. Pick one."
        )
    return CugaExecutorFactory(
        harness,
        trace_root=trace_root,
        recorder=recorder,
        wrapper_factory=wrapper_factory,
        trace_config=trace_config,
        worker_pool=worker_pool,
    )


def preflight(
    harness: HarnessVersion,
    *,
    max_workers: int,
    tasks: int,
    allow_unsafe_concurrency: bool = False,
    isolation: str = THREAD_ISOLATION,
) -> None:
    """Reject a configuration that cannot produce a valid measurement.

    Runs before any task, because every failure mode caught here would
    otherwise surface after minutes or hours of billed model calls.

    :param isolation: :data:`THREAD_ISOLATION` (default) runs rollouts in this
        process and is safe only at ``max_workers=1``. :data:`PROCESS_ISOLATION`
        gives each worker its own CUGA process, and is the only mode in which
        ``max_workers > 1`` is permitted.
    :param allow_unsafe_concurrency: escape hatch that permits threaded
        concurrency anyway, for experiments that knowingly accept losing tasks
        and cross-contaminated harnesses. Off by default: a measurement that
        silently drops half its tasks, or runs them against the wrong harness, is
        worse than no measurement.
    """
    if isolation not in _ISOLATION_MODES:
        raise CugaExecutorError(
            f"unknown isolation {isolation!r}; expected one of "
            f"{', '.join(_ISOLATION_MODES)}"
        )
    parallel = max_workers > 1
    if parallel and isolation == THREAD_ISOLATION:
        if harness.requires_workspace and not allow_unsafe_concurrency:
            raise ConcurrentWorkspaceError(
                f"harness {harness.version!r} materializes a CUGA workspace "
                f"({harness.artifact_summary}) and cannot run in-process with "
                f"--max-workers {max_workers}: candidate isolation is bound to "
                f"the process-global CUGA_FOLDER environment variable, so "
                f"concurrent threads would read each other's artifacts and the "
                f"run would measure a harness that never existed. Use "
                f"--max-workers 1, or process isolation."
            )
        if not allow_unsafe_concurrency:
            raise ConcurrencyUnsupportedError(
                f"real CUGA execution cannot run in-process with --max-workers "
                f"{max_workers}. Two process-global singletons collide, and both "
                f"were verified directly. (1) The knowledge engine holds an "
                f"exclusive flock on <persist_dir>/.lock, so the second "
                f"concurrent rollout dies with 'Knowledge engine already running "
                f"in another process'; observed on tiny5, where --max-workers 2 "
                f"answered 1 of 2 tasks. That one is fixable with a per-worker "
                f"persist_dir. (2) CUGA_FOLDER is a single environment variable "
                f"shared by every thread, read by CUGA's sandbox executors and "
                f"prepare_node during invoke(); two threads that each bound a "
                f"different workspace were both left observing the second one's, "
                f"so a task can run against another candidate's harness while "
                f"its trace still stamps its own harness_version. That one is "
                f"not fixable in-process. Use process isolation for real "
                f"parallelism."
            )
    if tasks <= 0:
        raise CugaExecutorError("no tasks selected; nothing to execute")
    require_model_env()
    if parallel and isolation == THREAD_ISOLATION:
        # Only threads race CPython's import locks; a subprocess warms its own
        # graph on its own single thread.
        try:
            warm_up_cuga_imports()
        except Exception as exc:  # noqa: BLE001 - see warm_up_cuga_imports
            # Report, do not abort: without the warmup a parallel run may lose
            # tasks to import deadlocks, but that is strictly better than
            # refusing to run at all.
            print(
                f"warning: CUGA import warmup failed ({type(exc).__name__}: "
                f"{exc}); a parallel run may lose tasks to import deadlocks"
            )


def missing_trace_task_ids(
    tasks: Sequence[BenchmarkTask], recorder: TraceRecorder
) -> tuple[str, ...]:
    """Tasks that produced no trace, in input order. Empty is the good case."""
    recorded = {record.task_id for record in recorder.records}
    return tuple(task.task_id for task in tasks if task.task_id not in recorded)
