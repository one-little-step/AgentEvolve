"""Run each benchmark worker's CUGA rollouts in its own subprocess.

Why this module exists
---------------------
``--execute --max-workers 2`` was refused, and correctly so, but the refusal
made real research expensive: 42 tasks at ~40-200s each is hours serially, and an
evolution loop multiplies that by candidates x iterations. This module removes
the refusal without removing the correctness it protected.

Threading was ruled out by measurement, not by taste. Two process-global
singletons collide inside one CUGA process:

1. **The knowledge engine's exclusive lock.** ``KnowledgeEngine.__init__`` takes
   a non-blocking ``flock`` on ``<persist_dir>/.lock`` and raises
   ``RuntimeError("Knowledge engine already running in another process")``
   otherwise. ``flock`` conflicts between two file objects in one process, so
   threads collide exactly like processes. **This one is fixable in-process:**
   two engines with distinct ``persist_dir`` both start, verified directly
   (``terminal_output/benchmarks/probe-knowledge-persist-dir.log``).

2. **``CUGA_FOLDER``, which is not fixable in-process.** ``cuga_wrapper``
   binds a candidate's workspace by exporting that one environment variable,
   because the constructor argument does not reach CUGA's sandbox executors or
   ``prepare_node`` on this build -- both call ``os.getenv("CUGA_FOLDER", ...)``
   directly, during ``invoke()``. Threads share ``os.environ``. Two threads that
   each called the real ``_construct_agent`` with a different workspace were both
   left observing the **second** thread's workspace
   (``probe-cuga-folder-threadsafety.log``). Serializing construction cannot help:
   the read happens later.

Point 2 is what decides it, and it is worse than a lost task. A task could
execute against another candidate's skills while its trace still stamped its own
``harness_version`` -- that field is copied from the harness config, so it cannot
detect the swap. The run would look clean and measure a harness that never
existed. So parallelism is per **process**: a subprocess has its own
``os.environ``, so ``CUGA_FOLDER`` becomes per-worker by construction, and its
own knowledge and policy stores, so the ``flock`` never collides.

The minimal-delta design
------------------------
Each worker is a plain ``python -u -m agent_evolve.benchmarks.cuga_process_pool``
child speaking newline-delimited JSON on stdin/stdout, and **the child's working
directory is deliberately left at the repository root**. cwd is load-bearing for
CUGA: ``settings.toml`` is discovered via ``os.getcwd()``, ``.cuga/skills`` and
``.cuga/playbooks`` are cwd-relative, and moving it would silently change the
configuration under measurement -- turning a throughput change into an
uncontrolled experiment.

Only what actually collides is redirected, by environment variable:

* ``DYNACONF_KNOWLEDGE__PERSIST_DIR`` -- the knowledge lock and vector store.
  ``persist_dir`` is a supported override: CUGA's own
  ``knowledge_settings.toml`` ships ``persist_dir = ""`` commented "empty =
  default ``<cwd>/.cuga/knowledge/``". Verified honoured on a **cold** child
  start, which is the case that matters (``probe-env-only-isolation.log``).
* ``CUGA_DBS_DIR`` -- the policy store. ``DBS_DIR`` otherwise defaults inside
  site-packages and is shared by every process, while
  ``_construct_agent`` passes ``reset_policy_storage=True`` for any workspace
  harness; two workers resetting one store would delete each other's playbooks.
* ``CUGA_FOLDER`` is **removed**, never forwarded. A value inherited from the
  parent is exactly the stale-candidate leak this design exists to prevent; the
  child's own ``_construct_agent`` binds it per task.

Traces still land in one shared ``trace_root``, because evidence should not
require knowing the worker topology to locate.

Honest limits
-------------
* Answers, not agent objects, cross the process boundary. That is all the
  benchmark runner needs (it consumes an answer string plus a trace path on
  disk), but it means this module offers no in-memory agent-state sharing.
* Failure is data. A child that dies, hangs past its budget, or answers
  unparseably raises here, so the runner records ``ok=False`` and the task leaves
  the scoring denominator instead of entering it as a wrong answer.

Why workers are seeded, not started empty
-----------------------------------------
The first working version of this module gave each worker a fresh empty knowledge
store, and that quietly broke scoring -- which is worse than the throughput
problem it was solving. Measured on the same 4 tiny5 tasks, same harness, with
the worker's tool surface verified byte-identical to the parent's:

===============================================  ==========
run                                              passed
===============================================  ==========
serial, in-process                               3/4
serial, in-process (repeat)                      3/4
process-isolated, empty store, **1** worker      0/3
process-isolated, empty store, 4 workers         0/3
process-isolated, **seeded** store, 2 workers    restored
===============================================  ==========

The 1-worker column is the important one: it rules concurrency out as the cause.
The difference is the store. A serial run inherits the repository's populated
``.cuga/knowledge`` (two indexed documents on this checkout) and the agent's
knowledge-search tool finds them; an empty store makes those searches come back
dry, and the model answers differently. A pass rate that moves with
``--isolation`` is not a measurement, so every worker is seeded with a **copy** of
the same reference store a serial run would have used. Copying, not sharing:
sharing would put two writers on one store, which is the flock collision this
module exists to avoid.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic as _monotonic
from typing import Any, Mapping

from agent_evolve.core.run_logging import LogCaptureConfig, RunLogSink

__all__ = [
    "DEFAULT_MAX_ROLLOUTS_PER_WORKER",
    "DEFAULT_WORKER_START_TIMEOUT",
    "default_knowledge_seed",
    "CugaProcessPool",
    "WorkerCrashedError",
    "WorkerPoolError",
    "WorkerProtocolError",
    "WorkerStartError",
    "main",
]

#: How long a worker gets to report readiness. A cold CUGA import graph is ~10s
#: on the observed machine; the margin covers a first-run embedding-model load.
DEFAULT_WORKER_START_TIMEOUT = 600.0

#: How many rollouts one worker serves before it is replaced.
#:
#: This is the fix for the dominant term in the 2026-08-19 memory exhaustion (a
#: 3-round RHO run reached ~90 GB and killed the machine). Each worker builds one
#: ``CugaWrapper`` and then serves every task with it. ``run_task`` drives the
#: full CUGA agent graph per call, and nothing between calls releases the SDK's
#: per-invocation state -- message histories, the in-memory instructions cache,
#: context-summariser buffers, LangGraph run trees. With 12 workers and hundreds
#: of rollouts that accumulation is unbounded and monotonic.
#:
#: Replacing the process is the only reliable reset: ``gc.collect()`` cannot free
#: state the SDK still references, and we do not control its internals. Process
#: death frees everything by construction, including leaked Playwright children.
#:
#: 25 balances two real costs. Worker startup is expensive (a cold CUGA import
#: graph is ~10s), so recycling every rollout would add that to every task. Too
#: high and the leak has room to grow. At 25 the restart cost is amortised to
#: well under a second per rollout while capping steady-state RSS per worker.
DEFAULT_MAX_ROLLOUTS_PER_WORKER = 25

#: Sentinel the child prints once it can accept work.
_READY = "AGENT_EVOLVE_WORKER_READY"

#: Frames every reply on the protocol channel.
#:
#: Belt and braces on top of the fd redirection in :func:`main`. The child gives
#: itself a private duplicate of fd 1 and points fd 1 at stderr, so a library
#: that prints to stdout cannot reach this channel -- but a library that writes
#: to fd 1 *directly*, or a future CUGA that grabs the tty, still could. An
#: unframed line is therefore skipped rather than parsed, because a banner
#: mistaken for a reply is a fabricated result, and a fabricated result is the
#: one failure mode this whole design exists to prevent.
_REPLY_PREFIX = "AGENT_EVOLVE_REPLY "

#: Environment variables the child must NOT inherit, and why.
#:
#: ``CUGA_FOLDER`` is the whole point: an inherited value is a stale candidate's
#: workspace, and the child's own ``_construct_agent`` is responsible for binding
#: it per task.
_STRIPPED_ENV: tuple[str, ...] = ("CUGA_FOLDER",)

#: Never copied when seeding a worker's knowledge store: it is another process's
#: lock, and handing a worker a foreign lock file's state is meaningless at best.
_UNSEEDED_NAMES: frozenset[str] = frozenset({".lock"})


#: Distinguishes "caller said nothing" from "caller said None" for
#: ``knowledge_seed``, because ``None`` is a meaningful value there (seed
#: nothing) and must not collide with the default.
_UNSET: Any = object()


def default_knowledge_seed() -> Path:
    """The knowledge store a **serial** run would use.

    ``KnowledgeConfig.persist_dir`` resolves to ``<cwd>/.cuga/knowledge`` when
    unset, so this is what a worker must start from for its results to be
    comparable. Resolved lazily (not at import) because cwd is the caller's.
    """
    return Path.cwd() / ".cuga" / "knowledge"


class WorkerPoolError(RuntimeError):
    """Base class for worker failures. Each becomes one ``ok=False`` task."""


class WorkerStartError(WorkerPoolError):
    """A worker process could not be started, or never reported readiness."""


class WorkerCrashedError(WorkerPoolError):
    """A worker died, or stopped answering, mid-rollout."""


class WorkerProtocolError(WorkerPoolError):
    """A worker answered something this pool cannot interpret as a result."""


@dataclass
class _Worker:
    """One live child process and the stores only it may touch."""

    worker_id: str
    harness_version: str
    process: subprocess.Popen[str]
    knowledge_dir: Path
    dbs_dir: Path
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Rollouts this child has served since it started.
    #:
    #: Per-worker, not global: a global counter would trip every worker at the
    #: same moment and stall the whole pool while all N children restarted.
    rollouts_served: int = 0


class CugaProcessPool:
    """Leases one CUGA subprocess per worker thread.

    Implements the ``WorkerPool`` protocol in
    :mod:`agent_evolve.benchmarks.cuga_executor`, so ``run_benchmark`` needs no
    change: the runner still fans tasks out over threads and still resolves
    results in input order, but each thread's rollouts happen in a process that
    owns its environment.
    """

    def __init__(
        self,
        *,
        root: Path | str,
        trace_root: Path | str,
        python_executable: str | None = None,
        start_timeout: float = DEFAULT_WORKER_START_TIMEOUT,
        task_timeout: float | None = None,
        knowledge_seed: Path | str | None = _UNSET,
        log_capture: LogCaptureConfig | None = None,
        max_rollouts_per_worker: int = DEFAULT_MAX_ROLLOUTS_PER_WORKER,
    ) -> None:
        """
        :param root: directory under which each worker's private knowledge and
            policy stores are created.
        :param trace_root: shared directory every worker writes traces to.
        :param python_executable: interpreter for children; defaults to this one,
            which keeps the child in the same virtualenv.
        :param start_timeout: how long a worker may take to become ready.
        :param task_timeout: per-task budget enforced inside the pool. ``None``
            defers entirely to the runner's own ``task_timeout_seconds``.
        :param knowledge_seed: knowledge store each worker is seeded from.
            Defaults to :func:`default_knowledge_seed`, i.e. the store a serial
            run would use -- an empty worker store measurably changes the pass
            rate (see the module docstring). Pass ``None`` to start workers empty
            on purpose.
        :param log_capture: when enabled, each child's ``stderr`` is written to
            ``<root>/workers/<worker_id>.log`` instead of being discarded. That
            stream is the *only* place CUGA reports its routing decisions
            (``is_autonomous_subtask``, ``Routing to:``), so discarding it leaves
            a finished run unable to say why it routed as it did. Defaults to
            disabled, i.e. today's ``DEVNULL`` behaviour byte for byte.
        """
        self.root = Path(root)
        self.trace_root = Path(trace_root)
        self.python_executable = python_executable or sys.executable
        self.start_timeout = start_timeout
        self.task_timeout = task_timeout
        if max_rollouts_per_worker < 1:
            raise ValueError(
                "max_rollouts_per_worker must be >= 1; 0 or negative would "
                "recycle a worker before it could serve any task"
            )
        self.max_rollouts_per_worker = max_rollouts_per_worker
        self.log_capture = log_capture or LogCaptureConfig()
        self._log_sink = RunLogSink(config=self.log_capture, channel="workers")
        self.knowledge_seed = (
            default_knowledge_seed()
            if knowledge_seed is _UNSET
            else (None if knowledge_seed is None else Path(knowledge_seed))
        )
        #: cwd is intentionally the parent's. See the module docstring: CUGA
        #: resolves settings.toml and .cuga/* relative to it, so relocating it
        #: would change the configuration being measured.
        self.worker_cwd = Path.cwd()
        self._lock = threading.Lock()
        self._workers: list[_Worker] = []

    # -- isolation -------------------------------------------------------- #

    def worker_environment(self, worker_id: str, harness_version: str) -> dict[str, str]:
        """The child's environment: inherited, minus leaks, plus private stores.

        Kept a separate public method because it *is* the isolation contract, and
        a contract that can only be observed by starting a CUGA process is a
        contract nothing will check.
        """
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        knowledge_dir, dbs_dir = self._store_paths(worker_id)
        env["DYNACONF_KNOWLEDGE__PERSIST_DIR"] = str(knowledge_dir)
        env["CUGA_DBS_DIR"] = str(dbs_dir)
        env["AGENT_EVOLVE_TRACE_ROOT"] = str(self.trace_root.resolve())
        env["AGENT_EVOLVE_WORKER_ID"] = worker_id
        env["AGENT_EVOLVE_WORKER_HARNESS"] = harness_version
        return env

    def _store_paths(self, worker_id: str) -> tuple[Path, Path]:
        base = self.root.resolve() / worker_id
        return base / "knowledge", base / "dbs"

    def seed_knowledge_store(self, worker_id: str) -> Path:
        """Create this worker's knowledge store, copied from the reference.

        Returns the store's path. Idempotent per entry: an existing file is left
        alone, so a re-leased worker id keeps whatever it has already written.

        Copies rather than shares. Sharing would put two live writers on one
        store, which is exactly the ``flock`` collision this pool exists to avoid,
        and would let one worker's ingest leak into another's evidence.

        An absent reference is not an error: a fresh checkout has no store, and a
        run that would otherwise work must not be blocked by a missing cache.
        """
        knowledge_dir, _ = self._store_paths(worker_id)
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        seed = self.knowledge_seed
        if seed is None or not seed.is_dir():
            return knowledge_dir
        for entry in seed.iterdir():
            if entry.name in _UNSEEDED_NAMES:
                continue
            destination = knowledge_dir / entry.name
            if destination.exists():
                continue
            try:
                if entry.is_dir():
                    shutil.copytree(entry, destination)
                else:
                    shutil.copy2(entry, destination)
            except OSError:
                # Best-effort: a partially seeded worker still runs, and the
                # alternative -- failing the run over a cache copy -- is worse.
                continue
        return knowledge_dir

    # -- lifecycle -------------------------------------------------------- #

    def lease(self, worker_id: str, harness_version: str) -> _Worker:
        """Start one worker and wait until it can accept a task."""
        knowledge_dir = self.seed_knowledge_store(worker_id)
        _, dbs_dir = self._store_paths(worker_id)
        dbs_dir.mkdir(parents=True, exist_ok=True)
        self.trace_root.mkdir(parents=True, exist_ok=True)

        try:
            # DEVNULL only when capture is off. The child's stderr is the sole
            # channel for CUGA's routing decisions, so discarding it is a choice
            # the caller makes, not the default we impose silently.
            stderr = self._log_sink.open_stream(worker_id) or subprocess.DEVNULL
            process = subprocess.Popen(
                [
                    self.python_executable,
                    "-u",
                    "-m",
                    "agent_evolve.benchmarks.cuga_process_pool",
                ],
                cwd=str(self.worker_cwd),
                env=self.worker_environment(worker_id, harness_version),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise WorkerStartError(
                f"worker {worker_id!r} could not be started with "
                f"{self.python_executable!r}: {exc}"
            ) from exc

        worker = _Worker(
            worker_id=worker_id,
            harness_version=harness_version,
            process=process,
            knowledge_dir=knowledge_dir,
            dbs_dir=dbs_dir,
        )
        with self._lock:
            self._workers.append(worker)
        self._await_ready(worker)
        return worker

    def _await_ready(self, worker: _Worker) -> None:
        """Block until the child prints its ready sentinel, or fail loudly.

        Tolerates unframed chatter before the sentinel: CUGA prints banners
        during import, and a worker is not broken merely because a dependency is
        chatty. Anything that is not the sentinel and not a framed reply is
        discarded.
        """
        stdout = worker.process.stdout
        if stdout is None:  # pragma: no cover - Popen always gives us a pipe
            raise WorkerStartError(f"worker {worker.worker_id!r} has no stdout")

        remaining = self.start_timeout
        while True:
            started = _monotonic()
            line = _with_timeout(stdout.readline, remaining)
            remaining -= _monotonic() - started
            if line is _TIMED_OUT or (line is not None and remaining <= 0):
                self._kill(worker)
                raise WorkerStartError(
                    f"worker {worker.worker_id!r} never became ready within "
                    f"{self.start_timeout}s; it would have used knowledge store "
                    f"{worker.knowledge_dir}"
                )
            if not line:
                self._kill(worker)
                raise WorkerStartError(
                    f"worker {worker.worker_id!r} exited before becoming ready "
                    f"(code {worker.process.poll()}); it would have used "
                    f"knowledge store {worker.knowledge_dir}"
                )
            text = str(line).strip()
            if text == _READY:
                return
            # Not the sentinel: library noise on the channel. Keep waiting.

    # -- running ---------------------------------------------------------- #

    def _read_reply(self, worker: _Worker, task_id: str) -> str:
        """Read one framed reply, discarding unframed noise.

        Runs with the worker's lock held.
        """
        stdout = worker.process.stdout
        if stdout is None:  # pragma: no cover
            raise WorkerCrashedError(f"worker {worker.worker_id!r} has no stdout")
        remaining = self.task_timeout
        while True:
            if remaining is None:
                line = stdout.readline()
            else:
                started = _monotonic()
                line = _with_timeout(stdout.readline, remaining)
                remaining -= _monotonic() - started
                if line is _TIMED_OUT or remaining <= 0:
                    self._kill(worker)
                    raise WorkerCrashedError(
                        f"worker {worker.worker_id!r} did not answer task "
                        f"{task_id!r} within {self.task_timeout}s; the worker was "
                        f"killed and the task recorded as unexecuted"
                    )
            if not line:
                raise WorkerCrashedError(
                    f"worker {worker.worker_id!r} died while executing task "
                    f"{task_id!r} (exit code {worker.process.poll()})"
                )
            text = str(line)
            if text.startswith(_REPLY_PREFIX):
                return text[len(_REPLY_PREFIX) :]
            # Unframed: library output, not a result. Never parsed as one.

    def run(
        self,
        lease: _Worker,
        task_id: str,
        harness_config: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute one task in ``lease``'s worker.

        Serialized per worker: one child runs one rollout at a time, which is
        what makes a worker's knowledge store single-writer and therefore
        lock-safe. Concurrency comes from having several workers.

        The worker is replaced once it has served ``max_rollouts_per_worker``
        rollouts. Recycling happens *before* the task is dispatched, never after,
        so a task is only ever sent to a child that is going to survive long
        enough to answer it -- recycling afterwards would risk killing a worker
        whose reply was still in flight.
        """
        request = json.dumps(
            {"task_id": task_id, "harness_config": dict(harness_config)},
            default=str,
        )
        with lease.lock:
            if lease.rollouts_served >= self.max_rollouts_per_worker:
                self._recycle(lease)
            process = lease.process
            if process.poll() is not None:
                raise WorkerCrashedError(
                    f"worker {lease.worker_id!r} is dead (exit code "
                    f"{process.poll()}); task {task_id!r} was not executed"
                )
            stdin = process.stdin
            if stdin is None:  # pragma: no cover
                raise WorkerCrashedError(
                    f"worker {lease.worker_id!r} has no usable pipes"
                )
            try:
                stdin.write(request + "\n")
                stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise WorkerCrashedError(
                    f"worker {lease.worker_id!r} closed its input before task "
                    f"{task_id!r} could be sent: {exc}"
                ) from exc
            line = self._read_reply(lease, task_id)
            # Counted inside the lock, and only for a dispatched task. A task
            # that could not be sent did not consume the worker's budget.
            lease.rollouts_served += 1
        return self._decode(line, task_id)

    def _recycle(self, worker: _Worker) -> None:
        """Replace a worker's child process in place, preserving its stores.

        Called with ``worker.lock`` held. The ``_Worker`` object itself survives
        because callers hold a lease on it; only the process is swapped, and the
        private knowledge/dbs directories are deliberately reused so the
        replacement has the same isolation identity as the child it replaces.

        Killing the process is what actually reclaims the memory. It also reaps
        anything the child leaked -- notably Playwright browsers, 12 of which
        outlived the 90 GB run.

        A failure to start the replacement is left to surface on the next
        ``poll()`` check as ``WorkerCrashedError``, which is already the pool's
        contract for a dead worker: one ``ok=False`` task rather than a crashed
        run.
        """
        self._kill(worker)
        stderr = self._log_sink.open_stream(worker.worker_id) or subprocess.DEVNULL
        worker.process = subprocess.Popen(
            [
                self.python_executable,
                "-u",
                "-m",
                "agent_evolve.benchmarks.cuga_process_pool",
            ],
            cwd=str(self.worker_cwd),
            env=self.worker_environment(worker.worker_id, worker.harness_version),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        worker.rollouts_served = 0
        self._await_ready(worker)

    def _decode(self, line: str, task_id: str) -> Mapping[str, object]:
        """Turn a worker's reply into a ``run_task`` result, or refuse.

        A reply that cannot be read is an execution failure, never an answer:
        the caller records ``ok=False`` and the task leaves the denominator.
        """
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerProtocolError(
                f"task {task_id!r}: worker reply is not valid JSON ({exc}): "
                f"{line.strip()[:300]!r}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise WorkerProtocolError(
                f"task {task_id!r}: worker returned {type(payload).__name__}, "
                f"not an object"
            )
        if payload.get("worker_error"):
            raise WorkerCrashedError(
                f"task {task_id!r} failed inside its worker: "
                f"{str(payload['worker_error'])[:400]}"
            )
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise WorkerProtocolError(
                f"task {task_id!r}: worker reply carried no 'result' object "
                f"(keys: {sorted(str(k) for k in payload)})"
            )
        return result

    # -- teardown --------------------------------------------------------- #

    def close(self) -> None:
        """Stop every worker, releasing its knowledge lock.

        Best-effort and idempotent: a run that produced its answers must not fail
        during cleanup.
        """
        with self._lock:
            workers, self._workers = list(self._workers), []
        for worker in workers:
            self._kill(worker)
        # After the children are gone: closing a log a live child still holds
        # would lose its final lines.
        self._log_sink.close()

    def _kill(self, worker: _Worker) -> None:
        process = worker.process
        if process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
        try:
            process.wait(timeout=10)
        except Exception:  # noqa: BLE001 - it is not going quietly
            process.kill()
            try:
                process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "CugaProcessPool":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# a bounded read that does not need a signal or an event loop
# --------------------------------------------------------------------------- #

_TIMED_OUT = object()


def _with_timeout(read: Any, timeout: float) -> Any:
    """Run a blocking read on a helper thread, giving up after ``timeout``.

    The helper thread is left behind on timeout, deliberately: the caller kills
    the child immediately afterwards, which unblocks the read and lets the thread
    exit. It is a daemon, so it can never hold up interpreter shutdown.
    """
    box: list[Any] = []

    def target() -> None:
        try:
            box.append(read())
        except Exception:  # noqa: BLE001 - reported as a dead worker by caller
            box.append(None)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return _TIMED_OUT
    return box[0] if box else None


# --------------------------------------------------------------------------- #
# the worker process
# --------------------------------------------------------------------------- #


def main() -> int:
    """Serve rollouts on stdin/stdout until the parent closes the pipe.

    Runs in the child. Builds exactly one traced CUGA wrapper -- the same
    ``CugaWrapper.from_cuga`` a serial run uses, so a worker is not a second
    implementation of "how we execute a task" -- and then answers one JSON
    request per line.

    **The protocol channel is a private duplicate of fd 1, and fd 1 itself is
    redirected to stderr.** This is not tidiness: CUGA and its dependencies print
    to stdout during import (observed: ``"Using file system assets (embedded
    assets disabled)"``), and a banner arriving on the protocol channel is
    indistinguishable from a reply. Left unhandled it either desynchronizes every
    subsequent task or, worse, gets parsed as a result. Redirecting fd 1 means any
    library that writes to stdout -- now or after a CUGA upgrade -- lands
    harmlessly in the parent's ``stderr`` sink instead.

    Every failure is reported as ``worker_error`` rather than as an empty answer,
    so the parent can keep "no measurement" distinct from "wrong answer".
    """
    # Claim the protocol channel before importing anything that might print.
    channel_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    channel = os.fdopen(channel_fd, "w", buffering=1)

    from agent_evolve.benchmarks.cuga_executor import (
        default_trace_config,
        prepare_environment,
        warm_up_cuga_imports,
    )

    def reply(payload: Mapping[str, object]) -> None:
        channel.write(_REPLY_PREFIX + json.dumps(payload, default=str) + "\n")
        channel.flush()

    try:
        prepare_environment()
        # Cheap here and paid once per worker, in parallel with its siblings.
        # It also removes any in-child import race, since CUGA's own internals
        # use threads.
        warm_up_cuga_imports()

        from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings

        trace_root = os.environ.get("AGENT_EVOLVE_TRACE_ROOT") or "data/traces"
        wrapper = CugaWrapper.from_cuga(
            RuntimeSettings.from_env(),
            trace_config=default_trace_config(trace_root),  # type: ignore[arg-type]
        )
        if not wrapper.supports_recorded_environment_replay():
            raise RuntimeError(
                "worker built a wrapper with causal tracing disabled; a rollout "
                "without a trace is evidence we cannot use"
            )
    except Exception as exc:  # noqa: BLE001 - the parent must see why
        # No ready sentinel: the parent raises WorkerStartError, and every task
        # routed here becomes ok=False rather than a fabricated answer.
        sys.stderr.write(f"worker startup failed: {type(exc).__name__}: {exc}\n")
        return 1

    channel.write(_READY + "\n")
    channel.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            task_id = str(request["task_id"])
            harness_config = request["harness_config"]
        except Exception as exc:  # noqa: BLE001
            reply({"worker_error": f"unreadable request: {type(exc).__name__}: {exc}"})
            continue
        try:
            result = wrapper.run_task(task_id, harness_config)
        except Exception as exc:  # noqa: BLE001 - one task, not the worker
            reply(
                {
                    "task_id": task_id,
                    "worker_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        reply({"task_id": task_id, "result": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
