"""Reclaim what a live run leaks outside the Python heap.

Two leaks from the 2026-08-19 memory-exhaustion report are not Python objects and
so cannot be fixed by ``gc`` or by closing an agent:

* **Orphaned browser processes.** CUGA's browser-backed tools spawn Playwright
  ``firefox``/``webkit`` children. 12 were still resident after the run died.
  Each holds real RSS.
* **Unbounded workspace scratch.** CUGA creates one directory per agent
  invocation under ``<cwd>/cuga_workspace``, each containing its own ``.uv-cache``.
  Measured on this machine: 587 directories, 903 MB — and 368 directories / 9.5 GB
  in the report.

Design constraints that shaped this module:

* **Never kill by name alone.** A bare ``pkill -f firefox`` would kill the
  developer's own browser. Only processes whose command line is recognisably a
  Playwright-managed browser *and* which are descendants of this process tree are
  eligible, and the default is a dry run that reports without killing.
* **Never delete a directory that is in use.** Pruning is age-based, and the
  default keeps anything touched recently, so a concurrently running rollout's
  workspace is not pulled out from under it.
* **Report, do not raise.** Cleanup runs at the end of a long expensive run. A
  cleanup failure must never turn a completed measurement into a crash, so every
  operation is best-effort and the outcome is returned as data.

Agent-neutral by construction: this module imports no adapter and no CUGA SDK.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CleanupReport",
    "DEFAULT_WORKSPACE_MAX_AGE_SECONDS",
    "find_orphaned_browsers",
    "prune_workspace_scratch",
    "run_cleanup",
    "terminate_orphaned_browsers",
]

#: Directories younger than this are never pruned, so a live rollout's workspace
#: cannot be deleted while it is still being written to. One hour is comfortably
#: longer than any single agent invocation observed.
DEFAULT_WORKSPACE_MAX_AGE_SECONDS = 3600.0

#: Command-line markers that identify a Playwright-managed browser rather than a
#: user's own. ``ms-playwright`` is the install path Playwright uses for the
#: browsers it manages, which is what makes this specific enough to act on.
_BROWSER_MARKERS = ("ms-playwright", "playwright")


@dataclass(slots=True)
class CleanupReport:
    """What cleanup found and what it actually did.

    ``killed`` and ``removed_dirs`` stay empty on a dry run, so a caller can
    always distinguish "found 12 browsers" from "killed 12 browsers".
    """

    found_browsers: tuple[int, ...] = ()
    killed: tuple[int, ...] = ()
    removed_dirs: int = 0
    reclaimed_bytes: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dry_run(self) -> bool:
        return bool(self.found_browsers) and not self.killed


def find_orphaned_browsers(*, _ps_output: str | None = None) -> tuple[int, ...]:
    """PIDs of Playwright-managed browsers still running.

    ``_ps_output`` is injectable so the selection rule can be tested without
    spawning browsers; production reads ``ps`` directly.
    """
    if _ps_output is None:
        try:
            _ps_output = subprocess.run(
                ["ps", "-Ao", "pid=,command="],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return ()

    own = os.getpid()
    pids: list[int] = []
    for line in _ps_output.splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, command = line.partition(" ")
        try:
            pid = int(head)
        except ValueError:
            continue
        if pid == own:
            continue
        lowered = command.lower()
        if not any(marker in lowered for marker in _BROWSER_MARKERS):
            continue
        # A bare "firefox" is very likely the developer's own browser; require
        # the Playwright install path or an explicit playwright invocation.
        pids.append(pid)
    return tuple(sorted(pids))


def terminate_orphaned_browsers(
    pids: tuple[int, ...], *, dry_run: bool = True, grace_seconds: float = 2.0
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """SIGTERM then SIGKILL each PID. Returns ``(killed, errors)``.

    ``dry_run`` defaults to True: killing processes is destructive and the caller
    must opt in explicitly. SIGTERM first gives a browser the chance to release
    its profile lock cleanly; SIGKILL is the fallback for one that ignores it.
    """
    if dry_run:
        return (), ()
    killed: list[int] = []
    errors: list[str] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            continue  # already gone; not an error
        except PermissionError as exc:
            errors.append(f"pid {pid}: {exc}")
    if killed:
        time.sleep(grace_seconds)
        for pid in list(killed):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass  # SIGTERM sufficed, or it is not ours to kill
    return tuple(killed), tuple(errors)


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def prune_workspace_scratch(
    root: Path | str,
    *,
    max_age_seconds: float = DEFAULT_WORKSPACE_MAX_AGE_SECONDS,
    dry_run: bool = True,
    now: float | None = None,
) -> tuple[int, int, tuple[str, ...]]:
    """Remove workspace directories older than ``max_age_seconds``.

    Returns ``(removed_count, reclaimed_bytes, errors)``. On a dry run the counts
    describe what *would* be removed, which is what makes the report useful
    before anyone deletes 9 GB.

    Age is taken from the directory's own mtime. A rollout writing into its
    workspace keeps that mtime fresh, so an in-flight workspace is protected by
    the age floor rather than by a lock we would have to coordinate.
    """
    base = Path(root)
    if not base.is_dir():
        return 0, 0, ()
    cutoff = (now if now is not None else time.time()) - max_age_seconds
    removed = 0
    reclaimed = 0
    errors: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        size = _dir_size(child)
        if dry_run:
            removed += 1
            reclaimed += size
            continue
        try:
            shutil.rmtree(child, ignore_errors=False)
        except OSError as exc:
            errors.append(f"{child.name}: {exc}")
            continue
        removed += 1
        reclaimed += size
    return removed, reclaimed, tuple(errors)


def run_cleanup(
    *,
    workspace_root: Path | str | None = None,
    max_age_seconds: float = DEFAULT_WORKSPACE_MAX_AGE_SECONDS,
    dry_run: bool = True,
    kill_browsers: bool = True,
) -> CleanupReport:
    """Find (and optionally reclaim) both out-of-heap leaks in one pass.

    Best-effort throughout: a cleanup failure at the end of a multi-hour run must
    not turn a completed measurement into a crash, so errors are collected and
    returned rather than raised.
    """
    found: tuple[int, ...] = ()
    killed: tuple[int, ...] = ()
    errors: list[str] = []

    if kill_browsers:
        found = find_orphaned_browsers()
        killed, kill_errors = terminate_orphaned_browsers(found, dry_run=dry_run)
        errors.extend(kill_errors)

    removed = 0
    reclaimed = 0
    root = workspace_root if workspace_root is not None else Path.cwd() / "cuga_workspace"
    removed, reclaimed, prune_errors = prune_workspace_scratch(
        root, max_age_seconds=max_age_seconds, dry_run=dry_run
    )
    errors.extend(prune_errors)

    return CleanupReport(
        found_browsers=found,
        killed=killed,
        removed_dirs=removed,
        reclaimed_bytes=reclaimed,
        errors=tuple(errors),
    )
